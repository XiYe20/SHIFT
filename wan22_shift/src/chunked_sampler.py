import random
from torch.utils.data import Sampler

class ChunkedSampler(Sampler):
    """
    Splits the dataset into N chunks of chunk_size each and yields them
    in sequence. On each epoch, the entire dataset is shuffled, re-chunked,
    then yielded again.

    This version also stores the list of indices for each chunk in `self.epoch_chunks`,
    so you can retrieve them later (e.g., in your model).

    The Chunk and Buffer Mechanism (on a single GPU):
    1. Given len(dataset) = N, batch_sie = B, buffer_size = bs, passes_per_buffer = ppb
    2. The dataset indices [0, 1, 2, ..., N-1] are split into N // bs chunks, each chunk has bs indices
    3. For each epoch, the number of training iterations is N//B. batch_idx keeps increasing from 0 to N//B-1
    4. Each chunk will be resampled (reshufflled) ppb times, i.e., requires bs/B*ppb (buffer_update_frequency) iterations to finish all the passes
    5. Then, we start to sample on next chunk, do the same ppb repeat sampling.
        !!! When ppb > 1, we cannot iterate every sample in one epoch!!!!
    6. Untill batch_idx == N//B, reach the end of one epoch.
    7. The number of chunks we sampled for one epoch is K = min(N//bs, 1+N // (bs/B*ppb))
       e.g., N =3700, B = 1, bs = 100, ppb = 10 -> num of chunks = N//bs = 37
       But each chunk will be repeated for 10 times, our total iterations of one epoch = 3700/1
       When batch_idx = 1000, we finished all passes of first chunk. -> ... -> batch_idx = 3700, we finished some of the fourth chunk.
       So, K = min(37, 3700//(100 / 1 * 10) + 1) = min(37, 4) = 4 -> we only iterates 4 chunks in one epoch.
       !!! The probability that one sample is sampled in one epoch is p = K/num_chunks = K/(N//bs)!!!
       After M epochs, the probability that one sample has been seen at least once is 1 - (1-p)^M = 1 - (1-K/(N//bs))^M
    8. Updated: with passes_per_epoch, we can iterate every sample in one epoch by equivalently repeating the whole dataset ppb times.
    """
    def __init__(self, data_source, chunk_size=1000, shuffle=True, passes_per_epoch: int = 1):
        super().__init__(data_source)
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.passes_per_epoch = passes_per_epoch

        # For simplicity, require perfect divisibility
        assert len(self.data_source) % chunk_size == 0, \
            "Dataset size must be divisible by num_chunks."
        self.num_chunks = len(self.data_source) // chunk_size

        # Initialize with an empty list
        self.epoch_chunks = []
        
        # Pre-initialize chunks with a basic partition to avoid the first-access error
        self._initialize_chunks()
        
    def _initialize_chunks(self):
        """Pre-initialize chunks with a basic ordering"""
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            random.shuffle(indices)
            
        self.epoch_chunks = []
        for chunk_idx in range(self.num_chunks):
            start = chunk_idx * self.chunk_size
            end = start + self.chunk_size
            self.epoch_chunks.append(indices[start:end])

    def __len__(self):
        """
        Return the total number of samples that will be yielded by this sampler.
        """
        return len(self.data_source) * self.passes_per_epoch
    
    def __iter__(self):
        """
        Yield indices in chunks.
        """
        for _ in range(self.passes_per_epoch):
            # Start with a fresh shuffling for this pass
            self._initialize_chunks()
            
            # Yield all indices in order of chunks
            for chunk in self.epoch_chunks:
                for idx in chunk:
                    yield idx

    def get_chunk_indices(self, chunk_id):
        """
        Return the list of indices belonging to the chunk_id-th chunk (0-based).
        """
        # If epoch_chunks is not initialized yet, initialize it
        if not self.epoch_chunks:
            self._initialize_chunks()
            
        return self.epoch_chunks[chunk_id]