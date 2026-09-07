import torch
from torch.utils.data import Dataset
import os
from einops import rearrange
import cv2
import random
import numpy as np
from pathlib import Path

class DiscriminatorDataset(Dataset):
    def __init__(self, real_video_dir, fake_video_dir):
        """
        A PyTorch dataset for loading video mp4 files.
        For each real video, there are possible multiple fake videos.
        Thus, the dataset witll return a random fake video for each real video.
        Args:
            real_video_dir (str): Directory containing the real video mp4 files.
            fake_video_dir (str): Directory containing the fake video mp4 files.
            transform (callable, optional): A function/transform that takes in a tensor and returns a transformed version.
        """
        self.real_video_dir = real_video_dir
        self.fake_video_dir = fake_video_dir
        # get all the real video files
        self.real_video_files = [
            os.path.join(real_video_dir, fname) 
            for fname in os.listdir(real_video_dir) 
            if fname.endswith('.mp4')
        ]
        # get all the fake video files
        self.fake_video_files = [
            os.path.join(fake_video_dir, fname) 
            for fname in os.listdir(fake_video_dir) 
            if fname.endswith('.mp4')
        ]
        # Now pairing the real video files with the fake video files
        # fake video file example:/generated_videos/generated_d66b2502b834867555af871f77b3121358de3a0d1dcab237caf0ef10da9765fa_3.mp4
        # real video file example:real_videos/real_d66b2502b834867555af871f77b3121358de3a0d1dcab237caf0ef10da9765fa.mp4
        fake_index = {}
        for fp in self.fake_video_files:
            stem = Path(fp).stem              # e.g. "generated_<id>_3"
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            fake_id = parts[1]                # "<id>"
            fake_index.setdefault(fake_id, []).append(fp)

        self.file_list = []
        from tqdm import tqdm
        for real_video_file in tqdm(self.real_video_files):
            real_video_name = os.path.basename(real_video_file).split('.')[0]
            real_video_name = real_video_name.split('_')[1]
            fake_video_files = fake_index.get(real_video_name, [])
            if len(fake_video_files) == 0:
                continue

            self.file_list.append({
                "real_video": real_video_file,
                "fake_video": fake_video_files
            })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Return:
            real_video (tensor): A tensor of shape (t, c, h, w), with pixel values in [0, 1]
            fake_video (tensor): A tensor of shape (t, c, h, w), with pixel values in [0, 1]
        """
        file_path = self.file_list[idx]
        # Load the 'real_video' and 'fake_video'
        real_vid_path = self.file_list[idx]["real_video"]
        fake_vid_path = random.choice(self.file_list[idx]["fake_video"])
        real_video = self.read_video(real_vid_path)
        fake_video = self.read_video(fake_vid_path)
        real_video = torch.clip(real_video, 0, 1)
        fake_video = torch.clip(fake_video, 0, 1)

        return {"real_video": real_video, "fake_video": fake_video}

    def read_video(self, file_path):
        """
        Read videos from file path and return a tensor of shape (f, c, h, w)
        Args:
            file_path (str): The path to the video file.
        Returns:
            video (tensor): A tensor of shape (f, c, h, w)
        """
        cap = cv2.VideoCapture(file_path)
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(0, total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the current frame position
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                frames.append(frame)
            else:
                break
        cap.release()

        # Convert to a numpy array (T, H, W, C)
        video_np = np.array(frames)
        # Convert from (T, H, W, C) to (T, C, H, W) format
        video_np = np.transpose(video_np, (0, 3, 1, 2))
        # Convert to PyTorch tensor
        video_tensor = torch.from_numpy(video_np).float()
        video_tensor = video_tensor / 255.0  # Normalize to [0, 1]
        return video_tensor

if __name__ == "__main__":
    dataset = DiscriminatorDataset(real_video_dir="/path/to/MotionAlignmentVDM/reward_model_pretrain/fake_videos_generation/svd_examples_collision/real_videos", 
                                   fake_video_dir="/path/to/MotionAlignmentVDM/reward_model_pretrain/fake_videos_generation/svd_examples_collision/generated_videos")
    print(len(dataset))
    x = dataset.__getitem__(1)
    import pdb; pdb.set_trace()