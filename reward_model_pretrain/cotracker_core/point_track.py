import torch
import torch.nn as nn
from einops import repeat, rearrange
import cv2
import numpy as np
from pathlib import Path
from cotracker_source.predictor import CoTrackerPredictor

class Cotracker(nn.Module):
    def __init__(self, backward_tracking=False, spatio_query_method='Random', time_query_method='First', max_num_queries=1024):
        super().__init__()
        # self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.model = CoTrackerPredictor(
                checkpoint='/path/to/cotracker_checkpoints/scaled_offline.pth',
                v2=False,
                offline=True,
                window_len=60,
                train_updateformer=True
            )
        
        self.model.eval()
        self.model.to(torch.float16)
        self.backward_tracking = backward_tracking
        self.spatio_query_method = spatio_query_method
        assert self.spatio_query_method in ['Random', 'Grid'], f"Unsupported spatio query method: {self.spatio_query_method}"
        self.max_num_queries = max_num_queries
        if self.spatio_query_method == 'SIFT':
            self.sift = cv2.SIFT_create(nfeatures=max_num_queries)

        self.time_query_method = time_query_method
        assert self.time_query_method in ['First', 'Last', 'Random'], f"Unsupported time query method: {self.time_query_method}"

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, x, visualize_filename=None):
        """
        Args:
            x: input video tensor with shape of (B T C H W), pixel value range: [0,1]
        Returns:
            pred_tracks: the predicted tracks of the query points, tensor with shape of (B T N 2), where N is the number of query points
                !!! Last dim of pred_tracks is [x, y], where x is "W-dim and y is H-dim"!!!!
            pred_visibility: the predicted visibility of the query points, tensor with shape of (B T N)
            queries: the queries for the cotracker, tensor with shape of (B N 3), where N is the number of query points;
                !!!Last dim of queries is [t, x, y], where x is "W-dim and y is H-dim"!!!!
        """
        # assert that x is in range of [0, 1]
        assert x.min() >= 0 and x.max() <= 1, f"input video for cotracker is not in range of [0, 1], min: {x.min()}, max: {x.max()}"
        # convert x to range of [0, 255]
        x = x * 255
        # generate the queries
        queries = self.__queries_generate__(x)
        queries = queries.to(x.dtype)
        # pred_tracks, pred_visibility = self.model(x, grid_size=20, backward_tracking=self.backward_tracking, grid_query_frame=8) # B T N 2,  B T N 1
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred_tracks, pred_visibility = self.model(x, queries=queries, backward_tracking=self.backward_tracking) # B T N 2,  B T N 1
        if visualize_filename is not None:
            self.visualize_tracks(x, pred_tracks, pred_visibility, visualize_filename)
        # pred_tracks = pred_tracks.to(x.dtype)
        return pred_tracks, pred_visibility, queries

    def __queries_generate__(self, vid):
        """Generate the qureies for the cotracker
        # the queries coordinates (for spatial_query_method == 'Random') are also the index of queries point embedding
        Args:
            vid: input video tensor with shape of (B T C H W), pixel value range: [0,255]
        Returns:
            queries: the queries for the cotracker, tensor with shape of (B N 3), where N is the number of query points
            Last dim of queries is [t, x, y], where x is "W-dim and y is H-dim"!!!!
        """
        # generate the query time
        B, T, C, H, W = vid.shape
        
        if self.spatio_query_method == 'SIFT':
            # indeed, we only need to localize the query points, no need for th SIFT descriptor
            # SIFT algorithm takes the DoG (Difference of Gaussian) method to detect the key points
            # Not suitable for the training of track discriminator
            query_frames = vid[torch.arange(B), query_t, ...].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            query_s = []
            for i in range(query_frames.shape[0]):
                img = cv2.cvtColor(query_frames[i, ...], cv2.COLOR_RGB2GRAY)
                # detect SIFT key points
                key_points = self.sift.detect(img, None)
                key_points = sorted(key_points, key=lambda x: -x.response)[:self.max_num_queries]
                key_points = np.array([[kp.pt[0], kp.pt[1]] for kp in key_points])
                print(key_points.shape)
                query_s.append(key_points)
            
            # take the minimum number of key points among all the frames
            min_num_key_points = min([key_points.shape[0] for key_points in query_s])
            query_s = np.array([key_points[:min_num_key_points, :] for key_points in query_s])
            query_s = torch.from_numpy(query_s).to(vid.device).to(torch.float32)
        
        elif self.spatio_query_method == 'Edge_Weighted':
            # We aim to track the points on the edge of the first frame
            # 1. Get the temporal difference between the first frame and the second frame
            frame_diff = torch.abs(vid[:, 10, ...] - vid[:, 11, ...]) #(B, C, H, W)
            frame_diff = frame_diff.sum(dim=1) #(B, H, W)
            frame_diff = rearrange(frame_diff, 'b h w -> b (w h)')

            # 2. Normalize the frame_diff, to make the sum of frame_diffs to 1
            frame_diff = frame_diff / frame_diff.sum(dim=-1, keepdim=True) #(B, H*W)

            # 3. Get the grid_y and grid_x
            extent = (H, W)
            range_y = (0, extent[0])
            range_x = (0, extent[1])
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, H, device=vid.device),
                torch.arange(0, W, device=vid.device),
            )
            query_s = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2) #(h, w, 2)
            assert self.max_num_queries <= H*W, "max_num_queries should be less than HxW"
            query_s = repeat(query_s, 'h w c -> b (h w) c', b=B)

            # 4. Randomly sample max_num_queries points from query_s
            # Using torch.multinomial with frame_diff as the sampling probability:
            # For each batch we sample max_num_queries indices from the flattened grid
            sampled_indices = torch.multinomial(frame_diff, num_samples=self.max_num_queries, replacement=False)  # (B, max_num_queries)
            print(sampled_indices.shape, 'sample_indices')
            sampled_edge_points = query_s[torch.arange(B), sampled_indices, :]
            print(sampled_edge_points.shape, 'sampled_edge_points')
            # Now, gather the corresponding grid points from query_s.
            # We need to expand sampled_indices to have the same last dimension as query_s (which is 2 for x,y coordinates)
            # sampled_edge_points = torch.gather(query_s, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, 2))  # (B, max_num_queries, 2)
            query_s = sampled_edge_points


        elif self.spatio_query_method == 'Grid':
            # official spatial grid query method of CoTracker, Not suitable for the training of track discriminator
            # query_s = get_points_on_a_grid(size=int(self.max_num_queries**0.5), extent=(H, W), device=vid.device)
            # modified from https://github.com/facebookresearch/co-tracker/blob/main/cotracker/models/core/model_utils.py#L83
            extent = (H, W)
            center = [extent[0] / 2, extent[1] / 2]
            margin = W / 64
            range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
            range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(*range_y, int(self.max_num_queries**0.5), device=vid.device),
                torch.linspace(*range_x, int(self.max_num_queries**0.5), device=vid.device),
                indexing="ij",
            )
            #note that here grid for W-dim/grid_x is placed before grid for H-dim/grid_y
            query_s = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
            query_s = repeat(query_s, 'h w c -> b (h w) c', b=B)

        elif self.spatio_query_method == 'Random':
            # randomly select max_num_queries points from the dense grid, which is the default setting for training
            extent = (H, W)
            range_y = (0, extent[0])
            range_x = (0, extent[1])
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, H, device=vid.device),
                torch.arange(0, W, device=vid.device),
            )
            query_s = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
            query_s = rearrange(query_s, 'h w c -> (h w) c')
            # randomly select max_num_queries points
            assert self.max_num_queries <= query_s.shape[0], "max_num_queries should be less than HxW"
            indices = torch.randperm(query_s.shape[0])[:self.max_num_queries]
            query_s = query_s[indices, :]
            query_s = repeat(query_s, 'n c -> b n c', b=B).to(torch.float32)
        else:
            raise NotImplementedError(f"Unsupported spatio query method: {self.spatio_query_method}")
        
        # generate the query spatial coordinates
        if self.time_query_method == 'First':
            query_t = torch.zeros(1, dtype=torch.long)
            query_t = repeat(query_t, '1 -> b n 1', b=B, n=query_s.shape[1])
        elif self.time_query_method == 'Last':
            query_t = torch.ones(1, dtype=torch.long)*(T-1)
            query_t = repeat(query_t, '1 -> b n 1', b=B, n=query_s.shape[1])
        elif self.time_query_method == 'Random':
            # randomly select B*query_s.shape[1] time points from the range (0, T)
            query_t = torch.randint(0, T, (B*query_s.shape[1],))
            query_t = rearrange(query_t, '(b n) -> b n 1', b=B, n=query_s.shape[1])
        else:
            raise NotImplementedError(f"Unsupported time query method: {self.time_query_method}")
        query_t = query_t.to(vid.device)

        #concatenate the time and spatial queries
        queries = torch.cat([query_t, query_s], dim=-1)

        return queries
    
    def visualize_tracks(self, vid, pred_tracks, pred_visibility, filename='point_track_example'):
        from cotracker_source.utils.visualizer import Visualizer
        vis = Visualizer(save_dir='./', pad_value=0, linewidth=2, tracks_leave_trace=-1)
        vis.visualize(vid, pred_tracks, pred_visibility, filename=filename)
    
class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.files = list(Path(data_root).glob('*.pt'))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Returns:
            example: a dictionary contains the track information,
            {'real_tracks': (B T N 2), 'real_visibility': (B T N) 'query_index_real': (B N 3)': , 
            'fake_tracks': (B T N 2), 'fake_visibility': (B T N), 'query_index_fake': (B  N 3)}
        """
        file_path = self.files[idx]
        example = torch.load(file_path, map_location='cpu')
        for k, v in example.items():
            example[k] = v.cpu()
        return example
    
if __name__ == '__main__':
    # get_points_on_a_grid(20, (240, 320))
    import imageio.v3 as iio

    device = 'cuda:9'
    frames = iio.imread('/path/to/Dataset/UCF-101/ApplyLipstick/v_ApplyLipstick_g01_c01.avi', plugin="FFMPEG")  # plugin="pyav"
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)[:, 0:16, ...]  # B T C H W
    print('video shape', video.shape, video.min(), video.max())
    video = video/255.0

    cotrac = Cotracker(time_query_method='First', spatio_query_method='Random', max_num_queries=800).to(device)
    pred_tracks, pred_visibility, _ = cotrac(video)
    print(pred_tracks[..., 0].min(), pred_tracks[..., 0].max())
    print(pred_tracks[..., 1].min(), pred_tracks[..., 1].max())
    print(video.shape, pred_tracks.shape, pred_visibility.shape)
    cotrac.visualize_tracks(video, pred_tracks, pred_visibility, 'point_track1')

    pred_tracks, pred_visibility, queries = cotrac(video)
    print(pred_tracks.shape)
    print(pred_tracks.min(), pred_tracks.max())
    cotrac.visualize_tracks(video, pred_tracks, pred_visibility, 'point_track2')
    print(queries[:, :, 1].max(), queries[:, :, 2].max())

    # track_set = TrackDataset('/path/to/PhysicsVidDiff/point_track_outputs_random_1024')
    # track_loader = torch.utils.data.DataLoader(track_set, batch_size=8, shuffle=True)
    # for batch in track_loader:
    #     for k, v in batch.items():
    #         print(k, v.shape)
        
    #     break