import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
import torch_scatter
import torch.nn.functional as F
from .resnet3d_discriminator import Discriminator3D
from .point_track import Cotracker

"""
Copy from https://github.com/XiYe20/VPTR/blob/main/utils/position_encoding.py
"""
class PositionEmbeddding1D(nn.Module):
    """
    1D position encoding
    Based on Attetion is all you need paper and DETR PositionEmbeddingSine class
    """
    def __init__(self, temperature = 10000, normalize = False, scale = None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, L: int, N: int, E: int):
        """
        Args:
            L for length, N for batch size, E for embedding size (dimension of transformer).

        Returns:
            pos: position encoding, with shape [L, N, E]
        """
        pos_embed = torch.ones(N, L, dtype = torch.float32).cumsum(axis = 1)
        dim_t = torch.arange(E, dtype = torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / E)
        if self.normalize:
            eps = 1e-6
            pos_embed = pos_embed / (L + eps) * self.scale

        pos_embed = pos_embed[:, :, None] / dim_t
        pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim = 3).flatten(2)
        pos_embed = pos_embed.permute(1, 0, 2)
        pos_embed.requires_grad_(False)
        
        return pos_embed

# create a toy trajectory dataset to test the discriminator
class ToyTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, vid_size = (16, 256, 256), num_points = 256, num_samples=1024):
        self.num_samples = num_samples
        self.vid_size = vid_size
        self.num_points = num_points

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        T, H, W = self.vid_size
        # All real trajectories start from the top left corner to the bottom right corner

        # step 2: randomly sample starting space grid point at frame t
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H//3),
            torch.arange(0, W//3),
        )
        query_s = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        query_s = rearrange(query_s, 'h w c -> (h w) c')
        indices = torch.randperm(query_s.shape[0])[:self.num_points]
        query_s = query_s[indices, :]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H//3 * 2, H),
            torch.arange(W//3 * 2, W),
        )
        query_e = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        query_e = rearrange(query_e, 'h w c -> (h w) c')
        indices = torch.randperm(query_e.shape[0])[:self.num_points]
        query_e = query_e[indices, :]
        
        # step3: connect the starting and ending points with a straight line
        queries = []
        direction = query_e - query_s
        for i in range(T):
            temp_query = query_s + direction * i / T
            #append the time step
            temp_query = torch.cat([torch.ones(temp_query.shape[0], 1) * i, temp_query], dim=-1)
            queries.append(temp_query)
        queries = torch.stack(queries, dim=0)
        # step4: we round all the spatial coordinates to the nearest integer
        queries = torch.round(queries).to(torch.long)

        real_tracks = queries
        real_visibility = torch.ones(real_tracks.shape[0], real_tracks.shape[1])
        query_index_real = real_tracks[0, ...].squeeze(0)

        example = {}
        example['real_tracks'] = real_tracks#[..., 1:]
        example['real_visibility'] = real_visibility
        example['query_index_real'] = query_index_real

        # all fake trajectories start from the bottom left corner to the top right corner, create fake tracks similarly
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H//3 * 2, H),
            torch.arange(W//3 * 2, W),
        )
        query_s = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        query_s = rearrange(query_s, 'h w c -> (h w) c')
        indices = torch.randperm(query_s.shape[0])[:self.num_points]
        query_s = query_s[indices, :]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H//3),
            torch.arange(0, W//3),
        )
        query_e = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        query_e = rearrange(query_e, 'h w c -> (h w) c')
        indices = torch.randperm(query_e.shape[0])[:self.num_points]
        query_e = query_e[indices, :]

        queries = []
        direction = query_e - query_s
        for i in range(T):
            temp_query = query_s + direction * i / T
            temp_query = torch.cat([torch.ones(temp_query.shape[0], 1) * i, temp_query], dim=-1)
            queries.append(temp_query)
        queries = torch.stack(queries, dim=0)
        # step4: we round all the spatial coordinates to the nearest integer
        queries = torch.round(queries).to(torch.long)

        fake_tracks = queries
        fake_visibility = torch.ones(fake_tracks.shape[0], fake_tracks.shape[1])
        query_index_fake = fake_tracks[0, ...].squeeze(0)

        example['fake_tracks'] = fake_tracks#[..., 1:]
        example['fake_visibility'] = fake_visibility
        example['query_index_fake'] = query_index_fake

        return example
    
    def visualize_example(self, example):
        from matplotlib import pyplot as plt

        real_tracks = example['real_tracks'] #(T, N, 3)
        real_visibility = example['real_visibility'] #(T, N)
        query_index_real = example['query_index_real'] #(N, 3)
        fake_tracks = example['fake_tracks'] #(T, N, 3)
        fake_visibility = example['fake_visibility'] #(T, N)
        query_index_fake = example['query_index_fake'] #(N, 3)

        # plot each real track in a 3d matplotlib figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for n in range(real_tracks.shape[1]):
            ax.plot(real_tracks[:, n, 0], real_tracks[:, n, 1], real_tracks[:, n, 2], color='green')
        
        # plot each fake track in a 3d matplotlib figure
        for n in range(fake_tracks.shape[1]):
            ax.plot(fake_tracks[:, n, 0], fake_tracks[:, n, 1], fake_tracks[:, n, 2], color='red')

        plt.savefig('real_and_fake_tracks.png')
        plt.close()

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, model_dim = 64, frame_size = [256, 256], temporal_size = 16, nfilter = 64,
                 backward_tracking=False, spatio_query_method='Random', time_query_method='First', 
                 max_num_queries=1024):#, feature_preprocess_method='bilinear_scatter'):
        
        super(TrajectoryDiscriminator, self).__init__()
        self.model_dim = model_dim

        # the input feature is (B, T, N, 64), backbone is a 3d resent
        assert frame_size[0] == frame_size[1], "frame_size must be a square for the backbone"
        self.backbone = Discriminator3D(in_channels=model_dim, in_frame_size=frame_size[0], temporal_size=temporal_size, num_classes=1, nfilter=nfilter)

        self.frame_size = frame_size
        embedding = self.create_embedding()
        self.register_buffer('point_embedding',embedding)

        # init the cotracker
        self.cotracker = Cotracker(backward_tracking=backward_tracking, 
                                   spatio_query_method=spatio_query_method, 
                                   time_query_method=time_query_method, 
                                   max_num_queries=max_num_queries)
        self.cotracker.eval()
        for p in self.cotracker.parameters():
            p.requires_grad_(False)
        
        # assert feature_preprocess_method in ['discrete_round', 'bilinear_scatter'], "feature_preprocess_method must be either 'discrete_round' or 'bilinear_scatter'"
        # self.feature_preprocess_method = feature_preprocess_method
        # if feature_preprocess_method == 'discrete_round':
        self.preprocess_batch = self.preprocess_batch_discrete_round
        # elif feature_preprocess_method == 'bilinear_scatter':
        #     self.preprocess_batch = self.preprocess_batch_bilinear_scatter

    def forward(self, forward_mode, vid, real=True):
        if forward_mode == 'g_step':
            cls_logits = self.forward_g_step(vid)
            return cls_logits
        elif forward_mode == 'd_step':
            if real:
                cls_logits, x = self.forward_d_step(vid, real)
                return cls_logits, x
            else:
                cls_logits = self.forward_d_step(vid, real)
                return cls_logits
        elif forward_mode == 'pretrain':
            cls_logits = self.forward_pretrain(vid)
            return cls_logits
        else:
           raise ValueError(f"Unsupported forward mode: {forward_mode}")
        
    def forward_g_step(self, vid):
        """Forward pass for GAN generator update step
        Args:
            vid: input video tensor with shape of (B T C H W)
        Returns:
            cls_logits: (B, 1), the logits for the discriminator
        """
        with torch.no_grad():
            self.cotracker.eval()
            pred_tracks, pred_visibility, _ = self.cotracker(vid)
            # pred_tracks = pred_tracks.to(vid.dtype) #enable this line for deepspeed training
        x = self.preprocess_batch(pred_tracks, pred_visibility)
        x = x.permute(0, 1, 4, 2, 3) #(B, T, model_dim, H, W)
        
        # we copy the gradient of x to the input vid tensor
        # vid has a shape of (B, T, 3, H, W), so \nabla_{vid} should have a shape of (B, T, 3, H, W)
        # But \nabla_{x} has a shape of (B, T, model_dim, H, W)
        # Here we average the \nabla_{x} along the model_dim dimension, results in a shape of (B, T, 1, H, W)
        # Then we copy the \nabla_{x} to each color channel of \nabla_{vid}, results in a gradient with shape of (B, T, 3, H, W)
        # we enable the specified gradient copy by reparametrization trick
        x = GradientCopyFn.apply(x, vid)
        # x.retain_grad()
        cls_logits = self.backbone(x)

        return cls_logits
        # return cls_logits, x
    
    def forward_d_step(self, vid, real=True):
        with torch.no_grad():
            self.cotracker.eval()
            pred_tracks, pred_visibility, _ = self.cotracker(vid)
            # pred_tracks = pred_tracks.to(vid.dtype) #enable this line for deepspeed training
        x = self.preprocess_batch(pred_tracks, pred_visibility)
        x = x.permute(0, 1, 4, 2, 3) #(B, T, model_dim, H, W)
        if real:
            # for later R1 gradient penalty
            x.requires_grad_(True)
            x.retain_grad()
        
        cls_logits = self.backbone(x)
        if real:
            return cls_logits, x
        return cls_logits
    
    def forward_pretrain(self, vid):
        with torch.no_grad():
            self.cotracker.eval()
            pred_tracks, pred_visibility, _ = self.cotracker(vid)
            # pred_tracks = pred_tracks.to(vid.dtype) #enable this line for deepspeed training
        x = self.preprocess_batch(pred_tracks, pred_visibility)
        x = x.permute(0, 1, 4, 2, 3) #(B, T, model_dim, H, W)
        cls_logits = self.backbone(x)

        return cls_logits

    def create_embedding(self):
        """
        Return:
            pos_emb: (H, W, E)
        """
        pos_emb_1d = PositionEmbeddding1D()
        H, W = self.frame_size
        pos_emb = pos_emb_1d(L = H*W, N=1, E=self.model_dim)
        pos_emb = pos_emb.squeeze(1)

        return pos_emb
    
    def preprocess_batch_discrete_round(self, p_track, p_visibility):
        """
        Filtering out the invalid tracks (out the range of the grid)
        Preparing the input for the PointTransformerV3
        Args:
            p_track: (B, T, N, 2), track points tensor, last dim is (x, y), i.e., width-dim (W) and height-dim (H)
            p_visibility: (B, T, N), visibility tensor
        
        Returns:
            feat: (B, T, H, W, model_dim), the feature tensor
        """
        # roudn the p_track to the nearest integer
        dtype = p_track.dtype
        p_track = torch.round(p_track).to(torch.long)

        B, T, N, _ = p_track.shape
        H, W = self.frame_size
        
        # if the point is out of the frame, we also set the visibility to 0
        in_frame = (p_track[:, :, :, 0] < W) & (p_track[:, :, :, 0] >= 0) & (p_track[:, :, :, 1] < H) & (p_track[:, :, :, 1] >= 0)
        in_frame = in_frame.to(torch.float)
        p_visibility = p_visibility * in_frame

        feat = torch.zeros(B, T, H, W, self.model_dim).to(p_track.device)

        # Generate random indices
        # Assuming random_idx is same as original implementation
        # Shape: (B, N)
        random_idx = torch.randperm(self.frame_size[0] * self.frame_size[1], device=p_track.device).unsqueeze(0).expand(B, -1)[:, :N]

        # Lookup embeddings: (B, N, model_dim)
        embeddings = self.point_embedding[random_idx]  # Shape: (B, N, model_dim)

        # Expand embeddings across time: (B, T, N, model_dim)
        embeddings = embeddings.unsqueeze(1).expand(-1, T, -1, -1)  # Shape: (B, T, N, model_dim)

        # Expand visibility: (B, T, N, 1)
        visibility = p_visibility.unsqueeze(-1)  # Shape: (B, T, N, 1)

        # Compute contributions: (B, T, N, model_dim)
        contributions = embeddings * visibility  # Shape: (B, T, N, model_dim)
        # Get spatial indices
        x_coords = p_track[:, :, :, 0].long()  # Shape: (B, T, N)
        y_coords = p_track[:, :, :, 1].long()  # Shape: (B, T, N)
        # Clamp indices to ensure they are within bounds
        x_coords = x_coords.clamp(0, W - 1)
        y_coords = y_coords.clamp(0, H - 1)
        # Flatten batch and time dimensions
        # New shape: (B*T*N, ...)
        contributions = contributions.view(B * T * N, -1)  # Shape: (B*T*N, model_dim)
        y_coords = y_coords.reshape(B * T * N)
        x_coords = x_coords.reshape(B * T * N)

        # Compute batch indices and time indices
        batch_indices = torch.arange(B, device=p_track.device).unsqueeze(1).unsqueeze(2).expand(B, T, N).reshape(-1)  # Shape: (B*T*N)
        time_indices = torch.arange(T, device=p_track.device).unsqueeze(0).unsqueeze(2).expand(B, T, N).reshape(-1)  # Shape: (B*T*N)

        # Initialize feat
        # Shape: (B, T, H, W, model_dim)
        # To perform scatter_add, we need to compute a linear index
        linear_indices = (batch_indices * T * H * W) + (time_indices * H * W) + (y_coords * W + x_coords)  # Shape: (B*T*N)
        # Reshape feat for scatter_add
        feat1_flat = feat.view(-1, self.model_dim)  # Shape: (B*T*H*W, model_dim)
        # Perform scatter_add_
        feat1_flat.index_add_(0, linear_indices, contributions)  # Accumulate contributions
        # Reshape back to original dimensions
        feat = feat1_flat.view(B, T, H, W, self.model_dim)

        # This is the original implementation using loop
        # feat1 = torch.zeros(B, T, H, W, self.model_dim).to(p_track.device)
        # for b in range(B):
        #     random_idx = torch.randperm(self.frame_size[0] * self.frame_size[1])[0:N]
        #     for n in range(N):
        #         for t in range(T):
        #             x, y = p_track[b, t, n, 1], p_track[b, t, n, 2]
        #             feat1[b, t, y, x, :] += self.point_embedding[random_idx[n]] * p_visibility[b, t, n]
        # Verify the difference
        # diff_mean = (feat - feat1).abs().mean()
        # diff_sum = (feat - feat1).abs().sum()
        # diff_max = (feat - feat1).abs().max()
        # diff_min = (feat - feat1).abs().min()
        # print('diff', diff_mean, diff_sum)
        # print('diff', diff_max, diff_min)
        feat = feat.to(dtype)
        return feat
    
    def preprocess_batch_bilinear_scatter(self, p_track, p_visibility):
        """
        Filtering out the invalid tracks (out the range of the grid)
        Preparing the input for the PointTransformerV3 with bilinear interpolation
        Args:
            p_track: (B, T, N, 2), track points tensor, last dim is (x, y), i.e., width-dim (W) and height-dim (H)
            p_visibility: (B, T, N), visibility tensor
        
        Returns:
            feat: (B, T, H, W, model_dim), the feature tensor
        """
        dtype = p_track.dtype
        B, T, N, _ = p_track.shape
        H, W = self.frame_size
        
        # Check if points are within frame bounds (with a small margin for interpolation)
        in_frame = (p_track[:, :, :, 0] < W-0.001) & (p_track[:, :, :, 0] >= 0.001) & \
                   (p_track[:, :, :, 1] < H-0.001) & (p_track[:, :, :, 1] >= 0.001)
        in_frame = in_frame.to(torch.float)
        p_visibility = p_visibility * in_frame
        
        # Initialize feature tensor
        feat = torch.zeros(B, T, H, W, self.model_dim, device=p_track.device)
        
        # Generate random indices for embeddings
        random_idx = torch.randperm(self.frame_size[0] * self.frame_size[1], device=p_track.device).unsqueeze(0).expand(B, -1)[:, :N]
        
        # Lookup embeddings: (B, N, model_dim)
        embeddings = self.point_embedding[random_idx]  # Shape: (B, N, model_dim)
        
        # Expand embeddings across time: (B, T, N, model_dim)
        embeddings = embeddings.unsqueeze(1).expand(-1, T, -1, -1)  # Shape: (B, T, N, model_dim)
        
        # Expand visibility: (B, T, N, 1)
        visibility = p_visibility.unsqueeze(-1)  # Shape: (B, T, N, 1)
        
        # Compute contributions: (B, T, N, model_dim)
        contributions = embeddings * visibility  # Shape: (B, T, N, model_dim)
        
        # Get floor coordinates for bilinear interpolation
        x0 = torch.floor(p_track[:, :, :, 0]).long()  # Shape: (B, T, N)
        y0 = torch.floor(p_track[:, :, :, 1]).long()  # Shape: (B, T, N)
        
        # Ensure coordinates are within bounds
        x0 = torch.clamp(x0, 0, W-2)  # Leave room for x1
        y0 = torch.clamp(y0, 0, H-2)  # Leave room for y1
        
        # Calculate x1, y1 (next integer coordinates)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Calculate interpolation weights
        wx1 = (p_track[:, :, :, 0] - x0.float())  # Shape: (B, T, N)
        wx0 = 1.0 - wx1
        wy1 = (p_track[:, :, :, 1] - y0.float())  # Shape: (B, T, N)
        wy0 = 1.0 - wy1
        
        # Reshape for broadcasting with contributions
        wx0 = wx0.unsqueeze(-1)  # Shape: (B, T, N, 1)
        wx1 = wx1.unsqueeze(-1)  # Shape: (B, T, N, 1)
        wy0 = wy0.unsqueeze(-1)  # Shape: (B, T, N, 1)
        wy1 = wy1.unsqueeze(-1)  # Shape: (B, T, N, 1)
        
        # Calculate weighted contributions for each corner
        w00 = wx0 * wy0  # top-left weight
        w01 = wx0 * wy1  # bottom-left weight
        w10 = wx1 * wy0  # top-right weight
        w11 = wx1 * wy1  # bottom-right weight
        
        # Compute contributions for each corner
        c00 = contributions * w00  # Shape: (B, T, N, model_dim)
        c01 = contributions * w01
        c10 = contributions * w10
        c11 = contributions * w11
        
        # Flatten batch and time dimensions for scatter operations
        c00_flat = c00.reshape(B * T * N, -1)  # Shape: (B*T*N, model_dim)
        c01_flat = c01.reshape(B * T * N, -1)
        c10_flat = c10.reshape(B * T * N, -1)
        c11_flat = c11.reshape(B * T * N, -1)
        
        # Flatten coordinate tensors
        x0_flat = x0.reshape(-1)  # Shape: (B*T*N)
        y0_flat = y0.reshape(-1)
        x1_flat = x1.reshape(-1)
        y1_flat = y1.reshape(-1)
        
        # Compute batch and time indices
        batch_indices = torch.arange(B, device=p_track.device).unsqueeze(1).unsqueeze(2).expand(B, T, N).reshape(-1)
        time_indices = torch.arange(T, device=p_track.device).unsqueeze(0).unsqueeze(2).expand(B, T, N).reshape(-1)
        
        # Compute linear indices for each corner
        idx00 = (batch_indices * T * H * W) + (time_indices * H * W) + (y0_flat * W + x0_flat)
        idx01 = (batch_indices * T * H * W) + (time_indices * H * W) + (y1_flat * W + x0_flat)
        idx10 = (batch_indices * T * H * W) + (time_indices * H * W) + (y0_flat * W + x1_flat)
        idx11 = (batch_indices * T * H * W) + (time_indices * H * W) + (y1_flat * W + x1_flat)
        
        # Reshape feat for scatter_add
        feat_flat = feat.reshape(-1, self.model_dim)  # Shape: (B*T*H*W, model_dim)
        
        # Add contributions to each corner using scatter_add
        feat_flat.index_add_(0, idx00, c00_flat)
        feat_flat.index_add_(0, idx01, c01_flat)
        feat_flat.index_add_(0, idx10, c10_flat)
        feat_flat.index_add_(0, idx11, c11_flat)
        
        # Reshape back to original dimensions
        feat = feat_flat.reshape(B, T, H, W, self.model_dim)
        
        # Convert to original dtype
        feat = feat.to(dtype)

        # sanity check
        # Implementation with loop
        # feat1 = torch.zeros(B, T, H, W, self.model_dim, device=p_track.device)
        # for b in range(B):
        #     for t in range(T):
        #         for n in range(N):
        #             if p_visibility[b, t, n] > 0:
        #                 # Get floating point coordinates
        #                 x, y = p_track[b, t, n, 0], p_track[b, t, n, 1]
                        
        #                 # Get integer coordinates of the four neighboring pixels
        #                 x0, y0 = int(torch.floor(x)), int(torch.floor(y))
        #                 x1, y1 = x0 + 1, y0 + 1
                        
        #                 # Ensure we're within bounds
        #                 x1 = min(x1, W-1)
        #                 y1 = min(y1, H-1)
                        
        #                 # Calculate interpolation weights
        #                 wx1 = x - x0
        #                 wx0 = 1 - wx1
        #                 wy1 = y - y0
        #                 wy0 = 1 - wy1
                        
        #                 # Weighted contribution to each of the four neighboring pixels
        #                 w00 = wx0 * wy0 * p_visibility[b, t, n]
        #                 w01 = wx0 * wy1 * p_visibility[b, t, n]
        #                 w10 = wx1 * wy0 * p_visibility[b, t, n]
        #                 w11 = wx1 * wy1 * p_visibility[b, t, n]
                        
        #                 # Add weighted embeddings to the four neighboring pixels
        #                 feat1[b, t, y0, x0] += embeddings[b, t, n] * w00
        #                 feat1[b, t, y1, x0] += embeddings[b, t, n] * w01
        #                 feat1[b, t, y0, x1] += embeddings[b, t, n] * w10
        #                 feat1[b, t, y1, x1] += embeddings[b, t, n] * w11
        
        return feat

# Define a custom autograd function for the gradient reparameterization
class GradientCopyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vid):
        """
        Forward pass: simply return x.
        """
        ctx.vid_shape = vid.shape  # Save original shape of vid for the backward pass
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        Backward pass:
          - grad_x: gradient of the loss with respect to point track representationx (shape: [B, T, model_dim, H, W])
          - We average grad_x along the model_dim channel (dim=2) -> shape becomes [B, T, 1, H, W].
          - Then we replicate the averaged gradient along the channel dimension to match vid (3 channels).
        """
        grad_vid = grad_x.mean(dim=2, keepdim=True)  # shape: [B, T, 1, H, W]
        grad_vid = grad_vid.expand(ctx.vid_shape)      # expand to [B, T, 3, H, W]
        return grad_x, grad_vid


if __name__ == '__main__':
    from accelerate import Accelerator
    import wandb
    import numpy as np
    from tqdm.auto import tqdm
    import argparse

    device = 'cuda:5'
    
    track_disc = TrajectoryDiscriminator().to(device)
    # generate random video with in range [0, 255]
    video = torch.randint(0, 256, (1, 16, 3, 256, 256)).to(device)
    vid = video / 255.0

    # vid = torch.randn(1, 16, 3, 256, 256).to(device)
    vid.requires_grad_(True)

    for p in track_disc.parameters():
        p.requires_grad_(False)

    cls_logits, x = track_disc.forward_g_step(vid)
    loss = -cls_logits.mean()

    loss.backward()
    print(vid.grad.shape)
    print(x.grad.shape)

    temp = x.grad.mean(dim=2, keepdim=True)
    diff = (vid.grad[:, :, 2:3, ...] - temp).abs()
    print(diff.mean(), diff.sum(), diff.max(), diff.min())
    
    # vid = vid.detach()
    # cls_logits, real_track_feature = track_disc.forward_d_step(vid, real=True)
    # loss = -cls_logits.mean()
    # loss.backward()
    # print(real_track_feature.grad.shape)



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 accelerate launch --num_processes 10 --main_process_port 29513 track_discriminator.py --epochs 100 --batch_size 20
# CUDA_VISIBLE_DEVICES=5,6,7,8,9 accelerate launch --num_processes 5 --main_process_port 29513 track_discriminator.py --epochs 100 --batch_size 80
