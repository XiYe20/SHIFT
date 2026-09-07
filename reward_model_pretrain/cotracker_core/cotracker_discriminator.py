import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from .cotracker_source.predictor import CoTrackerPredictor
from .track_former import TrackFormer

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, pretrain_cotracker_ckpt_file='/path/to/cotracker_checkpoints/scaled_offline.pth',
                 spatio_query_method='Grid', time_query_method='First',
                 max_num_queries=1024, tracking_num_iters=3, track_former_input_dim=320, mlp_ratio=4.0):#, feature_preprocess_method='bilinear_scatter'):
        super(TrajectoryDiscriminator, self).__init__()
        self.spatio_query_method = spatio_query_method
        self.time_query_method = time_query_method
        self.max_num_queries = max_num_queries

        # init the cotracker
        self.cotracker = CoTrackerPredictor(
                checkpoint=pretrain_cotracker_ckpt_file,
                v2=False,
                offline=True,
                window_len=60,
            )
        
        # freeze all parameters of the cotracker
        self.cotracker.eval()
        for p in self.cotracker.parameters():
            p.requires_grad_(False)
        
        self.tracking_num_iters = tracking_num_iters  # or from kwargs
        # Infer defaults from the pretrained tracking updateformer
        uf = getattr(self.cotracker.model, "updateformer", None)
        assert uf is not None, "Expected self.cotracker.model.updateformer to exist"

        def _safe_get(obj, name, default=None):
            return getattr(obj, name, default)

        infer_hidden_size = _safe_get(uf, "hidden_size", 384)
        infer_num_heads = _safe_get(uf, "num_heads", 8)
        infer_num_virtual_tracks = _safe_get(uf, "num_virtual_tracks", 64)

        # depth: count blocks (robust)
        infer_time_depth = len(_safe_get(uf, "time_blocks", []))
        infer_space_depth = len(_safe_get(uf, "space_virtual_blocks", []))

        # add_space_attn: either explicit flag or presence of space blocks
        infer_add_space_attn = bool(_safe_get(uf, "add_space_attn", hasattr(uf, "space_virtual_blocks")))

        # Input dim: either user override, or use tracker model input_dim (matches track_former_input)
        infer_input_dim = getattr(self.cotracker.model, "input_dim", None)
        assert infer_input_dim is not None, "Expected self.cotracker.model.input_dim to exist"

        input_dim = track_former_input_dim
        # Init TrackFormer using tracker-matched defaults (except input_dim)
        self.track_former = TrackFormer(
            input_dim=input_dim,
            space_depth=int(infer_space_depth),
            time_depth=int(infer_time_depth),
            hidden_size=int(infer_hidden_size),
            num_heads=int(infer_num_heads),
            mlp_ratio=float(mlp_ratio),
            num_virtual_tracks=int(infer_num_virtual_tracks),
            add_space_attn=bool(infer_add_space_attn),
        )
        
    def forward(self, vid, queries_point=None):
        return self.cotracker_forward(vid, queries_point)
        
    def cotracker_forward(self, x, queries_point=None, visualize_filename=None, visualize_dir='/path/to/dev_dir'):
        """
        Args:
            x: input video tensor with shape of (B T C H W), pixel value range: [0,1]
        Returns:
            cls_logits: the logits for the discriminator, tensor with shape of (B, 1)
        """
        # assert that x is in range of [0, 1]
        assert x.min() >= 0 and x.max() <= 1, f"input video for cotracker is not in range of [0, 1], min: {x.min()}, max: {x.max()}"
        # convert x to range of [0, 255]
        x = x * 255
        # generate the queries
        if queries_point is not None:
            # use the provided queries point
            queries = queries_point
        else:
            queries = self.__queries_generate__(x)
        queries = queries.to(x.dtype)
        tracks, visibilities, confidence, track_former_input = self.cotracker(x, queries=queries, iters=self.tracking_num_iters) 
        # visualize_filename = 'track_examples'
        if visualize_filename is not None:
            if getattr(self, 'example_id', None) is None:
                self.example_id = 0
            else:
                self.example_id += 1
            visualize_filename = f'{self.example_id}_{visualize_filename}'
            # same visualizer used in point_track.py
            from .cotracker_source.utils.visualizer import Visualizer
            vis = Visualizer(save_dir=visualize_dir, pad_value=0, linewidth=2, tracks_leave_trace=-1)
            # Visualizer expects video in [0,255] (float/uint8 both ok) and tracks (B,T,N,2), vis (B,T,N)
            H, W = x.shape[-2], x.shape[-1]
            interp_H, interp_W = self.cotracker.interp_shape  # (H, W) of model resolution
            scale = tracks.new_tensor([(W - 1) / (interp_W - 1), (H - 1) / (interp_H - 1)])
            print(interp_H, interp_H, W, H)
            tracks_vis = tracks * scale  # (B,T,N,2)

            vis.visualize(x, tracks_vis, visibilities, filename=visualize_filename)

        # track_former_input is already shaped for TrackFormer: (B, N, T, D)
        cls_logits = self.track_former(track_former_input)  # (B,1)

        return cls_logits, queries
    
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
        if self.spatio_query_method == 'Grid':
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

if __name__ == '__main__':
    from accelerate import Accelerator
    import wandb
    import numpy as np
    from tqdm.auto import tqdm
    import argparse

    device = 'cuda:5'
    
    track_disc = TrajectoryDiscriminator(train_updateformer=True).to(device)

    #print total number of parameters
    print('total parameters:', sum(p.numel() for p in track_disc.parameters()))
    #print number of trainable parameters
    print('trainable parameters:', sum(p.numel() for p in track_disc.parameters() if p.requires_grad))
    # generate random video with in range [0, 255]
    video = torch.randint(0, 256, (1, 16, 3, 320, 576)).to(device)
    vid = video / 255.0

    # vid = torch.randn(1, 16, 3, 256, 256).to(device)
    vid.requires_grad_(True)

    cls_logits = track_disc.forward_g_step(vid)
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