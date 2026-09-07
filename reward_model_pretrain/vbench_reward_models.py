import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torchvision.transforms as transforms

# Add evaluation/vbench to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
vbench_dir = os.path.join(root_dir, 'evaluation', 'vbench')
if vbench_dir not in sys.path:
    sys.path.append(vbench_dir)

try:
    import clip
    from dreamsim import dreamsim
    from pyiqa.archs.musiq_arch import MUSIQ
    from vbench.aesthetic_quality import get_aesthetic_model
    from vbench2_beta_i2v.utils import init_submodules
except ImportError as e:
    print(f"Warning: Failed to import some VBench dependencies: {e}")

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

class VBenchAppearanceRewardModel(nn.Module):
    def __init__(self, device='cuda', cache_dir=None, torch_home=None):
        super().__init__()
        self.device = device
        
        # 1. Set environment variables explicitly if provided
        if cache_dir:
            os.environ['VBENCH_CACHE_DIR'] = cache_dir
            print(f"VBenchRewardModel: Set VBENCH_CACHE_DIR={cache_dir}")
            
        if torch_home:
            os.environ['TORCH_HOME'] = torch_home
            print(f"VBenchRewardModel: Set TORCH_HOME={torch_home}")

        # 2. Auto-configure from shared paths if NOT set in env (fallback)
        if 'VBENCH_CACHE_DIR' not in os.environ:
            shared_vbench_cache = "/path/to/vbench_cache_dir"
            if os.path.exists(shared_vbench_cache):
                os.environ['VBENCH_CACHE_DIR'] = shared_vbench_cache
                print(f"VBenchRewardModel: Auto-configured VBENCH_CACHE_DIR={shared_vbench_cache}")
        
        if 'TORCH_HOME' not in os.environ:
            shared_torch_home = "/path/to/custom_torch_home"
            if os.path.exists(shared_torch_home):
                os.environ['TORCH_HOME'] = shared_torch_home
                print(f"VBenchRewardModel: Auto-configured TORCH_HOME={shared_torch_home}")

        # Initialize submodules paths using vbench util
        # We dummy call init_submodules to get paths/download weights
        dims = ['subject_consistency', 'background_consistency', 'aesthetic_quality', 'imaging_quality', 'i2v_subject', 'i2v_background']
        # We need to suppress print/logging from init_submodules if possible, but it's fine
        self.submodules_paths = init_submodules(dims, local=False) # Use local=False to match evaluate_i2v.py defaults
        
        # 1. Subject Consistency & I2V Subject (DINO ViT-B/16)
        # They use the same model architecture but maybe different transforms
        self.dino_model = self._load_dino()
        
        # 2. Background Consistency (CLIP ViT-B/32)
        self.clip_b32_model = self._load_clip_b32()
        
        # 3. Aesthetic Quality (CLIP ViT-L/14 + Linear)
        self.clip_l14_model, self.aesthetic_linear = self._load_aesthetic()
        
        # 4. Imaging Quality (MUSIQ)
        self.musiq_model = self._load_musiq()
        
        # 5. I2V Background (DreamSim)
        self.dreamsim_model = self._load_dreamsim()
        
        # Define Transforms
        self.normalize_dino = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize_clip = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        
        # Weights for i2v score aggregation
        self.i2v_weights = {'max': 0.4, 'mean': 0.3, 'min': 0.3}

    def _load_dino(self):
        import sys, os, torch
        from pathlib import Path
        info = self.submodules_paths['subject_consistency']

        # 1) Temporarily drop project paths from sys.path
        orig_path = list(sys.path)
        sys.path = [p for p in sys.path if "MotionAlignmentVDM" not in p]

        # 2) If a project-level utils is cached, evict it
        mod = sys.modules.get("utils")
        if mod and getattr(mod, "__file__", "") and "MotionAlignmentVDM" in mod.__file__:
            del sys.modules["utils"]
        try:
            model = torch.hub.load(**info)
        finally:
            sys.path = orig_path  # restore after loading

        model.to(self.device)
        model.eval()
        return model

    def _load_clip_b32(self):
        path = self.submodules_paths['background_consistency'][0]
        model, _ = clip.load(path, device=self.device, download_root=os.path.join(os.environ.get("VBENCH_CACHE_DIR"), "clip") if os.environ.get("VBENCH_CACHE_DIR") else None)
        model.eval()
        return model

    def _load_aesthetic(self):
        vit_path = self.submodules_paths['aesthetic_quality'][0]
        aes_path = self.submodules_paths['aesthetic_quality'][1]
        
        clip_model, _ = clip.load(vit_path, device=self.device, download_root=os.path.join(os.environ.get("VBENCH_CACHE_DIR"), "clip") if os.environ.get("VBENCH_CACHE_DIR") else None)
        aesthetic_model = get_aesthetic_model(os.path.dirname(aes_path)) # get_aesthetic_model expects folder? No, cache_folder. 
        
        clip_model.eval()
        aesthetic_model.to(self.device)
        aesthetic_model.eval()
        return clip_model, aesthetic_model

    def _load_musiq(self):
        path = self.submodules_paths['imaging_quality']['model_path']
        model = MUSIQ(pretrained_model_path=path)
        model.to(self.device)
        model.eval()
        model.training = False
        return model

    def _load_dreamsim(self):
        # DreamSim handles its own caching
        cache_root = os.environ.get('VBENCH_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'vbench'))
        dreamsim_cache_dir = os.path.join(cache_root, 'dreamsim')
        model, _ = dreamsim(pretrained=True, cache_dir=dreamsim_cache_dir)
        model.to(self.device)
        model.eval()
        return model

    def freeze_params(self):
        for model in [self.dino_model, self.clip_b32_model, self.clip_l14_model, self.aesthetic_linear, self.musiq_model, self.dreamsim_model]:
            for param in model.parameters():
                param.requires_grad = False
                
    def forward(self, vid, ref_img=None):
        """
        Args:
            vid: (B, T, C, H, W) tensor, values in [0, 1]
            ref_img: (B, C, H, W) tensor, values in [0, 1]. If None, uses vid[:, 0]
        Returns:
            mean_score: (B,) tensor
            details: dict of individual scores
        """
        # Ensure inputs are on the correct device
        vid = vid.to(self.device)
        if ref_img is not None:
            ref_img = ref_img.to(self.device)
        else:
            ref_img = vid[:, 0] # Use first frame as reference if not provided

        B, T, C, H, W = vid.shape
            
        scores = {}
        
        # 1. Subject Consistency
        scores['subject_consistency'] = self.compute_subject_consistency(vid)
        
        # 2. Background Consistency
        scores['background_consistency'] = self.compute_background_consistency(vid)
        
        # 3. Aesthetic Quality
        scores['aesthetic_quality'] = self.compute_aesthetic_quality(vid)
        
        # 4. Imaging Quality
        scores['imaging_quality'] = self.compute_imaging_quality(vid)

        # 5. I2V Subject
        scores['i2v_subject'] = self.compute_i2v_subject(vid, ref_img)
        
        # 6. I2V Background
        scores['i2v_background'] = self.compute_i2v_background(vid, ref_img)
        
        # Calculate Mean
        total_score = 0
        valid_dims = 0
        for k, v in scores.items():
            total_score = total_score + v
            valid_dims += 1
            
        mean_score = total_score / valid_dims
        return mean_score, scores

    def _transform_images(self, images, size, norm_type='dino'):
        # images: (B, C, H, W) or (B*T, C, H, W)
        
        if norm_type == 'internet':
            # VBench dino_transform_internet: Resize(256), CenterCrop(224)
            out = transforms.Resize(256, interpolation=BICUBIC, antialias=False)(images)
            out = transforms.CenterCrop(224)(out)
            out = self.normalize_dino(out)
        elif norm_type == 'dino':
            # VBench dino_transform
            out = transforms.Resize(size, interpolation=BICUBIC, antialias=False)(images)
            out = transforms.CenterCrop(size)(out)
            out = self.normalize_dino(out)
        elif norm_type == 'clip':
            # VBench clip_transform
            out = transforms.Resize(size, interpolation=BICUBIC, antialias=False)(images)
            out = transforms.CenterCrop(size)(out)
            out = self.normalize_clip(out)
        elif norm_type == 'dreamsim':
            # VBench dreamsim_transform
            out = transforms.Resize((size, size), interpolation=BICUBIC, antialias=False)(images)
        elif norm_type == 'musiq':
            # VBench imaging_quality (uses default Bilinear, antialias=False)
            out_list = []
            for i in range(images.shape[0]):
                img = images[i]
                h, w = img.shape[1], img.shape[2]
                if max(h, w) > 512:
                    scale = 512. / max(h, w)
                    new_h, new_w = int(scale * h), int(scale * w)
                    img = transforms.Resize((new_h, new_w), antialias=False)(img)
                out_list.append(img)
            return out_list
            
        return out

    def compute_subject_consistency(self, vid):
        B, T, C, H, W = vid.shape
        flat_vid = vid.view(B * T, C, H, W)
        processed = self._transform_images(flat_vid, 224, 'dino')
        feats = self.dino_model(processed)  # (B*T, dim)
        feats = F.normalize(feats, dim=-1, p=2).view(B, T, -1)

        if T <= 1:
            return torch.zeros(B, device=self.device, dtype=feats.dtype)

        prev_feats = feats[:, :-1, :]
        curr_feats = feats[:, 1:, :]
        first_feats = feats[:, :1, :].expand(-1, T - 1, -1)

        sim_pre = F.cosine_similarity(prev_feats, curr_feats, dim=-1).clamp(min=0.0)
        sim_fir = F.cosine_similarity(first_feats, curr_feats, dim=-1).clamp(min=0.0)

        return ((sim_pre + sim_fir) * 0.5).mean(dim=1)

    def compute_background_consistency(self, vid):
        B, T, C, H, W = vid.shape
        flat_vid = vid.view(B * T, C, H, W)
        processed = self._transform_images(flat_vid, 224, 'clip')
        with torch.no_grad():
            feats = self.clip_b32_model.encode_image(processed)  # (B*T, dim)
            feats = F.normalize(feats, dim=-1, p=2).view(B, T, -1)

        if T <= 1:
            return torch.zeros(B, device=self.device, dtype=feats.dtype)

        prev_feats = feats[:, :-1, :]
        curr_feats = feats[:, 1:, :]
        first_feats = feats[:, :1, :].expand(-1, T - 1, -1)

        sim_pre = F.cosine_similarity(prev_feats, curr_feats, dim=-1).clamp(min=0.0)
        sim_fir = F.cosine_similarity(first_feats, curr_feats, dim=-1).clamp(min=0.0)

        return ((sim_pre + sim_fir) * 0.5).mean(dim=1)

    def compute_aesthetic_quality(self, vid):
        B, T, C, H, W = vid.shape
        flat_vid = vid.view(B*T, C, H, W)
        processed = self._transform_images(flat_vid, 224, 'clip')
        
        with torch.no_grad():
            feats = self.clip_l14_model.encode_image(processed).float()
            feats = F.normalize(feats, dim=-1, p=2)
            aes_scores = self.aesthetic_linear(feats).squeeze(-1) # (B*T,)
        
        aes_scores = aes_scores.view(B, T)
        normalized = aes_scores / 10.0
        return normalized.mean(dim=1)

    def compute_imaging_quality(self, vid):
        B, T, C, H, W = vid.shape
        flat_vid = vid.view(B * T, C, H, W)
        imgs_list = self._transform_images(flat_vid, None, 'musiq')
        imgs_batch = torch.stack(imgs_list, dim=0)

        with torch.no_grad():
            scores = self.musiq_model(imgs_batch).float()

        # Keep behavior equivalent to per-frame scalar averaging:
        # average per-frame MUSIQ over time, then normalize by 100.
        scores = scores.reshape(B, T, -1).mean(dim=-1)
        return scores.mean(dim=1) / 100.0

    def compute_i2v_subject(self, vid, ref_img):
        B, T, C, H, W = vid.shape
        
        ref_processed = self._transform_images(ref_img, 224, 'internet')
        flat_vid = vid.view(B * T, C, H, W)
        vid_processed = self._transform_images(flat_vid, 224, 'internet')
        with torch.no_grad():
            ref_feats = self.dino_model(ref_processed)  # (B, dim)
            ref_feats = F.normalize(ref_feats, dim=-1, p=2)
            vid_feats = self.dino_model(vid_processed)  # (B*T, dim)
            vid_feats = F.normalize(vid_feats, dim=-1, p=2).view(B, T, -1)

        conformity_scores = F.cosine_similarity(
            vid_feats, ref_feats.unsqueeze(1).expand(-1, T, -1), dim=-1
        ).clamp(min=0.0)  # (B, T)
        conformity_max = conformity_scores.max(dim=1).values  # (B,)

        if T <= 1:
            consec_mean = torch.full_like(conformity_max, float("nan"))
            consec_min = torch.full_like(conformity_max, float("nan"))
        else:
            consec_scores = F.cosine_similarity(vid_feats[:, :-1, :], vid_feats[:, 1:, :], dim=-1).clamp(min=0.0)  # (B, T-1)
            consec_mean = consec_scores.mean(dim=1)
            consec_min = consec_scores.min(dim=1).values

        return (
            self.i2v_weights['max'] * conformity_max
            + self.i2v_weights['mean'] * consec_mean
            + self.i2v_weights['min'] * consec_min
        )

    def compute_i2v_background(self, vid, ref_img):
        B, T, C, H, W = vid.shape
        
        ref_processed = self._transform_images(ref_img, 224, 'dreamsim')
        flat_vid = vid.view(B * T, C, H, W)
        vid_processed = self._transform_images(flat_vid, 224, 'dreamsim')
        with torch.no_grad():
            ref_feats = self.dreamsim_model.embed(ref_processed)  # (B, dim)
            ref_feats = F.normalize(ref_feats, dim=-1, p=2)
            vid_feats = self.dreamsim_model.embed(vid_processed)  # (B*T, dim)
            vid_feats = F.normalize(vid_feats, dim=-1, p=2).view(B, T, -1)

        conformity_scores = F.cosine_similarity(
            vid_feats, ref_feats.unsqueeze(1).expand(-1, T, -1), dim=-1
        ).clamp(min=0.0)  # (B, T)
        conformity_max = conformity_scores.max(dim=1).values  # (B,)

        if T <= 1:
            consec_mean = torch.full_like(conformity_max, float("nan"))
            consec_min = torch.full_like(conformity_max, float("nan"))
        else:
            consec_scores = F.cosine_similarity(vid_feats[:, :-1, :], vid_feats[:, 1:, :], dim=-1).clamp(min=0.0)  # (B, T-1)
            consec_mean = consec_scores.mean(dim=1)
            consec_min = consec_scores.min(dim=1).values

        return (
            self.i2v_weights['max'] * conformity_max
            + self.i2v_weights['mean'] * consec_mean
            + self.i2v_weights['min'] * consec_min
        )

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import cv2
    
    # Add parent directory to sys.path to allow importing from sibling modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from reward_model_pretrain.utils import DiscriminatorDataset
    except ImportError:
        # Fallback if running from within the directory
        from utils import DiscriminatorDataset

    parser = argparse.ArgumentParser()
    # Default paths from user's environment/previous context
    parser.add_argument("--real_dir", type=str, default="/path/to/Track4GenData/correct/valid/videos", help="Path to real videos")
    parser.add_argument("--fake_dir", type=str, default="/path/to/Track4GenData/correct/valid/fakevideos/t4gepoch0", help="Path to fake videos")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Initialize Reward Model
    reward_model = VBenchAppearanceRewardModel(
        device=device,
        cache_dir="/path/to/vbench_cache_dir",
        torch_home="/path/to/Track4Gen/torch_cache"
    )
    
    # 2. Initialize Dataset and DataLoader
    if not os.path.exists(args.real_dir) or not os.path.exists(args.fake_dir):
        print(f"Error: Video directories not found.\nReal: {args.real_dir}\nFake: {args.fake_dir}")
        sys.exit(1)

    print(f"Loading dataset from:\nReal: {args.real_dir}\nFake: {args.fake_dir}")
    
    # Use DiscriminatorDataset as requested
    dataset = DiscriminatorDataset(real_video_dir=args.real_dir, fake_video_dir=args.fake_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")

    # 3. Evaluation Loop
    real_scores_sum = 0.0
    fake_scores_sum = 0.0
    total_samples = 0

    
    # Metrics to track separately
    metrics_sum = {
        'real': {},
        'fake': {}
    }

    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # batch contains 'real_video' and 'fake_video': (B, T, C, H, W)
            real_vid = batch['real_video'].to(device)
            fake_vid = batch['fake_video'].to(device)
            
            # Compute rewards
            # Real Videos
            r_mean, r_details = reward_model(real_vid, ref_img=None)
            
            # Fake Videos (Use real video first frame as reference for i2v metrics)
            # Assuming paired data where real_vid corresponds to fake_vid
            f_mean, f_details = reward_model(fake_vid, ref_img=real_vid[:, 0])
            
            # Accumulate Mean Rewards
            batch_size = real_vid.shape[0]
            real_scores_sum += r_mean.sum().item()
            fake_scores_sum += f_mean.sum().item()
            total_samples += batch_size
            
            # Accumulate Detail Metrics
            for k, v in r_details.items():
                if k not in metrics_sum['real']: metrics_sum['real'][k] = 0.0
                metrics_sum['real'][k] += v.sum().item()
                
            for k, v in f_details.items():
                if k not in metrics_sum['fake']: metrics_sum['fake'][k] = 0.0
                metrics_sum['fake'][k] += v.sum().item()

    # 4. Print Results
    print("\n" + "="*50)
    print(f"Evaluation Results (N={total_samples})")
    print("="*50)
    
    print(f"{'Metric':<25} | {'Real Video':<10} | {'Fake Video':<10}")
    print("-" * 50)
    
    # Print Mean Reward
    avg_real = real_scores_sum / total_samples
    avg_fake = fake_scores_sum / total_samples
    print(f"{'Mean VBench Score':<25} | {avg_real:.4f}     | {avg_fake:.4f}")
    print("-" * 50)
    
    # Print Details
    all_keys = sorted(list(metrics_sum['real'].keys()))
    for k in all_keys:
        r_val = metrics_sum['real'][k] / total_samples
        f_val = metrics_sum['fake'][k] / total_samples
        print(f"{k:<25} | {r_val:.4f}     | {f_val:.4f}")
    
    print("="*50)
# ==================================================
# Evaluation Results (N=414)
# ==================================================
# Metric                    | Real Video | Fake Video
# --------------------------------------------------
# Mean VBench Score         | 0.8169     | 0.7701
# --------------------------------------------------
# aesthetic_quality         | 0.5181     | 0.4731
# background_consistency    | 0.9432     | 0.9408
# i2v_background            | 0.9725     | 0.8979
# i2v_subject               | 0.9657     | 0.8684
# imaging_quality           | 0.5931     | 0.5372
# subject_consistency       | 0.9087     | 0.9029
# ==================================================
    
