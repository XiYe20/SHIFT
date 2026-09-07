from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.loaders.unet_loader_utils import _maybe_expand_lora_scales
from peft import LoraConfig, set_peft_model_state_dict
import torch
import imageio.v3 as iio
import os
import numpy as np
from torchvision import transforms
import argparse
import multiprocessing as mp
from edm_ancestral_scheduler import set_edm_ancestral
import cv2
from pathlib import Path
import re

# For the native SVD UNet class
_SET_ADAPTER_SCALE_FN_MAPPING.setdefault("UNetSpatioTemporalConditionModel", _maybe_expand_lora_scales)
# For the wrapped version, if you're using attach_peft_mixin that changes the class name
_SET_ADAPTER_SCALE_FN_MAPPING.setdefault("PeftWrappedUNetSpatioTemporalConditionModel", _maybe_expand_lora_scales)

def read_video(vid_path, target_size, target_fps=7, match_target_fps=True):
    """Read video file into a pytorch tensor, with shape (T, C, H, W), pixel range [0, 1]
    Args:
        vid_path: path to the video file
        target_size: the target size of the video, tuple of (T, H, W)
        target_fps: target frame rate (default 7)
        match_target_fps: whether to match the target fps of the video (default True)
    Returns:
        video_tensor: pytorch tensor of shape (T, C, H, W), pixel range [0, 1]
    """
    # check if the video is a mp4 file
    if vid_path.endswith('.mp4'):
        print(f"Reading video: {vid_path}")
        cap = cv2.VideoCapture(vid_path)
        
        # Get the original frame rate (FPS) of the video
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the number of frames to skip and ensure it is at least 1
        if match_target_fps:
            frame_skip = max(int(original_fps // target_fps), 1)
        else:
            frame_skip = 1

        print(f"Original FPS: {original_fps}, Target FPS: {target_fps}, Frame skip: {frame_skip}")
        
        frames = []
        # Skip frames, taking one frame every `frame_skip` frames
        for i in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the current frame position
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # <<< convert to RGB here
                frames.append(frame)
            else:
                break
        cap.release()

    # Convert to a numpy array (T, H, W, C)
    video_np = np.array(frames)
    assert video_np.ndim == 4, "Video must have 4 dimensions (T, H, W, C)"
    
    # Convert from (T, H, W, C) to (T, C, H, W) format
    video_np = np.transpose(video_np, (0, 3, 1, 2))
    
    # Convert to PyTorch tensor
    video_tensor = torch.from_numpy(video_np).float()
    video_tensor = video_tensor / 255.0  # Normalize to [0, 1]
    
    # Keep the target number of frames (target_size[0])
    if video_tensor.shape[0] < target_size[0]:
        print(f"Video {vid_path} has less than target_size[0] frames, Skip this real video")
        return None


    #新resize逻辑
    # Resize and center crop to target height and width
    target_h, target_w = target_size[1], target_size[2]

    # if the spatio ratio of target_h/target_w does not match the original spatio ratio of the video, then do center crop to the target size
    if video_tensor.shape[2] / float(video_tensor.shape[3]) != target_h / float(target_w):
        # Create transform pipeline for resizing and cropping
        scale_h = float(target_h) / float(video_tensor.shape[2])
        scale_w = float(target_w) / float(video_tensor.shape[3])
        scale_factor = max(scale_h, scale_w)

        # Calculate intermediate size
        intermediate_h = int(round(video_tensor.shape[2] * scale_factor))
        intermediate_w = int(round(video_tensor.shape[3] * scale_factor))

        # Create transform pipeline
        resize_transform = transforms.Compose([
            transforms.Resize((intermediate_h, intermediate_w), antialias=True),
            transforms.CenterCrop((target_h, target_w)),
        ])
    else:
        resize_transform = transforms.Resize((target_h, target_w), antialias=True)

    
    # Apply the resize transform to each frame
    resized_frames = [resize_transform(frame).unsqueeze(0) for frame in video_tensor]
    video_tensor = torch.cat(resized_frames, dim=0)

    # segment into target_size[0] frames
    num_clips = video_tensor.shape[0] // target_size[0]
    if num_clips == 0:
        return None
    clips = []
    for i in range(num_clips):
        clips.append(video_tensor[i*target_size[0]:(i+1)*target_size[0]])

    return clips


def attach_peft_mixin(model):
    """
    Make an *already-instantiated* UNet gain the PEFT/LoRA API
    (add_adapter, set_adapter, enable_adapters, …) without touching
    the original source code.

    Returns the same object, now with the extra methods.
    """
    # 1.  Do nothing if it already has the mixin
    if isinstance(model, PeftAdapterMixin):
        return model

    # 2.  Create a new class that merges the current class + mixin
    NewCls = type(
        f"PeftWrapped{model.__class__.__name__}",
        (model.__class__, PeftAdapterMixin),
        {}
    )

    # 3.  Change the instance's __class__ to the new one
    model.__class__ = NewCls
    return model


def load_sft_lora(pipeline, lora_ckpt_path, lora_scale=1.0):
    """
    Load SFT LoRA weights into the SVD pipeline
    """
    if lora_ckpt_path is None:
        return pipeline

    lora_ckpt_path = Path(lora_ckpt_path)
    if not lora_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {lora_ckpt_path}")

    # Load the lora ckpt
    try:
        checkpoint = torch.load(lora_ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {lora_ckpt_path}: {e}")

    pipeline.unet = attach_peft_mixin(pipeline.unet)

    train_cfg = checkpoint.get("config_dict", None)
    if train_cfg is None or "lora_config" not in train_cfg:
        raise ValueError("Checkpoint is missing 'config_dict.lora_config'; cannot rebuild LoRA adapter.")

    lora_config = dict(train_cfg["lora_config"])  # shallow copy
    unet_lora_config = LoraConfig(**lora_config)

    adapter_name = train_cfg.get("lora_adapter_name")
    pipeline.unet.add_adapter(unet_lora_config, adapter_name=adapter_name)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

        # Extract LoRA weights (remove 'unet.' prefix)
        lora_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unet.'):
                # Remove the 'unet.' prefix to get the LoRA parameter name
                lora_key = key[5:]  # Remove 'unet.' prefix
                lora_state_dict[lora_key] = value

        # Load the LoRA weights into the model
        if lora_state_dict:
            set_peft_model_state_dict(pipeline.unet, lora_state_dict, adapter_name=adapter_name)
            print(f"Successfully loaded {len(lora_state_dict)} LoRA parameters from checkpoint: {lora_ckpt_path}")
        else:
            print(f"Warning: No LoRA weights found in checkpoint: {lora_ckpt_path}")
            return pipeline
    else:
        print(f"Warning: No state_dict found in checkpoint: {lora_ckpt_path}")
        return pipeline

    pipeline.unet.set_adapters([adapter_name], weights=[lora_scale])
    print(f"SFT LoRA loaded with scale {lora_scale}")
    return pipeline


@torch.no_grad()
def generate_svd_example(svd_pipeline, output_dir, real_video_dir, target_size, target_fps=7, match_target_fps=True, num_inference_steps=25, noise_aug_strength=0.02, num_fake_example_per_real_video=4, video_files=None):
    """
    Generate example videos using SVD. Take the first frame of the real video example as input.
    Args:
        svd_pipeline: the SVD pipeline
        output_dir: the directory to save the example videos
        real_video_dir: the path to the real video, each video should be saved as a video file
        target_size: the target size of the video, tuple of (T, H, W)
        num_fake_example_per_real_video: number of examples to generate per video
        video_files: list of specific video files to process (if None, process all in real_video_dir)
    Saved:
        example_{real_vid}_{i}.pt: the example video, which contains the real video and the fake video
        Value range:
            real_video: [0, 1]
            fake_video: [0, 1]
    """
    # print(f"save_pt_files is set to: {save_pt_files}")

    # create the output directory if not exists
    real_vid_output_dir = os.path.join(output_dir, "real_videos")
    if not os.path.exists(real_vid_output_dir):
        os.makedirs(real_vid_output_dir, exist_ok=True)
    fake_vid_output_dir = os.path.join(output_dir, "generated_videos")
    if not os.path.exists(fake_vid_output_dir):
        os.makedirs(fake_vid_output_dir, exist_ok=True)
    print(f"Output real videos to directory: {real_vid_output_dir}")
    print(f"Output generated videos directory: {fake_vid_output_dir}")

    # Get list of videos to process
    if video_files is None:
        video_files = os.listdir(real_video_dir)

    # load the real video
    for real_vid in video_files:
        real_video_path = os.path.join(real_video_dir, real_vid)
        if not os.path.isfile(real_video_path):
            continue
        real_clips = read_video(real_video_path, target_size, target_fps, match_target_fps=match_target_fps)
        if real_clips is None:
            print(f"Skipping {real_video_path} because it has less than target_size[0] frames")
            continue
        for n, real_video in enumerate(real_clips):
            real_vid_n = real_vid.split('.')[0]+f"-seg{n}"
            # export the corresponding real video
            real_output_path = os.path.join(real_vid_output_dir, f"real_{real_vid_n}.mp4")
            real_video = real_video.permute(0, 2, 3, 1)
            real_video = real_video * 255.0
            real_video = real_video.numpy().astype(np.uint8)
            iio.imwrite(real_output_path, real_video)
            init_frame = real_video[0, ...]
            # convert init_frame to PIL image
            init_frame = transforms.ToPILImage()(init_frame)
            height, width = target_size[1], target_size[2]
            for i in range(num_fake_example_per_real_video):
                # Check if this example already exists
                fake_output_path = os.path.join(fake_vid_output_dir, f"generated_{real_vid_n}_{i}.mp4")
                if os.path.exists(fake_output_path) and os.path.exists(real_output_path):
                    print(f"Skipping existing example: {fake_output_path} and real video: {real_output_path}")
                    continue

                # generate the fake example for each real one
                fake_video = svd_pipeline(init_frame, height=height, width=width, num_frames=target_size[0], num_inference_steps=num_inference_steps, fps=target_fps, noise_aug_strength=noise_aug_strength, return_dict=False, output_type='pt')[0, ...]
                fake_video = fake_video.to('cpu')
                print(f"Generated video for {real_vid_n}, example {i}: {fake_video.shape}, range [{fake_video.min()}, {fake_video.max()}]")
                # print(f"Real video: {real_video.shape}, range [{real_video.min()}, {real_video.max()}]")

                # if save_pt_files:
                #     fake_video = fake_video.cpu()
                #     example = {'real_video': real_video, 'fake_video': fake_video}
                #     print(f"Preparing to save example to {output_path}")
                #     torch.save(example, output_path)
                #     print(f"Saved example to {output_path}")

                # export fake_video as a video file
                fake_video = fake_video.permute(0, 2, 3, 1)
                fake_video = fake_video * 255.0
                fake_video = fake_video.numpy().astype(np.uint8)
                iio.imwrite(fake_output_path, fake_video)

def process_video_chunk(gpu_id, video_files, real_video_dir, output_dir, target_size,
                        target_fps=7, match_target_fps=True, num_inference_steps=25, noise_aug_strength=0.02,
                        num_fake_example_per_real_video=4, scheduler_type='edm_ancestral', pretrained_svd_path="stabilityai/stable-video-diffusion-img2vid-xt",
                        sft_lora=None, lora_scale=1.0):
    """
    Process a chunk of videos on a specific GPU
    """
    # Set the device
    device = f'cuda:{gpu_id}'
    print(f"Process using device: {device}")
    
    # Initialize the pipeline on this GPU
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_svd_path, 
        torch_dtype=torch.float16, 
        )
    if scheduler_type == 'euler_discrete':
        pass #keep the original EulerDiscreteScheduler of svd pipeline
    elif scheduler_type == 'edm_ancestral':
        set_edm_ancestral(pipeline)
        print('use edm ancestral scheduler')
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    pipeline.to(device)

    # Load SFT LoRA if specified
    if sft_lora is not None:
        print('loaded sft lora', sft_lora)
        pipeline = load_sft_lora(pipeline, sft_lora, lora_scale)

    # Generate examples for this chunk of videos
    generate_svd_example(
        pipeline, 
        output_dir, 
        real_video_dir, 
        target_size, 
        target_fps=target_fps,
        match_target_fps=match_target_fps,
        num_inference_steps=num_inference_steps,
        noise_aug_strength=noise_aug_strength,
        num_fake_example_per_real_video=num_fake_example_per_real_video,
        video_files=video_files
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Generate SVD examples with multi-GPU support')
    parser.add_argument('--output_dir', type=str, default='/path/to/datasets/wisa_80k/svd_example/default', help='Output directory for generated examples and corresponding real videos')
    parser.add_argument('--real_video_dir', type=str, default='/path/to/datasets/wisa_80k/organized_videos/collision', help='Directory containing real videos')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated list of specific GPU IDs to use (e.g., "0,1,3")')
    parser.add_argument('--num_process_per_gpu', type=int, default=1, help='Number of worker processes to spawn per GPU')

    parser.add_argument('--num_fake_example_per_real_video', type=int, default=4, help='Number of examples to generate per video')
    parser.add_argument('--target_frames', type=int, default=24, help='Number of frames in target video')
    parser.add_argument('--target_height', type=int, default=320, help='Height of target video')
    parser.add_argument('--target_width', type=int, default=576, help='Width of target video')
    parser.add_argument('--target_fps', type=int, default=7, help='Target FPS of the video')
    parser.add_argument('--match_target_fps', action='store_true', default=True, help='Whether to match the target FPS of the video')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
    parser.add_argument('--noise_aug_strength', type=float, default=0., help='Noise augmentation strength')
    parser.add_argument('--scheduler_type', type=str, choices=['euler_discrete', 'edm_ancestral'], default='edm_ancestral', required=True, help='Scheduler type')
    parser.add_argument('--pretrained_svd_path', type=str, default="stabilityai/stable-video-diffusion-img2vid-xt", help='Pretrained SVD model path')
    parser.add_argument('--sft_lora', type=str, default=None, help='Path to SFT LoRA checkpoint (optional)')
    parser.add_argument('--lora_scale', type=float, default=1.0, help='Scale for the LoRA weights')
    args = parser.parse_args()
    
    # Set target size
    target_size = (args.target_frames, args.target_height, args.target_width)
    
    # Get list of all video files
    all_video_files = [f for f in os.listdir(args.real_video_dir) if os.path.isfile(os.path.join(args.real_video_dir, f))]
    print(f"Found {len(all_video_files)} video files to process")
    
    # Determine which GPUs to use
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]

    # Divide videos into chunks for each GPU
    video_chunks = []
    chunk_size = len(all_video_files) // len(gpu_ids)
    for i in range(len(gpu_ids)):
        if i == len(gpu_ids) - 1:  # Last GPU gets remaining videos
            video_chunks.append(all_video_files[i * chunk_size:])
        else:
            video_chunks.append(all_video_files[i * chunk_size:(i + 1) * chunk_size])
    
    # Process videos in parallel
    import multiprocessing as mp
    processes = []

    for gpu_id, video_chunk in zip(gpu_ids, video_chunks):
        if not video_chunk:
            continue

        # Further split this GPU's videos into sub‑chunks for each process
        num_procs = max(1, args.num_process_per_gpu)
        sub_chunk_size = max(1, len(video_chunk) // num_procs)

        for proc_idx in range(num_procs):
            start = proc_idx * sub_chunk_size
            end = len(video_chunk) if proc_idx == num_procs - 1 else (proc_idx + 1) * sub_chunk_size
            sub_videos = video_chunk[start:end]
            if not sub_videos:
                continue

            print(f"GPU {gpu_id}, process {proc_idx} will process {len(sub_videos)} videos")
            p = mp.Process(
                target=process_video_chunk,
                args=(
                    gpu_id,
                    sub_videos,
                    args.real_video_dir,
                    args.output_dir,
                    target_size,
                    args.target_fps,
                    args.match_target_fps,
                    args.num_inference_steps,
                    args.noise_aug_strength,
                    args.num_fake_example_per_real_video,
                    args.scheduler_type,
                    args.pretrained_svd_path,
                    args.sft_lora,
                    args.lora_scale
                )
            )
            p.start()
            processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All videos processed successfully!")

    # usage
    # python svd_example_generation.py --gpu_ids "0,1,2" --output_dir ./svd_examples --real_video_dir ./real_videos