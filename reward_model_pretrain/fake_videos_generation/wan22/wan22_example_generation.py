import torch
import imageio.v3 as iio
import os
import numpy as np
from torchvision import transforms
import argparse
import multiprocessing as mp
import cv2
from pathlib import Path
import re
import json

# WAN2.2 TI2V (mirrors wan22_inference.py)
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from transformers import CLIPVisionModel

# add this helper near the bottom (same logic style as sampling_videos/sample_i2v.py)
def _get_node_rank_and_num_nodes(args):
    # On your k8s cluster: NODE_RANK + WORLD_SIZE are provided (see launch_sample_eval.sh)
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", os.environ.get("RANK", "0"))))

    num_nodes = args.num_nodes
    if num_nodes is None:
        num_nodes = int(os.environ.get("WORLD_SIZE", os.environ.get("NUM_NODES", os.environ.get("NNODES", "1"))))

    if num_nodes < 1:
        num_nodes = 1
    if node_rank < 0:
        node_rank = 0
    return node_rank, num_nodes

def load_prompt_map(prompt_json_file: str) -> dict[str, str]:
    """
    JSON format: [ {"video_name": "...mp4", "captions": "..."}, ... ]
    Returns: { "xxx.mp4": "caption text", ... }
    """
    with open(prompt_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["video_name"]: item["captions"] for item in data}

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

@torch.no_grad()
def generate_wan_example(
    wan_pipeline, 
    output_dir, 
    real_video_dir, 
    target_size, 
    target_fps=24, 
    match_target_fps=True, 
    num_inference_steps=40,
    num_fake_example_per_real_video=4, 
    video_files=None,
    guidance_scale=5.0,
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    prompt_json_file=None,
    generate_fake_videos=True
    ):
    """
    Generate example videos using WAN2.2 TI2V. Take the first frame of the real video example as input.
    Args:
        wan_pipeline: the WAN pipeline
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
    prompt_map = load_prompt_map(prompt_json_file)
    # load the real video
    for real_vid in video_files:
        real_video_path = os.path.join(real_video_dir, real_vid)
        if not os.path.isfile(real_video_path):
            continue
        real_clips = read_video(real_video_path, target_size, target_fps, match_target_fps=match_target_fps)
        if real_clips is None:
            print(f"Skipping {real_video_path} because it has less than target_size[0] frames")
            continue
        prompt = prompt_map.get(Path(real_vid).name)
        if not prompt:
            print(f"[WARN] No caption found for {real_vid}; skipping.")
            continue
        # print('prompt:', prompt)
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
            if generate_fake_videos:
                for i in range(num_fake_example_per_real_video):
                    # Check if this example already exists
                    fake_output_path = os.path.join(fake_vid_output_dir, f"generated_{real_vid_n}_{i}.mp4")
                    if os.path.exists(fake_output_path) and os.path.exists(real_output_path):
                        print(f"Skipping existing example: {fake_output_path} and real video: {real_output_path}")
                        continue

                    # generate the fake example for each real one
                    fake_video = wan_pipeline(
                        image=init_frame,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_frames=target_size[0],
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                    ).frames[0]
                    print(f"Generated video for {real_vid_n}, example {i}: {fake_video.shape}, range [{fake_video.min()}, {fake_video.max()}]")
                    # export fake_video as a video file
                    fake_video = fake_video * 255.0
                    fake_video = fake_video.astype(np.uint8)
                    iio.imwrite(fake_output_path, fake_video)

def process_video_chunk(gpu_id, video_files, real_video_dir, output_dir, target_size, 
                        target_fps=24, match_target_fps=True, num_inference_steps=40, guidance_scale=5.0,
                        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        num_fake_example_per_real_video=4, pretrained_wan_path="Wan2.2-TI2V-5B-Diffusers",
                        prompt_json_file=None, only_process_real_videos=False):
    """
    Process a chunk of videos on a specific GPU
    """
    if not only_process_real_videos:
        # Set the device
        device = f'cuda:{gpu_id}'
        print(f"Process using device: {device}")
        
        # Initialize the pipeline on this GPU
        vae = AutoencoderKLWan.from_pretrained(
            pretrained_wan_path, subfolder="vae", torch_dtype=torch.float32
        )

        pipeline = WanImageToVideoPipeline.from_pretrained(
            pretrained_wan_path,
            vae=vae,
            image_encoder=None,
            torch_dtype=torch.bfloat16,
        )
        pipeline.to(device)
        generate_fake_videos=True
    else:
        pipeline = None
        generate_fake_videos=False
    # Generate examples for this chunk of videos
    generate_wan_example(
        pipeline, 
        output_dir, 
        real_video_dir, 
        target_size, 
        target_fps=target_fps,
        match_target_fps=match_target_fps,
        num_inference_steps=num_inference_steps,
        num_fake_example_per_real_video=num_fake_example_per_real_video,
        video_files=video_files,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        prompt_json_file=prompt_json_file,
        generate_fake_videos=generate_fake_videos
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Generate SVD examples with multi-GPU support')
    parser.add_argument('--output_dir', type=str, default='/path/to/datasets/wisa_80k/svd_example/default', help='Output directory for generated examples and corresponding real videos')
    parser.add_argument('--real_video_dir', type=str, default='/path/to/datasets/wisa_80k/organized_videos/collision', help='Directory containing real videos')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated list of specific GPU IDs to use (e.g., "0,1,3")')
    parser.add_argument('--num_process_per_gpu', type=int, default=1, help='Number of worker processes to spawn per GPU')
    parser.add_argument("--num_nodes", type=int, default=1, help="Num nodes (override env WORLD_SIZE)")
    parser.add_argument('--num_fake_example_per_real_video', type=int, default=4, help='Number of examples to generate per video')
    parser.add_argument('--target_frames', type=int, default=24, help='Number of frames in target video')
    parser.add_argument('--target_height', type=int, default=704, help='Height of target video')
    parser.add_argument('--target_width', type=int, default=1280, help='Width of target video')
    parser.add_argument('--target_fps', type=int, default=24, help='Target FPS of the video')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--negative_prompt', type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help='Negative prompt')
    parser.add_argument('--match_target_fps', action='store_true', default=True, help='Whether to match the target FPS of the video')
    parser.add_argument('--num_inference_steps', type=int, default=40, help='Number of inference steps')
    parser.add_argument('--pretrained_wan_path', type=str, default="/path/to/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635", help='Pretrained WAN2.2 TI2V model path')
    parser.add_argument('--prompt_json_file', type=str, default='/path/to/datasets/MotionAlignmentDatasets/wisa80k_correct/wisa_80k_captions.json', help='Prompt json file')
    parser.add_argument('--only_process_real_videos', action='store_true', default=False, help='True for only processing real videos, for example, for finetuning, but skip fake videos generation')
    args = parser.parse_args()
    
    # Set target size
    target_size = (args.target_frames, args.target_height, args.target_width)
    
    # Get list of all video files
    all_video_files = [
        f for f in os.listdir(args.real_video_dir)
        if os.path.isfile(os.path.join(args.real_video_dir, f))
    ]
    all_video_files.sort()

    node_rank, num_nodes = _get_node_rank_and_num_nodes(args)
    if num_nodes > 1:
        all_video_files = all_video_files[node_rank::num_nodes]
        print(f"[node {node_rank}/{num_nodes}] Sharded inputs: {len(all_video_files)} videos")
    else:
        print(f"Found {len(all_video_files)} video files to process")
    # Determine which GPUs to use
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]

    # Divide videos into chunks for each GPU
    video_chunks = []
    chunk_size = max(1, len(all_video_files) // len(gpu_ids))
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
                    args.guidance_scale,
                    args.negative_prompt,
                    args.num_fake_example_per_real_video,
                    args.pretrained_wan_path,
                    args.prompt_json_file,
                    args.only_process_real_videos
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