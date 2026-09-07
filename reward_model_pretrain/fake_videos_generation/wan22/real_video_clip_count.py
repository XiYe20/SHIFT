import argparse
import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def load_prompt_map(prompt_json_file: str) -> dict[str, str]:
    """
    JSON format: [ {"video_name": "...mp4", "captions": "..."}, ... ]
    Returns: { "xxx.mp4": "caption text", ... }
    """
    with open(prompt_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["video_name"]: item["captions"] for item in data}


def _count_sampled_frames(total_frames: int, original_fps: float, target_fps: int, match_target_fps: bool) -> int:
    if total_frames <= 0:
        return 0

    if match_target_fps:
        frame_skip = max(int(original_fps // target_fps), 1)
    else:
        frame_skip = 1

    # Matches the behavior of: for i in range(0, total_frames, frame_skip)
    return (total_frames + frame_skip - 1) // frame_skip


def count_real_video_clips(
    base_real_video_dir: str,
    subdir_name: str,
    target_frames: int = 49,
    target_fps: int = 24,
    match_target_fps: bool = True,
    prompt_json_file: str | None = None,
    show_progress: bool = True,
) -> int:
    """
    Count how many real clips would be produced by wan22_example_generation.py
    (without generating fake videos).

    A "real clip" is one segment of length `target_frames` after optional FPS matching.

    Args:
        base_real_video_dir: Base folder containing class subfolders (e.g. .../original_videos).
        subdir_name: Subdirectory to process (e.g. "deformation").
        target_frames: Clip length used in generation (default 49).
        target_fps: Target FPS used for frame skipping (default 24).
        match_target_fps: Same meaning as in generation code.
        prompt_json_file: Optional caption json path. If provided, videos without captions
                          are skipped to mirror generation behavior.
        show_progress: Whether to display a tqdm progress bar.
    """
    real_video_dir = os.path.join(base_real_video_dir, subdir_name)
    if not os.path.isdir(real_video_dir):
        raise FileNotFoundError(f"Subdirectory not found: {real_video_dir}")

    prompt_map = load_prompt_map(prompt_json_file) if prompt_json_file else None

    video_names = sorted(os.listdir(real_video_dir))
    iterator = tqdm(
        video_names,
        desc=f"Counting real clips [{subdir_name}]",
        unit="video",
        disable=not show_progress,
    )

    total_real_clips = 0
    for video_name in iterator:
        video_path = os.path.join(real_video_dir, video_name)
        if not os.path.isfile(video_path):
            continue
        if not video_name.lower().endswith(".mp4"):
            continue

        if prompt_map is not None and Path(video_name).name not in prompt_map:
            # Mirrors generation behavior: missing caption => skip video.
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        sampled_frames = _count_sampled_frames(
            total_frames=total_frames,
            original_fps=original_fps,
            target_fps=target_fps,
            match_target_fps=match_target_fps,
        )
        num_clips = sampled_frames // target_frames
        total_real_clips += num_clips

    return total_real_clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Count real video clips without fake generation.")
    parser.add_argument("--base_real_video_dir", type=str, required=True, help="Base real video directory.")
    parser.add_argument("--subdir_name", type=str, required=True, help='Subdir name, e.g. "deformation".')
    parser.add_argument("--target_frames", type=int, default=49, help="Target frames per clip.")
    parser.add_argument("--target_fps", type=int, default=24, help="Target FPS.")
    parser.set_defaults(match_target_fps=True)
    parser.add_argument("--match_target_fps", dest="match_target_fps", action="store_true", help="Match target FPS.")
    parser.add_argument(
        "--no-match_target_fps",
        dest="match_target_fps",
        action="store_false",
        help="Do not match target FPS (use every frame).",
    )
    parser.add_argument(
        "--prompt_json_file",
        type=str,
        default=None,
        help="Optional prompt json file. If set, only videos with captions are counted.",
    )
    args = parser.parse_args()

    clip_count = count_real_video_clips(
        base_real_video_dir=args.base_real_video_dir,
        subdir_name=args.subdir_name,
        target_frames=args.target_frames,
        target_fps=args.target_fps,
        match_target_fps=args.match_target_fps,
        prompt_json_file=args.prompt_json_file,
    )
    print(clip_count)


if __name__ == "__main__":
    main()
