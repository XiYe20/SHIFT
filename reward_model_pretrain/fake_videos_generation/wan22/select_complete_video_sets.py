import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def _is_video_not_corrupted(video_path: Path) -> bool:
    """
    Basic integrity check for an mp4 file:
    - can be opened by OpenCV
    - can decode the first frame
    - can decode middle and last frames (when available)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False

    check_indices = [0]
    if frame_count > 2:
        check_indices.extend([frame_count // 2, frame_count - 1])

    for idx in check_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return False

    cap.release()
    return True


def sample_and_copy_complete_video_sets(
    src_dir: str,
    tgt_dir: str,
    sample_size: int = 5555,
    num_fake_per_real: int = 5,
    seed: int = 42,
    show_progress: bool = True,
    copy_in_parallel: bool = False,
    num_copy_workers: int = 8,
) -> list[str]:
    """
    Randomly sample valid real-video sets and copy to target directory.

    Expected source structure:
      src_dir/
        real_videos/
          real_<video_key>.mp4
        generated_videos/
          generated_<video_key>_0.mp4
          ...
          generated_<video_key>_4.mp4

    A set is valid only if:
    - real video exists and is not corrupted
    - all generated videos [0..num_fake_per_real-1] exist and are not corrupted

    Returns:
      List of sampled video keys (without "real_" prefix and ".mp4" suffix).
    """
    src_path = Path(src_dir)
    src_real_dir = src_path / "real_videos"
    src_fake_dir = src_path / "generated_videos"

    if not src_real_dir.is_dir():
        raise FileNotFoundError(f"Missing source real_videos dir: {src_real_dir}")
    if not src_fake_dir.is_dir():
        raise FileNotFoundError(f"Missing source generated_videos dir: {src_fake_dir}")

    tgt_path = Path(tgt_dir)
    tgt_real_dir = tgt_path / "real_videos"
    tgt_fake_dir = tgt_path / "generated_videos"
    tgt_real_dir.mkdir(parents=True, exist_ok=True)
    tgt_fake_dir.mkdir(parents=True, exist_ok=True)

    real_files = sorted(src_real_dir.glob("real_*.mp4"))
    valid_sets: list[tuple[Path, list[Path], str]] = []

    iterator = tqdm(
        real_files,
        desc="Validating video sets",
        unit="real",
        disable=not show_progress,
    )

    for real_path in iterator:
        video_key = real_path.stem[len("real_") :]
        fake_paths = [
            src_fake_dir / f"generated_{video_key}_{fake_idx}.mp4"
            for fake_idx in range(num_fake_per_real)
        ]

        if any(not p.exists() for p in fake_paths):
            continue

        if not _is_video_not_corrupted(real_path):
            continue
        if any(not _is_video_not_corrupted(fake_path) for fake_path in fake_paths):
            continue

        valid_sets.append((real_path, fake_paths, video_key))

    if len(valid_sets) < sample_size:
        raise ValueError(
            f"Not enough valid sets to sample {sample_size}. "
            f"Found only {len(valid_sets)} valid sets."
        )

    rng = random.Random(seed)
    sampled_sets = rng.sample(valid_sets, sample_size)

    sampled_keys: list[str] = []

    def _copy_one_set(real_path: Path, fake_paths: list[Path], video_key: str) -> str:
        shutil.copy2(real_path, tgt_real_dir / real_path.name)
        for fake_path in fake_paths:
            shutil.copy2(fake_path, tgt_fake_dir / fake_path.name)
        return video_key

    if copy_in_parallel and num_copy_workers > 1:
        with ThreadPoolExecutor(max_workers=num_copy_workers) as executor:
            futures = [
                executor.submit(_copy_one_set, real_path, fake_paths, video_key)
                for real_path, fake_paths, video_key in sampled_sets
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Copying sampled sets (parallel)",
                unit="set",
                disable=not show_progress,
            ):
                sampled_keys.append(future.result())
    else:
        copy_iterator = tqdm(
            sampled_sets,
            desc="Copying sampled sets",
            unit="set",
            disable=not show_progress,
        )
        for real_path, fake_paths, video_key in copy_iterator:
            sampled_keys.append(_copy_one_set(real_path, fake_paths, video_key))

    return sampled_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample complete non-corrupted real/fake video sets.")
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory with real_videos/ and generated_videos/.")
    parser.add_argument("--tgt_dir", type=str, required=True, help="Target directory with real_videos/ and generated_videos/.")
    parser.add_argument("--sample_size", type=int, default=5555, help="How many valid sets to sample.")
    parser.add_argument("--num_fake_per_real", type=int, default=5, help="Number of fake videos required per real video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.set_defaults(copy_in_parallel=False)
    parser.add_argument("--copy_in_parallel", dest="copy_in_parallel", action="store_true", help="Copy sampled files with a thread pool.")
    parser.add_argument("--no-copy_in_parallel", dest="copy_in_parallel", action="store_false", help="Disable parallel copy.")
    parser.add_argument("--num_copy_workers", type=int, default=8, help="Thread workers used when --copy_in_parallel is enabled.")
    parser.set_defaults(show_progress=True)
    parser.add_argument("--show_progress", dest="show_progress", action="store_true", help="Show tqdm progress bars.")
    parser.add_argument("--no-show_progress", dest="show_progress", action="store_false", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    sampled_keys = sample_and_copy_complete_video_sets(
        src_dir=args.src_dir,
        tgt_dir=args.tgt_dir,
        sample_size=args.sample_size,
        num_fake_per_real=args.num_fake_per_real,
        seed=args.seed,
        show_progress=args.show_progress,
        copy_in_parallel=args.copy_in_parallel,
        num_copy_workers=args.num_copy_workers,
    )
    print(f"Sampled and copied {len(sampled_keys)} complete sets.")


if __name__ == "__main__":
    main()
