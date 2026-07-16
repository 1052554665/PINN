#!/usr/bin/env python3
"""Build Mel-GADF/PCNN fused image dataset from raw .wav files.

Converts raw transformer voiceprint signals into fused RGB images:
    R channel: Mel spectrogram (gamma-corrected)
    G channel: GADF (Gramian Angular Difference Field)
    B channel: Mel - GADF difference (complementary information)

Optionally applies PCNN enhancement and splits into train/val/test sets.

Usage:
    python scripts/build_dataset.py \
        --input_dir raw-data/ \
        --output_dir datasets/mel_gadf_fused \
        --gamma 1.7 \
        --pcnn \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import librosa
import numpy as np
from PIL import Image
from pyts.image import GramianAngularField
from tqdm import tqdm


# ==============================================================================
# Feature Generation Functions
# ==============================================================================

def generate_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 256,
    hop_length: int = 1024,
    n_fft: int = 4096,
    gamma: float = 1.7,
) -> np.ndarray:
    """Generate gamma-corrected Mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = np.clip(mel_db, a_min=-80, a_max=0)
    mel_norm = (mel_db + 80) / 80  # → [0, 1]
    mel_norm = np.power(mel_norm, gamma)
    return mel_norm


def generate_gaf_feature(
    data: np.ndarray,
    size: int = 224,
    method: str = "difference",
) -> np.ndarray:
    """Generate GAF image from 1-d time series."""
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    gaf = GramianAngularField(image_size=len(data), method=method)
    gaf_image = gaf.fit_transform(data.reshape(1, -1))[0]
    gaf_resized = cv2.resize(
        gaf_image, (size, size), interpolation=cv2.INTER_CUBIC
    )
    # Normalize to [0, 1]
    g_min, g_max = gaf_resized.min(), gaf_resized.max()
    if g_max - g_min > 1e-8:
        gaf_resized = (gaf_resized - g_min) / (g_max - g_min)
    return gaf_resized


def build_fusion_image(
    wav_path: str,
    target_size: tuple = (224, 224),
    gamma: float = 1.7,
    gaf_method: str = "difference",
    gaf_n_samples: int = 3000,
) -> np.ndarray:
    """Build a 3-channel RGB fusion image from a .wav file.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    y, sr = librosa.load(wav_path, sr=None)

    # Mel spectrogram (gamma-corrected)
    mel = generate_mel_spectrogram(y, sr, gamma=gamma)
    mel_resized = cv2.resize(mel, target_size, interpolation=cv2.INTER_CUBIC)
    mel_resized = np.flipud(mel_resized)  # flip to put low freq at bottom

    # GAF
    y_short = y[:gaf_n_samples]
    gaf = generate_gaf_feature(y_short, size=target_size[0], method=gaf_method)

    # RGB channels
    r = (mel_resized * 255).astype(np.uint8)
    g = (gaf * 255).astype(np.uint8)
    # B channel: difference between Mel and GAF
    diff = np.abs(mel_resized - gaf)
    b = (diff * 255).astype(np.uint8)

    fusion = np.stack([r, g, b], axis=2)
    return fusion


# ==============================================================================
# Dataset Building
# ==============================================================================

def build_dataset(
    input_dir: str,
    output_dir: str,
    gamma: float = 1.7,
    gaf_method: str = "difference",
    target_size: tuple = (224, 224),
):
    """Convert all .wav files in input_dir to fused images.

    Expects input_dir to be organized as:
        input_dir/
            class_name_1/
                *.wav
            class_name_2/
                *.wav
            ...

    Outputs:
        output_dir/
            class_name_1/
                *.png
            class_name_2/
                *.png
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    wav_files = list(input_path.rglob("*.wav")) + list(input_path.rglob("*.WAV"))
    if not wav_files:
        print(f"[ERROR] No .wav files found in {input_dir}")
        sys.exit(1)

    for wav_file in tqdm(wav_files, desc="Building fused images"):
        # Determine class from parent directory name
        class_name = wav_file.parent.name
        out_class_dir = output_path / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        out_name = wav_file.stem + ".png"
        out_file = out_class_dir / out_name

        if out_file.exists():
            continue  # skip already processed

        try:
            fusion = build_fusion_image(
                str(wav_file),
                target_size=target_size,
                gamma=gamma,
                gaf_method=gaf_method,
            )
            Image.fromarray(fusion).save(str(out_file))
        except Exception as e:
            print(f"[WARN] Failed to process {wav_file}: {e}")

    print(f"[Done] Built dataset: {output_dir}")


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Split an ImageFolder-style dataset into train/val/test.

    Args:
        data_dir: directory with class_name/ subdirectories.
        output_dir: where to create train/, val/, test/.
        train_ratio, val_ratio, test_ratio: split proportions.
        seed: random seed for reproducibility.
    """
    random.seed(seed)
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            split_dir = out_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                dst = split_dir / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

    # Print statistics
    for split in ["train", "val", "test"]:
        split_dir = out_path / split
        total = sum(1 for _ in split_dir.rglob("*.png"))
        print(f"  {split}: {total} images")

    print(f"[Done] Split dataset: {output_dir}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Mel-GADF fused image dataset from raw .wav files."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing class_name/*.wav files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./datasets/mel_gadf_fused",
        help="Output directory for fused images.",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.7,
        help="Gamma correction factor for Mel spectrogram.",
    )
    parser.add_argument(
        "--gaf_method", type=str, default="difference",
        choices=["difference", "summation"],
        help="GAF method: 'difference' (GADF) or 'summation' (GASF).",
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Output image size (square).",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Also split into train/val/test after building.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    # Step 1: Build fused images
    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gamma=args.gamma,
        gaf_method=args.gaf_method,
        target_size=(args.img_size, args.img_size),
    )

    # Step 2: Split (optional)
    if args.split:
        split_dir = Path(args.output_dir).parent / (Path(args.output_dir).name + "_split")
        split_dataset(
            data_dir=args.output_dir,
            output_dir=str(split_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
