#!/usr/bin/env python3
"""Apply PCNN (Pulse-Coupled Neural Network) processing to a dataset.

Processes all images in an input directory using the PCNN layer and
saves them to an output directory, preserving the directory structure.

Usage:
    python scripts/apply_pcnn.py \
        --input_dir datasets/mel_gadf_fused \
        --output_dir datasets/mel_gadf_pcnn \
        --num_steps 10 --VT 0.8
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pcnn import PCNNLayer


def process_directory_rgb(
    input_root: str,
    output_root: str,
    pcnn: PCNNLayer,
    img_size: int = 224,
):
    """Apply PCNN to each RGB channel independently and save.

    Args:
        input_root: directory containing class_name/*.png files.
        output_root: output directory (preserves structure).
        pcnn: PCNN layer instance.
        img_size: target image size.
    """
    rgb_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    input_path = Path(input_root)
    output_path = Path(output_root)

    image_files = (
        list(input_path.rglob("*.png"))
        + list(input_path.rglob("*.jpg"))
        + list(input_path.rglob("*.jpeg"))
    )

    if not image_files:
        print(f"[ERROR] No images found in {input_root}")
        sys.exit(1)

    for img_file in tqdm(image_files, desc="Applying PCNN"):
        rel_path = img_file.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            continue

        try:
            image = Image.open(str(img_file)).convert("RGB")
            tensor = rgb_transform(image).unsqueeze(0)  # [1, 3, H, W]

            with torch.no_grad():
                # Process each channel independently
                channels = []
                for c in range(3):
                    ch = tensor[:, c:c+1, :, :]
                    ch_out = pcnn(ch)
                    channels.append(ch_out)
                processed = torch.cat(channels, dim=1)  # [1, 3, H, W]

            vutils.save_image(processed.squeeze(0), str(out_file))
        except Exception as e:
            print(f"[WARN] Failed to process {img_file}: {e}")


def process_directory_grayscale(
    input_root: str,
    output_root: str,
    pcnn: PCNNLayer,
    img_size: int = 224,
):
    """Apply PCNN to grayscale images.

    Args:
        input_root: directory containing class_name/*.png files.
        output_root: output directory.
        pcnn: PCNN layer instance.
        img_size: target image size.
    """
    gray_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    input_path = Path(input_root)
    output_path = Path(output_root)

    image_files = (
        list(input_path.rglob("*.png"))
        + list(input_path.rglob("*.jpg"))
        + list(input_path.rglob("*.jpeg"))
    )

    for img_file in tqdm(image_files, desc="Applying PCNN (gray)"):
        rel_path = img_file.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            continue

        try:
            image = Image.open(str(img_file)).convert("L")
            tensor = gray_transform(image).unsqueeze(0)  # [1, 1, H, W]

            with torch.no_grad():
                processed = pcnn(tensor)

            vutils.save_image(processed.squeeze(0), str(out_file))
        except Exception as e:
            print(f"[WARN] Failed to process {img_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply PCNN enhancement to a dataset of images."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory with class_name/*.png structure.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for PCNN-processed images.",
    )
    parser.add_argument(
        "--mode", type=str, default="rgb",
        choices=["rgb", "grayscale"],
        help="Processing mode: per-channel RGB or grayscale.",
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="Number of PCNN pulse iterations.",
    )
    parser.add_argument(
        "--alpha_F", type=float, default=0.1,
        help="PCNN feeding decay constant.",
    )
    parser.add_argument(
        "--beta", type=float, default=0.2,
        help="PCNN linking strength coefficient.",
    )
    parser.add_argument(
        "--VT", type=float, default=0.8,
        help="PCNN initial threshold voltage.",
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Image size.",
    )
    args = parser.parse_args()

    pcnn = PCNNLayer(
        num_steps=args.num_steps,
        alpha_F=args.alpha_F,
        beta=args.beta,
        VT=args.VT,
        accumulate=True,
    ).eval()

    if args.mode == "rgb":
        process_directory_rgb(
            args.input_dir, args.output_dir, pcnn, args.img_size,
        )
    else:
        process_directory_grayscale(
            args.input_dir, args.output_dir, pcnn, args.img_size,
        )

    print(f"[Done] PCNN-processed dataset: {args.output_dir}")


if __name__ == "__main__":
    main()
