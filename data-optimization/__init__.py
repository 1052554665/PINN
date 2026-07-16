# Data Optimization Utilities

"""Image preprocessing and optimization utilities.

This module contains:
- Gamma correction for Mel spectrogram enhancement
- Image normalization and resizing
- PCNN-based batch processing
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float = 1.7,
    c: float = 1.0,
) -> np.ndarray:
    """Apply gamma correction to enhance image contrast.

    I_out = c * I_in^gamma

    A gamma value > 1 enhances dark regions (useful for Mel spectrograms
    where low-frequency energy is important). Gamma < 1 enhances bright
    regions.

    Args:
        image: input image (H, W) or (H, W, C) in range [0, 1] or [0, 255].
        gamma: gamma correction factor (default: 1.7).
        c: scaling constant (default: 1.0).

    Returns:
        Gamma-corrected image in the same value range as input.
    """
    # Determine input range
    if image.max() <= 1.0:
        in_range = 1.0
    else:
        in_range = 255.0

    normalized = image.astype(np.float32) / in_range
    corrected = c * np.power(normalized, gamma)
    corrected = np.clip(corrected, 0.0, 1.0)

    return (corrected * in_range).astype(image.dtype)


def generate_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 256,
    hop_length: int = 1024,
    n_fft: int = 4096,
    gamma: float = 1.7,
    clip_db: bool = True,
) -> np.ndarray:
    """Generate enhanced Mel spectrogram from raw audio.

    Applies logarithmic compression, optional clipping, and gamma
    correction for better low-energy feature visibility.

    Args:
        y: raw audio signal.
        sr: sample rate.
        n_mels: number of Mel filter banks.
        hop_length: hop length for STFT.
        n_fft: FFT window size.
        gamma: gamma correction factor (1.7 recommended).
        clip_db: if True, clip dB range to [-80, 0].

    Returns:
        Mel spectrogram array normalized to [0, 1], shape (n_mels, T).
    """
    import librosa

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    if clip_db:
        mel_db = np.clip(mel_db, a_min=-80, a_max=0)
        mel_norm = (mel_db + 80) / 80  # [-80, 0] → [0, 1]
    else:
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Gamma correction
    mel_norm = np.power(mel_norm, gamma)

    return mel_norm


def generate_gaf_image(
    data: np.ndarray,
    size: int = 224,
    method: str = "difference",
) -> np.ndarray:
    """Generate Gramian Angular Field image from time-series data.

    Args:
        data: 1-d time-series array.
        size: output image size (square).
        method: 'summation' for GASF, 'difference' for GADF.

    Returns:
        GAF image array of shape (size, size), values in [0, 1].
    """
    from pyts.image import GramianAngularField

    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    gaf = GramianAngularField(image_size=len(data), method=method)
    gaf_image = gaf.fit_transform(data.reshape(1, -1))[0]

    # Resize to target size using bicubic interpolation
    gaf_resized = cv2.resize(
        gaf_image, (size, size), interpolation=cv2.INTER_CUBIC
    )

    # Normalize to [0, 1]
    g_min, g_max = gaf_resized.min(), gaf_resized.max()
    if g_max - g_min > 1e-8:
        gaf_resized = (gaf_resized - g_min) / (g_max - g_min)

    return gaf_resized
