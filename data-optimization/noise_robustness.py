"""Noise robustness evaluation utilities.

Adds Gaussian noise, impulse noise, and background noise to test images
to evaluate model robustness under degraded conditions.
"""

import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 0.05,
) -> np.ndarray:
    """Add Gaussian noise to an image.

    Args:
        image: input image in [0, 1] or [0, 255].
        mean: noise mean.
        std: noise standard deviation (relative to [0,1] range).

    Returns:
        Noisy image, clipped to valid range.
    """
    if image.max() > 1.0:
        scale = 255.0
    else:
        scale = 1.0

    noise = np.random.normal(mean, std * scale, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, scale).astype(image.dtype)


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.02,
) -> np.ndarray:
    """Add salt-and-pepper noise to an image.

    Args:
        image: input image.
        amount: proportion of pixels to corrupt.

    Returns:
        Noisy image.
    """
    noisy = image.copy()
    if image.max() > 1.0:
        vmax = 255
        vmin = 0
    else:
        vmax = 1.0
        vmin = 0.0

    n_pixels = image.size
    n_salt = int(amount * n_pixels / 2)
    n_pepper = int(amount * n_pixels / 2)

    # Salt (white)
    coords = [np.random.randint(0, i, n_salt) for i in image.shape]
    noisy[tuple(coords)] = vmax

    # Pepper (black)
    coords = [np.random.randint(0, i, n_pepper) for i in image.shape]
    noisy[tuple(coords)] = vmin

    return noisy
