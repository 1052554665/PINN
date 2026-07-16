"""Laplacian (Physics-Informed) Regularization Loss.

Implements the heat-diffusion-inspired smoothness prior for PINN training.
The Laplacian regularization encourages local smoothness in the predicted
physical field, enforcing the approximate Laplace equation Δu ≈ 0.

This acts as a physics-informed soft constraint that:
    - Improves generalization on limited/noisy data
    - Mitigates overfitting
    - Highlights fault-induced irregularities as deviations from smoothness
"""

import torch
import torch.nn.functional as F


def laplacian_loss(u: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian (smoothness) regularization loss.

    Approximates the discrete Laplacian operator ∇²u using second-order
    central finite differences:

        ∇²u ≈ u(x+1,y) + u(x-1,y) + u(x,y+1) + u(x,y-1) - 4·u(x,y)

    The loss is the mean squared Laplacian, which encourages the field
    to be as smooth as possible (Δu → 0).

    Args:
        u: physical field tensor of shape (B, 1, H, W) or (B, C, H, W).

    Returns:
        Scalar Laplacian regularization loss.
    """
    # Second-order central differences along x and y
    u_xx = (
        u[:, :, :-2, 1:-1]
        - 2 * u[:, :, 1:-1, 1:-1]
        + u[:, :, 2:, 1:-1]
    )
    u_yy = (
        u[:, :, 1:-1, :-2]
        - 2 * u[:, :, 1:-1, 1:-1]
        + u[:, :, 1:-1, 2:]
    )
    lap = u_xx + u_yy
    return torch.mean(lap ** 2)


def total_loss(
    pred_cls: torch.Tensor,
    true_cls: torch.Tensor,
    phys_field: torch.Tensor,
    lambda_phy: float = 0.1,
):
    """Compute the combined PINN training loss.

    L_total = L_classification + λ_phy · L_laplacian

    Args:
        pred_cls: predicted class logits (B, num_classes).
        true_cls: ground-truth class labels (B,).
        phys_field: physical field tensor (B, 1, H, W).
        lambda_phy: weight for the physics-informed regularization term.

    Returns:
        Tuple of (total_loss, classification_loss, physics_loss).
    """
    loss_cls = F.cross_entropy(pred_cls, true_cls)
    loss_phy = laplacian_loss(phys_field)
    return loss_cls + lambda_phy * loss_phy, loss_cls, loss_phy
