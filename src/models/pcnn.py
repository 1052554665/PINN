"""Pulse-Coupled Neural Network (PCNN) Layer.

A biologically-inspired neural network module that emulates the spike
firing mechanism of biological neurons. PCNN enhances discriminative
features while suppressing background noise through iterative
pulse-coupled dynamics.

Reference:
    Wang et al., "Pulse-coupled neural networks," IEEE TNN, 2010.
"""

import torch
import torch.nn as nn


class PCNNLayer(nn.Module):
    """Pulse-Coupled Neural Network layer for image enhancement.

    The PCNN iteratively fires pulses based on local excitation and
    global threshold dynamics, producing an enhanced feature map
    where salient regions are emphasized and noise is suppressed.

    Key parameters:
        - alpha_F: decay constant for the feeding input
        - beta: linking strength (lateral coupling)
        - VT: initial dynamic threshold (higher → stronger noise suppression)
        - num_steps: number of pulse iterations

    Args:
        num_steps: number of iterative pulse steps (default: 10).
        alpha_F: feeding decay constant (default: 0.1).
        beta: linking strength coefficient (default: 0.2).
        VT: initial threshold voltage (default: 0.8).
        accumulate: if True, returns accumulated normalized pulse map;
                    if False, returns binary pulse output.
    """

    def __init__(
        self,
        num_steps: int = 10,
        alpha_F: float = 0.1,
        beta: float = 0.2,
        VT: float = 0.8,
        accumulate: bool = True,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.alpha_F = alpha_F
        self.beta = beta
        self.VT = VT
        self.accumulate = accumulate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PCNN processing.

        Args:
            x: input tensor of shape (B, C, H, W) in range [0, 1].

        Returns:
            Processed tensor of same shape, values in [0, 1].
        """
        B, C, H, W = x.shape
        Y = torch.zeros_like(x)
        F = x.clone()
        T = self.VT * torch.ones_like(x)

        if self.accumulate:
            Y_accum = torch.zeros_like(x)

        for _ in range(self.num_steps):
            # Internal excitation: feeding input modulated by linking
            U = F * (1 + self.beta * Y)

            # Spike generation: fire if excitation exceeds threshold
            Y_new = (U > T).float()

            # Dynamic threshold update
            T = T * (1 + self.alpha_F) - Y_new * T * self.alpha_F
            Y = Y_new

            if self.accumulate:
                Y_accum += Y_new

        if self.accumulate:
            # Normalize accumulated pulses to [0, 1]
            return Y_accum / self.num_steps
        else:
            return Y

    def extra_repr(self) -> str:
        return (
            f"num_steps={self.num_steps}, alpha_F={self.alpha_F}, "
            f"beta={self.beta}, VT={self.VT}, accumulate={self.accumulate}"
        )
