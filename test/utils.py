"""Utility functions for testing."""

from torch import Tensor, sin
from torch.nn import Module

VMAPS = [False, True]
VMAP_IDS = [f"vmap={v}" for v in VMAPS]


class Sin(Module):
    """Sine activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer.

        Args:
            x: Input tensor.

        Returns:
            The output of the layer.
        """
        return sin(x)
