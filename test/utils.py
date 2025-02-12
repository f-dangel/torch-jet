"""Utility functions for testing."""

from time import perf_counter
from typing import Callable, Tuple

import numpy as np
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


def measure_time(
    f: Callable, name: str, num_repeats: int = 100, warmup: int = 10
) -> Tuple[float, float]:
    """Measure the CPU time of a function.

    Args:
        f: Function to measure the time of.
        name: Name of the function. Will be used to print the timings.
        num_repeats: Number of times to repeat the measurement. Default: `100`.
        warmup: Number of warmup runs before measuring the time. Default: `10`.

    Returns:
        Mean and standard deviation of the measured times.
    """
    for _ in range(warmup):
        f()

    times = []
    for _ in range(num_repeats):
        start = perf_counter()
        _ = f()
        times.append(perf_counter() - start)

    mu, sigma = np.mean(times), np.std(times)
    print(f"{name}: {mu:.5f} ± {sigma:.5f} s")

    return float(mu), float(sigma)
