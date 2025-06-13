"""Utility functions for testing."""

from torch import Tensor


def report_nonclose(
    a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8, name: str = "Tensors"
):
    """Report non-closeness of two tensors.

    Args:
        a: First tensor.
        b: Second tensor.
        rtol: Relative tolerance. Default: `1e-5`.
        atol: Absolute tolerance. Default: `1e-8`.
        name: Name of the tensors. Default: `"Tensors"`.
    """
    assert a.shape == b.shape, f"Shapes are not equal: {a.shape} != {b.shape}"
    close = a.allclose(b, rtol=rtol, atol=atol)
    if not close:
        for idx, (x, y) in enumerate(zip(a.flatten(), b.flatten())):
            if not x.isclose(y, rtol=rtol, atol=atol):
                print(f"Index {idx}: {x} != {y} (ratio: {x / y})")
    else:
        print(f"{name} are close.")
    assert close, f"{name} are not close."
