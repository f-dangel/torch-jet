"""Utility functions for testing."""

from typing import Any

from torch import Tensor
from torch.utils._pytree import TreeSpec, tree_flatten


def report_nonclose(
    a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8, name: str = "Tensors"
) -> None:
    """Assert two tensors are element-wise close, printing mismatches on failure.

    Args:
        a: First tensor.
        b: Second tensor.
        rtol: Relative tolerance. Default: ``1e-5``.
        atol: Absolute tolerance. Default: ``1e-8``.
        name: Label for error messages. Default: ``"Tensors"``.

    Raises:
        AssertionError: If shapes differ or values are not close.
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


def _assert_specs_compatible(spec1: TreeSpec, spec2: TreeSpec, name: str = "") -> None:
    """Assert two tree specs are structurally compatible.

    Tolerates ``dict`` vs ``immutable_dict`` differences introduced by
    ``make_fx`` tracing by normalizing container type names before comparing.

    Args:
        spec1: First tree spec.
        spec2: Second tree spec.
        name: Label for error messages.

    Raises:
        AssertionError: If the tree specs are not structurally compatible.
    """
    if spec1 == spec2:
        return
    s1 = str(spec1).replace("immutable_dict", "dict")
    s2 = str(spec2).replace("immutable_dict", "dict")
    assert s1 == s2, f"{name} tree structure mismatch: {spec1} vs {spec2}"


def assert_pytrees_close(
    tree1: Any,
    tree2: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "",
) -> None:
    """Assert two pytrees have the same structure and close tensor leaves.

    Tolerates ``dict`` vs ``immutable_dict`` differences introduced by
    ``make_fx`` tracing.

    Args:
        tree1: First pytree.
        tree2: Second pytree.
        rtol: Relative tolerance for leaf comparison. Default: ``1e-5``.
        atol: Absolute tolerance for leaf comparison. Default: ``1e-8``.
        name: Label prefix for error messages.

    Raises:
        AssertionError: If tree structures differ or tensor leaves are not close.
    """
    flat1, spec1 = tree_flatten(tree1)
    flat2, spec2 = tree_flatten(tree2)
    _assert_specs_compatible(spec1, spec2, name)
    for j, (t1, t2) in enumerate(zip(flat1, flat2)):
        report_nonclose(t1, t2, rtol=rtol, atol=atol, name=f"{name} leaf {j}")
