"""Utility functions for testing."""

from typing import Any, Callable

from pytest import param
from torch import Tensor, float64, manual_seed, rand, rand_like
from torch.testing import assert_close
from torch.utils._pytree import tree_map

import jet
from jet import rev_collapsed_jet, rev_jet

#: Valid ``(K, collapsed)`` pairs for the standard-vs-collapsed mode sweep.
#: ``K=0`` (primal-only) and ``K=1`` (Jacobian-vector product) are
#: standard-mode boundaries that exercise the no-recursion branches of the
#: jet rules; ``K=2`` is the collapsed-mode floor; ``K=5`` is a representative
#: high order. Intermediate orders exercise the same code paths and don't
#: earn their own coverage at the primitive / composition layer.
K_AND_MODE = [
    param(0, False, id="K=0-standard"),
    param(1, False, id="K=1-standard"),
    param(2, False, id="K=2-standard"),
    param(5, False, id="K=5-standard"),
    param(2, True, id="K=2-collapsed"),
    param(5, True, id="K=5-collapsed"),
]


def shape(*dims: int) -> Callable[[], tuple[Tensor]]:
    """``args_fn`` factory: one ``rand`` tensor of the given shape (float64)."""
    return lambda: (rand(*dims, dtype=float64),)


def shapes(*shape_pairs) -> Callable[[], tuple[Tensor, ...]]:
    """``args_fn`` factory: one ``rand`` tensor per shape (all float64)."""
    return lambda: tuple(rand(*s, dtype=float64) for s in shape_pairs)


def setup_case(
    config: dict[str, Any],
) -> tuple[Callable[..., Tensor], tuple[Tensor, ...]]:
    """Instantiate the function and the arguments to evaluate it on.

    The arguments are taken verbatim from ``config["args_fn"]()``
    -- if the case wants a batched input it should encode the batch dimension
    into its ``args_fn`` directly. Callers that re-pass the returned
    tuple to a tracing API (where the library calls the parameter
    ``mock_args``) may rebind it locally to ``mock_args``.

    Args:
        config: Configuration dictionary of the test case. Must have ``"f"``
            and ``"args_fn"`` keys.

    Returns:
        Tuple ``(f, args)`` where ``args`` is a tuple of tensors (one entry
        per positional argument of ``f``).
    """
    manual_seed(0)
    return config["f"], config["args_fn"]()


def make_jet_args(
    args: tuple[Any, ...], K: int, collapsed: bool = False, R: int = 2
) -> tuple[Any, ...]:
    """Build jet args from ``args`` by attaching Taylor coefficients per leaf.

    Each tensor leaf ``t`` in ``args`` becomes a jet tuple
    ``(t, c_1, ..., c_K)`` of tensors -- ``t`` itself is the primal. Shape
    contract by mode:

    - **standard** (``collapsed=False``): every ``c_k`` has shape ``t.shape``.
      ``R`` is ignored.
    - **collapsed** (``collapsed=True``): ``c_1..c_{K-1}`` have shape
      ``(R, *t.shape)`` (each carries the ``R`` directions); ``c_K`` has
      shape ``t.shape``.

    All coefficients are independently randomly drawn.
    """
    manual_seed(42)

    def make_leaf(t: Tensor) -> tuple[Tensor, ...]:
        if not collapsed:
            return (t, *(rand_like(t) for _ in range(K)))
        batched = [rand(R, *t.shape, dtype=t.dtype) for _ in range(K - 1)]
        return (t, *batched, rand_like(t))

    return tree_map(make_leaf, args)


def assert_jet_matches_oracle(config: dict[str, Any], K: int, collapsed: bool) -> None:
    """Assert ``jet(f, mock_args, collapsed)`` matches its mode-specific oracle.

    The oracle is :func:`jet.rev_jet` (standard) or :func:`jet.rev_collapsed_jet`
    (collapsed). Both are built on nested reverse-mode AD and are independent
    of the FX-trace + interpreter machinery under test.
    """
    f = config["f"]
    mock_args = config["args_fn"]()
    jet_args = make_jet_args(mock_args, K, collapsed=collapsed)
    oracle = rev_collapsed_jet(f) if collapsed else rev_jet(f)
    actual = jet.jet(f, mock_args, collapsed=collapsed)(*jet_args)
    expected = oracle(*jet_args)
    assert_close(actual, expected)
