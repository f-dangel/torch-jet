"""Utility functions for testing."""

from typing import Any, Callable

from pytest import param
from torch import Tensor, float64, manual_seed, rand, rand_like, stack, zeros_like
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

import jet
from jet import _is_jet_leaf, rev_jet

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


def rev_collapsed_jet(f: Callable[..., Any]) -> Callable[..., Any]:
    """Reference implementation for collapsed Taylor mode via :func:`jet.rev_jet`.

    Built on :func:`jet.rev_jet` (nested reverse-mode AD), so independent of
    the FX-trace and interpreter machinery. See :func:`jet.jet` for the
    collapsed-mode shape contract.

    The implementation exploits that the K-th output coefficient depends
    linearly on the K-th input coefficient (Faa di Bruno: only the term with
    multi-index ``m_K = 1, m_{<K} = 0`` involves ``c_K``, contributing
    ``Df(c_0) c_K``). So summing per-direction K-th outputs is the same as
    putting the full ``c_K`` into one direction and zeros into the rest,
    which is what we do.
    """
    std_jet = rev_jet(f)

    def cjet_f(*args: Any) -> Any:
        in_leaves, in_spec = tree_flatten(args, is_leaf=_is_jet_leaf)
        K = len(in_leaves[0]) - 1
        R = in_leaves[0][1].shape[0]

        def per_direction_args(r: int) -> tuple[Any, ...]:
            per_leaf = [
                (
                    leaf[0],
                    *(leaf[order][r] for order in range(1, K)),
                    leaf[K] if r == 0 else zeros_like(leaf[K]),
                )
                for leaf in in_leaves
            ]
            return tree_unflatten(per_leaf, in_spec)

        per_direction_outputs = [std_jet(*per_direction_args(r)) for r in range(R)]
        flat_per_r = [
            tree_flatten(out, is_leaf=_is_jet_leaf)[0] for out in per_direction_outputs
        ]
        _, out_spec = tree_flatten(per_direction_outputs[0], is_leaf=_is_jet_leaf)

        collapsed_leaves = []
        for leaf_idx in range(len(flat_per_r[0])):
            per_r = [flat_per_r[r][leaf_idx] for r in range(R)]
            o_0 = per_r[0][0]
            middles = tuple(
                stack([per_r[r][k] for r in range(R)], dim=0) for k in range(1, K)
            )
            o_K = stack([per_r[r][K] for r in range(R)], dim=0).sum(0)
            collapsed_leaves.append((o_0, *middles, o_K))

        return tree_unflatten(collapsed_leaves, out_spec)

    return cjet_f


def assert_jet_matches_oracle(config: dict[str, Any], K: int, collapsed: bool) -> None:
    """Assert ``jet(f, mock_args, collapsed)`` matches its mode-specific oracle.

    The oracle is :func:`jet.rev_jet` (standard) or :func:`rev_collapsed_jet`
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
