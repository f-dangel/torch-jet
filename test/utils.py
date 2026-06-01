"""Utility functions for testing."""

from typing import Any, Callable

from pytest import param
from torch import (
    Tensor,
    cuda,
    float32,
    float64,
    manual_seed,
    rand,
    rand_like,
    sigmoid,
    stack,
    zeros_like,
)
from torch.backends import mps
from torch.nn import Linear, Sequential, Tanh
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten, tree_map

from jet import _is_jet_leaf, jet, rev_jet

#: Devices the test suite parametrizes over. CPU is always present; CUDA and
#: MPS are added when their respective backends are available.
DEVICES = ["cpu"]
if cuda.is_available():
    DEVICES.append("cuda")
if mps.is_available():
    DEVICES.append("mps")


def dtype_for_device(device: str):
    """MPS doesn't support float64; CPU/CUDA use float64."""
    return float32 if device == "mps" else float64


def tolerances_for_device(device: str) -> dict[str, float]:
    """Relaxed ``assert_close`` tolerances for float32 devices (MPS).

    Default ``rtol=1.3e-6`` is calibrated for single-op float32 precision;
    higher-order Taylor coefficients accumulate roundoff and need headroom.
    Float64 keeps the default (no relaxation).
    """
    if device == "mps":
        return {"rtol": 1e-3, "atol": 1e-3}
    return {}


#: Valid ``(K, collapsed)`` pairs for the standard-vs-collapsed mode sweep.
#: ``K=0`` (primal-only) and ``K=1`` (Jacobian-vector product) are
#: standard-mode boundaries that exercise the no-recursion branches of the
#: jet rules; ``K=2`` is the collapsed-mode floor; ``K=5`` is a representative
#: high order. Intermediate orders exercise the same code paths and don't
#: earn their own coverage at the primitive / composition layer.
K_AND_MODE = [
    *(param(K, False, id=f"K={K}-standard") for K in (0, 1, 2, 5)),
    *(param(K, True, id=f"K={K}-collapsed") for K in (2, 5)),
]


#: Per-device cache for the two-layer tanh MLP used by composition / laplacian
#: / bilaplacian / exp01 tests. Sequential's Linear weights are concrete
#: tensors at trace time, which FX requires.
_MLP_CACHE: dict[str, Sequential] = {}


def mlp(device: str) -> Sequential:
    """Build (or retrieve cached) two-layer tanh-activated MLP on ``device``."""
    if device not in _MLP_CACHE:
        manual_seed(0)
        net = Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        )
        _MLP_CACHE[device] = net.to(device=device, dtype=dtype_for_device(device))
    return _MLP_CACHE[device]


#: Scalar-output cases shared by the laplacian + bilaplacian consumer tests.
#: ``f`` is a builder ``device -> Callable``; tensor inputs are described
#: declaratively as ``input_shapes`` (one shape tuple per positional arg).
SCALAR_OUTPUT_CASES = [
    {"f": mlp, "input_shapes": [(5,)], "id": "two-layer-tanh-mlp"},
    {
        "f": lambda device: lambda x: sigmoid(sigmoid(x)),
        "input_shapes": [(3,)],
        "id": "sigmoid-sigmoid",
    },
]


def setup_case(
    config: dict[str, Any], device: str = "cpu"
) -> tuple[Callable[..., Tensor], tuple[Any, ...]]:
    """Instantiate the function and the arguments on ``device``.

    Each case dict carries a function builder ``"f": device -> Callable`` and
    either ``"input_shapes": list[tuple[int, ...]]`` (flat tuple of plain
    tensors, materialized here) or ``"args_builder": device -> tuple`` (for
    pytree-shaped inputs).

    Args:
        config: Configuration dictionary of the test case.
        device: Device to materialize the function and arguments on.

    Returns:
        Tuple ``(f, args)`` where ``args`` is the materialized positional
        argument tuple.
    """
    manual_seed(0)
    f = config["f"](device)
    if "input_shapes" in config:
        dtype = dtype_for_device(device)
        args = tuple(
            rand(*s, dtype=dtype, device=device) for s in config["input_shapes"]
        )
    else:
        args = config["args_builder"](device)
    return f, args


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

    All coefficients are independently randomly drawn on the same device and
    dtype as the primal.
    """
    manual_seed(42)

    def make_leaf(t: Tensor) -> tuple[Tensor, ...]:
        if not collapsed:
            return (t, *(rand_like(t) for _ in range(K)))
        batched = [
            rand(R, *t.shape, dtype=t.dtype, device=t.device) for _ in range(K - 1)
        ]
        return (t, *batched, rand_like(t))

    return tree_map(make_leaf, args)


def rev_collapsed_jet(f: Callable[..., Any]) -> Callable[..., Any]:
    """Reference implementation for collapsed Taylor mode via :func:`jet.rev_jet`.

    Built on :func:`jet.rev_jet` (nested reverse-mode AD), so independent of
    the FX-trace and interpreter machinery. See :func:`jet.jet` for the
    collapsed-mode shape contract.

    Run a standard ``rev_jet`` per direction (full ``c_K`` into direction 0,
    zeros into the rest, exploiting that ``o_K`` is linear in ``c_K``), then
    combine: orders 1..K-1 stack across ``R``, order K sums. (``rev_jet``
    uses :func:`torch.autograd.grad` which does not compose with
    :func:`torch.func.vmap`, hence the explicit Python loop over ``R``.)
    """
    std_jet = rev_jet(f)

    def cjet_f(*args: Any) -> Any:
        leaves, _ = tree_flatten(args, is_leaf=_is_jet_leaf)
        K = len(leaves[0]) - 1
        R = leaves[0][1].shape[0]
        for leaf in leaves:
            if len(leaf) - 1 != K:
                raise ValueError(f"K mismatch across leaves: {K} vs {len(leaf) - 1}.")
            if leaf[1].shape[0] != R:
                raise ValueError(
                    f"R mismatch across leaves: {R} vs {leaf[1].shape[0]}."
                )

        def direction(leaf: tuple[Tensor, ...], r: int) -> tuple[Tensor, ...]:
            c_K = leaf[K] if r == 0 else zeros_like(leaf[K])
            return (leaf[0], *(leaf[k][r] for k in range(1, K)), c_K)

        per_dir = [
            std_jet(
                *tree_map(
                    lambda leaf, r=r: direction(leaf, r), args, is_leaf=_is_jet_leaf
                )
            )
            for r in range(R)
        ]

        def combine(*jets: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            return (
                jets[0][0],
                *(stack([j[k] for j in jets]) for k in range(1, K)),
                sum(j[K] for j in jets),
            )

        return tree_map(combine, *per_dir, is_leaf=_is_jet_leaf)

    return cjet_f


def assert_jet_matches_oracle(
    config: dict[str, Any], K: int, collapsed: bool, device: str = "cpu"
) -> None:
    """Assert ``jet(f, mock_args, collapsed)`` matches its mode-specific oracle.

    The oracle is :func:`jet.rev_jet` (standard) or :func:`rev_collapsed_jet`
    (collapsed). Both are built on nested reverse-mode AD and are independent
    of the FX-trace + interpreter machinery under test.
    """
    f, mock_args = setup_case(config, device)
    jet_args = make_jet_args(mock_args, K, collapsed=collapsed)
    oracle = rev_collapsed_jet(f) if collapsed else rev_jet(f)
    actual = jet(f, mock_args, collapsed=collapsed)(*jet_args)
    expected = oracle(*jet_args)
    assert_close(actual, expected, **tolerances_for_device(device))
