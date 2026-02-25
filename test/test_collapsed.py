"""Tests for collapsed Taylor mode (interpreter-level collapsing)."""

from typing import Any

from pytest import mark
from torch import (
    cos,
    eye,
    float64,
    manual_seed,
    rand,
    sigmoid,
    sin,
    tanh,
    zeros,
    zeros_like,
)
from torch.func import hessian, vmap
from torch.nn import Linear, Sequential, Tanh

import jet
from jet.collapsed import collapsed_jet
from jet.laplacian import laplacian
from test.utils import report_nonclose

manual_seed(0)

# ---------------------------------------------------------------------------
# Test cases: MLP and elementwise functions
# ---------------------------------------------------------------------------

COLLAPSED_CASES = [
    {"f": sin, "shape": (2,), "id": "sin"},
    {"f": cos, "shape": (3,), "id": "cos"},
    {"f": tanh, "shape": (5,), "id": "tanh"},
    {"f": sigmoid, "shape": (4,), "id": "sigmoid"},
    {"f": lambda x: x**2, "shape": (5,), "id": "pow-2"},
    {"f": lambda x: x + 2.0, "shape": (5,), "id": "add-scalar"},
    {"f": lambda x: x * 3.0, "shape": (5,), "id": "mul-scalar"},
    {"f": lambda x: x * x, "shape": (5,), "id": "mul-x-x"},
    {"f": lambda x: sin(sin(x)), "shape": (2,), "id": "sin-sin"},
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
    {"f": Linear(4, 2), "shape": (4,), "id": "linear"},
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    {"f": lambda x: sin(x) + x, "shape": (3,), "id": "sin-residual"},
    {"f": lambda x: x.sum(0), "shape": (3, 5), "id": "sum-3"},
]

COLLAPSED_IDS = [c["id"] for c in COLLAPSED_CASES]


def _compare_collapsed_vs_standard(f, x, K, R=None):
    """Compare collapsed jet output against standard jet + vmap + sum.

    For a function f with derivative order K and R directions:
    - Standard: vmap over R directions, each a full K-jet, then sum K-th coeff
    - Collapsed: single call with batched 1..K-1 and collapsed K-th = 0

    These should produce identical K-th order outputs.
    """
    from torch.nn import Module

    if isinstance(f, Module):
        f = f.double()
    x = x.double()
    in_shape = x.shape
    in_dim = x.numel()
    if R is None:
        R = in_dim

    mock_x = zeros(*in_shape, dtype=float64)

    # Directions: identity basis (or random if R != in_dim)
    manual_seed(42)
    if R == in_dim:
        E = eye(R, dtype=float64).reshape(R, *in_shape)
    else:
        E = rand(R, *in_shape, dtype=float64)

    # --- Standard approach: vmap + sum ---
    jet_f = jet.jet(f, K, (mock_x,))
    z = zeros_like(x)
    series_zeros = tuple((z,) for _ in range(K))

    def single_jet(x1):
        series = ((x1,),) + tuple((z,) for _ in range(K - 1))
        return jet_f((x,), series)

    vmapped = vmap(
        single_jet, randomness="different", out_dims=(None, tuple(0 for _ in range(K)))
    )
    F0_std, Fs_std = vmapped(E)
    FK_std_summed = Fs_std[K - 1].sum(0)

    # --- Collapsed approach ---
    cjet_f = collapsed_jet(f, K, (mock_x,))
    z = zeros_like(x)
    # series: K-1 batched entries + 1 collapsed entry
    batched_series = tuple(
        (E,) if i == 0 else (zeros(R, *in_shape, dtype=float64),) for i in range(K - 1)
    )
    collapsed_series = ((z,),)
    series = batched_series + collapsed_series
    F0_col, Fs_col = cjet_f((x,), series)

    FK_col = Fs_col[K - 1]

    # Compare
    report_nonclose(F0_std, F0_col, name="Primals")
    report_nonclose(FK_std_summed, FK_col, name=f"Collapsed K={K} coefficient")

    return FK_std_summed, FK_col


# ---------------------------------------------------------------------------
# Test: collapsed K=2 matches standard jet + sum for various functions
# ---------------------------------------------------------------------------


@mark.parametrize("config", COLLAPSED_CASES, ids=COLLAPSED_IDS)
def test_collapsed_k2(config: dict[str, Any]):
    """Collapsed 2-jet matches standard 2-jet + vmap + sum."""
    manual_seed(0)
    f = config["f"]
    shape = config["shape"]
    from torch.nn import Module

    if isinstance(f, Module):
        f = f.double()

    x = rand(*shape, dtype=float64)
    _compare_collapsed_vs_standard(f, x, K=2)


# ---------------------------------------------------------------------------
# Test: collapsed Laplacian matches Hessian trace
# ---------------------------------------------------------------------------

LAPLACIAN_CASES = [
    {"f": sin, "shape": (3,), "id": "sin"},
    {"f": lambda x: sin(sin(x)), "shape": (2,), "id": "sin-sin"},
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    {"f": lambda x: sigmoid(sigmoid(x)), "shape": (3,), "id": "sigmoid-sigmoid"},
]

LAPLACIAN_IDS = [c["id"] for c in LAPLACIAN_CASES]


@mark.parametrize("config", LAPLACIAN_CASES, ids=LAPLACIAN_IDS)
def test_collapsed_laplacian(config: dict[str, Any]):
    """Collapsed Laplacian matches Hessian trace."""
    manual_seed(0)
    f = config["f"]
    shape = config["shape"]
    from torch.nn import Module

    if isinstance(f, Module):
        f_orig = f.double()
    else:
        f_orig = f

    x = rand(*shape, dtype=float64)
    mock_x = zeros(*shape, dtype=float64)

    # Reference: Hessian trace
    H = hessian(f_orig)(x)
    # H has shape (*out_shape, *in_shape, *in_shape)
    # For scalar output: (in_dim, in_dim), trace = sum of diagonal
    # For vector output: (*out_shape, *in_shape, *in_shape)
    out = f_orig(x)
    if out.ndim == 0:
        lap_ref = H.trace()
    else:
        # Flatten Hessian to (..., D, D) and take trace over last 2 dims
        D = x.numel()
        H_flat = H.reshape(*out.shape, D, D)
        lap_ref = H_flat.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Collapsed Taylor mode Laplacian
    lap_f = laplacian(f_orig, mock_x)
    _, _, lap_col = lap_f(x)

    report_nonclose(lap_ref, lap_col, name="Laplacian")


# ---------------------------------------------------------------------------
# Test: collapsed K=3 and K=4 (higher orders)
# ---------------------------------------------------------------------------


@mark.parametrize("K", [3, 4], ids=["K=3", "K=4"])
@mark.parametrize(
    "config",
    [
        {"f": sin, "shape": (2,), "id": "sin"},
        {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
        {"f": Linear(3, 2), "shape": (3,), "id": "linear"},
    ],
    ids=["sin", "tanh-tanh", "linear"],
)
def test_collapsed_higher_order(config: dict[str, Any], K: int):
    """Collapsed higher-order jets match standard jets + sum."""
    manual_seed(0)
    f = config["f"]
    shape = config["shape"]
    from torch.nn import Module

    if isinstance(f, Module):
        f = f.double()

    x = rand(*shape, dtype=float64)
    _compare_collapsed_vs_standard(f, x, K=K)
