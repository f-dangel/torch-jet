from test.utils import VMAP_IDS, VMAPS, Sin
from typing import Callable

from pytest import mark
from torch import Tensor, eye, manual_seed, rand, vmap, zeros_like
from torch.autograd.functional import hessian
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Sigmoid, Tanh

from jet import jet


def laplacian_jet_loop(f, x):
    jet_f = jet(f, 2, verbose=True)
    lap = 0

    v2 = zeros_like(x)

    for i in range(x.numel()):
        v1 = zeros_like(x).flatten()
        v1[i] = 1.0
        v1 = v1.reshape(x.shape)
        _, _, d2i = jet_f(x, v1, v2)
        lap += d2i

    return lap


def laplacian_jet_torch_vmap(f, x):
    jet_f = jet(f, 2, verbose=True)
    v2 = zeros_like(x)

    def d2_f(v1):
        return jet_f(x, v1, v2)[2]

    dim_x = x.numel()
    v1s = eye(dim_x, dtype=x.dtype, device=x.device).reshape(dim_x, *x.shape)

    vmap_d2_f = vmap(d2_f)
    lap = vmap_d2_f(v1s).sum()

    return lap


def laplacian_jet_manual_vmap(f, x):
    dim = x.numel()
    jet_f = jet(f, k=2, vmap=True, verbose=True)
    X = x.unsqueeze(0).expand(dim, *x.shape)
    V1 = eye(x.numel(), dtype=x.dtype, device=x.device).reshape(dim, *x.shape)
    V2 = zeros_like(X)

    return jet_f(X, V1, V2)[2].sum()


def laplacian(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Compute the Laplacian of a scalar function.

    Args:
        f: The scalar function to compute the Hessian of.
        x: The point at which to compute the Hessian.

    Returns:
        The Hessian of the function f at the point x.
    """
    return hessian(f, x).trace()


def test_laplacian_vmap():
    """Compare Laplacian implementations."""
    manual_seed(0)
    mlp = Sequential(
        Linear(5, 4, bias=False),
        Tanh(),
        Linear(4, 3, bias=True),
        Sin(),
        Linear(3, 1, bias=True),
        Sigmoid(),
    )
    x = rand(5)

    # reference: Using PyTorch
    lap_rev = laplacian(mlp, x)

    # Using a jet to for-loop over the diagonal elements
    lap_jet_loop = laplacian_jet_loop(mlp, x)
    assert lap_rev.allclose(lap_jet_loop)
    print("Functorch and jet (loop) Laplacians match.")

    # Using a torch.vmap-ed jet to compute all diagonal elements
    # NOTE: This function is not trace-able, therefore we cannot
    # symbolically simplify it.
    lap_jet_torch_vmap = laplacian_jet_torch_vmap(mlp, x)
    assert lap_rev.allclose(lap_jet_torch_vmap)
    print("Functorch and jet (vmap) Laplacians match.")

    # Using a manually vmap-ed jet to compute all diagonal elements.
    # NOTE: This function is trace-able, therefore we can symbolically
    # simplify it.
    lap_jet_manual_vmap = laplacian_jet_manual_vmap(mlp, x)
    assert lap_rev.allclose(lap_jet_manual_vmap)
    print("Functorch and manually-jetted Laplacians match.")


@mark.parametrize("vmap", VMAPS, ids=VMAP_IDS)
def test_symbolic_trace_jet(vmap: bool):
    """Test whether the function produced by jet can be traced.

    Args:
        vmap: Whether to use vmap.
    """
    mlp = Sequential(
        Linear(5, 4, bias=False),
        Tanh(),
        Linear(4, 3, bias=True),
        Sin(),
        Linear(3, 1, bias=True),
        Sigmoid(),
    )
    # generate the jet's compute graph
    jet_f = jet(mlp, 2, vmap=vmap)

    # try tracing it
    print("Compute graph of jet function:")
    mod = symbolic_trace(jet_f)
    print(mod.graph)
