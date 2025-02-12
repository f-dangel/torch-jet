from test.utils import VMAP_IDS, VMAPS, Sin
from typing import Callable

from pytest import mark
from torch import Size, Tensor, eye, manual_seed, rand, zeros_like
from torch.autograd.functional import hessian
from torch.fx import symbolic_trace
from torch.nn import Linear, Module, Sequential, Sigmoid, Tanh

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


class Laplacian(Module):
    def __init__(self, f, x_shape):
        super().__init__()
        self.jet_f = jet(f, 2, vmap=True)
        self.dim = x_shape.numel()
        self.shape = x_shape

    def forward(self, x):
        X = x.unsqueeze(0).expand(self.dim, *[-1 * len(self.shape)])
        V1 = eye(self.dim).reshape(self.dim, *self.shape)
        V2 = zeros_like(X)
        return self.jet_f(X, V1, V2)[2].sum(0)


def laplacian(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Compute the Laplacian of a scalar function.

    Args:
        f: The scalar function to compute the Hessian of.
        x: The point at which to compute the Hessian.

    Returns:
        The Hessian of the function f at the point x.
    """
    return hessian(f, x).trace()


def test_laplacian():
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

    # Using a manually-vmapped jet that is traceable
    # NOTE: This module is trace-able, therefore we can symbolically simplify it.
    lap_mod = Laplacian(mlp, x.shape)(x)
    assert lap_rev.allclose(lap_mod)
    print("Functorch and module-vmapped traceable Laplacian module match.")


@mark.parametrize("vmap", VMAPS, ids=VMAP_IDS)
def test_symbolic_trace_jet(vmap: bool):
    """Test whether the function produced by jet can be traced.

    Args:
        vmap: Whether to use vmap.
    """
    mlp = Sequential(
        Linear(5, 1, bias=False),
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


def test_symbolic_trace_Laplacian():
    """Test whether the Laplacian module is trace-able."""
    mlp = Sequential(
        Linear(5, 1, bias=False),
        Tanh(),
        Linear(4, 3, bias=True),
        Sin(),
        Linear(3, 1, bias=True),
        Sigmoid(),
    )
    lap = Laplacian(mlp, Size([5]))

    # try tracing the Laplacian module
    print("Compute graph of manually Laplacian module:")
    mod = symbolic_trace(lap)
    print(mod.graph)
