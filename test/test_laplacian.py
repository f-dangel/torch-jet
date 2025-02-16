from functools import partial
from test.utils import VMAP_IDS, VMAPS, Sin
from typing import Callable

from pytest import mark
from torch import Tensor, manual_seed, rand, zeros, zeros_like
from torch.autograd.functional import hessian
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Sigmoid, Tanh

from jet import jet
from jet.laplacian import Laplacian


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


def laplacian(f: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Compute the Laplacian of a tensor-to-tensor function.

    Args:
        f: The function to compute the Laplacian of.
        x: The point at which to compute the Laplacian.

    Returns:
        The Laplacian of the function f at the point x, evaluated
        for each element f[i](x). Has same shape as f(x)
    """
    out = f(x)

    def f_flat(x_flat, i):
        return f(x_flat.reshape_as(x)).flatten()[i]

    lap = zeros_like(out).flatten()
    for i in range(out.numel()):
        f_i = partial(f_flat, i=i)
        lap[i] = hessian(f_i, x.flatten()).trace()

    return lap.reshape_as(out)


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
    _, _, lap_mod = Laplacian(mlp, x, is_batched=False)(x)
    assert lap_rev.allclose(lap_mod)
    print("Functorch and module-vmapped traceable Laplacian module match.")


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


def test_symbolic_trace_Laplacian():
    """Test whether the Laplacian module is trace-able."""
    mlp = Sequential(
        Linear(5, 4, bias=False),
        Tanh(),
        Linear(4, 3, bias=True),
        Sin(),
        Linear(3, 1, bias=True),
        Sigmoid(),
    )
    x_dummy = zeros(5)
    lap = Laplacian(mlp, x_dummy, is_batched=False)

    # try tracing the Laplacian module
    print("Compute graph of manually Laplacian module:")
    mod = symbolic_trace(lap)
    print(mod.graph)
