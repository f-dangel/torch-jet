from functools import partial
from test.utils import VMAP_IDS, VMAPS, Sin
from typing import Callable

from numpy import mean, std
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
    ).double()
    x = rand(5).double()

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

    # TODO Make sure that if we use the randomized Laplacian and we crank up
    # the number of Monte-Carlo samples, it converges.
    manual_seed(0)  # make deterministic
    # max_samples = 100_000
    # num_samples = 900_000

    # lap_randlacian = 0
    # used_samples = 0
    # converged = False

    for dist in ["normal", "rademacher"]:

        print(dist)
        num_repeats = 10
        for num_samples in [16, 64, 250, 1_000, 4_000, 16_000, 64_000, 256_000]:
            lap_rands = [
                Laplacian(mlp, x, is_batched=False, randomize=(dist, num_samples))(x)[
                    2
                ].detach()
                for _ in range(num_repeats)
            ]
            lap_mean = mean(lap_rands)
            lap_std = std(lap_rands)

            rel_error = (lap_mean - lap_rev).abs() / lap_rev.abs()
            rel_std = lap_std / lap_rev.abs()
            print(num_samples, rel_error.item())  # , rel_std.item())
    raise Exception("Fail")

    # while not converged or used_samples < max_samples:
    #     # Using a manually-vmapped jet that is traceable
    #     _, _, lap_rand = Laplacian(mlp, x, is_batched=False, randomize=("normal", num_samples))(x)

    #     # update the MC estimator
    #     raise NotImplementedError("TODO")

    #     used_samples += num_samples

    #     rel_error = (lap_rand - lap_rev).abs() / lap_rev.abs()

    #     if lap_rev.allclose(lap_randlacian):
    #         converged = True

    # raise NotImplementedError("TODO")


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
