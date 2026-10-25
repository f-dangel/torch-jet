"""End-to-end Taylor-mode tests on torchvision CNNs (eval mode, CPU, float64).

torchvision pins torch exactly, so it is not a package extra: CI installs it in
a separate step, and this module is skipped when it is absent.
"""

from math import factorial
from typing import Callable

from pytest import importorskip, mark
from torch import Tensor, float64, manual_seed, randn, zeros_like
from torch.func import jvp
from torch.testing import assert_close

from jet import jet

models = importorskip("torchvision.models")

#: Models that trace end to end, with the input resolution they need.
MODELS: dict[str, tuple[Callable, int]] = {
    "alexnet": (models.alexnet, 64),
    "vgg11_bn": (models.vgg11_bn, 32),
    "resnet18": (models.resnet18, 32),
    "squeezenet1_0": (models.squeezenet1_0, 64),
    "densenet121": (models.densenet121, 32),
    "googlenet": (lambda: models.googlenet(init_weights=False, aux_logits=False), 96),
    "regnet_x_400mf": (models.regnet_x_400mf, 32),
    "mnasnet0_5": (models.mnasnet0_5, 32),
    "mobilenet_v2": (models.mobilenet_v2, 32),
}


def _directional_derivatives(f: Callable, x: Tensor, v: Tensor, K: int) -> list[Tensor]:
    """Compute ``d^k/dt^k f(x + t v)`` at ``t = 0`` for ``k = 0..K`` by nested ``jvp``.

    Forward-over-forward is far cheaper on a CNN than the nested reverse-mode
    oracle of the other tests (seconds vs. minutes per order).
    """
    ds, g = [f(x)], f
    for _ in range(K):
        g = lambda y, g=g: jvp(g, (y,), (v,))[1]  # noqa: E731
        ds.append(g(x))
    return ds


@mark.parametrize("K", [2, 3], ids=lambda k: f"K{k}")
@mark.parametrize("name", MODELS)
def test_torchvision_model(name: str, K: int):
    """``jet`` of a torchvision CNN matches its nested-``jvp`` directional derivatives."""
    manual_seed(0)
    ctor, res = MODELS[name]
    model = ctor().eval().to(float64)
    x, v = randn(1, 3, res, res, dtype=float64), randn(1, 3, res, res, dtype=float64)
    coeffs = jet(model, (x,))((x, v, *[zeros_like(x)] * (K - 1)))
    for k, d in enumerate(_directional_derivatives(model, x, v, K)):
        assert_close(coeffs[k] * factorial(k), d)
