"""Guards for ``DifferentiableSubstitutes.SUBSTITUTES`` (see ``test.utils``).

One test per entry asserts, per device, the PyTorch limitation that justifies
the entry and that the substitute does not share it. When such a test fails,
PyTorch has caught up: drop the entry and the test.
"""

from typing import Callable

from pytest import raises
from torch import Tensor, manual_seed, rand, randn, zeros_like
from torch.autograd import grad
from torch.nn.functional import hardsigmoid, max_pool2d
from torch.testing import assert_close

from test.utils import DifferentiableSubstitutes, device_kw


def _second_derivative(f: Callable, x: Tensor) -> Tensor:
    """Differentiate ``f(x).sum()`` twice w.r.t. ``x``.

    A first derivative autograd already sees as constant (piecewise-linear
    ``f`` written with ``where``) has second derivative zero.
    """
    (g,) = grad(f(x).sum(), x, create_graph=True)
    return grad(g.sum(), x)[0] if g.requires_grad else zeros_like(g)


def test_max_pool2d_double_backward(device: str):
    """On MPS, ``max_pool2d`` after a nonlinearity is not twice differentiable.

    The ``return_indices=True`` substitute is, on every device, and agrees
    with the plain op where that works.
    """
    manual_seed(0)
    x = rand(1, 2, 6, 6, requires_grad=True, **device_kw(device))

    def squared_pool(pool: Callable) -> Callable:
        return lambda x: pool(x, kernel_size=2) ** 2

    expected = _second_derivative(
        squared_pool(DifferentiableSubstitutes.SUBSTITUTES[max_pool2d]), x
    )
    if device == "mps":
        with raises(RuntimeError, match="not infinitely differentiable"):
            _second_derivative(squared_pool(max_pool2d), x)
    else:
        assert_close(_second_derivative(squared_pool(max_pool2d), x), expected)


def test_hardsigmoid_double_backward(device: str):
    """``hardsigmoid`` is not twice differentiable on any device; its substitute is."""
    manual_seed(0)
    x = randn(10, requires_grad=True, **device_kw(device))
    with raises(RuntimeError, match="hardsigmoid_backward is not implemented"):
        _second_derivative(hardsigmoid, x)
    _second_derivative(DifferentiableSubstitutes.SUBSTITUTES[hardsigmoid], x)
