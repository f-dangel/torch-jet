"""Guards for ``DifferentiableSubstitutes.SUBSTITUTES`` (see ``test.utils``).

One test per entry asserts, per device, the PyTorch limitation that justifies
the entry and that the substitute does not share it. When such a test fails,
PyTorch has caught up: drop the entry and the test.
"""

from typing import Callable

from pytest import raises
from torch import Tensor, manual_seed, rand, randn, zeros, zeros_like
from torch.autograd import grad
from torch.nn.functional import hardsigmoid, layer_norm, max_pool2d
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


def test_layer_norm_third_derivative(device: str):
    """``layer_norm``'s third derivative is wrong on every device; its substitute's is not.

    pytorch/pytorch#175754. The pending upstream fix raises on the third
    derivative instead of returning a wrong value; either way this cell flags it.

    Without affine parameters, ``layer_norm(x).sum()`` is identically zero, so
    its third directional derivative must vanish.
    """
    manual_seed(0)
    kw = device_kw(device)
    x, v = randn(2, 8, **kw), randn(2, 8, **kw)

    def third_directional(f: Callable) -> Tensor:
        t = zeros((), requires_grad=True, **kw)
        y = f(x + t * v, (8,)).sum()
        for _ in range(3):
            (y,) = grad(y, t, create_graph=True)
        return y

    with raises(AssertionError):
        assert_close(third_directional(layer_norm), zeros((), **kw))
    substitute = DifferentiableSubstitutes.SUBSTITUTES[layer_norm]
    assert_close(third_directional(substitute), zeros((), **kw))
