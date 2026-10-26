"""Guards for ``DifferentiableSubstitutes.SUBSTITUTES`` (see ``test.utils``).

One test per entry asserts, per device, the PyTorch limitation that justifies
the entry and that the substitute does not share it. When such a test fails,
PyTorch has caught up: drop the entry and the test.
"""

from typing import Callable

from pytest import raises
from torch import Tensor, rand
from torch.autograd import grad
from torch.nn.functional import max_pool2d

from test.utils import DifferentiableSubstitutes, device_kw


def _second_derivative(f: Callable, x: Tensor) -> Tensor:
    """Differentiate ``f(x).sum()`` twice w.r.t. ``x``."""
    (g,) = grad(f(x).sum(), x, create_graph=True)
    return grad(g.sum(), x)[0]


def test_max_pool2d_double_backward(device: str):
    """On MPS, ``max_pool2d`` after a nonlinearity is not twice differentiable.

    Other devices are fine, and so is the ``return_indices=True`` substitute.
    """
    x = rand(1, 2, 6, 6, requires_grad=True, **device_kw(device))

    def squared_pool(pool: Callable) -> Callable:
        return lambda x: pool(x, kernel_size=2) ** 2

    if device == "mps":
        with raises(RuntimeError, match="not infinitely differentiable"):
            _second_derivative(squared_pool(max_pool2d), x)
    else:
        _second_derivative(squared_pool(max_pool2d), x)
    _second_derivative(
        squared_pool(DifferentiableSubstitutes.SUBSTITUTES[max_pool2d]), x
    )
