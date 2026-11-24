"""Taylor mode through a training loop with PyTorch's functional SGD kernel.

``torch.optim.sgd.sgd`` updates the parameters and momentum buffers it is
given in place, so the traced loop clones them first. The reference is the same
loop written out of place by hand, in the nested reverse-mode oracle. The
optimizer classes (``torch.optim.SGD``) are not supported.
"""

from functools import partial
from typing import Any

from pytest import mark, raises
from torch import Tensor, manual_seed, rand, tanh, zeros_like
from torch.func import grad
from torch.optim.sgd import sgd
from torch.testing import assert_close

from jet import jet
from jet._jet import _rev_jet
from test.test_torchvision import ORDER_IDS, ORDERS
from test.utils import device_kw, make_jet_args, rev_collapsed_jet, tolerances_for

#: Shapes of the MLP's parameters ``[W1, b1, W2]`` and of one batch ``(x, y)``.
PARAM_SHAPES, BATCH_SHAPES = [(2, 3), (2,), (1, 2)], [(4, 3), (4, 1)]
#: Number of training steps (each on its own batch) and the learning rate.
STEPS, LR = 2, 0.1
#: Plain SGD in the kernel's hyper-parameters; the cases override some of them.
PLAIN = dict(weight_decay=0, momentum=0, dampening=0, nesterov=False, maximize=False)
SGD_CASES = {
    "plain": PLAIN,
    "momentum": PLAIN | dict(momentum=0.9),
    "momentum_dampening_weight_decay": PLAIN
    | dict(momentum=0.9, dampening=0.5, weight_decay=0.1),
    "nesterov_weight_decay_maximize": PLAIN
    | dict(momentum=0.9, nesterov=True, weight_decay=0.1, maximize=True),
}


def _setup(device: str) -> tuple[list[Tensor], list[list[Tensor]]]:
    """Draw the initial parameters and one batch per training step."""
    manual_seed(0)
    kw = device_kw(device)
    params = [rand(*shape, **kw) for shape in PARAM_SHAPES]
    return params, [[rand(*s, **kw) for s in BATCH_SHAPES] for _ in range(STEPS)]


def _loss(params: list[Tensor], x: Tensor, y: Tensor) -> Tensor:
    """Mean squared error of a two-layer tanh MLP."""
    W1, b1, W2 = params
    return ((tanh(x @ W1.T + b1) @ W2.T - y) ** 2).mean()


def _train(params: list[Tensor], batches: list, zero_bufs: bool = True, **hyper: Any):
    """Train with the kernel; return the parameters and momentum buffers."""
    params = [p.clone() for p in params]
    bufs = [zeros_like(p) if zero_bufs else None for p in params]
    for x, y in batches:
        sgd(params, grad(_loss)(params, x, y), bufs, foreach=False, lr=LR, **hyper)
    return params, bufs


def _train_ref(params, batches, weight_decay, momentum, dampening, nesterov, maximize):
    """The loop of :func:`_train` out of place, written by hand."""
    bufs = [zeros_like(p) for p in params]
    for x, y in batches:
        new_params, new_bufs = [], []
        for p, g, b in zip(params, grad(_loss)(params, x, y), bufs):
            g = (-g if maximize else g) + weight_decay * p
            if momentum != 0:
                b = momentum * b + (1 - dampening) * g
                g = g + momentum * b if nesterov else b
            new_params.append(p - LR * g)
            new_bufs.append(b)
        params, bufs = new_params, new_bufs
    return params, bufs


@mark.parametrize("hyper", SGD_CASES.values(), ids=SGD_CASES)
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_sgd(hyper: dict[str, Any], K: int, collapsed: bool, device: str):
    """The jet of the kernel's training loop matches the hand-written loop's."""
    params, batches = _setup(device)
    jet_args = make_jet_args((params,), K, collapsed)
    train = partial(_train, batches=batches, **hyper)
    oracle = rev_collapsed_jet if collapsed else _rev_jet
    expected = oracle(partial(_train_ref, batches=batches, **hyper))(*jet_args)
    actual = jet(train, (params,), collapsed=collapsed)(*jet_args)
    assert_close(actual, expected, **tolerances_for(device))


def test_sgd_lazy_momentum_buffers_raise():
    """``None`` buffers are seeded with ``grad.detach()``, which has no rule."""
    params, batches = _setup("cpu")
    train = partial(_train, batches=batches, zero_bufs=False, **SGD_CASES["momentum"])
    with raises(NotImplementedError, match="detach"):
        jet(train, (params,))(*make_jet_args((params,), K=1))
