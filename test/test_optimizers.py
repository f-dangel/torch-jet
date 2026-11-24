"""Taylor mode through a training loop with PyTorch's functional optimizer kernels.

``torch.optim.sgd.sgd``, ``torch.optim.adam.adam`` and ``torch.optim.adamw.adamw``
update the parameters and state tensors they are given in place, so the traced
loop clones the parameters first. The reference is the same loop written out of
place by hand, in the nested reverse-mode oracle. The optimizer classes
(``torch.optim.SGD``, ...) are not supported.
"""

from functools import partial
from typing import Any, Callable

from pytest import mark, raises, skip
from torch import Tensor, manual_seed, rand, tanh, tensor, zeros_like
from torch.func import grad
from torch.optim.adam import adam
from torch.optim.adamw import adamw
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
#: Plain Adam in the kernels' hyper-parameters; the cases override some of them.
ADAM = dict(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, maximize=False)
ADAM_CASES = {
    "plain": ADAM,
    "weight_decay": ADAM | dict(weight_decay=0.1),
    "maximize": ADAM | dict(maximize=True),
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


def _train_adam(params: list[Tensor], batches: list, kernel: Callable, **hyper: Any):
    """Train with an Adam kernel; return the parameters and both moment estimates."""
    params = [p.clone() for p in params]
    # The kernel needs moments made from the traced parameters and untraced counts.
    m, v = ([zeros_like(p) for p in params] for _ in range(2))
    steps = [tensor(0.0) for _ in params]
    for x, y in batches:
        grads = grad(_loss)(params, x, y)
        kernel(params, grads, m, v, [], steps, foreach=False, amsgrad=False, **hyper)
    return params, m, v


def _train_adam_ref(
    params, batches, kernel, lr, beta1, beta2, eps, weight_decay, maximize
):
    """The loop of :func:`_train_adam` out of place, written by hand."""
    m, v = ([zeros_like(p) for p in params] for _ in range(2))
    for t, (x, y) in enumerate(batches, start=1):
        new = []
        for p, g, m_p, v_p in zip(params, grad(_loss)(params, x, y), m, v):
            g = -g if maximize else g
            if kernel is adamw:  # decoupled weight decay
                p = p * (1 - lr * weight_decay)
            else:
                g = g + weight_decay * p
            m_p = beta1 * m_p + (1 - beta1) * g
            v_p = beta2 * v_p + (1 - beta2) * g * g
            denom = v_p.sqrt() / (1 - beta2**t) ** 0.5 + eps
            new.append((p - lr / (1 - beta1**t) * m_p / denom, m_p, v_p))
        params, m, v = (list(state) for state in zip(*new))
    return params, m, v


def _assert_jet_matches(
    train: Callable, ref: Callable, K: int, collapsed: bool, device: str
):
    """The jet of the kernel's loop ``train`` matches the hand-written ``ref``'s."""
    params, batches = _setup(device)
    jet_args = make_jet_args((params,), K, collapsed)
    oracle = rev_collapsed_jet if collapsed else _rev_jet
    expected = oracle(partial(ref, batches=batches))(*jet_args)
    jet_train = jet(partial(train, batches=batches), (params,), collapsed=collapsed)
    assert_close(jet_train(*jet_args), expected, **tolerances_for(device))


@mark.parametrize("hyper", SGD_CASES.values(), ids=SGD_CASES)
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_sgd(hyper: dict[str, Any], K: int, collapsed: bool, device: str):
    """The jet of the kernel's training loop matches the hand-written loop's."""
    train, ref = partial(_train, **hyper), partial(_train_ref, **hyper)
    _assert_jet_matches(train, ref, K, collapsed, device)


@mark.parametrize("kernel", [adam, adamw], ids=["adam", "adamw"])
@mark.parametrize("hyper", ADAM_CASES.values(), ids=ADAM_CASES)
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_adam(
    kernel: Callable, hyper: dict[str, Any], K: int, collapsed: bool, device: str
):
    """The jet of an Adam kernel's training loop matches the hand-written loop's."""
    if device == "mps":
        skip("Adam is ill-conditioned in float32, the only float type on MPS.")
    train = partial(_train_adam, kernel=kernel, **hyper)
    ref = partial(_train_adam_ref, kernel=kernel, **hyper)
    _assert_jet_matches(train, ref, K, collapsed, device)


def test_sgd_lazy_momentum_buffers_raise():
    """``None`` buffers are seeded with ``grad.detach()``, which has no rule."""
    params, batches = _setup("cpu")
    train = partial(_train, batches=batches, zero_bufs=False, **SGD_CASES["momentum"])
    with raises(NotImplementedError, match="detach"):
        jet(train, (params,))(*make_jet_args((params,), K=1))
