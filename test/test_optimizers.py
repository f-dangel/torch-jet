"""Taylor mode through a training loop with PyTorch's functional optimizer kernels.

``torch.optim.sgd.sgd``, ``torch.optim.adam.adam`` and ``torch.optim.adamw.adamw``
update the parameters and state tensors they are given in place, so the traced
loop clones the parameters first. The reference is the same loop written out of
place by hand, in the nested reverse-mode oracle. The optimizer classes
(``torch.optim.SGD``, ...) are not supported.
"""

from functools import partial
from typing import Any, Callable

from pytest import FixtureRequest, fixture, mark, raises, skip
from torch import Tensor, arange, manual_seed, maximum, rand, tanh, tensor, zeros_like
from torch.func import grad
from torch.optim.adam import adam
from torch.optim.adamw import adamw
from torch.optim.sgd import sgd
from torch.testing import assert_close
from torch.utils._pytree import tree_map

from jet import freeze, jet
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
ADAM = dict(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
ADAM |= dict(weight_decay=0, maximize=False, amsgrad=False)
ADAM_CASES = {
    "plain": ADAM,
    "weight_decay": ADAM | dict(weight_decay=0.1),
    "maximize": ADAM | dict(maximize=True),
    # With a small ``beta2``, old and new second moments both win the maximum.
    "amsgrad": ADAM | dict(amsgrad=True, beta2=0.5),
}
#: Why the Adam tests skip MPS; the input feature that the zero-gradient test zeroes.
MPS_SKIP, DEAD = "Adam is ill-conditioned in float32, the only float type on MPS.", 0


@fixture(params=[False, True], ids=["single", "foreach"])
def foreach(request: FixtureRequest) -> bool:
    """Run the test on the kernel's single-tensor and multi-tensor (foreach) path."""
    return request.param


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
        sgd(params, grad(_loss)(params, x, y), bufs, lr=LR, **hyper)
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
    """Train with an Adam kernel; return the parameters and the moment estimates."""
    params = [p.clone() for p in params]
    # The kernel needs moments made from the traced parameters and untraced counts.
    m, v, v_max = ([zeros_like(p) for p in params] for _ in range(3))
    steps = [tensor(0.0) for _ in params]
    max_v = v_max if hyper["amsgrad"] else []  # only amsgrad takes the maxima
    for x, y in batches:
        grads = grad(_loss)(params, x, y)
        kernel(params, grads, m, v, max_v, steps, **hyper)
    return params, m, v, v_max


def _train_adam_ref(
    params, batches, kernel, lr, beta1, beta2, eps, weight_decay, maximize, amsgrad
):
    """The loop of :func:`_train_adam` out of place, written by hand."""
    m, v, v_max = ([zeros_like(p) for p in params] for _ in range(3))
    for t, (x, y) in enumerate(batches, start=1):
        new, grads = [], grad(_loss)(params, x, y)
        for p, g, m_p, v_p, v_max_p in zip(params, grads, m, v, v_max):
            g = -g if maximize else g
            if kernel is adamw:  # decoupled weight decay
                p = p * (1 - lr * weight_decay)
            else:
                g = g + weight_decay * p
            m_p = beta1 * m_p + (1 - beta1) * g
            v_p = beta2 * v_p + (1 - beta2) * g * g
            v_max_p = maximum(v_max_p, v_p) if amsgrad else v_max_p
            denom = (v_max_p if amsgrad else v_p).sqrt() / (1 - beta2**t) ** 0.5 + eps
            new.append((p - lr / (1 - beta1**t) * m_p / denom, m_p, v_p, v_max_p))
        params, m, v, v_max = (list(state) for state in zip(*new))
    return params, m, v, v_max


def _adam_step(params: list, state: dict, t: Tensor, x: Tensor, y: Tensor, **hyper):
    """One Adam step after ``t`` steps: the new parameters and moment estimates."""
    m, v = state["exp_avg"], state["exp_avg_sq"]
    params, m, v = ([p.clone() for p in ps] for ps in (params, m, v))
    steps = [t.clone() for _ in params]
    adam(params, grad(_loss)(params, x, y), m, v, [], steps, **hyper)
    # No step count in the output: fed back it is a jet, whose ``item()`` has no rule.
    return params, {"exp_avg": m, "exp_avg_sq": v}


def _jets(train: Callable, ref: Callable, params: list, K: int, collapsed: bool):
    """Evaluate the jets of the kernel's loop ``train`` and the hand-written ``ref``."""
    jet_args = make_jet_args((params,), K, collapsed)
    oracle = rev_collapsed_jet if collapsed else _rev_jet
    return jet(train, (params,), collapsed=collapsed)(*jet_args), oracle(ref)(*jet_args)


def _assert_jet_matches(
    train: Callable, ref: Callable, K: int, collapsed: bool, device: str
):
    """The jet of the kernel's loop ``train`` matches the hand-written ``ref``'s."""
    params, batches = _setup(device)
    train, ref = partial(train, batches=batches), partial(ref, batches=batches)
    assert_close(*_jets(train, ref, params, K, collapsed), **tolerances_for(device))


@mark.parametrize("hyper", SGD_CASES.values(), ids=SGD_CASES)
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_sgd(hyper: dict, foreach: bool, K: int, collapsed: bool, device: str):
    """The jet of the kernel's training loop matches the hand-written loop's."""
    train, ref = partial(_train, foreach=foreach, **hyper), partial(_train_ref, **hyper)
    _assert_jet_matches(train, ref, K, collapsed, device)


@mark.parametrize("kernel", [adam, adamw], ids=["adam", "adamw"])
@mark.parametrize("hyper", ADAM_CASES.values(), ids=ADAM_CASES)
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_adam(
    kernel: Callable, hyper: dict, foreach: bool, K: int, collapsed: bool, device: str
):
    """The jet of an Adam kernel's training loop matches the hand-written loop's."""
    if device == "mps":
        skip(MPS_SKIP)
    train = partial(_train_adam, kernel=kernel, foreach=foreach, **hyper)
    ref = partial(_train_adam_ref, kernel=kernel, **hyper)
    _assert_jet_matches(train, ref, K, collapsed, device)


@mark.parametrize("case", ["plain", "amsgrad"])
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_adam_zero_gradient(
    case: str, foreach: bool, K: int, collapsed: bool, device: str
):
    """An entry whose gradient jet is exactly zero stays where it is, without NaN.

    A zero input feature gives its column of ``W1`` such a gradient: there Adam
    divides zero by the square root of zero, which has no derivatives. The other
    entries train as if the feature did not exist, which is the reference.
    """
    if device == "mps":
        skip(MPS_SKIP)
    params, batches = _setup(device)
    live = arange(params[0].shape[1], device=device) != DEAD
    hyper = dict(kernel=adam, **ADAM_CASES[case])
    dead_batches = [[x * live, y] for x, y in batches]
    train = partial(_train_adam, batches=dead_batches, foreach=foreach, **hyper)

    def ref(params: list[Tensor]):
        (W1, *rest), live_batches = params, [[x[:, live], y] for x, y in batches]
        return _train_adam_ref([W1[:, live], *rest], live_batches, **hyper)

    actual, expected = _jets(train, ref, params, K, collapsed)
    for got, want in zip(actual, expected):  # the parameters and each moment estimate
        assert_close([tuple(c[..., live] for c in got[0]), *got[1:]], want)
    # At the dead entries, the parameter keeps its jet and the moments are zero.
    (old,) = make_jet_args((params,), K, collapsed)
    new_W1, *moments = (state[0] for state in actual)
    for c, c_old in zip(new_W1, old[0]):
        assert_close(c[..., ~live], c_old[..., ~live], rtol=0, atol=0)
    assert all((c[..., ~live] == 0).all() for moment in moments for c in moment)


@mark.parametrize("frozen", [False, True], ids=["jet", "frozen"])
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_adam_stepwise(
    frozen: bool, foreach: bool, K: int, collapsed: bool, device: str
):
    """A step traced once and called on its own outputs matches the whole loop."""
    if device == "mps":
        skip(MPS_SKIP)
    (params, batches), hyper = _setup(device), dict(foreach=foreach, **ADAM)
    # The initial state, the number of steps done and the batch are constants.
    state = {key: [zeros_like(p) for p in params] for key in ("exp_avg", "exp_avg_sq")}
    mock = (params, state, tensor(0.0), *batches[0])
    jet_step = jet(partial(_adam_step, **hyper), mock, collapsed=collapsed)
    jet_args = make_jet_args((params,), K, collapsed)
    (jet_params,) = jet_args
    if frozen:  # ``freeze`` fixes the structure: the state is a zero jet from the start
        (state,) = tree_map(zeros_like, make_jet_args((state,), K, collapsed))
        jet_step = freeze(jet_step, (jet_params, state, *mock[2:]))
    # The frozen step is traced after 0 steps and also called after 1: a step count
    # baked into the trace would be silently wrong there.
    for t, (x, y) in enumerate(batches):
        jet_params, state = jet_step(jet_params, state, tensor(float(t)), x, y)
    train = partial(_train_adam, batches=batches, kernel=adam, **hyper)
    *expected, _ = jet(train, (params,), collapsed=collapsed)(*jet_args)
    actual = [jet_params, state["exp_avg"], state["exp_avg_sq"]]
    assert_close(actual, expected, **tolerances_for(device))


def test_sgd_lazy_momentum_buffers_raise(foreach: bool):
    """``None`` buffers are seeded with ``grad.detach()``, which has no rule."""
    params, batches = _setup("cpu")
    hyper = dict(zero_bufs=False, foreach=foreach, **SGD_CASES["momentum"])
    train = partial(_train, batches=batches, **hyper)
    with raises(NotImplementedError, match="detach"):
        jet(train, (params,))(*make_jet_args((params,), K=1))
