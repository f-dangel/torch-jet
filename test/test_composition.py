"""Composition tests for Taylor mode.

Exercise the interpreter / tracing machinery on realistic compute graphs.
Not primitive coverage -- see ``test_primitives.py`` for that.
"""

from functools import partial
from typing import Any, Callable

from pytest import mark
from torch import Tensor, cos, manual_seed, rand, randn, sin, tanh
from torch.nn import (
    Hardsigmoid,
    Hardswish,
    Linear,
    Module,
    ReLU,
    ReLU6,
    Sequential,
    SiLU,
)
from torch.nn.functional import cross_entropy, rms_norm

from test.utils import (
    _stateless,
    assert_jet_matches_oracle,
    class_index_loss,
    device_kw,
    mlp,
)


def _deep_pytree_f(x: Tensor, params: list) -> tuple[Tensor, dict[str, Tensor]]:
    """Pytree-shaped composition: nested list input, mixed tuple/dict output."""
    w, (b0, b1) = params
    h = sin(x) * w
    return (h + b0, {"a": cos(h) * b1, "b": tanh(h + b0 + b1)})


def _inplace_mlp(device: str, activation: type[Module]) -> Sequential:
    """MLP with in-place activations -- torchvision's pattern (ReLU, ReLU6, ...)."""
    manual_seed(0)
    kw = device_kw(device)
    return Sequential(
        Linear(5, 4, **kw),
        activation(inplace=True),
        Linear(4, 3, **kw),
        activation(inplace=True),
        Linear(3, 1, **kw),
    )


def _mobilenet_mlp(device: str) -> Callable[[Tensor], Tensor]:
    """In-place-ReLU6 MLP gated by a ``Hardsigmoid`` -- MobileNetV3's pattern."""
    net = _inplace_mlp(device, ReLU6)
    manual_seed(1)
    gate = Sequential(Linear(5, 1, **device_kw(device)), Hardsigmoid(inplace=True))
    return lambda x: net(x) * gate(x)


def _channel_shuffle(x: Tensor, groups: int = 2) -> Tensor:
    """Channel shuffle as in torchvision's ShuffleNet: view, transpose, contiguous, view."""
    N, C, H, W = x.shape
    x = x.view(N, groups, C // groups, H, W).transpose(1, 2).contiguous()
    return x.view(N, C, H, W)


COMPOSITION_CASES = [
    {
        "id": "scalar_Rn_to_R",
        "f": _stateless(lambda x: (sin(x) * x).sum(0)),
        "args_fn": lambda: (rand(5),),
    },
    {"id": "mlp", "f": mlp, "args_fn": lambda: (rand(5),)},
    {"id": "mlp_batched", "f": mlp, "args_fn": lambda: (rand(10, 5),)},
    {
        "id": "relu_inplace_mlp",
        "f": partial(_inplace_mlp, activation=ReLU),
        "args_fn": lambda: (randn(5),),
    },
    {
        "id": "relu6_inplace_mlp",
        "f": partial(_inplace_mlp, activation=ReLU6),
        "args_fn": lambda: (3 * randn(5),),
    },
    {"id": "mobilenet_mlp", "f": _mobilenet_mlp, "args_fn": lambda: (3 * randn(5),)},
    {
        "id": "hardswish_inplace_mlp",
        "f": partial(_inplace_mlp, activation=Hardswish),
        "args_fn": lambda: (4 * randn(5),),
    },
    {
        "id": "silu_inplace_mlp",
        "f": partial(_inplace_mlp, activation=SiLU),
        "args_fn": lambda: (randn(5),),
    },
    {
        "id": "deep_pytree",
        "f": _stateless(_deep_pytree_f),
        "args_fn": lambda: (rand(3), [rand(3), [rand(3), rand(3)]]),
    },
    {
        "id": "sin_residual",
        # variable aliasing: ``x`` appears both inside ``sin`` and outside.
        "f": _stateless(lambda x: sin(x) + x),
        "args_fn": lambda: (rand(3),),
    },
    {
        "id": "multi_input",
        "f": _stateless(lambda x, y: sin(x) * cos(y)),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "multi_input_tuple_output",
        "f": _stateless(lambda x, y: (x + y, x * y)),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "multi_input_dict_output",
        "f": _stateless(lambda x, y: {"sum": x + y, "prod": x * y}),
        "args_fn": lambda: (rand(4), rand(4)),
    },
    {
        "id": "dict_only_input",
        "f": _stateless(lambda d: sin(d["a"]) * d["b"]),
        "args_fn": lambda: ({"a": rand(4), "b": rand(4)},),
    },
    {
        "id": "dict_first_input",
        "f": _stateless(lambda params, x: params["scale"] * sin(x) + params["bias"]),
        "args_fn": lambda: ({"scale": rand(3), "bias": rand(3)}, rand(3)),
    },
    # ``nn.CrossEntropyLoss`` (class-index targets) decomposes into
    # ``_log_softmax`` + ``nll_loss_forward`` + ``getitem``. Differentiate
    # w.r.t. the logits; the integer target (and optional per-class weight) are
    # frozen constants.
    *(
        {
            "id": f"cross_entropy_class_index_{reduction}"
            + ("_weighted" if weighted else ""),
            "f": class_index_loss(cross_entropy, reduction, 8, 5, weighted=weighted),
            "args_fn": lambda: (rand(8, 5),),
        }
        for reduction in ("mean", "sum", "none")
        for weighted in (False, True)
    ),
    # ``nn.RMSNorm`` traces to ``pow`` / ``mean`` / ``add`` / ``rsqrt`` / ``mul``
    # (the fused CUDA/MPS kernel is decomposed); the weight is a second jet.
    # Inputs are shifted off zero so the rsqrt derivatives stay well-conditioned.
    {
        "id": "rms_norm",
        "f": _stateless(lambda x: rms_norm(x, (3,))),
        "args_fn": lambda: (rand(2, 4, 3) + 0.5,),
    },
    {
        "id": "rms_norm_weighted",
        "f": _stateless(lambda x, w: rms_norm(x, (3,), w)),
        "args_fn": lambda: (rand(2, 4, 3) + 0.5, rand(3)),
    },
    {
        "id": "channel_shuffle",
        # view -> transpose -> contiguous (clone) -> view, then a nonlinearity so
        # the higher-order coefficients are non-trivial.
        "f": _stateless(lambda x: sin(_channel_shuffle(x)) * x),
        "args_fn": lambda: (rand(2, 4, 3, 3),),
    },
]


@mark.parametrize("config", COMPOSITION_CASES, ids=lambda c: c["id"])
def test_composition(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(composition)`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
