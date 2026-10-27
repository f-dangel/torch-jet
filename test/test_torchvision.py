"""End-to-end Taylor-mode tests on torchvision CNNs (eval mode).

torchvision pins torch exactly, so it is a ``test`` extra, not a dependency.
A missing torchvision fails this module rather than skipping it, so a broken
install cannot pass silently.
"""

from typing import Any, Callable

from pytest import mark
from torch import Tensor, manual_seed, randn
from torchvision import models

from test.utils import assert_jet_matches_oracle, device_kw

#: Mark the whole module so CI can run it in its own job (``-m models``).
pytestmark = mark.models

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
    "shufflenet_v2_x0_5": (models.shufflenet_v2_x0_5, 32),
    "efficientnet_b0": (models.efficientnet_b0, 32),
    "mobilenet_v3_small": (models.mobilenet_v3_small, 32),
    "convnext_tiny": (models.convnext_tiny, 32),
    "fcn_resnet50": (
        lambda: models.segmentation.fcn_resnet50(
            weights=None, weights_backbone=None, aux_loss=False
        ),
        32,
    ),
    "deeplabv3_mobilenet_v3_large": (
        lambda: models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, aux_loss=False
        ),
        32,
    ),
    "lraspp_mobilenet_v3_large": (
        lambda: models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None
        ),
        32,
    ),
}


def _out(model: Callable, x: Tensor) -> Tensor:
    """Forward ``x``; segmentation models return a dict with the logits at ``"out"``."""
    out = model(x)
    return out["out"] if isinstance(out, dict) else out


def _projected(ctor: Callable, res: int) -> Callable[[str], Callable]:
    """Build ``x -> sum(model(x) * w)`` with a fixed random ``w`` on ``device``.

    The scalar output keeps the nested reverse-mode oracle cheap (it
    differentiates each output element separately) while every logit still
    contributes. The parameters are frozen: only the input is differentiated,
    so autograd need not record graphs for them.
    """

    def builder(device: str) -> Callable:
        manual_seed(0)
        kw = device_kw(device)
        model = ctor().eval().requires_grad_(False).to(**kw)
        w = randn(1, *_out(model, randn(2, 3, res, res, **kw)).shape[1:], **kw)
        return lambda x: (_out(model, x) * w).sum()

    return builder


TORCHVISION_CASES = [
    {
        "id": name,
        "f": _projected(ctor, res),
        "args_fn": lambda res=res: (randn(2, 3, res, res),),
    }
    for name, (ctor, res) in MODELS.items()
]


#: Taylor orders and modes for the CNN cells, overriding the ``K`` and
#: ``collapsed`` fixtures. Orders stop at 3: the nested reverse-mode oracle
#: dominates the cost and grows steeply with ``K`` on whole CNNs, while every
#: rule the CNNs hit is tested at ``K=5`` in ``test_primitives``. Collapsed
#: mode with ``K < 2`` only raises (tested there too), so those cells are left out.
ORDERS = [
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (2, True),
    (3, True),
]


@mark.parametrize("config", TORCHVISION_CASES, ids=lambda c: c["id"])
@mark.parametrize(
    "K, collapsed",
    ORDERS,
    ids=[f"K{K}-{'collapsed' if c else 'standard'}" for K, c in ORDERS],
)
def test_torchvision_model(
    config: dict[str, Any], K: int, collapsed: bool, device: str
):
    """``jet`` of a torchvision CNN matches the nested reverse-mode oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
