"""End-to-end Taylor-mode tests on torchvision CNNs (eval mode, CPU).

torchvision pins torch exactly, so it is not a package extra: CI installs it in
a separate step. A missing torchvision fails this module rather than skipping
it, so a broken install step cannot pass silently.
"""

from typing import Any, Callable

from pytest import mark
from torch import manual_seed, randn
from torchvision import models

from test.utils import assert_jet_matches_oracle, device_kw

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


def _projected(ctor: Callable) -> Callable[[str], Callable]:
    """Build ``x -> sum(model(x) * w)`` with a fixed random ``w`` on ``device``.

    The scalar output keeps the nested reverse-mode oracle cheap (it
    differentiates each output element separately) while every logit still
    contributes.
    """

    def builder(device: str) -> Callable:
        manual_seed(0)
        kw = device_kw(device)
        model = ctor().eval().to(**kw)
        w = randn(1, 1000, **kw)
        return lambda x: (model(x) * w).sum()

    return builder


TORCHVISION_CASES = [
    {
        "id": name,
        "f": _projected(ctor),
        "args_fn": lambda res=res: (randn(1, 3, res, res),),
    }
    for name, (ctor, res) in MODELS.items()
]


@mark.parametrize("config", TORCHVISION_CASES, ids=lambda c: c["id"])
def test_torchvision_model(config: dict[str, Any], K: int, collapsed: bool):
    """``jet`` of a torchvision CNN matches the nested reverse-mode oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device="cpu")
