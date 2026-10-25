"""End-to-end Taylor-mode tests on torchvision CNNs (eval mode).

torchvision pins torch exactly, so it is not a package extra: ``make
install-test`` installs it separately (``make install-torchvision``, versioned
via ``TORCHVISION_VERSION``). A missing torchvision fails this module rather
than skipping it, so a broken install step cannot pass silently.
"""

from typing import Any, Callable

from pytest import mark, raises
from torch import manual_seed, randn
from torchvision import models

from test.utils import assert_jet_matches_oracle, device_kw

#: Devices on which PyTorch cannot differentiate ``max_pool2d`` twice when a
#: nonlinearity follows it (unless called with ``return_indices=True``). The
#: oracle, not ``jet``, breaks there; the test asserts the known error so that
#: a PyTorch fix (or a test-side substitute) shows up as a failure to remove.
ORACLE_MAX_POOL_FAILS_ON = {"mps"}

#: Models that trace end to end: input resolution and the devices on which the
#: oracle is known to fail for ``K >= 2``.
MODELS: dict[str, tuple[Callable, int, set[str]]] = {
    "alexnet": (models.alexnet, 64, ORACLE_MAX_POOL_FAILS_ON),
    "vgg11_bn": (models.vgg11_bn, 32, ORACLE_MAX_POOL_FAILS_ON),
    "resnet18": (models.resnet18, 32, ORACLE_MAX_POOL_FAILS_ON),
    "squeezenet1_0": (models.squeezenet1_0, 64, ORACLE_MAX_POOL_FAILS_ON),
    "densenet121": (models.densenet121, 32, ORACLE_MAX_POOL_FAILS_ON),
    "googlenet": (
        lambda: models.googlenet(init_weights=False, aux_logits=False),
        96,
        ORACLE_MAX_POOL_FAILS_ON,
    ),
    "regnet_x_400mf": (models.regnet_x_400mf, 32, set()),
    "mnasnet0_5": (models.mnasnet0_5, 32, set()),
    "mobilenet_v2": (models.mobilenet_v2, 32, set()),
}


def _projected(ctor: Callable) -> Callable[[str], Callable]:
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
        w = randn(1, 1000, **kw)
        return lambda x: (model(x) * w).sum()

    return builder


TORCHVISION_CASES = [
    {
        "id": name,
        "f": _projected(ctor),
        "args_fn": lambda res=res: (randn(1, 3, res, res),),
        "oracle_fails_on": oracle_fails_on,
    }
    for name, (ctor, res, oracle_fails_on) in MODELS.items()
]


@mark.parametrize("config", TORCHVISION_CASES, ids=lambda c: c["id"])
def test_torchvision_model(
    config: dict[str, Any], K: int, collapsed: bool, device: str
):
    """``jet`` of a torchvision CNN matches the nested reverse-mode oracle.

    Where the oracle itself cannot differentiate the model twice, assert that
    known failure instead, so its disappearance does not go unnoticed.
    """
    if device in config["oracle_fails_on"] and K >= 2:
        with raises(RuntimeError, match="not infinitely differentiable"):
            assert_jet_matches_oracle(config, K, collapsed, device)
        return
    assert_jet_matches_oracle(config, K, collapsed, device)
