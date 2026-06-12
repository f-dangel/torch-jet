"""Shared pytest fixtures for the test suite."""

from pytest import FixtureRequest, fixture
from torch import cuda
from torch.backends import mps

#: Devices the test suite parametrizes over. CPU is always present; CUDA and
#: MPS are added when their respective backends are available.
DEVICES = ["cpu"]
if cuda.is_available():
    DEVICES.append("cuda")
if mps.is_available():
    DEVICES.append("mps")


@fixture(params=DEVICES)
def device(request: FixtureRequest) -> str:
    """Run the requesting test once per available device (see ``DEVICES``)."""
    return request.param


@fixture(params=[False, True], ids=["standard", "collapsed"])
def collapsed(request: FixtureRequest) -> bool:
    """Run the requesting test in standard and collapsed Taylor mode."""
    return request.param


@fixture(params=[False, True], ids=["unscaled", "scaled"])
def scale_coeffs(request: FixtureRequest) -> bool:
    """Run the requesting test with the default and scaled coefficient bases."""
    return request.param


@fixture(params=[0, 1, 2, 5], ids=lambda k: f"K{k}")
def K(request: FixtureRequest) -> int:
    """Run the requesting test once per Taylor order in ``{0, 1, 2, 5}``."""
    return request.param
