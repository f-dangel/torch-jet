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
    """Run the requesting test once per locally available device.

    CPU is always present; CUDA and MPS are added when their backends are
    available (see ``DEVICES``). Any test taking a ``device`` argument is
    parametrized over these.
    """
    return request.param
