"""Shared pytest fixtures for the test suite."""

from pytest import FixtureRequest, fixture

from test.utils import DEVICES


@fixture(params=DEVICES)
def device(request: FixtureRequest) -> str:
    """Run the requesting test once per locally available device.

    CPU is always present; CUDA and MPS are added when their backends are
    available (see ``DEVICES``). Any test taking a ``device`` argument is
    parametrized over these.
    """
    return request.param
