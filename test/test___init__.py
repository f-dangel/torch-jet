"""Tests for ``jet/__init__.py`` API surface.

Primitive correctness is in ``test_primitives.py``; composition coverage is
in ``test_composition.py``; constant-output handling is in
``test_constants.py``. This file keeps the small set of API-rejection /
smoke-test cases that don't fit any of those layers.
"""

from pytest import raises
from torch import sin, zeros

import jet
from jet.tracing import capture_graph


def test_capture_graph_rejects_non_tuple_mock_args():
    """capture_graph requires mock_args to be a tuple (not a bare tensor)."""
    with raises(TypeError, match="must be a tuple"):
        capture_graph(sin, zeros(3))


def test_jet_rejects_unsupported_tuple_dict_signature():
    """Reject only the (tensor/tuple, dict) two-argument signature (make_fx bug).

    All other dict signatures are supported, so they must not raise.
    """
    t, d = zeros(3), {"a": zeros(3)}

    # Unsupported: two args, first tensor/tuple, second dict.
    f = lambda x, params: x * params["a"]  # noqa: E731
    match = r"pytorch/pytorch#185640"  # pin to the tracked upstream issue
    for collapsed in (False, True):
        with raises(NotImplementedError, match=match):
            jet.jet(f, (t, d), collapsed=collapsed)

    # Supported dict signatures must not raise.
    jet.jet(lambda d: d["a"] * 2, (d,))  # single dict arg
    jet.jet(lambda d, x: d["a"] + x, (d, t))  # dict first
    jet.jet(lambda x, y, d: x + y + d["a"], (t, t, d))  # three args, trailing dict
