"""Tests for jet/_jet.py (the jet() / _rev_jet transforms)."""

from pytest import raises
from torch import sin, zeros

from jet import jet


def test_collapsed_jet_rejects_order_below_2():
    """jet(..., collapsed=True) raises ValueError at call time for K < 2."""
    cjet_f = jet(sin, (zeros(3),), collapsed=True)
    x = zeros(3)

    # K=1: jet tuple has length 2 -> only a primal and one coefficient.
    # K=0: jet tuple has length 1 -> only a primal.
    for jet_tuple in [(x, x), (x,)]:
        with raises(ValueError, match="collapsed mode requires K >= 2"):
            cjet_f(jet_tuple)


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
            jet(f, (t, d), collapsed=collapsed)

    # Supported dict signatures must not raise.
    jet(lambda d: d["a"] * 2, (d,))  # single dict arg
    jet(lambda d, x: d["a"] + x, (d, t))  # dict first
    jet(lambda x, y, d: x + y + d["a"], (t, t, d))  # three args, trailing dict
