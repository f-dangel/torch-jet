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
