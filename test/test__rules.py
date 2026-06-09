"""Tests for the jet rule registry (``jet/_rules.py``)."""

from pytest import raises
from torch import ops, sin, zeros

from jet import jet
from jet._rules import RULES


def test_every_op_has_a_standard_rule():
    """``RULES[op]`` always carries the standard (``False``) rule.

    Collapsed-only is invalid: the registry may hold a standard rule without a
    collapsed sibling (standard-only op), but never the reverse.
    """
    assert RULES
    assert all(False in rule for rule in RULES.values())


def test_collapsed_request_without_collapsed_rule_errors(monkeypatch):
    """``jet(..., collapsed=True)`` over a standard-only op raises a precise error."""
    # Make ``sin`` standard-only (drop its collapsed rule) for this test.
    standard_only = {False: RULES[ops.aten.sin.default][False]}
    monkeypatch.setitem(RULES, ops.aten.sin.default, standard_only)

    cjet_sin = jet(sin, (zeros(3),), collapsed=True)
    x = zeros(3)
    # Valid collapsed 2-jet: c_1 batched ``(R, *S)``, c_2 collapsed ``S``.
    jet_in = (x, zeros(5, 3), x)
    with raises(NotImplementedError, match="standard jet rule but no collapsed"):
        cjet_sin(jet_in)
