"""Tests for ``jet/__init__.py`` API surface.

Primitive correctness is in ``test_primitives.py``; composition coverage is
in ``test_composition.py``; constant-output handling is in
``test_constants.py``. This file keeps the small set of API-rejection /
smoke-test cases that don't fit any of those layers.
"""

from pytest import raises
from torch import Tensor, float64, rand, sin, zeros

import jet
from jet.tracing import capture_graph


def test_collapsed_jet_rejects_order_below_2():
    """jet(..., collapsed=True) raises ValueError at call time for K < 2."""
    cjet_f = jet.jet(sin, (zeros(3),), collapsed=True)
    x = zeros(3)

    # K=1: jet tuple has length 2 -> only a primal and one coefficient.
    # K=0: jet tuple has length 1 -> only a primal.
    for jet_tuple in [(x, x), (x,)]:
        with raises(ValueError, match="collapsed mode requires K >= 2"):
            cjet_f(jet_tuple)


def test_collapsed_jet_constant_output_uses_collapsed_shape():
    """F1: constant outputs in collapsed mode get c_1..c_{K-1} of shape (R, *S).

    A function with a constant output leaf (a tensor independent of the inputs)
    must still produce coefficients matching the collapsed-mode shape contract,
    or downstream consumers see broken broadcasts.
    """
    R = 3  # K=2 implicit from passing one coefficient slot to cjet_f below
    out_shape = (4,)

    def f(x: Tensor) -> tuple[Tensor, Tensor]:
        return sin(x), zeros(*out_shape, dtype=float64)  # second leaf is constant

    cjet_f = jet.jet(f, (zeros(3, dtype=float64),), collapsed=True)
    primal = rand(3, dtype=float64)
    c1 = rand(R, 3, dtype=float64)
    cK = zeros(3, dtype=float64)
    (_, _, _), (const, const_c1, const_cK) = cjet_f((primal, c1, cK))
    assert const.shape == out_shape
    assert const_c1.shape == (R, *out_shape)
    assert const_cK.shape == out_shape


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
