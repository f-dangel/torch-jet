"""Tests for jet/_jet.py (the jet() / _rev_jet transforms)."""

from pytest import mark, raises
from torch import manual_seed, rand, sin, zeros
from torch.testing import assert_close

from jet import jet
from jet._jet import _uncollapsed_via_vmap
from test.utils import make_jet_args, rev_collapsed_jet


@mark.parametrize("K", [2, 3])
def test_uncollapsed_via_vmap_nonzero_c_K(K: int, scale_coeffs: bool) -> None:
    """_uncollapsed_via_vmap matches the collapsed oracle for random non-zero c_K.

    The order-K coefficient must enter a single vmap direction (zeros into the
    rest), as in rev_collapsed_jet. Feeding it into every direction (the old
    bug) overcounts it R-fold; this is masked only when c_K == 0.
    """
    manual_seed(0)
    x = rand(4)

    def f(t):
        return (t**3).sin()

    mock_args = (x,)
    jet_args = make_jet_args(mock_args, K, collapsed=True, R=3)
    # Sanity-check the regression premise: c_K is non-zero for every leaf.
    assert jet_args[0][K].abs().max() > 0

    actual = _uncollapsed_via_vmap(f, mock_args, None, scale_coeffs=scale_coeffs)(
        *jet_args
    )
    expected = rev_collapsed_jet(f)(*jet_args)
    assert_close(actual, expected)


def test_collapsed_jet_rejects_order_below_2(scale_coeffs: bool):
    """jet(..., collapsed=True) raises ValueError at call time for K < 2."""
    cjet_f = jet(sin, (zeros(3),), collapsed=True, scale_coeffs=scale_coeffs)
    x = zeros(3)

    # K=1: jet tuple has length 2 -> only a primal and one coefficient.
    # K=0: jet tuple has length 1 -> only a primal.
    for jet_tuple in [(x, x), (x,)]:
        with raises(ValueError, match="collapsed mode requires K >= 2"):
            cjet_f(jet_tuple)


def test_jet_rejects_unsupported_tuple_dict_signature(scale_coeffs: bool):
    """Reject only the (tensor/tuple, dict) two-argument signature (make_fx bug).

    All other dict signatures are supported, so they must not raise.
    """
    t, d = zeros(3), {"a": zeros(3)}

    # Unsupported: two args, first tensor/tuple, second dict.
    f = lambda x, params: x * params["a"]  # noqa: E731
    match = r"pytorch/pytorch#185640"  # pin to the tracked upstream issue
    for collapsed in (False, True):
        with raises(NotImplementedError, match=match):
            jet(f, (t, d), collapsed=collapsed, scale_coeffs=scale_coeffs)

    # Supported dict signatures must not raise.
    jet(lambda d: d["a"] * 2, (d,), scale_coeffs=scale_coeffs)  # single dict arg
    jet(lambda d, x: d["a"] + x, (d, t), scale_coeffs=scale_coeffs)  # dict first
    jet(
        lambda x, y, d: x + y + d["a"], (t, t, d), scale_coeffs=scale_coeffs
    )  # three args, trailing dict


@mark.parametrize("collapsed", [False, True], ids=["standard", "collapsed"])
@mark.parametrize("K", [0, 1, 3])
def test_scaled_coeffs_matches_default(collapsed: bool, K: int) -> None:
    """The opt-in scaled basis preserves the public jet semantics."""

    def f(x, y):
        return (x * y).sin() + x.exp()

    x = zeros(4)
    mock_args = (x, x)
    jet_args = make_jet_args(mock_args, K, collapsed=collapsed, R=3)
    if collapsed and K < 2:
        for scale_coeffs in (False, True):
            with raises(ValueError, match="collapsed mode requires K >= 2"):
                jet(f, mock_args, collapsed=collapsed, scale_coeffs=scale_coeffs)(
                    *jet_args
                )
        return

    scaled = jet(f, mock_args, collapsed=collapsed, scale_coeffs=True)(*jet_args)
    default = jet(f, mock_args, collapsed=collapsed, scale_coeffs=False)(*jet_args)
    assert_close(scaled, default)
