"""Nesting Taylor mode: ``jet(jet(f, mock_inner), mock_outer)``.

The inner :func:`jet` returns a callable whose positional argument is a jet
tuple ``(primal, c_1, ..., c_{K_inner})``. To trace it with an outer
:func:`jet`, ``mock_outer`` mirrors that jet-tuple shape literally -- one
container holding ``K_inner + 1`` tensor leaves. The outer then assigns
``K_outer + 1`` Taylor coefficients to each leaf at call time. Output
structure: a pytree mirroring the inner output, with each tensor leaf
replaced by a ``K_outer + 1``-tuple.

This is one analytically-verifiable smoke test (sin, ``K_inner = K_outer = 1``).
The expected coefficients come from differentiating the closed-form inner
jet ``inner(x_p, x_c) = (sin(x_p), cos(x_p) * x_c)`` along the outer
direction.
"""

from torch import cos, float64, manual_seed, rand, sin, zeros
from torch.testing import assert_close

import jet


def test_nested_jet_sin_K_inner_1_K_outer_1():
    """jet(jet(sin, mock_inner), mock_outer) matches the analytical expansion."""
    manual_seed(0)
    inner = jet.jet(sin, (zeros(3, dtype=float64),))
    # mock_outer mirrors inner's jet-tuple shape: 1 positional arg whose
    # container is (primal, c_1) of (K_inner + 1) = 2 tensor leaves.
    mock_outer = ((zeros(3, dtype=float64), zeros(3, dtype=float64)),)
    outer = jet.jet(inner, mock_outer)

    p_p = rand(3, dtype=float64)  # primal at inner-primal position
    p_c = rand(3, dtype=float64)  # outer c_1 at inner-primal position
    q_p = rand(3, dtype=float64)  # primal at inner-c_1 position
    q_c = rand(3, dtype=float64)  # outer c_1 at inner-c_1 position
    result = outer(
        (((p_p, p_c), (q_p, q_c))),
    )

    # ``inner(x_p, x_c) = (sin(x_p), cos(x_p) * x_c)``. Differentiating w.r.t.
    # (x_p, x_c) along the outer direction (p_c, q_c) gives the outer c_1.
    (r00, r01), (r10, r11) = result
    assert_close(r00, sin(p_p))
    assert_close(r01, cos(p_p) * p_c)
    assert_close(r10, cos(p_p) * q_p)
    assert_close(r11, -sin(p_p) * q_p * p_c + cos(p_p) * q_c)
