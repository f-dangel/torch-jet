"""Constant-output handling under both Taylor-mode regimes.

Constant outputs (tensor leaves of ``f``'s return value that don't depend on
any input) must still produce coefficients matching the mode's shape
contract -- zero standard-mode coefficients of the primal's shape, or
collapsed-mode batched zeros (``(R, *S)``) for ``c_1..c_{K-1}`` and an
unbatched zero (``S``) at the collapsed slot ``c_K``.
"""

from torch import Tensor, float64, rand, sin, zeros

import jet


def test_standard_constant_output_zero_coeffs():
    """Standard mode: a constant output leaf gets K zero coefficients of S."""
    K = 3
    out_shape = (4,)

    def f(x: Tensor) -> tuple[Tensor, Tensor]:
        return sin(x), zeros(*out_shape, dtype=float64)

    jet_f = jet.jet(f, (zeros(3, dtype=float64),))
    primal = rand(3, dtype=float64)
    coeffs = tuple(rand(3, dtype=float64) for _ in range(K))
    (_, _, _, _), (const, *const_coeffs) = jet_f((primal, *coeffs))

    assert const.shape == out_shape, (
        f"constant primal shape {const.shape} != {out_shape}"
    )
    assert len(const_coeffs) == K, f"got {len(const_coeffs)} coeffs, expected {K}"
    for k, c in enumerate(const_coeffs, start=1):
        assert c.shape == out_shape, f"constant c_{k} shape {c.shape} != {out_shape}"


def test_collapsed_constant_output_uses_collapsed_shape():
    """F1: collapsed mode -- ``c_1..c_{K-1}`` are ``(R, *S)``, ``c_K`` is ``S``.

    A function with a constant output leaf must still match the collapsed-mode
    shape contract, or downstream consumers see broken broadcasts.
    """
    R = 3  # K=2 implicit from passing one coefficient slot to cjet_f below
    out_shape = (4,)

    def f(x: Tensor) -> tuple[Tensor, Tensor]:
        return sin(x), zeros(*out_shape, dtype=float64)

    cjet_f = jet.jet(f, (zeros(3, dtype=float64),), collapsed=True)
    primal = rand(3, dtype=float64)
    c1 = rand(R, 3, dtype=float64)
    cK = zeros(3, dtype=float64)
    (_, _, _), (const, const_c1, const_cK) = cjet_f((primal, c1, cK))

    assert const.shape == out_shape
    assert const_c1.shape == (R, *out_shape)
    assert const_cK.shape == out_shape
