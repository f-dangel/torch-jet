"""Constant-output handling under both Taylor-mode regimes.

Constant outputs (tensor leaves of ``f``'s return value that don't depend on
any input) must still produce coefficients matching the mode's shape
contract -- zero standard-mode coefficients of the primal's shape, or
collapsed-mode batched zeros (``(R, *S)``) for ``c_1..c_{K-1}`` and an
unbatched zero (``S``) at the collapsed slot ``c_K``.
"""

from pytest import mark
from torch import Tensor, sin, zeros

from jet import jet
from test.utils import DEVICES, dtype_for_device, make_jet_args


@mark.parametrize("device", DEVICES)
@mark.parametrize("collapsed", [False, True], ids=["standard", "collapsed"])
def test_constant_output_shape(collapsed: bool, device: str):
    """Constant output leaves match the mode-specific shape contract.

    - Standard mode: each coefficient ``c_1..c_K`` is shape ``S``.
    - Collapsed mode: ``c_1..c_{K-1}`` are ``(R, *S)`` and ``c_K`` is ``S``.
    """
    K, R, out_shape = 2, 3, (4,)
    dtype = dtype_for_device(device)
    mock_args = (zeros(3, dtype=dtype, device=device),)

    def f(x: Tensor) -> tuple[Tensor, Tensor]:
        return sin(x), zeros(*out_shape, dtype=dtype, device=device)

    args = make_jet_args(mock_args, K, collapsed=collapsed, R=R)
    _, (const, *const_coeffs) = jet(f, mock_args, collapsed=collapsed)(*args)

    expected = (
        [(R, *out_shape)] * (K - 1) + [out_shape] if collapsed else [out_shape] * K
    )
    assert const.shape == out_shape
    assert len(const_coeffs) == K
    for k, (c, exp) in enumerate(zip(const_coeffs, expected), start=1):
        assert c.shape == exp, f"c_{k} shape {c.shape} != {exp}"
