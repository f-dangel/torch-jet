"""Tests for jet/validation.py.

The validator is independent of the rest of the jet pipeline; these tests
exercise it directly without going through jet()/collapsed_jet().
"""

from pytest import raises
from torch import zeros

from jet.validation import validate_input_jet


def _ones_args(shape, K, R=None, leading_dim_on_kK=False):
    """Build a jet tuple ``(primal, c_1, ..., c_K)`` of zero tensors.

    R: if given, c_1..c_{K-1} have shape (R, *shape) (collapsed mode).
    leading_dim_on_kK: deliberately give c_K an extra leading dim
        (negative-test helper).
    """
    primal = zeros(*shape)
    coeffs = []
    for k in range(1, K + 1):
        if R is not None and k < K:
            coeffs.append(zeros(R, *shape))
        elif leading_dim_on_kK and k == K:
            coeffs.append(zeros(R, *shape))
        else:
            coeffs.append(zeros(*shape))
    return (primal, *coeffs)


# ---------------------------------------------------------------------------
# Standard mode
# ---------------------------------------------------------------------------


def test_validate_standard_K_mismatch():
    """K must be consistent across leaves."""
    mock = (zeros(3), zeros(3))
    args = (_ones_args((3,), K=2), _ones_args((3,), K=3))
    with raises(ValueError, match="K=3 disagrees with K=2"):
        validate_input_jet(mock, args, collapsed=False)


def test_validate_rejects_non_tuple_leaf():
    """A bare tensor where a jet tuple is expected is rejected."""
    mock = (zeros(3),)
    args = (zeros(3),)  # bare tensor instead of jet tuple
    with raises(ValueError, match="expected a jet tuple"):
        validate_input_jet(mock, args, collapsed=False)


def test_validate_rejects_non_tensor_in_tuple():
    """Python scalars inside the tuple are rejected."""
    mock = (zeros(3),)
    args = ((zeros(3), 1.0),)
    with raises(ValueError, match="every entry of a jet tuple must be a Tensor"):
        validate_input_jet(mock, args, collapsed=False)


def test_validate_rejects_wrong_primal_shape():
    """Primal shape mismatch against mock is rejected."""
    mock = (zeros(3),)
    args = ((zeros(5), zeros(5)),)
    with raises(ValueError, match="primal shape .* does not match mock shape"):
        validate_input_jet(mock, args, collapsed=False)


def test_validate_rejects_wrong_coefficient_shape():
    """Coefficient shape mismatch is rejected."""
    mock = (zeros(3),)
    args = ((zeros(3), zeros(5)),)
    with raises(ValueError, match="coefficient c_1 has shape"):
        validate_input_jet(mock, args, collapsed=False)


# ---------------------------------------------------------------------------
# Collapsed mode
# ---------------------------------------------------------------------------


def test_validate_collapsed_rejects_R_mismatch_across_leaves():
    """R must agree across all collapsed leaves."""
    mock = (zeros(3), zeros(3))
    args = (
        _ones_args((3,), K=2, R=4),
        _ones_args((3,), K=2, R=5),
    )
    with raises(ValueError, match="leaf's R=5 disagrees with R=4"):
        validate_input_jet(mock, args, collapsed=True)


def test_validate_collapsed_rejects_collapsed_slot_with_leading_dim():
    """c_K must not carry a leading direction dim."""
    mock = (zeros(3),)
    args = (_ones_args((3,), K=2, R=4, leading_dim_on_kK=True),)
    with raises(ValueError, match=r"coefficient c_2 has shape"):
        validate_input_jet(mock, args, collapsed=True)


def test_validate_collapsed_rejects_unbatched_c1():
    """c_1 in collapsed mode must be batched (R, *S)."""
    mock = (zeros(3),)
    # c_1 should be (R, 3); pass plain (3,) instead
    args = ((zeros(3), zeros(3), zeros(3)),)
    with raises(ValueError, match=r"coefficient c_1 has shape .* expected \(R, "):
        validate_input_jet(mock, args, collapsed=True)


# ---------------------------------------------------------------------------
# Pytree structure
# ---------------------------------------------------------------------------


def test_validate_dict_pytree():
    """Validator walks dict pytrees correctly."""
    mock = ({"a": zeros(3), "b": zeros(3)},)
    args = ({"a": _ones_args((3,), K=2), "b": _ones_args((3,), K=2)},)
    leaves, K, _ = validate_input_jet(mock, args, collapsed=False)
    assert K == 2
    assert len(leaves) == 2


def test_validate_rejects_non_tensor_mock_leaf():
    """Non-Tensor mock leaves (e.g. a stray Python int) get a typed error."""
    mock = (zeros(3), 0)  # the int sneaks in as a pytree leaf
    args = (_ones_args((3,), K=2), _ones_args((3,), K=2))
    with raises(ValueError, match="mock_args leaf must be a Tensor"):
        validate_input_jet(mock, args, collapsed=False)


def test_validate_rejects_empty_mock():
    """Tensor-free mock_args is rejected up front with a clear error."""
    mock = ({},)  # empty dict -> zero tensor leaves
    args = ({},)
    with raises(ValueError, match="No jet leaves found"):
        validate_input_jet(mock, args, collapsed=False)


def test_validate_rejects_structure_mismatch():
    """Structural mismatch surfaces flatten_up_to's error."""
    mock = ({"a": zeros(3)},)
    args = ([_ones_args((3,), K=2)],)  # list instead of dict
    # PyTorch's flatten_up_to raises ValueError or RuntimeError depending on
    # whether the Python or C++ pytree backend is in use; match either, and
    # the message text loosely to avoid pinning the upstream wording.
    with raises((ValueError, RuntimeError), match=r"[Mm]ismatch"):
        validate_input_jet(mock, args, collapsed=False)
