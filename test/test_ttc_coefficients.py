"""Test for ttc_coefficient methods."""

from fractions import Fraction
from math import comb

from jet.ttc_coefficients import (
    binomi,
    compute_all_gammas,
    generate_multi_indices,
    generate_restricted_multi_indices,
)


def test_generate_multi_indices_empty_degree_zero_cases():
    """If derivative_order = 0, only the all-zero multi-index should appear."""
    assert generate_multi_indices(1, 0) == [(0,)]
    assert generate_multi_indices(2, 0) == [(0, 0)]
    assert generate_multi_indices(3, 0) == [(0, 0, 0)]


def test_generate_multi_indices_constraint_holds():
    """All generated multi-indices must have entries summing to derivative_order,
    must be less than d and must be greater or equal to 0.
    """  # noqa: D205
    P = 13
    d = 7
    indices = generate_multi_indices(P, d)
    for j in indices:
        assert len(j) == P
        assert sum(j) == d
        assert all(isinstance(x, int) and x >= 0 for x in j)
        assert max(j) <= d


def test_generate_multi_indices_combinatorial_formula():
    """Test that the number of multi-indices with P components summing to d is (d + P - 1) choose (P - 1)."""
    for P, d in [(1, 5), (2, 4), (3, 3), (4, 2)]:
        indices = generate_multi_indices(P, d)
        expected_count = comb(d + P - 1, P - 1)
        assert len(indices) == expected_count


def test_generate_multi_indices_structure_for_small_dimensions():
    """Test for explicit indices for P=2, derivative_order=2 and P=3, derivative_order=2."""
    assert set(generate_multi_indices(2, 2)) == {(0, 2), (1, 1), (2, 0)}

    expected = {
        (0, 0, 2),
        (0, 1, 1),
        (0, 2, 0),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 0),
    }
    assert set(generate_multi_indices(3, 2)) == expected


def test_generate_restricted_empty_case():
    """Edge case: i_vec = (0,0,0) should yield an empty list (no positive sum possible)."""
    i_vec = (0, 0, 0)
    m_list = generate_restricted_multi_indices(i_vec)
    assert m_list == []


def test_generate_restricted_multi_indices_constrain_holds():
    """All entries must satisfy 0 ≤ m_p ≤ i_p, and 0 < sum(m) ≤ sum(i)."""
    i_vec = (2, 1, 7)
    m_vec_list = generate_restricted_multi_indices(i_vec)
    for m_vec in m_vec_list:
        assert len(m_vec) == len(i_vec)
        assert all(0 <= m_vec[p] <= i_vec[p] for p in range(len(i_vec)))
        assert 0 < sum(m_vec) <= sum(i_vec)
    assert i_vec in m_vec_list


def test_generate_restricted_expected_count():
    """Expected count = (i1+1)*(i2+1)*...*(iP+1) - 1 (subtract 1 for excluding the zero multi-index)."""
    cases = [(1,), (2,), (1, 1), (2, 1), (2, 2)]
    for i_vec in cases:
        m_vec_list = generate_restricted_multi_indices(i_vec)
        expected_count = 1
        for i_p in i_vec:
            expected_count *= i_p + 1
        expected_count -= 1
        assert len(m_vec_list) == expected_count


def test_generate_restricted_symmetry():
    """Test for symmetry if `i_vec` has identical entries. For example, for i_vec=(2,2), (1,0) and (0,1) must both appear."""
    i_vec = (8, 8)
    m_vec_list = generate_restricted_multi_indices(i_vec)

    # count occurences where indices like (2, 1) and (1, 2) are mapped to the same
    entry_counter = {}
    for m_vec in m_vec_list:
        key = frozenset(m_vec)
        entry_counter[key] = entry_counter.get(key, 0) + 1

    # all entries with key of size 1 are come from (1, 1), (2, 2), ... and should have count 1
    # all other entries occur twice where (2, 1) and (1, 2), etc are mapped to the same key
    for key, val in entry_counter.items():
        assert val == (1 if len(key) == 1 else 2)


def test_binomi_base_cases():
    """Base cases for b = 0 and b = 1."""
    # binom(a, 0) == 1 for any a
    for a in [Fraction(0), Fraction(3), Fraction(5, 2), Fraction(-2, 3)]:
        assert binomi(a, 0) == Fraction(1)
        assert isinstance(binomi(a, 0), Fraction)

    # binom(a, 1) == a for any a
    for a in [Fraction(7), Fraction(3, 2), Fraction(-4, 3)]:
        assert binomi(a, 1) == a


def test_binomi_integer_a_matches_standard_binomial():
    """For integer a >= b, should match ordinary combinatorial 'n choose k'."""
    for n in range(0, 8):
        for k in range(0, n + 1):
            assert binomi(Fraction(n), k) == Fraction(comb(n, k))


def test_binomi_fractional_a_known_values():
    """Check specific known rational results."""
    # (5/2 choose 2) = (5/2 * 3/2) / 2 = 15/8
    assert binomi(Fraction(5, 2), 2) == Fraction(15, 8)
    # (1/2 choose 2) = (1/2 * -1/2) / 2 = -1/8
    assert binomi(Fraction(1, 2), 2) == Fraction(-1, 8)
    # (-1/2 choose 2) = (-1/2 * -3/2) / 2 = 3/8
    assert binomi(Fraction(-1, 2), 2) == Fraction(3, 8)
    # (3/2 choose 3) = (3/2 * 1/2 * -1/2) / 6 = -1/16
    assert binomi(Fraction(3, 2), 3) == Fraction(-1, 16)


def test_gamma_scalar_case_matches_binomial():
    """1D sanity check."""
    for n in range(1, 6):
        i_vec = (n,)
        gammas = compute_all_gammas(i_vec)
        # E17 reduces to this in 1D
        expected = sum(
            [
                Fraction((-1) ** (n - j)) * binomi(Fraction(n), j) * Fraction(j, n) ** n
                for j in range(1, n + 1)
            ]
        )
        assert gammas[(n,)] == expected, f"Mismatch for {n=}"


def test_gamma_laplacian():
    """Test if gammas are correct for the laplacian."""
    i_vec = (2,)
    gamma = compute_all_gammas(i_vec)
    assert gamma == {(2,): Fraction(1, 2)}


def test_gamma_biharmonic():
    """Test known if the coefficients are correct for biharmonic."""
    i_vec = (2, 2)
    gammas = compute_all_gammas(i_vec)
    assert len(gammas) == 5
    assert gammas[(2, 2)] == Fraction(5, 8)
    assert gammas[(4, 0)] == Fraction(13, 192) == gammas[(0, 4)]
    assert gammas[(1, 3)] == Fraction(-1, 3) == gammas[(3, 1)]
