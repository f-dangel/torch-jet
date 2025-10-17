"""Gamma coefficient computation for higher-order differential operators.
=====================================================================

This module implements the γ_{i,j} coefficients defined in the appendix of:

    Collapsing Taylor Mode Automatic Differentiation (https://arxiv.org/abs/2505.13644/v1)

Specifically, it implements Equation (E17) from Appendix E,
which defines the general formula for γ_{i,j}:

    γ_{i,j}
      = ∑_{m ∈ ℕ^P, 0 < |m| ≤ |i|}
          (-1)^{‖i‖₁ - ‖m‖₁}
          ( ∏_{p=1}^P ( i_p choose m_p ) )
          ( ∏_{p=1}^P  ( ( (‖i‖₁ * m_p) / ‖m‖₁ ) choose j_p ) )
          ( ‖m‖₁ / ‖i‖₁ )^{‖i‖₁}.

All quantities are computed exactly using Python’s ``fractions.Fraction`` class.

Mathematical ↔ Code variable mapping
------------------------------------
    i        →  ``i_vec``        : multi-index representing mixed partials (e.g., (2,2) for the biharmonic)
    P        →  ``P``            : length of i, i.e., number of directions
    j        →  ``j_vec``        : multi-index of length P with |j| = |i|
    m        →  ``m_vec``        : elements of ``generate_restricted_multi_indices(i_vec)``
    ‖i‖₁     →  ``norm_i = sum(i_vec)``
    ‖m‖₁     →  ``norm_m = sum(m)``
    (a choose b)  →  ``binomi(a, b)``, supporting generalized binomials with fractional a

Mathematical expression ↔ Code function mapping
-----------------------------------------------
    generate_multi_indices(P, derivative_order) → ``j_vec_list``                 : Enumerate all j ∈ ℕ^P with ∑ j_p = derivative_order.

    generate_restricted_multi_indices(i_vec)    → {m ∈ ℕ^P, 0 < |m| ≤ |i|}       : Enumerate all admissible m used in the summation (0 < |m| ≤ |i|).

    compute_gamma_for_j(i_vec, j_vec)           → γ_{i,j}                        : Compaute γ_{i,j} for fixed i and j.

    compute_all_gammas(i_vec)                   → {γ_{i,j}, j ∈ ℕ^P, |j| = |i|}  : Compute all γ_{i,j} for multi-indices j of length P with |j| = |i|.

Numerical details
-----------------
* Exact arithmetic based on ``fractions.Fraction``.
* Caching of binomial coefficients (via ``functools.cache``).
"""  # noqa: D205

from fractions import Fraction
from functools import cache
from itertools import product

MultiIndex = tuple[int, ...]


def generate_multi_indices(
    P: int,
    derivative_order: int,
    position: int = 0,
    remainder: int | None = None,
    current: list[int] | None = None,
) -> list[MultiIndex]:
    """Recursively generate all multi-indices j ∈ ℕ^P with sum(j) == derivative_order.

    The method explores all combinations of nonnegative integers (j_1, ..., j_P)
    such that ∑_p j_p = derivative_order. Each recursion step iterates through all
    possible values based on the selected values of the previous recursion steps.
    In every iteration the selected component is added to the current multi-index
    and the remaining total is passed to the next recursion. At the last position
    the remainder dictates the only possible value and the multi-index is returned.

    Args:
        P: Length of the multi-index.
        derivative_order: Total derivative order (∑ jₚ).
        position: Current recursion depth (index of the coordinate being filled).
        remainder: Remaining value to distribute among remaining coordinates.
        current: Partial multi-index under construction.

    Returns:
        A list of all multi-indices j ∈ ℕ^P with ∑ jₚ = derivative_order.
    """
    remainder = derivative_order if remainder is None else remainder
    current = [] if current is None else current

    # Base case - last coordinate
    if position == P - 1:
        return [tuple(current + [remainder])]

    results: list[MultiIndex] = []
    for v in range(remainder + 1):
        # flatten the returned lists
        results.extend(
            generate_multi_indices(
                P, derivative_order, position + 1, remainder - v, current + [v]
            )
        )
    return results


def generate_restricted_multi_indices(i_vec: MultiIndex) -> list[MultiIndex]:
    """All m with 0 <= m_p <= i_p for each p, and 0 < sum(m) <= sum(i).

    Args:
        i_vec: mixed_partial derivative

    Returns:
       All m with 0 <= m_p <= i_p for each p, and 0 < sum(m) <= sum(i).
    """
    ranges = [range(int(i_p) + 1) for i_p in i_vec]
    m_vec_list = []
    for m_vec in product(*ranges):
        if sum(m_vec) > 0:
            m_vec_list.append(m_vec)
    return m_vec_list


def generate_restricted_multi_indices2(i_vec: MultiIndex) -> list[MultiIndex]:
    """All m with 0 <= m_p <= i_p for each p, and 0 < sum(m) <= sum(i).

    Args:
        i_vec: mixed_partial derivative

    Returns:
       All m with 0 <= m_p <= i_p for each p, and 0 < sum(m) <= sum(i).

    """
    return [
        m_vec
        for m_vec in product(*[range(int(i_p) + 1) for i_p in i_vec])
        if sum(m_vec) > 0
    ]


@cache
def binomi(a: Fraction, b: int) -> Fraction:
    """Generalized binomial (a choose b). a is a Fraction, b is int.

    Args:
        a: upper part of a choose b
        b: lower part of a choose b

    Returns:
        a choose b
    """
    res = Fraction(1)
    if b == 0:
        return res
    for i in range(1, b + 1):
        res *= Fraction(a - (i - 1), i)
    return res


@cache
def compute_gamma_for_j(i_vec: MultiIndex, j_vec: MultiIndex) -> Fraction:
    """Computes γ_{i,j} as in Eq. (E17) of the paper, using exact Fraction arithmetic.

    Args:
        i_vec: Multi-index 'i' representing the mixed partial derivative (e.g., (2,2) for biharmonic).
        j_vec: Multi-index 'j' with sum(j_vec) == sum(i_vec).

    Returns:
        Fraction: γ_{i,j} coefficient.

    Raises:
        ValueError: if sum(j_vec) != sum(i_vec)
    """
    P = len(i_vec)
    norm_i = sum(i_vec)
    if sum(j_vec) != norm_i:
        raise ValueError(
            f"The 1-norm of {sum(j_vec)=} and {sum(i_vec)=} must be equal!"
        )

    result = Fraction(0)

    # Precompute all m (the summation domain): 0 < |m| <= |i| and 0 <= m_p <= i_p
    m_vec_list = generate_restricted_multi_indices(i_vec)

    for m_vec in m_vec_list:
        norm_m = sum(m_vec)
        if norm_m == 0:
            continue

        # sign = (-1)^{‖i‖₁ − ‖m‖₁}
        sign = Fraction(-1) if (norm_i - norm_m) % 2 == 1 else Fraction(1)

        # binom_i_m = ∏_{p=1}^P  ( i_p choose m_p )
        binom_i_m = Fraction(1)
        for p in range(P):
            binom_i_m *= binomi(Fraction(i_vec[p]), m_vec[p])

        # binom_a_j = ∏_{p=1}^P  ( ( (‖i‖₁ * m_p) / ‖m‖₁ ) choose j_p )
        prod_binom_a_j = Fraction(1)
        for p in range(P):
            a_p = Fraction(norm_i * m_vec[p], norm_m)
            prod_binom_a_j *= binomi(a_p, j_vec[p])

        # (||m|| / ||i||)^{||i||}
        scale_power = Fraction(norm_m, norm_i) ** norm_i

        result += sign * binom_i_m * prod_binom_a_j * scale_power

    return result


def compute_all_gammas(i_vec: MultiIndex) -> dict[MultiIndex, Fraction]:
    """Compute γ_{i,j} for all j with sum(j) == sum(i).

    Args:
        i_vec: Multi-index representing the derivative for which gammas are computed.

    Returns:
        Mapping from j_vec to γ_{i,j}.

    Examples:
        >>> compute_all_gammas((2, 2))
        {(0, 4): Fraction(13, 192), (1, 3): Fraction(-1, 3), (2, 2): Fraction(5, 8), (3, 1): Fraction(-1, 3), (4, 0): Fraction(13, 192)}
    """
    P = len(i_vec)
    norm_i = sum(i_vec)
    return {
        j_vec: compute_gamma_for_j(i_vec, j_vec)
        for j_vec in generate_multi_indices(P, norm_i)
    }
