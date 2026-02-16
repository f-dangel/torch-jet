"""Test `jet.utils`."""

from jet.utils import integer_partitions


def test_integer_partitions():
    """Test the computation of integer partitions."""
    assert list(integer_partitions(1)) == [(1,)]
    assert list(integer_partitions(2)) == [(2,), (1, 1)]
    assert list(integer_partitions(3)) == [(3,), (1, 2), (1, 1, 1)]
    assert list(integer_partitions(4)) == [
        (4,),
        (1, 3),
        (1, 1, 2),
        (1, 1, 1, 1),
        (2, 2),
    ]
    assert list(integer_partitions(5)) == [
        (5,),
        (1, 4),
        (1, 1, 3),
        (1, 1, 1, 2),
        (1, 1, 1, 1, 1),
        (1, 2, 2),
        (2, 3),
    ]
