"""Test `jet.utils`."""

from torch import zeros
from torch.fx import Graph

import jet.utils
from jet.utils import integer_partitions, standardize_signature


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


def test_standardize_signature():
    """Test standardizing args and kwargs of nodes."""
    # create some dummy inputs
    x = zeros(3, 2)
    times = 4

    # different combinations a use may pass arguments to replicate
    args_and_kwargs = [
        [(), {"x": x, "times": times}],
        [(x,), {"times": times}],
        [(x,), {"times": times, "pos": 2}],
        [(x, times, 1), {}],
    ]

    # make sure they are all standardized
    for args, kwargs in args_and_kwargs:
        g = Graph()
        n = g.create_node(
            "call_function", jet.utils.replicate, args=args, kwargs=kwargs
        )
        n.args, n.kwargs = standardize_signature(
            n.target, n.args, n.kwargs, verbose=True
        )
        assert n.args == (x, times)
        pos = args[2] if len(args) == 3 else kwargs.get("pos", 0)
        assert n.kwargs == {"pos": pos}
