"""Test `jet.utils`."""

from pathlib import Path

from pytest import mark, raises
from torch import rand
from torch.func import vmap
from torch.fx.experimental.proxy_tensor import make_fx

from jet.utils import integer_partitions, visualize_graph
from test.utils import peak_memory


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


@mark.parametrize("suffix", [".pdf", ".png", ".svg"])
def test_visualize_graph(tmp_path: Path, suffix: str):
    """Test that visualize_graph writes a file in the requested format."""
    mod = make_fx(lambda x: x.sum())(rand(3))
    savefile = str(tmp_path / f"graph{suffix}")
    visualize_graph(mod, savefile)
    assert Path(savefile).stat().st_size > 0


def test_visualize_graph_unsupported_format(tmp_path: Path):
    """Test that visualize_graph raises on an unsupported extension."""
    mod = make_fx(lambda x: x.sum())(rand(3))
    with raises(ValueError, match="Unsupported file format"):
        visualize_graph(mod, str(tmp_path / "graph.bmp"))


def test_peak_memory(device: str):
    """``peak_memory`` matches peaks that can be stated by hand."""
    a, b, c = (rand(4, 64, 64, device=device) for _ in range(3))
    RS, S = a.untyped_storage().nbytes(), a.untyped_storage().nbytes() // 4
    known = {
        lambda a, b, c: a.transpose(0, 1): 0,  # a view
        lambda a, b, c: 2 * a: RS,  # the wrapped scalar does not count
        lambda a, b, c: a * b * c: 2 * RS,  # the first product and the result
        lambda a, b, c: (a * b).sum(0): RS + S,
        lambda a, b, c: vmap(lambda a, b: a.T * b.T)(a, b): RS,
        lambda a, b, c: a.transpose(0, 1).contiguous(): RS,  # a copy
    }
    for f, peak in known.items():
        assert peak_memory(lambda f=f: f, a, b, c) == peak
