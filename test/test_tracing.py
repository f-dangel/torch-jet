"""Test `jet.tracing`.

``make_fx(functionalize(...))`` is supposed to replace all in-place ops with
functional equivalents, but fails to do so for ``squeeze_.dim`` (e.g. when
tracing ``Linear(3, 1)`` with a 1-d input). The first test documents this
PyTorch limitation and acts as a regression detector -- if PyTorch fixes this
upstream, the test will fail, signaling that ``_replace_inplace_ops`` in
``jet/tracing.py`` may no longer be needed.

The second test verifies that ``capture_graph`` works around this limitation by
explicitly replacing ``squeeze_.dim`` after tracing, ensuring the resulting
graph is fully functional.
"""

from torch import ops, rand
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Linear

from jet.tracing import capture_graph


def _uses_squeeze_inplace(mod) -> bool:
    """Check if any node in the graph uses the in-place squeeze_ op."""
    return any(
        node.op == "call_function" and node.target == ops.aten.squeeze_.dim
        for node in mod.graph.nodes
    )


def test_make_fx_functionalize_does_not_replace_squeeze_():
    """make_fx(functionalize(...)) does not replace squeeze_ with squeeze."""
    f = Linear(3, 1)
    x = rand(3)
    mod = make_fx(functionalize(f))(x)
    assert _uses_squeeze_inplace(mod)


def test_capture_graph_replaces_squeeze_():
    """capture_graph replaces squeeze_ with its out-of-place equivalent."""
    f = Linear(3, 1)
    x = rand(3)
    mod = capture_graph(f, x)
    assert not _uses_squeeze_inplace(mod)
