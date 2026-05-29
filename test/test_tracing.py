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

from pytest import mark
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


@mark.xfail(
    strict=True,
    reason=(
        "make_fx codegen mishandles a dict positional arg that follows a "
        "tuple/list positional arg: it drops dict keys, producing a graph that "
        "raises at call time. Because every jet wraps its leaves in tuples, this "
        "blocks dict arguments in the jet transforms (see jet._assert_no_dicts). "
        "When PyTorch fixes this, the test will XPASS (strict) and signal that "
        "the dict restriction can be lifted."
    ),
)
def test_make_fx_supports_dict_arg_after_tuple_arg():
    """make_fx should handle a dict positional arg following a tuple arg.

    Minimal reproduction of the limitation that forces the jet transforms to
    reject ``dict`` arguments: a function whose first positional arg is a tuple
    and whose second is a dict. ``make_fx`` builds a broken input-reconstruction
    template for the dict (dropping keys), so calling the traced module fails.
    """

    def f(a: tuple, d: dict):
        return a[0] + d["x"] + d["y"]

    mock = ((rand(2),), {"x": rand(2), "y": rand(2)})
    graph = make_fx(f)(*mock)

    args = ((rand(2),), {"x": rand(2), "y": rand(2)})
    expected = args[0][0] + args[1]["x"] + args[1]["y"]
    assert graph(*args).allclose(expected)
