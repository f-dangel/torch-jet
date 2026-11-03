"""A symbolic-zero (``None``) coefficient equals a zero tensor in every rule."""

from typing import Any

from pytest import mark
from torch import zeros_like
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten, tree_map

from jet import capture_graph
from jet.jet_interpreter import JetInterpreter
from jet.utils import _is_jet_leaf
from test.test_primitives import PRIMITIVE_CASES
from test.utils import make_jet_args, setup_case, tolerances_for


def _run_unchecked(f: Any, mock_args: tuple, collapsed: bool, args: tuple) -> Any:
    """``jet(f, mock_args, collapsed)(*args)`` without the validator.

    The validator rejects ``None`` entries; the interpreter is driven directly.
    """
    with sdpa_kernel(SDPBackend.MATH):
        mod, _ = capture_graph(f, mock_args)
    leaves = tree_flatten(mock_args)[1].flatten_up_to(args)
    K = len(leaves[0]) - 1
    R = leaves[0][1].shape[0] if collapsed else None
    return JetInterpreter(mod, collapsed=collapsed).run(K, R, *leaves)


@mark.parametrize("K", [2, 3], ids=["K2", "K3"])
@mark.parametrize("config", PRIMITIVE_CASES, ids=lambda c: c["id"])
def test_symbolic_zero_tail_matches_dense(
    config: dict[str, Any], K: int, collapsed: bool, device: str
):
    """``None`` in the coefficients of order ``2..K`` equals zero tensors there."""
    f, mock_args = setup_case(config, device)
    args = make_jet_args(mock_args, K, collapsed)
    dense = tree_map(
        lambda leaf: (*leaf[:2], *(zeros_like(c) for c in leaf[2:])),
        args,
        is_leaf=_is_jet_leaf,
    )
    symbolic = tree_map(
        lambda leaf: (*leaf[:2], *([None] * (K - 1))), args, is_leaf=_is_jet_leaf
    )
    want = _run_unchecked(f, mock_args, collapsed, dense)
    got = _run_unchecked(f, mock_args, collapsed, symbolic)
    assert_close(got, want, **tolerances_for(device))
