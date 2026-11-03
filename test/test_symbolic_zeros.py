"""A symbolic-zero (``None``) coefficient equals a zero tensor in every rule."""

from typing import Any

from pytest import mark
from torch import ops, rand, zeros_like
from torch.testing import assert_close
from torch.utils._pytree import tree_map

from jet import jet
from jet._rules import RULES
from jet.primitives import JetTuple, jet_add, jet_cat
from jet.utils import _is_jet_leaf
from test.test_composition import COMPOSITION_CASES
from test.test_primitives import PRIMITIVE_CASES
from test.utils import make_jet_args, setup_case, tolerances_for


@mark.parametrize("K", [2, 3], ids=["K2", "K3"])
@mark.parametrize("config", PRIMITIVE_CASES + COMPOSITION_CASES, ids=lambda c: c["id"])
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
    jet_f = jet(f, mock_args, collapsed=collapsed)
    assert_close(jet_f(*symbolic), jet_f(*dense), **tolerances_for(device))


def test_symbolic_zero_survives_linear_rules(collapsed: bool, device: str):
    """``add``, ``cat`` and a constant-output op keep a ``None`` coefficient ``None``."""
    x = rand(3, device=device)
    c_1 = rand(2, 3, device=device) if collapsed else rand(3, device=device)
    a = JetTuple((x, c_1, None), collapsed=collapsed)
    assert jet_add(a, a)[2] is None
    assert jet_cat([a, x])[2] is None
    assert RULES[ops.aten.zeros_like.default](a)[2] is None
