"""PyTorch's function transforms apply to a jet function.

A jet function is an ordinary PyTorch function, so ``torch.func`` transforms
can wrap it. Each test transforms the jet of a case and compares with a
reference.
"""

from functools import partial
from operator import itemgetter
from typing import Any, Callable

from pytest import mark
from torch import Tensor, autograd, rand_like, stack
from torch.func import grad, vjp, vmap
from torch.testing import assert_close
from torch.utils._pytree import tree_leaves, tree_map

from jet import jet
from jet._jet import _rev_jet
from jet.utils import _is_jet_leaf
from test.test_torchvision import ORDER_IDS, ORDERS
from test.test_transformer_encoder import ENCODER_CASES
from test.utils import (
    SCALAR_OUTPUT_CASES,
    DifferentiableSubstitutes,
    make_jet_args,
    rev_collapsed_jet,
    setup_case,
    tolerances_for,
)

#: The encoder case belongs in CI's ``-m models`` job; mark the whole module.
pytestmark = mark.models

#: A small MLP and a tiny transformer encoder (its attention puts the
#: scaled-dot-product-attention adapter of the running device on the path).
CASES = [SCALAR_OUTPUT_CASES[0], ENCODER_CASES[0]]


def _jet_setup(
    config: dict[str, Any], K: int, collapsed: bool, device: str
) -> tuple[Callable, Callable, tuple]:
    """The function of a case, its jet function, and matching jet args."""
    f, mock_args = setup_case(config, device)
    jet_fn = jet(f, mock_args, collapsed=collapsed)
    return f, jet_fn, make_jet_args(mock_args, K, collapsed)


def _map_coeffs(fn: Callable, jet_args: tuple) -> tuple:
    """Apply ``fn`` to every coefficient of ``jet_args``; the primals stay."""
    return tree_map(
        lambda leaf: (leaf[0], *map(fn, leaf[1:])), jet_args, is_leaf=_is_jet_leaf
    )


@mark.parametrize("config", CASES, ids=lambda c: c["id"])
@mark.parametrize("K, collapsed", ORDERS[1:], ids=ORDER_IDS[1:])
def test_vmap(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``vmap`` of a jet over ``R`` sets of input coefficients matches a loop.

    The primals are not batched. ``K=0`` is left out: nothing to batch. In
    collapsed mode this batches sets of ``R`` directions, one set per lane.
    """
    R = 3
    _, jet_fn, jet_args = _jet_setup(config, K, collapsed, device)
    batched = _map_coeffs(lambda c: stack([rand_like(c) for _ in range(R)]), jet_args)
    in_dims = tree_map(lambda _: (None, *K * (0,)), jet_args, is_leaf=_is_jet_leaf)
    outs = [jet_fn(*_map_coeffs(itemgetter(r), batched)) for r in range(R)]
    want = tree_map(lambda *o: stack(o), *outs)
    assert_close(vmap(jet_fn, in_dims)(*batched), want, **tolerances_for(device))


def _weighted_sum(fn: Callable, weights: Any, args: tuple) -> Tensor:
    """``fn(*args)`` with every output coefficient weighted and summed to a scalar."""
    return sum((o * w).sum() for o, w in zip(tree_leaves(fn(*args)), weights))


@mark.parametrize("config", CASES, ids=lambda c: c["id"])
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_grad_vjp(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``grad`` and ``vjp`` of a jet w.r.t. primals and coefficients match the oracle.

    The scalar is a fixed random weighting of every output coefficient. The
    oracle nests ``torch.autograd.grad``, which ``torch.func`` cannot wrap, so
    the reference differentiates it with ``torch.autograd.grad``.
    """
    f, jet_fn, jet_args = _jet_setup(config, K, collapsed, device)
    weights = tree_map(rand_like, jet_fn(*jet_args))
    ref_args = tree_map(lambda t: t.detach().requires_grad_(), jet_args)
    oracle = (rev_collapsed_jet if collapsed else _rev_jet)(f, detach=False)
    with DifferentiableSubstitutes():
        scalar = _weighted_sum(oracle, tree_leaves(weights), ref_args)
        want = autograd.grad(scalar, tree_leaves(ref_args))
    tol = tolerances_for(device)
    got = grad(partial(_weighted_sum, jet_fn, tree_leaves(weights)))(jet_args)
    assert_close(tree_leaves(got), list(want), **tol)
    assert_close(tree_leaves(vjp(jet_fn, *jet_args)[1](weights)), list(want), **tol)
