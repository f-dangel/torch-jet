"""Jets of the vector-Jacobian product of every primitive case.

A jet of a function that differentiates inside traces to a graph of autograd's
backward ops, so this file checks the jet rules for those ops. A case whose
backward op has no rule yet must raise ``NotImplementedError``; once it passes,
its prefix must leave ``UNSUPPORTED``.
"""

from functools import partial
from typing import Any

from pytest import mark, raises, skip
from torch import bincount, manual_seed, rand, tensor
from torch.func import vjp
from torch.nn.functional import embedding
from torch.testing import assert_close

from jet import jet
from test.test_primitives import PRIMITIVE_CASES
from test.utils import assert_jet_matches_oracle, device_kw, make_jet_args, vjp_case

#: Prefixes of the case ids whose VJP needs a backward op without a jet rule.
UNSUPPORTED: tuple[str, ...] = ()


@mark.parametrize("config", map(vjp_case, PRIMITIVE_CASES), ids=lambda c: c["id"])
def test_backward(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(vjp(primitive))`` matches its mode-specific oracle.

    A case listed in ``UNSUPPORTED`` must raise ``NotImplementedError`` on every
    cell that reaches a rule; ``collapsed`` with ``K < 2`` is rejected before
    that and keeps the oracle's check of the guard.
    """
    if config["id"].startswith(UNSUPPORTED) and not (collapsed and K < 2):
        with raises(NotImplementedError, match="No jet rule"):
            assert_jet_matches_oracle(config, K, collapsed, device)
    else:
        assert_jet_matches_oracle(config, K, collapsed, device)


def test_embedding_backward_scale_grad_by_freq(K: int, collapsed: bool, device: str):
    """``scale_grad_by_freq`` divides each row of the VJP by its index's count.

    The substitutes do not model the flag, so the jet with it is checked against
    the jet without it, rescaled by the known counts.
    """
    if collapsed and K < 2:
        skip("collapsed mode requires K >= 2")
    manual_seed(0)
    indices = tensor([0, 2, 0, 1, 0, 2], device=device)  # counts 3, 1, 2
    weight, v = rand(4, 3, **device_kw(device)), rand(6, 3, **device_kw(device))

    def vjp_embedding(weight, v, scale: bool):
        f = partial(embedding, indices, scale_grad_by_freq=scale)
        return vjp(f, weight)[1](v)[0]

    args = make_jet_args((weight, v), K, collapsed)
    scaled, plain = (
        jet(partial(vjp_embedding, scale=scale), (weight, v), collapsed=collapsed)(
            *args
        )
        for scale in (True, False)
    )
    counts = bincount(indices, minlength=weight.shape[0]).clamp(min=1).unsqueeze(-1)
    for s, p in zip(scaled, plain):
        assert_close(s, p / counts)
