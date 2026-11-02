"""Jets of the vector-Jacobian product of every primitive case.

A jet of a function that differentiates inside traces to a graph of autograd's
backward ops, so this file checks the jet rules for those ops. A case whose
backward op has no rule yet is a strict expected failure on one cell of the
grid; once it passes, its prefix must leave ``UNSUPPORTED``.
"""

from functools import partial
from typing import Any

from pytest import FixtureRequest, mark, skip
from torch import bincount, manual_seed, rand, tensor
from torch.func import vjp
from torch.nn.functional import embedding
from torch.testing import assert_close

from jet import jet
from test.test_primitives import PRIMITIVE_CASES
from test.utils import assert_jet_matches_oracle, device_kw, make_jet_args, vjp_case

#: Prefixes of the case ids whose VJP needs a backward op without a jet rule.
UNSUPPORTED = (
    "sdpa",
    "conv",
)
CASES = [vjp_case(c) for c in PRIMITIVE_CASES]
SUPPORTED = [c for c in CASES if not c["id"].startswith(UNSUPPORTED)]
XFAILED = [c for c in CASES if c["id"].startswith(UNSUPPORTED)]
#: On MPS and CUDA the fused attention op has no autograd formula, so the VJP
#: of ``sdpa`` takes the math path, whose ops all have rules.
SUPPORTED_ON = {
    device: {c["id"] for c in CASES if c["id"].startswith("sdpa")}
    for device in ("mps", "cuda")
}


@mark.parametrize("config", SUPPORTED, ids=lambda c: c["id"])
def test_backward(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(vjp(primitive))`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)


@mark.parametrize("config", XFAILED, ids=lambda c: c["id"])
def test_backward_unsupported(
    config: dict[str, Any], device: str, request: FixtureRequest
):
    """A VJP whose backward op has no jet rule raises ``NotImplementedError``."""
    if config["id"] not in SUPPORTED_ON.get(device, ()):
        request.applymarker(
            mark.xfail(strict=True, raises=NotImplementedError, reason="no jet rule")
        )
    assert_jet_matches_oracle(config, K=1, collapsed=False, device=device)


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
