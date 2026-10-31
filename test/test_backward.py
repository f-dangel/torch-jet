"""Jets of the vector-Jacobian product of every primitive case.

A jet of a function that differentiates inside traces to a graph of autograd's
backward ops, so this file checks the jet rules for those ops. A case whose
backward op has no rule yet is a strict expected failure on one cell of the
grid; once it passes, its prefix must leave ``UNSUPPORTED``.
"""

from typing import Any

from pytest import FixtureRequest, mark

from test.test_primitives import PRIMITIVE_CASES
from test.utils import assert_jet_matches_oracle, vjp_case

#: Prefixes of the case ids whose VJP needs a backward op without a jet rule.
UNSUPPORTED = (
    "sdpa",
    "batch_norm",
    "layer_norm",
    "nll_loss",
    "mse_loss",
    "hardtanh",
    "relu",
    "gelu",
    "upsample",
    "index",
    "as_strided",
    "hardsigmoid",
    "hardswish",
    "log_softmax",
    "softmax",
    "embedding",
    "sigmoid",
    "silu",
    "max_pool",
    "select",
    "slice",
    "adaptive",
    "avg_pool",
)
CASES = [vjp_case(c) for c in PRIMITIVE_CASES]
SUPPORTED = [c for c in CASES if not c["id"].startswith(UNSUPPORTED)]
XFAILED = [c for c in CASES if c["id"].startswith(UNSUPPORTED)]
#: MPS has no fused attention kernel, so ``sdpa`` with constant query and key
#: differentiates through ``softmax(scores) @ value``, which needs no new rule.
SUPPORTED_ON = {"mps": {c["id"] for c in CASES if c["id"].startswith("sdpa_CCJ")}}


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
