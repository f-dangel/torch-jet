"""End-to-end Taylor-mode tests on ``nn.TransformerEncoder`` (eval mode).

Three configurations, at the torchvision suite's orders (``ORDERS``).
"""

from collections.abc import Iterator
from typing import Any, Callable

from pytest import fixture, mark
from torch.backends import mha
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from test.test_torchvision import ORDER_IDS, ORDERS, _model_case
from test.utils import assert_jet_matches_oracle

#: Mark the whole module so CI can run it in its own job (``-m models``).
pytestmark = mark.models


@fixture(autouse=True, scope="module")
def no_fast_path() -> Iterator[None]:
    """Run the encoder on its plain path, as a jet does, for this module.

    Tracing and torch-function modes skip PyTorch's fused transformer fast
    path, so only the plain eager reference call would take it. On CUDA, the
    fused kernel computes ``gelu`` with the tanh approximation although the
    layer was built with the exact one (about ``1e-4`` per element, in
    ``float64`` too), so the reference would disagree with every jet result.
    """
    enabled = mha.get_fastpath_enabled()
    mha.set_fastpath_enabled(False)
    yield
    mha.set_fastpath_enabled(enabled)


def _encoder(**kw: Any) -> Callable:
    """Two ``batch_first`` layers (``d_model=16``, two heads, ``dim_feedforward=32``)."""
    return lambda: TransformerEncoder(
        TransformerEncoderLayer(16, 2, dim_feedforward=32, batch_first=True, **kw),
        num_layers=2,
        enable_nested_tensor=False,
    )


#: Models that trace end to end, with the shape of one input (no batch dim).
MODELS: dict[str, tuple[Callable, tuple[int, ...]]] = {
    "transformer_encoder": (_encoder(), (5, 16)),
    "transformer_encoder_norm_first": (_encoder(norm_first=True), (5, 16)),
    "transformer_encoder_gelu": (_encoder(activation="gelu"), (5, 16)),
}

ENCODER_CASES = [
    _model_case(name, ctor, shape) for name, (ctor, shape) in MODELS.items()
]


@mark.parametrize("config", ENCODER_CASES, ids=lambda c: c["id"])
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_transformer_encoder(
    config: dict[str, Any], K: int, collapsed: bool, device: str
):
    """``jet`` of an encoder matches the nested reverse-mode oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
