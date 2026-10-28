"""End-to-end Taylor-mode tests on transformers (eval mode).

``nn.TransformerEncoder`` in three configurations and torchvision's
``vit_b_16``, at the CNN suite's orders (``ORDERS``).
"""

from typing import Any, Callable

from pytest import mark
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import vit_b_16

from test.test_torchvision import ORDER_IDS, ORDERS, _model_case
from test.utils import assert_jet_matches_oracle

#: Mark the whole module so CI can run it in its own job (``-m models``).
pytestmark = mark.models


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
    "vit_b_16": (lambda: vit_b_16(weights=None), (3, 224, 224)),
}

TRANSFORMER_CASES = [
    _model_case(name, ctor, shape) for name, (ctor, shape) in MODELS.items()
]


@mark.parametrize("config", TRANSFORMER_CASES, ids=lambda c: c["id"])
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_transformer(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet`` of a transformer matches the nested reverse-mode oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
