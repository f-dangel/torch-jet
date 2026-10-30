"""End-to-end Taylor-mode tests on tiny Hugging Face transformers (eval mode).

GPT-2 (decoder, causal attention, with and without the language model head)
and BERT (encoder), built from small configs with random weights, so nothing
is downloaded. The models take the sequence of input embeddings as the
differentiated input. ``transformers`` is a ``test`` extra like torchvision.
"""

from typing import Any, Callable

from pytest import mark
from torch import Tensor
from torch.nn import Module
from transformers import (
    BertConfig,
    BertModel,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
    PreTrainedModel,
)

from test.test_torchvision import ORDER_IDS, ORDERS, _model_case
from test.utils import assert_jet_matches_oracle

#: Mark the whole module so CI can run it in its own job (``-m models``).
pytestmark = mark.models

#: Embedding size of every model (the width of one input token).
D = 16


class _FromEmbeddings(Module):
    """Call ``model`` on a sequence of input embeddings and return its tensor output."""

    def __init__(self, model: PreTrainedModel) -> None:
        """Store the Hugging Face ``model``."""
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """Return the logits (language model head) or the last hidden state."""
        out = self.model(inputs_embeds=x)
        return out.logits if hasattr(out, "logits") else out.last_hidden_state


def _gpt2(head: bool = False, **kw: Any) -> Callable:
    """Two-layer GPT-2 (two heads, 50 tokens, 8 positions), optionally with the LM head."""
    config = GPT2Config(
        n_layer=2, n_embd=D, n_head=2, vocab_size=50, n_positions=8, **kw
    )
    return lambda: _FromEmbeddings((GPT2LMHeadModel if head else GPT2Model)(config))


def _bert(**kw: Any) -> Callable:
    """Two-layer BERT (two heads, 50 tokens, 8 positions, feed-forward width 32)."""
    config = BertConfig(
        num_hidden_layers=2,
        hidden_size=D,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=50,
        max_position_embeddings=8,
        **kw,
    )
    return lambda: _FromEmbeddings(BertModel(config))


#: Models that trace end to end, with the shape of one input (no batch dim).
#: The default attention implementation is ``sdpa`` (the fused op per device);
#: ``eager`` spells attention out (``softmax``, ``where`` for the causal mask).
MODELS: dict[str, tuple[Callable, tuple[int, ...]]] = {
    "gpt2": (_gpt2(), (5, D)),
    "gpt2_eager": (_gpt2(attn_implementation="eager"), (5, D)),
    "gpt2_lm_head": (_gpt2(head=True), (5, D)),
    "bert": (_bert(), (5, D)),
    "bert_eager": (_bert(attn_implementation="eager"), (5, D)),
}

HUGGINGFACE_CASES = [
    _model_case(name, ctor, shape) for name, (ctor, shape) in MODELS.items()
]


@mark.parametrize("config", HUGGINGFACE_CASES, ids=lambda c: c["id"])
@mark.parametrize("K, collapsed", ORDERS, ids=ORDER_IDS)
def test_huggingface_model(
    config: dict[str, Any], K: int, collapsed: bool, device: str
):
    """``jet`` of a Hugging Face transformer matches the nested reverse-mode oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)
