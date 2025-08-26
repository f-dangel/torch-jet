"""Tests for the PyTorch built-in signature parser module."""

from inspect import Parameter, Signature
from typing import Callable

from pytest import mark
from torch.nn.functional import linear

from jet.signature_parser import parse_torch_builtin

POK = Parameter.POSITIONAL_OR_KEYWORD  # shortcut

# NOTE These test cases contain hard-coded ground truth signatures which may change
# in future PyTorch releases.
TORCH_BUILTINS = [
    (
        linear,
        [
            Parameter("input", POK),
            Parameter("weight", POK),
            Parameter("bias", POK, default=None),
        ],
    ),
]
TORCH_BUILTIN_IDS = [f"{b[0].__module__}.{b[0].__name__}" for b in TORCH_BUILTINS]


@mark.parametrize("config", TORCH_BUILTINS, ids=TORCH_BUILTIN_IDS)
def test_parse_torch_builtin(config: tuple[Callable, list[Parameter]]):
    """Test parsing function signatures from PyTorch built-ins.

    Args:
        config: A tuple containing the function and its expected parameters.
    """
    f, params = config
    expected_sig = Signature(params)
    parsed_sig = parse_torch_builtin(f)
    assert parsed_sig == expected_sig
