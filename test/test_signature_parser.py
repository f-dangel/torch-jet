"""Tests for the PyTorch built-in signature parser module."""

from inspect import Parameter, Signature
from typing import Callable

from pytest import mark
from torch import allclose
from torch.nn.functional import celu, conv1d, linear

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
    (
        conv1d,  # has integer-valued default arguments
        [
            Parameter("input", POK),
            Parameter("weight", POK),
            Parameter("bias", POK, default=None),
            Parameter("stride", POK, default=1),
            Parameter("padding", POK, default=0),
            Parameter("dilation", POK, default=1),
            Parameter("groups", POK, default=1),
        ],
    ),
    (
        celu,  # has float-valued default arguments without scientific notation
        [Parameter("self", POK), Parameter("alpha", POK, default=1.0)],
    ),
    (
        allclose,  # has float-valued default arguments with scientific notation
        [
            Parameter("self", POK),
            Parameter("other", POK),
            Parameter("rtol", POK, default=1e-05),
            Parameter("atol", POK, default=1e-08),
            Parameter("equal_nan", POK, default=False),
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
