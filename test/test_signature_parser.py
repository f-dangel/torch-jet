"""Tests for the PyTorch built-in signature parser module."""

import urllib.request
from importlib.metadata import version as get_version
from inspect import Parameter, Signature
from pathlib import Path
from typing import Callable
from warnings import catch_warnings, simplefilter

from packaging.version import parse
from pytest import MonkeyPatch, mark, raises
from torch import allclose, cos, sigmoid, sin, tanh
from torch.nn.functional import celu, conv1d, conv2d, linear
from yaml import dump

from jet.signature_parser import (
    _get_native_functions_yaml,
    _preprocess,
    _str_to_default_value,
    _str_to_param,
    parse_torch_builtin,
)

POK = Parameter.POSITIONAL_OR_KEYWORD  # shortcut


def test_get_native_functions_yaml_existing():
    """Test if return existing file if present."""
    version = parse(get_version("torch"))
    tag = f"v{version.major}.{version.minor}.{version.micro}"
    expected_suffix = f"native_functions_{tag.replace('.', '_')}.yaml"
    result = _get_native_functions_yaml()
    assert isinstance(result, Path)
    assert result.name == expected_suffix
    assert result.exists()


def test_get_native_functions_yaml_download(monkeypatch: MonkeyPatch):
    """Test that missing file is downloaded.

    Args:
        monkeypatch: A mocking utility to mock the function urllib.requests.urlretrieve.
    """
    # ensure file is removed to check for dummy content
    result = _get_native_functions_yaml()
    result.unlink()

    called = {}

    def fake_urlretrieve(url, dest):
        called["url"] = url
        dest.write_text("downloaded")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    with catch_warnings(record=True) as w:
        simplefilter("always")
        result = _get_native_functions_yaml()
        assert any("Attempting to download" in str(warn.message) for warn in w)

    assert "url" in called
    assert result.exists()
    assert result.read_text() == "downloaded"

    # delete file with dummy content
    result.unlink()


mock_yaml_path = Path("mock_native_functions.yaml")
mock_yaml_content = [
    {"func": "linear(Tensor input, Tensor weight, Tensor? bias=None)"},
    {"func": "tanh(Tensor self)"},
]
mock_yaml_path.write_text(dump(mock_yaml_content))


def test_preprocess():
    """Test whether _preprocess work correctly for expected yaml format."""
    func_map = _preprocess(mock_yaml_path)
    assert func_map["linear"] == "Tensor input, Tensor weight, Tensor? bias=None"
    assert func_map["tanh"] == "Tensor self"


TORCH_BUILTINS = [
    (sin, [Parameter("self", POK)]),
    (cos, [Parameter("self", POK)]),
    (tanh, [Parameter("self", POK)]),
    (sigmoid, [Parameter("self", POK)]),
    (
        linear,
        [
            Parameter("input", POK),
            Parameter("weight", POK),
            Parameter("bias", POK, default=None),
        ],
    ),
    (
        conv1d,
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
        conv2d,
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
        celu,
        [Parameter("self", POK), Parameter("alpha", POK, default=1.0)],
    ),
    (
        allclose,
        [
            Parameter("self", POK),
            Parameter("other", POK),
            Parameter("rtol", POK, default=1e-05),
            Parameter("atol", POK, default=1e-08),
            Parameter("equal_nan", POK, default=False),
        ],
    ),
]
TORCH_BUILTIN_IDS = [f"{f.__module__}.{f.__name__}" for f, _ in TORCH_BUILTINS]


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


def test_str_to_param():
    """Test various the output of _str_to_param for various string inputs."""
    assert _str_to_param("Tensor input") == Parameter(
        "input", Parameter.POSITIONAL_OR_KEYWORD
    )
    assert _str_to_param("Tensor input=-1.0e-3") == Parameter(
        "input", Parameter.POSITIONAL_OR_KEYWORD, default=-0.001
    )
    assert _str_to_param("Tensor? input") == Parameter(
        "input", Parameter.POSITIONAL_OR_KEYWORD, default=None
    )
    assert _str_to_param("int[3]? bias") == Parameter(
        "bias", Parameter.POSITIONAL_OR_KEYWORD, default=None
    )
    assert _str_to_param("float[2]? test=None") == Parameter(
        "test", Parameter.POSITIONAL_OR_KEYWORD, default=None
    )


def test_str_to_default_value():
    """Test the whether _str_to_default_value parses the input correctly."""
    assert _str_to_default_value(True, "None") is None
    assert _str_to_default_value(False, "None") is None
    assert _str_to_default_value(True, None) is None
    assert _str_to_default_value(True, "-1") == -1
    assert _str_to_default_value(True, "1") == 1
    assert _str_to_default_value(True, "13") == 13
    assert _str_to_default_value(True, "-1.3") == -1.3
    assert _str_to_default_value(True, "21.3") == 21.3
    assert _str_to_default_value(True, "1e-05") == 1e-05
    assert _str_to_default_value(True, "1e05") == 1e05
    assert _str_to_default_value(True, "1.3e-05") == 1.3e-05
    assert _str_to_default_value(True, "-1.3e-05") == -1.3e-05
    with raises(NotImplementedError):
        _str_to_default_value(True, "ggg")
