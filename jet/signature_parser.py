"""Parse signatures of specific PyTorch built-in functions."""

import re
import urllib.request
from functools import cache
from inspect import Parameter, Signature
from os import path
from typing import Any, Callable

import torch
from packaging.version import parse


def download_native_functions_yaml() -> str:
    """Download native_functions.yaml from PyTorch GitHub repository.

    This function downloads the `native_functions.yaml` file corresponding to the
    installed version of PyTorch. The file is saved in the same directory as this
    script. Only downloads if the file does not already exists.

    Returns:
        The path to the downloaded (or existing) `native_functions.yaml` file for the
        current PyTorch version.
    """
    # Get the installed PyTorch version
    version = parse(torch.__version__)
    tag = f"v{version.major}.{version.minor}.{version.micro}"

    # Maybe download the native_functions.yaml
    heredir = path.dirname(path.abspath(__file__))
    savepath = path.join(heredir, f"native_functions_{tag.replace('.', '_')}.yaml")
    if not path.exists(savepath):
        url = (
            f"https://raw.githubusercontent.com/pytorch/pytorch/{tag}/"
            + "aten/src/ATen/native/native_functions.yaml"
        )
        urllib.request.urlretrieve(url, savepath)

    return savepath


@cache
def parse_torch_builtin(f: Callable) -> Signature:
    """Parse signature of a PyTorch built-in C++ function.

    This function handles specific PyTorch built-in functions that don't have
    Python signatures accessible via inspect.signature().

    Args:
        f: The callable whose signature is to be parsed.

    Returns:
        Signature object representing the function's signature.

    Raises:
        ValueError: If the function is not supported or recognized.
    """
    # Find the path to native_functions.yaml relative to this file
    with open(download_native_functions_yaml(), "r") as file:
        yaml_content = file.read()

    # Find the function definition in the YAML file
    # Look for "- func: function_name(" pattern
    pattern = rf"- func: {re.escape(f.__name__)}\((.*?)\)"
    search_result = re.search(pattern, yaml_content, re.DOTALL)

    if not search_result:
        raise ValueError(f"Function {f.__name__} not found in native_functions.yaml")

    # Parse the function signature
    signature_str = search_result[1].strip()

    # Split into arguments
    param_strings = signature_str.split(",")
    param_strings = [p.strip() for p in param_strings]

    # Convert parameter strings to Parameter objects
    parameters = [_str_to_param(param_str) for param_str in param_strings]
    parameters = [p for p in parameters if p is not None]

    return Signature(parameters)


def _str_to_param(param_str: str) -> Parameter | None:
    """Convert a parameter string from native_functions.yaml to a Parameter object.

    Args:
        param_str: The parameter string to be converted.

    Returns:
        A Parameter object representing the parameter, or None if parsing fails.
    """
    # Parse each parameter: "Type[dims] name=default" or "Type? name=default"
    # Examples:
    # "Tensor input", "Tensor? bias=None", "bool[3] output_mask"
    # Check if parameter is optional (has ? after type)
    is_optional = "?" in param_str.split()[0] if param_str else False

    # Remove array notation like [3] and optional marker ?
    param_str_clean = re.sub(r"\[.*?\]", "", param_str)
    param_str_clean = param_str_clean.replace("?", "")

    # Split by = to get default value if present
    if "=" in param_str_clean:
        param_part, default_str = param_str_clean.split("=", 1)
        default_str = default_str.strip()
    else:
        param_part = param_str_clean
        default_str = None

    # Split the parameter part to get type and name
    parts = param_part.strip().split()
    if len(parts) >= 2:
        # Type and name are separate
        param_name = parts[-1]
    elif len(parts) == 1:
        # Only name provided (rare case)
        param_name = parts[0]
    else:
        return None

    kwargs = {}
    if default_str is not None or is_optional:
        kwargs["default"] = _str_to_default_value(is_optional, default_str)

    return Parameter(param_name, Parameter.POSITIONAL_OR_KEYWORD, **kwargs)


def _str_to_default_value(is_optional: bool, default_str: str | None) -> Any:
    """Convert the default value string to an actual Python value.

    Args:
        is_optional: Whether the parameter is optional (indicated by a `?` in the type).
        default_str: The default value as a string, or None if not specified.

    Returns:
        The default value in appropriate Python type, or None if not specified.

    Raises:
        NotImplementedError: If the default value string cannot be converted.
    """
    if default_str == "None" or (is_optional and default_str is None):
        default_value = None
    else:
        assert isinstance(default_str, str)
        if default_str == "True":
            default_value = True
        elif default_str == "False":
            default_value = False
        # handle integers
        elif default_str.isdigit() or (
            default_str[0] == "-" and default_str[1:].isdigit()
        ):
            default_value = int(default_str)
        # handle floats
        elif re.match(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$", default_str):
            default_value = float(default_str)
        else:
            raise NotImplementedError(f"Converting {default_str=} not supported.")

    return default_value
