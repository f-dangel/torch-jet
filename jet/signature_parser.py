"""Parse signatures of specific PyTorch built-in functions."""

import re
import urllib.request
import warnings
from functools import cache
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import yaml
from packaging.version import parse


def _get_native_functions_yaml(
    path_to_native_functions: Optional[Path] = None,
) -> Path:
    """Return the path to the PyTorch native_functions.yaml.

        Downloads the file if it does not exist locally and stores
        it at `path_to_native_functions`. By default, the file is
        saved next to this script, named according to the installed
        PyTorch version.

    Args:
        path_to_native_functions: Optional custom path to store or
        look for the YAML file.

    Returns:
        Path to the downloaded (or existing) `native_functions.yaml`
        file for the current PyTorch version.

    Raises:
        RuntimeError: If the file could not be downloaded.
    """
    version = parse(torch.__version__)
    tag = f"v{version.major}.{version.minor}.{version.micro}"

    if not path_to_native_functions:
        heredir = Path(__file__).parent
        path_to_native_functions = (
            heredir / f"native_functions_{tag.replace(".", "_")}.yaml"
        )

    if not path_to_native_functions.exists():
        warnings.warn(
            f"{path_to_native_functions} not found! Attempting to download...",
            stacklevel=2,
        )
        url = (
            f"https://raw.githubusercontent.com/pytorch/pytorch/{tag}/"
            + "aten/src/ATen/native/native_functions.yaml"
        )
        try:
            urllib.request.urlretrieve(url, path_to_native_functions)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download {url}: {e.reason}") from e

    return path_to_native_functions


@cache
def _preprocess(path: Path) -> dict[str, str]:
    """Parse native_functions.yaml into a lookup dictionary.

    Args:
        path: Path to the native_functions.yaml file.

    Returns:
        A mapping from function name (e.g. "linear") to its raw argument string
        (the contents inside the parentheses of the `func` entry).

    Note:
        Assumes each entry in the YAML has a key "func" of the form
        "func_name(arg1, arg2, ...)".
    """
    with open(path, "r", encoding="utf-8") as file:
        yaml_content = yaml.safe_load(file)
    func_map = {}
    for entry in yaml_content:
        func_name = entry["func"].split("(")[0]
        func_map[func_name] = entry["func"].split("(")[1].split(")")[0]
    return func_map


def _search_signature(func_name: str, func_map: dict[str, str]) -> str:
    """Return the raw argument string for a given function name.

    Args:
        func_name: Name of the function (e.g. "linear").
        func_map: Preprocessed mapping of function names to argument strings.

    Returns:
        The raw argument string if found, otherwise an empty string.
    """
    return func_map.get(func_name, "")


@cache
def parse_torch_builtin(f: Callable) -> Signature:
    """Parse signature of a PyTorch built-in C++ function.

    This function handles specific PyTorch built-in functions that don't have
    Python signatures accessible via ``inspect.signature()``.

    Args:
        f: The callable whose signature is to be parsed.

    Returns:
        Signature object representing the function's signature.

    Raises:
        ValueError: If the function is not supported or recognized.

    Note:
        Assumes that native_functions.yaml contains entries with a "func"
        key formatted as ``"name(arg1, arg2, ...)"``.
    """
    search_result = _search_signature(
        f.__name__, _preprocess(_get_native_functions_yaml())
    )
    if not search_result:
        raise ValueError(f"Function {f.__name__} not found in native_functions.yaml")

    # Split into arguments
    param_strings = search_result.split(",")
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
        # matches e.g., -2 or 3
        elif re.match(r"^-?\d+$", default_str):
            default_value = int(default_str)
        # matches e.g., -1.2, 1.443e+04
        elif re.match(r"^-?\d+(?:\.\d+)?(?:[e][-+]?\d+)?$", default_str):
            default_value = float(default_str)
        else:
            raise NotImplementedError(f"Converting {default_str=} not supported.")

    return default_value
