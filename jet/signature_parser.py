"""Parse signatures of specific PyTorch built-in functions."""

from functools import cache
from importlib.metadata import version as get_version
from inspect import Parameter, Signature
from pathlib import Path
from re import match, sub
from typing import Any, Callable, Optional
from urllib.error import URLError
from urllib.request import urlretrieve
from warnings import warn

from packaging.version import parse
from yaml import safe_load


def _get_native_functions_yaml() -> Path:
    """Return the path to the PyTorch native_functions.yaml.

        Downloads the file if it does not exist locally and stores
        it next to this script, named according to the installed
        PyTorch version.

    Returns:
        Path to the downloaded (or existing) `native_functions.yaml`
        file for the current PyTorch version.

    Raises:
        RuntimeError: If the file could not be downloaded.
    """
    version = parse(get_version("torch"))
    tag = f"v{version.major}.{version.minor}.{version.micro}"

    heredir = Path(__file__).parent
    path_to_native_functions = (
        heredir / f"native_functions_{tag.replace('.', '_')}.yaml"
    )

    if not path_to_native_functions.exists():
        warn(
            f"{path_to_native_functions} not found! Attempting to download...",
            stacklevel=2,
        )
        url = (
            f"https://raw.githubusercontent.com/pytorch/pytorch/{tag}/"
            + "aten/src/ATen/native/native_functions.yaml"
        )
        try:
            urlretrieve(url, path_to_native_functions)
        except URLError as e:
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
        yaml_content = safe_load(file)
    return {
        entry["func"].split("(")[0]: entry["func"].split("(")[1].split(")")[0]
        for entry in yaml_content
    }


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
    func_map = _preprocess(_get_native_functions_yaml())
    search_result = func_map.get(f.__name__, "")

    if not search_result:
        raise ValueError(f"Function {f.__name__} not found in native_functions.yaml")

    # Split into arguments and convert parameter strings to Parameter objects.
    # After encountering "*", all following parameters are keyword-only.
    keyword_only = False
    parameters = []
    for param_str in map(str.strip, search_result.split(",")):
        if param_str == "*":
            keyword_only = True
        elif param := _str_to_param(param_str, keyword_only):
            parameters.append(param)

    return Signature(parameters)


def _str_to_param(
    param_str: str, keyword_only: Optional[bool] = False
) -> Parameter | None:
    """Convert a parameter string from native_functions.yaml to a Parameter object.

    Args:
        param_str: The parameter string to be converted.
        keyword_only: Inform if keyword is enforced.

    Returns:
        A Parameter object representing the parameter, or None if parsing fails.

    Examples:
        >>> _str_to_param("Tensor input")
        <Parameter "input">

        >>> _str_to_param("Tensor? bias=None")
        <Parameter "bias=None">

        >>> _str_to_param("bool[3] output_mask")
        <Parameter "output_mask">

        >>> _str_to_param("Tensor(a!) self")
        <Parameter "self">

        >>> _str_to_param("Tensor(a!) self", keyword_only=True)
        <Parameter "self=None">
    """
    # Check if parameter is optional (has ? after type)
    is_optional = "?" in param_str.split()[0] if param_str else False

    # Remove array notation like [3] and optional marker ? or mutable marker !
    param_str_clean = sub(r"\[.*?\]", "", param_str)
    param_str_clean = param_str_clean.replace("?", "").replace("!", "")

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

    default_str = "None" if default_str is None and keyword_only else default_str
    kwargs = (
        {"default": _str_to_default_value(is_optional, default_str)}
        if (default_str is not None or is_optional)
        else {}
    )

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
        elif match(r"^-?\d+$", default_str):
            default_value = int(default_str)
        # matches e.g., -1.2, 1.443e+04
        elif match(r"^-?\d+(?:\.\d+)?(?:[e][-+]?\d+)?$", default_str):
            default_value = float(default_str)
        else:
            raise NotImplementedError(f"Converting {default_str=} not supported.")

    return default_value
