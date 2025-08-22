"""Parse signatures of specific PyTorch built-in functions."""

import re
import urllib.request
from functools import cache
from inspect import Parameter, Signature
from os import path
from typing import Callable

import torch
from packaging.version import parse

HEREDIR = path.dirname(path.abspath(__file__))


def download_native_functions_yaml() -> str:
    """Download native_functions.yaml from PyTorch GitHub repository.

    This function downloads the `native_functions.yaml` file corresponding to the
    installed version of PyTorch. The file is saved in the same directory as this
    script. Only downloads if the file does not already exists.

    Returns:
        The path to the downloaded (or existing) `native_functions.yaml` file.
    """
    # Get the installed PyTorch version
    version = parse(torch.__version__)
    tag = f"v{version.major}.{version.minor}.{version.micro}"

    # Maybe download the native_functions.yaml
    savepath = path.join(HEREDIR, f"native_functions_{tag.replace('.','_')}.yaml")
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
    native_functions = download_native_functions_yaml()

    # Get the function name
    function_name = f.__name__

    # Find the path to native_functions.yaml relative to this file
    with open(native_functions, "r") as file:
        yaml_content = file.read()

    # Find the function definition in the YAML file
    # Look for "- func: function_name(" pattern
    pattern = rf"- func: {re.escape(function_name)}\((.*?)\)"
    match = re.search(pattern, yaml_content, re.DOTALL)

    if not match:
        raise ValueError(f"Function {function_name} not found in native_functions.yaml")

    # Parse the function signature
    signature_str = match[1].strip()

    # Split by comma, but be careful with nested parentheses
    param_strings = []
    current_param = ""
    paren_depth = 0
    for char in signature_str:
        if char == "," and paren_depth == 0:
            param_strings.append(current_param.strip())
            current_param = ""
        else:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            current_param += char
    if current_param.strip():
        param_strings.append(current_param.strip())

    parameters = []
    for param_str in param_strings:
        # Parse each parameter: "Type[dims] name=default" or "Type? name=default"
        # Examples:
        # "Tensor input"
        # "Tensor? bias=None"
        # "bool[3] output_mask"

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
            continue  # Skip empty parameters

        # Determine default value
        if default_str is not None or is_optional:
            # Parse default value
            if default_str == "None" or (is_optional and default_str is None):
                default_value = None
            elif default_str == "True":
                default_value = True
            elif default_str == "False":
                default_value = False
            elif (
                default_str and default_str.replace("-", "").replace(".", "").isdigit()
            ):
                # Handle integers and floats
                default_value = (
                    float(default_str) if "." in default_str else int(default_str)
                )
            elif (
                default_str
                and default_str.startswith("[")
                and default_str.endswith("]")
            ):
                # Handle list defaults
                # For simplicity, we'll use eval here, but in production code
                # you'd want a safer parser
                try:
                    default_value = eval(default_str)
                except Exception:
                    default_value = default_str
            else:
                # Keep as string for other cases
                default_value = default_str if default_str else None

            parameters.append(
                Parameter(
                    name=param_name,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default=default_value,
                )
            )
        else:
            # Required parameter
            parameters.append(
                Parameter(name=param_name, kind=Parameter.POSITIONAL_OR_KEYWORD)
            )

    return Signature(parameters)
