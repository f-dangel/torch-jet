"""Utility functions for the experiments."""

from statistics import mean, stdev
from subprocess import CalledProcessError, CompletedProcess, run
from time import perf_counter
from typing import Callable, List, Tuple, Union
from warnings import warn

import jax
from memory_profiler import memory_usage
from torch import cuda


def measure_time(
    f: Callable, name: str, is_cuda: bool, num_repeats: int = 50, warmup: int = 10
) -> Tuple[float, float, float]:
    """Measure the CPU time of a function.

    Args:
        f: Function to measure the time of.
        name: Name of the function. Will be used to print the timings.
        is_cuda: Whether the function is executed on a CUDA device.
        num_repeats: Number of times to repeat the measurement. Default: `50`.
        warmup: Number of warmup runs before measuring the time. Default: `10`.

    Returns:
        Mean, standard deviation, and best of measured times.
    """
    for _ in range(warmup):
        f()
        if is_cuda:
            cuda.synchronize()

    times = []
    for _ in range(num_repeats):
        start = perf_counter()
        _ = f()
        if is_cuda:
            cuda.synchronize()
        times.append(perf_counter() - start)

    mu, sigma = mean(times), stdev(times)
    best = min(times)
    print(f"{name}: {mu:.5f} ± {sigma:.5f} s (best: {best:.5f} s)")

    return mu, sigma, best


def measure_peak_memory(
    f: Callable, name: str, is_cuda: bool, use_jax: bool = False
) -> float:
    """Measure the peak memory usage of a function.

    Args:
        f: Function to measure the peak memory usage of.
        name: Name of the function. Will be used to print the peak memory usage.
        is_cuda: Whether the function is executed on a CUDA device.
        use_jax: Whether the function uses JAX instead of PyTorch. Default: `False`.

    Returns:
        The peak memory usage in GiB.
    """
    if is_cuda:
        if not use_jax:
            cuda.reset_peak_memory_stats()
        f()
        if use_jax:
            # See https://github.com/jax-ml/jax/issues/8096
            stats = jax.devices("cuda")[0].memory_stats()
            peakmem_bytes = stats["peak_bytes_in_use"]
        else:
            peakmem_bytes = cuda.max_memory_allocated()
    else:
        if use_jax:
            warn("Memory measurements of JAX code on CPU do not work.")
        peakmem_bytes = memory_usage(f, interval=1e-4, max_usage=True) * 2**20

    peakmem_gib = peakmem_bytes / 2**30
    print(f"{name}: {peakmem_gib:.2e} GiB")
    return peakmem_gib


def run_verbose(cmd: List[str]) -> CompletedProcess:
    """Run a command and print stdout & stderr if it fails.

    Args:
        cmd: The command to run.

    Returns:
        CompletedProcess: The result of the command.

    Raises:
        CalledProcessError: If the command fails.
    """
    print(f"Running {' '.join(cmd)}")

    def _print_formatted(message, name):
        clean = message.rstrip().replace("\n", "\n\t")
        if not clean:
            return
        print(f"{name}\n\t{clean}")

    try:
        job = run(cmd, capture_output=True, text=True, check=True)
        _print_formatted(job.stdout, "STDOUT:")
        _print_formatted(job.stderr, "STDERR:")
        return job
    except CalledProcessError as e:
        _print_formatted(e.stdout, "STDOUT:")
        _print_formatted(e.stderr, "STDERR:")
        raise e


def to_string(
    drop_none_values: bool = True,
    compact_bool_values: bool = True,
    **kwargs: Union[str, int],
) -> str:
    """Convert a dictionary to a string representation.

    Args:
        **kwargs: The arguments and their values.
        drop_none_values: Whether to drop arguments with value `None`. Default: `True`.
        compact_bool_values: Whether to convert boolean values. If a value is `True`,
            its key will be in the string. If it is `False`, the key will not.

    Returns:
        A string representation of the sorted arguments and their values.
    """
    sorted_keys = sorted(kwargs.keys())
    if drop_none_values:
        sorted_keys = [key for key in sorted_keys if kwargs[key] is not None]

    formatted = []
    for key in sorted_keys:
        value = kwargs[key]
        if isinstance(value, bool) and compact_bool_values and value:
            formatted.append(str(key))
        elif (
            isinstance(value, bool)
            and not compact_bool_values
            or not isinstance(value, bool)
        ):
            formatted.append(f"{key}_{value}")
    return "_".join(formatted)
