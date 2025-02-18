"""Execute the benchmark measurements and gather the results.

This script runs a series of benchmark experiments to measure the performance
of different strategies for computing the Laplacian of a function. It supports
various architectures, dimensions, batch sizes, and devices. The results are
gathered into a single data frame and saved to a CSV file.
"""

from itertools import product
from os import makedirs, path
from typing import List

from pandas import DataFrame, concat
from torch import cuda, linspace

from jet.exp.exp01_benchmark_laplacian.execute import HERE as SCRIPT
from jet.exp.exp01_benchmark_laplacian.execute import SUPPORTED_STRATEGIES
from jet.exp.exp01_benchmark_laplacian.execute import savepath as savepath_raw
from jet.exp.utils import run_verbose, to_string

HEREDIR = path.dirname(path.abspath(__file__))
GATHERDIR = path.join(HEREDIR, "gathered")
makedirs(GATHERDIR, exist_ok=True)


def measure(
    architectures: List[str],
    dims: List[int],
    batch_sizes: List[int],
    strategies: List[str],
    devices: List[str],
    name: str,
    skip_existing: bool = False,
    gather_every: int = 10,
):
    """Run benchmark measurements for all combinations of input parameters.

    Args:
        architectures: List of neural network architectures to test.
        dims: List of input dimensions for the architectures.
        batch_sizes: List of batch sizes to use in the experiments.
        strategies: List of strategies for computing the Laplacian.
        devices: List of devices to run the experiments on (e.g., `'cpu'`, `'cuda'`).
        skip_existing: Whether to skip experiments if results already exist.
            Default is `False`.
    """
    combinations = list(product(architectures, dims, batch_sizes, strategies, devices))
    for idx, (architecture, dim, batch_size, strategy, device) in enumerate(
        combinations
    ):
        print(f"\n{idx + 1}/{len(combinations)}")
        kwargs = {
            "architecture": architecture,
            "dim": dim,
            "batch_size": batch_size,
            "strategy": strategy,
            "device": device,
        }

        # maybe skip the computation
        raw = savepath_raw(**kwargs)
        if path.exists(raw) and skip_existing:
            print(f"Skipping because file already exists: {raw}.")
            continue

        # if the device is cuda and CUDA is not available, skip the experiment
        if device == "cuda" and not cuda.is_available():
            print("Skipping GPU measurement because CUDA is not available.")
            continue

        cmd = ["python", SCRIPT] + [f"--{key}={value}" for key, value in kwargs.items()]
        run_verbose(cmd)

        # gather data every few measurements so we can plot even before all are done
        if idx % gather_every == 0 or idx == len(combinations) - 1:
            df = gather_data(
                architectures,
                dims,
                batch_sizes,
                strategies,
                devices,
                allow_missing=True,
            )
            filename = savepath(name)
            print(f"Saving gathered data for experiment {name} to {filename}.")
            df.to_csv(filename, index=False)


def gather_data(
    architectures: List[str],
    dims: List[int],
    batch_sizes: List[int],
    strategies: List[str],
    devices: List[str],
    allow_missing: bool = False,
) -> DataFrame:
    """Create a data frame that collects all the results into a single table.

    Args:
        architectures: List of neural network architectures tested.
        dims: List of input dimensions for the architectures.
        batch_sizes: List of batch sizes used in the experiments.
        strategies: List of strategies for computing the Laplacian.
        devices: List of devices the experiments were run on.
        allow_missing: Whether to allow missing result files. Default is False.

    Returns:
        A pandas DataFrame containing the gathered results.
    """
    df = None

    # Iterate over all possible combinations of the input parameters
    for architecture, dim, batch_size, strategy, device in product(
        architectures, dims, batch_sizes, strategies, devices
    ):
        # Create a dictionary for each combination
        result = {
            "architecture": [architecture],
            "dim": [dim],
            "batch_size": [batch_size],
            "strategy": [strategy],
            "device": [device],
        }
        filename = savepath_raw(**{key: value[0] for key, value in result.items()})

        if not path.exists(filename) and allow_missing:
            print(f"Skipping missing file {filename}.")
            continue

        with open(filename, "r") as f:
            content = "\n".join(f.readlines())
            peakmem_no, peakmem, best, mu, sigma = [
                float(n) for n in content.split(", ")
            ]
            result["peakmem non-differentiable [GiB]"] = peakmem_no
            result["peakmem [GiB]"] = peakmem
            result["mean [s]"] = mu
            result["std [s]"] = sigma
            result["best [s]"] = best

        this_df = DataFrame.from_dict(result)
        df = this_df if df is None else concat([df, this_df], ignore_index=True)

    return df


def savepath(name: str) -> str:
    """Generate a file path for saving gathered data.

    Args:
        name: The name of the experiment.

    Returns:
        A string representing the file path where the data will be saved.
    """
    filename = to_string(name=name)
    return path.join(GATHERDIR, f"{filename}.csv")


EXPERIMENTS = [
    # Experiment 1:  Use the largest MLP from dangel2024kroneckerfactored with 10 in
    #                features; vary the batch size.
    (  # Experiment name, must be unique
        "dangel2024kroneckerfactored_vary_batch_size",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [10, 50],
            "batch_sizes": linspace(1, 2048, 25).int().unique().tolist(),
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cpu", "cuda"],
        },
    )
]

if __name__ == "__main__":
    for name, experiment in EXPERIMENTS:
        measure(**experiment, name=name, skip_existing=True, gather_every=10)
