"""Execute the benchmark measurements and gather the results.

This script runs a series of benchmark experiments to measure the performance
of different strategies for computing the Laplacian of a function. It supports
various architectures, dimensions, batch sizes, and devices. The results are
gathered into a single data frame and saved to a CSV file.
"""

from itertools import product
from os import makedirs, path
from typing import List, Optional

from pandas import DataFrame, concat
from torch import cuda, linspace

from jet.exp.exp01_benchmark_laplacian.execute import HERE as SCRIPT
from jet.exp.exp01_benchmark_laplacian.execute import RAWDIR, SUPPORTED_STRATEGIES
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
    distributions: Optional[List[str]] = None,
    nums_samples: Optional[List[int]] = None,
    operator: str = "laplacian",
    script_file: str = SCRIPT,
    rawdir: str = RAWDIR,
    gatherdir: str = GATHERDIR,
):
    """Run benchmark measurements for all combinations of input parameters.

    Args:
        architectures: List of neural network architectures to test.
        dims: List of input dimensions for the architectures.
        batch_sizes: List of batch sizes to use in the experiments.
        strategies: List of strategies for computing the Laplacian.
        devices: List of devices to run the experiments on (e.g., `'cpu'`, `'cuda'`).
        name: The name of the experiment. Must be unique.
        skip_existing: Whether to skip experiments if results already exist.
            Default is `False`.
        gather_every: How often to gather the data into a single table. Default is `10`.
        distributions: List of distributions for the randomized Laplacian. `None` means
            that the exact Laplacian will be benchmarked. Default is `None`.
        nums_samples: List of numbers of samples for the randomized Laplacian. `None`
            means that the exact Laplacian will be benchmarked. Default is `None`.
        operator: The differential operator to benchmark. Default is `'laplacian'`.
        gatherdir: The directory to save the gathered data into. Default is the gather
            directory of the PyTorch benchmark.
        script_file: The path to the script file that runs the benchmark. Default is the
            script of the PyTorch benchmark.
        rawdir: The directory to save the raw data into. Default is the raw directory
            of the PyTorch benchmark.
    """
    _distributions = [None] if distributions is None else distributions
    _nums_samples = [None] if nums_samples is None else nums_samples

    combinations = list(
        product(
            architectures,
            dims,
            batch_sizes,
            strategies,
            devices,
            _distributions,
            _nums_samples,
        )
    )
    for idx, (
        architecture,
        dim,
        batch_size,
        strategy,
        device,
        distribution,
        num_samples,
    ) in enumerate(combinations):
        print(f"\n{idx + 1}/{len(combinations)}")
        kwargs = {
            "architecture": architecture,
            "dim": dim,
            "batch_size": batch_size,
            "strategy": strategy,
            "device": device,
            "distribution": distribution,
            "num_samples": num_samples,
            "operator": operator,
        }

        # maybe skip the computation
        raw = savepath_raw(rawdir=rawdir, **kwargs)
        skip = False
        if path.exists(raw) and skip_existing:
            print(f"Skipping because file already exists: {raw}.")
            skip = True

        # if the device is cuda and CUDA is not available, skip the experiment
        if device == "cuda" and not cuda.is_available():
            print("Skipping GPU measurement because CUDA is not available.")
            skip = True

        if not skip:
            cmd = ["python", script_file] + [
                f"--{key}={value}" for key, value in kwargs.items() if value is not None
            ]
            run_verbose(cmd)

        # gather data every few measurements so we can plot even before all are done
        is_last = idx == len(combinations) - 1
        if idx % gather_every == 0 or is_last:
            df = gather_data(
                architectures,
                dims,
                batch_sizes,
                strategies,
                devices,
                _distributions,
                _nums_samples,
                operator,
                rawdir,
                allow_missing=not is_last,
            )
            filename = savepath(name, gatherdir=gatherdir)
            print(f"Saving gathered data for experiment {name} to {filename}.")
            df.to_csv(filename, index=False)


def gather_data(
    architectures: List[str],
    dims: List[int],
    batch_sizes: List[int],
    strategies: List[str],
    devices: List[str],
    distributions: List[Optional[str]],
    nums_samples: List[Optional[int]],
    operator: str,
    rawdir: str,
    allow_missing: bool = False,
) -> DataFrame:
    """Create a data frame that collects all the results into a single table.

    Args:
        architectures: List of neural network architectures tested.
        dims: List of input dimensions for the architectures.
        batch_sizes: List of batch sizes used in the experiments.
        strategies: List of strategies for computing the Laplacian.
        devices: List of devices the experiments were run on.
        distributions: List of distributions for the randomized Laplacian.
        nums_samples: List of numbers of samples for the randomized Laplacian.
        operator: The differential operator that was benchmarked.
        rawdir: The directory where the raw data is stored.
        allow_missing: Whether to allow missing result files. Default is False.

    Returns:
        A pandas DataFrame containing the gathered results.
    """
    df = None

    # Iterate over all possible combinations of the input parameters
    for (
        architecture,
        dim,
        batch_size,
        strategy,
        device,
        distribution,
        num_samples,
    ) in product(
        architectures,
        dims,
        batch_sizes,
        strategies,
        devices,
        distributions,
        nums_samples,
    ):
        # Create a dictionary for each combination
        result = {
            "architecture": [architecture],
            "dim": [dim],
            "batch_size": [batch_size],
            "strategy": [strategy],
            "device": [device],
            "distribution": [distribution],
            "num_samples": [num_samples],
        }
        filename = savepath_raw(
            **{key: value[0] for key, value in result.items()},
            operator=operator,
            rawdir=rawdir,
        )

        if not path.exists(filename) and allow_missing:
            print(f"Skipping missing file {filename}.")
            continue

        with open(filename, "r") as f:
            content = "\n".join(f.readlines())
            peakmem_no, peakmem, mu, sigma, best = [
                float(n) for n in content.split(", ")
            ]
            result["peakmem non-differentiable [GiB]"] = peakmem_no
            result["peakmem [GiB]"] = peakmem
            result["mean [s]"] = mu
            result["std [s]"] = sigma
            result["best [s]"] = best

        this_df = DataFrame.from_dict({k: v for k, v in result.items() if v != [None]})
        df = this_df if df is None else concat([df, this_df], ignore_index=True)

    return df


def savepath(name: str, gatherdir: str = GATHERDIR) -> str:
    """Generate a file path for saving gathered data.

    Args:
        name: The name of the experiment.
        gatherdir: The directory where the gathered data will be saved. Default is the
            gather directory of the PyTorch benchmark.

    Returns:
        A string representing the file path where the data will be saved.
    """
    filename = to_string(name=name)
    return path.join(gatherdir, f"{filename}.csv")


EXPERIMENTS = [
    # Experiment 1:  Use the largest MLP from dangel2024kroneckerfactored with 50
    #                in features; vary the batch size.
    (  # Experiment name, must be unique
        "laplacian_vary_batch_size",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [50],
            "batch_sizes": linspace(1, 2048, 10).int().unique().tolist(),
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cuda"],
            "operator": "laplacian",
        },
        # what to plot: x-axis is batch_sizes and each strategy is plotted in a curve
        ("batch_size", "strategy"),
    ),
    # Experiment 2:  Use the largest MLP from dangel2024kroneckerfactored with 50 in
    #                features, vary the batch size.
    (  # Experiment name, must be unique
        "laplacian_vary_num_samples",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [50],
            "batch_sizes": [2048],
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cuda"],
            "operator": "laplacian",
            "distributions": ["normal"],
            "nums_samples": linspace(1, 50, 10).int().unique().tolist(),
        },
        # what to plot: x-axis is nums_samples and each strategy is plotted in a curve
        ("num_samples", "strategy"),
    ),
    # Experiment 3:  Use the largest MLP from dangel2024kroneckerfactored and vary the
    #                in features, computing the Bi-Laplacian.
    (  # Experiment name, must be unique
        "bilaplacian_vary_dim",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": linspace(1, 10, 10).int().unique().tolist(),
            "batch_sizes": [256],
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cuda"],
            "operator": "bilaplacian",
        },
        # what to plot: x-axis is dims and each strategy is plotted in a curve
        ("dim", "strategy"),
    ),
    # Experiment 4:  Use the largest MLP from dangel2024kroneckerfactored with 50
    #                in features; vary the batch size, computing the weighted Laplacian.
    (  # Experiment name, must be unique
        "weighted_laplacian_vary_batch_size",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [50],
            "batch_sizes": linspace(1, 2048, 10).int().unique().tolist(),
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cuda"],
            "operator": "weighted-laplacian",
        },
        # what to plot: x-axis is batch size and each strategy is plotted in a curve
        ("batch_size", "strategy"),
    ),
    # Experiment 5:  Use the largest MLP from dangel2024kroneckerfactored and with
    #                50 in features, vary the MC samples computing the randomized
    #                weighted Laplacian.
    (  # Experiment name, must be unique
        "weighted_laplacian_vary_num_samples",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [50],
            "batch_sizes": [2048],
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cuda"],
            "operator": "weighted-laplacian",
            "distributions": ["normal"],
            "nums_samples": linspace(1, 50, 10).int().unique().tolist(),
        },
        # what to plot: x-axis is nums_samples and each strategy is plotted in a curve
        ("num_samples", "strategy"),
    ),
    # Experiment 6:  Use the largest MLP from dangel2024kroneckerfactored and with
    #                5 in features, vary the MC samples computing the randomized
    #                Bi-Laplacian.
    (  # Experiment name, must be unique
        "bilaplacian_vary_num_samples",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [5],
            "batch_sizes": [256],
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cuda"],
            "operator": "bilaplacian",
            "distributions": ["normal"],
            # exact takes 4.5 D**2 - 1.5 D + 4 = 109, randomized takes 2 + 3S, so
            # choosing S <= 36 because for S=36 we can compute the Bi-Laplacian exactly
            "nums_samples": linspace(1, 36, 10).int().unique().tolist(),
        },
        # what to plot: x-axis is nums_samples and each strategy is plotted in a curve
        ("num_samples", "strategy"),
    ),
    # Experiment 7:  Use the largest MLP from dangel2024kroneckerfactored with 5
    #                in features; vary the batch size, computing the Bi-Laplacian.
    (  # Experiment name, must be unique
        "bilaplacian_vary_batch_size",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [5],
            "batch_sizes": linspace(1, 1024, 10).int().unique().tolist(),
            "strategies": SUPPORTED_STRATEGIES,
            "devices": ["cuda"],
            "operator": "bilaplacian",
        },
        # what to plot: x-axis is batch size and each strategy is plotted in a curve
        ("batch_size", "strategy"),
    ),
]

if __name__ == "__main__":
    names = [name for (name, _, _) in EXPERIMENTS]
    if len(names) != len(set(names)):
        raise ValueError(f"Experiment names must be unique. Got: {names}.")

    for name, experiment, _ in EXPERIMENTS:
        measure(**experiment, name=name, skip_existing=True, gather_every=10)
