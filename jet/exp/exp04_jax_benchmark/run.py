"""Execute the JAX benchmark and gather the results.

Runs a series of benchmark experiments by calling out to a script that executes the
measurement of one experiment in a separate Python session to avoid memory allocations
from previous measurements to leak into the current one. The results are gathered in a
specified directory in csv files.
"""

from os import makedirs, path

from torch import linspace

from jet.exp.exp01_benchmark_laplacian.execute import SUPPORTED_STRATEGIES
from jet.exp.exp01_benchmark_laplacian.run import measure
from jet.exp.exp04_jax_benchmark.execute import HERE as SCRIPT
from jet.exp.exp04_jax_benchmark.execute import RAWDIR

HEREDIR = path.dirname(path.abspath(__file__))
GATHERDIR = path.join(HEREDIR, "gathered")
makedirs(GATHERDIR, exist_ok=True)

EXPERIMENTS = [
    # Experiment 1:  Use the largest MLP from dangel2024kroneckerfactored with 50
    #                in features; vary the batch size.
    (  # Experiment name, must be unique
        "jax_laplacian_vary_batch_size",
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
    # Experiment 2:  Use the largest MLP from dangel2024kroneckerfactored with 5
    #                in features; vary the batch size, computing the Bi-Laplacian.
    (  # Experiment name, must be unique
        "jax_bilaplacian_vary_batch_size",
        # Experiment parameters
        {
            "architectures": ["tanh_mlp_768_768_512_512_1"],
            "dims": [5],
            "batch_sizes": linspace(1, 512, 10).int().unique().tolist(),
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
        measure(
            **experiment,
            name=name,
            skip_existing=True,
            gather_every=10,
            script_file=SCRIPT,
            rawdir=RAWDIR,
            gatherdir=GATHERDIR,
        )
