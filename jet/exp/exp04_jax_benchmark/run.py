"""Execute the JAX benchmark and gather the results.

Runs a series of benchmark experiments by calling out to a script that executes the
measurement of one experiment in a separate Python session to avoid memory allocations
from previous measurements to leak into the current one. The results are gathered in a
specified directory in csv files.
"""

from argparse import ArgumentParser
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
    # Experiment 1: Exact Laplacian, vary batch size
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
    # Experiment 2: Stochastic Laplacian, vary MC samples
    (  # Experiment name, must be unique
        "jax_laplacian_vary_num_samples",
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
    # Experiment 3: Stochastic Bi-Laplacian, vary MC samples
    (  # Experiment name, must be unique
        "jax_bilaplacian_vary_num_samples",
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
    # Experiment 4: Exact Bi-Laplacian, vary batch size
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

# make sure experiment names are unique
names = [name for (name, _, _) in EXPERIMENTS]
if len(names) != len(set(names)):
    raise ValueError(f"Experiment names must be unique. Got: {names}.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run benchmark experiments")
    parser.add_argument(
        "--experiment_idx",
        type=int,
        choices=range(len(EXPERIMENTS)),
        help="Index of the experiment to run (if unspecified, run all experiments)",
        required=False,
    )
    args = parser.parse_args()

    idx = args.experiment_idx
    run_experiments = EXPERIMENTS if idx is None else [EXPERIMENTS[idx]]
    print(f"Running {'all experiments' if idx is None else f'experiment {idx}'}.")

    for name, experiment, _ in run_experiments:
        measure(
            **experiment,
            name=name,
            skip_existing=True,
            gather_every=10,
            script_file=SCRIPT,
            rawdir=RAWDIR,
            gatherdir=GATHERDIR,
        )
