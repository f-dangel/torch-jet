"""Visualize the benchmark results."""

from itertools import product
from os import makedirs, path
from typing import List

from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from tueplots import bundles

from jet.exp.exp01_benchmark_laplacian.run import EXPERIMENTS
from jet.exp.exp01_benchmark_laplacian.run import savepath as savepath_gathered
from jet.exp.utils import to_string

HEREDIR = path.dirname(path.abspath(__file__))
PLOTDIR = path.join(HEREDIR, "figures")
makedirs(PLOTDIR, exist_ok=True)

MARKERS = {"hessian_trace": "o", "jet_naive": "o", "jet_simplified": "o"}
COLORS = {"hessian_trace": "C0", "jet_naive": "C1", "jet_simplified": "C2"}
LINESTYLES = {"hessian_trace": "-", "jet_naive": "-", "jet_simplified": "-"}
LABELS = {
    "hessian_trace": "Hessian trace (baseline)",
    "jet_naive": "Naive jet",
    "jet_simplified": "Collapsed jet (ours)",
}


def savepath(name: str, architecture: str, dim: int, device: str) -> str:
    """Generate a file path for saving a plot.

    Args:
        name: The name of the experiment.
        architecture: The architecture of the network.
        dim: The dimension of the network.
        device: The device used in the experiment (e.g., 'cpu', 'cuda').

    Returns:
        A string representing the file path where the plot will be saved.
    """
    filename = to_string(name=name, architecture=architecture, dim=dim, device=device)
    return path.join(PLOTDIR, f"{filename}.pdf")


def plot_metric(
    df: DataFrame,
    strategies: List[str],
    architecture: str,
    dim: int,
    metric: str,
    device: str,
    ax: plt.Axes,
) -> None:
    """Plot a specified metric.

    Args:
        df: The DataFrame containing the data to plot.
        strategies: A list of strategies to plot.
        architecture: The architecture of the network to filter the data by.
        dim: The dimension of the network to filter the data by.
        metric: The metric to plot. Can be `'time'` or `'peak_memory'`.
        device: The device to filter the data by (e.g., `'cpu'`, `'cuda'`).
        ax: The axes to plot the data on.
    """
    ylabel = {"time": "Time [s]", "peak_memory": "Peak memory [GiB]"}[metric]
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)

    for strategy in strategies:
        mask = (
            (df["architecture"] == architecture)
            & (df["dim"] == dim)
            & (df["strategy"] == strategy)
            & (df["device"] == device)
        )
        sub_df = df[mask]
        batch_sizes = sub_df["batch_size"]

        column = {"time": "best [s]", "peak_memory": "peakmem [GiB]"}[metric]
        ax.plot(
            batch_sizes,
            sub_df[column],
            label=LABELS[strategy],
            marker=MARKERS[strategy],
            linestyle=LINESTYLES[strategy],
            color=COLORS[strategy],
            markersize=3,
        )


if __name__ == "__main__":
    METRICS = ["time", "peak_memory"]

    for name, _ in EXPERIMENTS:
        df = read_csv(savepath_gathered(name))
        strategies = df["strategy"].unique()
        architectures = df["architecture"].unique()
        devices = df["device"].unique()
        dims = df["dim"].unique()

        # go over all combinations
        for architecture, dim, device in product(architectures, dims, devices):
            with plt.rc_context(bundles.neurips2024(rel_width=1.0, ncols=2)):
                fig, axs = plt.subplots(ncols=2)
                for idx, (ax, metric) in enumerate(zip(axs, METRICS)):
                    plot_metric(df, strategies, architecture, dim, metric, device, ax)
                    # set ymin to 0
                    ax.set_ylim(bottom=0)
                    if idx == 0:
                        ax.legend()
                filename = savepath(name, architecture, dim, device)
                print(f"Saving plot for experiment {name} to {filename}.")
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)
