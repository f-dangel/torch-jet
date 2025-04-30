"""Visualize the benchmark results."""

from itertools import product
from os import makedirs, path
from typing import Dict, Union

from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from tueplots import bundles

from jet.exp.exp01_benchmark_laplacian.run import EXPERIMENTS
from jet.exp.exp01_benchmark_laplacian.run import savepath as savepath_gathered
from jet.exp.utils import to_string

HEREDIR = path.dirname(path.abspath(__file__))
PLOTDIR = path.join(HEREDIR, "figures")
makedirs(PLOTDIR, exist_ok=True)

MARKERS = {"hessian_trace": "o", "jet_naive": ">", "jet_simplified": "<"}
COLORS = {"hessian_trace": "C0", "jet_naive": "C1", "jet_simplified": "C2"}
LINESTYLES = {"hessian_trace": "-", "jet_naive": "-", "jet_simplified": "-"}
LABELS = {
    "hessian_trace": "Hessian trace (baseline)",
    "jet_naive": "Naive jet",
    "jet_simplified": "Collapsed jet (ours)",
}


def savepath(name: str, plotdir: str = PLOTDIR, **kwargs) -> str:
    """Generate a file path for saving a plot.

    Args:
        name: The name of the experiment.
        plotdir: The directory where the plot will be saved. Default is the figure
            directory of the PyTorch benchmark.
        **kwargs: Other parameters of the experiment.

    Returns:
        A string representing the file path where the plot will be saved.
    """
    filename = to_string(name=name, **kwargs)
    return path.join(plotdir, f"{filename}.pdf")


def fix_columns(df: DataFrame, fix: Dict[str, Union[str, int]]) -> DataFrame:
    """Fix specific columns of a DataFrame.

    Args:
        df: The DataFrame to fix columns in.
        fix: The columns to fix and their values.

    Returns:
        The DataFrame with only the rows where the fixed columns have the specified
        values.
    """
    keys = list(fix.keys())
    k0 = keys[0]
    mask = df[k0] == fix[k0]

    for k in keys[1:]:
        mask = mask & (df[k] == fix[k])

    return df[mask]


def plot_metric(
    df: DataFrame,
    metric: str,
    x: str,
    lines: str,
    ax: plt.Axes,
    xlabel: bool = True,
    ylabel: bool = True,
) -> None:
    """Plot a specified metric.

    Args:
        df: The DataFrame containing only the relevant data to plot.
        metric: The metric to plot. Can be `'time'` or `'peak_memory'`.
        x: The column of the values used as x-axis.
        lines: The column of the values used to distinguish lines in the plot.
        ax: The axes to plot the data on.
        xlabel: Whether to add an x-axis label. Defaults to `True`.
        ylabel: Whether to add a y-axis label. Defaults to `True`.
    """
    if xlabel:
        x_to_xlabel = {
            "batch_size": "Batch size",
            "num_samples": "Samples",
            "dim": "Input dimension",
        }
        ax.set_xlabel(x_to_xlabel.get(x, x))
    if ylabel:
        y_to_ylabel = {"time": "Time [s]", "peak_memory": "Mem. [GiB]"}
        ax.set_ylabel(y_to_ylabel.get(metric, metric))

    for line in df[lines].unique().tolist():
        mask = df[lines] == line
        sub_df = df[mask]
        xs = sub_df[x]

        column = {"time": "best [s]", "peak_memory": "peakmem [GiB]"}[metric]
        ax.plot(
            xs,
            sub_df[column],
            label=LABELS[line],
            marker=MARKERS[line],
            linestyle=LINESTYLES[line],
            color=COLORS[line],
            markersize=3,
        )
        if metric == "peak_memory":
            column = "peakmem non-differentiable [GiB]"
            ax.plot(
                xs,
                sub_df[column],
                marker=MARKERS[line],
                linestyle="--",
                color=COLORS[line],
                markersize=3,
                alpha=0.5,
            )


MEASUREMENT_COLUMNS = [
    "peakmem non-differentiable [GiB]",
    "peakmem [GiB]",
    "mean [s]",
    "std [s]",
    "best [s]",
]

if __name__ == "__main__":
    METRICS = ["time", "peak_memory"]

    for name, _, (x, lines) in EXPERIMENTS:
        df = read_csv(savepath_gathered(name))

        # find all columns of df that are not x and lines
        columns = [
            c for c in df.columns.tolist() if c not in [x, lines, *MEASUREMENT_COLUMNS]
        ]
        # find out for which combinations we have to generate plots
        combinations = [
            dict(zip(columns, combination))
            for combination in product(*[df[col].unique().tolist() for col in columns])
        ]
        print(f"Generating {len(combinations)} plots with x={x!r} for {lines!r} ")

        # go over all combinations and plot
        for fix in combinations:
            print(f"Processing combination: {fix}")
            with plt.rc_context(bundles.neurips2024(rel_width=0.27, ncols=1, nrows=2)):
                fig, axs = plt.subplots(nrows=2, sharex=True)
                # fix specific values, leaving only the data to be plotted
                df_fix = fix_columns(df, fix)
                for idx, (ax, metric) in enumerate(zip(axs, METRICS)):
                    plot_metric(df_fix, metric, x, lines, ax, xlabel=idx == 1)
                    # set ymin to 0
                    ax.set_ylim(bottom=0)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                filename = savepath(name=name, **fix)
                print(f"Saving plot for experiment {name} to {filename}.")
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)
