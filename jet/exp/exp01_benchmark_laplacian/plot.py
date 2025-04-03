"""Visualize the benchmark results."""

from itertools import product
from os import makedirs, path
from typing import Dict, Union

from matplotlib import pyplot as plt
from numpy import polyfit
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


def savepath(name: str, **kwargs) -> str:
    """Generate a file path for saving a plot.

    Args:
        name: The name of the experiment.
        **kwargs: Other parameters of the experiment.

    Returns:
        A string representing the file path where the plot will be saved.
    """
    filename = to_string(name=name, **kwargs)
    return path.join(PLOTDIR, f"{filename}.pdf")


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
) -> None:
    """Plot a specified metric.

    Args:
        df: The DataFrame containing only the relevant data to plot.
        metric: The metric to plot. Can be `'time'` or `'peak_memory'`.
        x: The column of the values used as x-axis.
        lines: The column of the values used to distinguish lines in the plot.
        ax: The axes to plot the data on.
    """
    ylabel = {"time": "Time [s]", "peak_memory": "Peak memory [GiB]"}[metric]
    x_to_xlabel = {"batch_size": "Batch size", "num_samples": "Monte-Carlo samples"}
    ax.set_xlabel(x_to_xlabel[x])
    ax.set_ylabel(ylabel)

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


def report_relative_performance(df: DataFrame, x: str, lines: str):
    """Report the relative performance between different lines.

    Fits a linear function to each line and reports the differences in slope.

    Args:
        df: The DataFrame containing only the relevant data to analyze.
        x: The column of the values used as x-axis.
        lines: The column of the values used to distinguish lines in the plot.
    """
    metrics = ["best [s]", "peakmem [GiB]", "peakmem non-differentiable [GiB]"]
    line_vals = df[lines].unique().tolist()

    # fit a linear function to each line
    offsets_and_slopes = {m: {val: {}} for m in metrics for val in line_vals}

    for line in line_vals:
        sub_df = df[df[lines] == line]
        xs = sub_df[x]

        for metric in metrics:
            ys = sub_df[metric]
            c0, c1 = polyfit(xs, ys, deg=1)
            offsets_and_slopes[metric][line] = (c0, c1)

    for metric in metrics:
        print(f"Linear fit of {metric} w.r.t. x={x}:")
        c1_max = max(c1 for (_, c1) in offsets_and_slopes[metric].values())
        for line in line_vals:
            c0, c1 = offsets_and_slopes[metric][line]
            print(f"\t{line}:\t{c0:.5f} + {c1:.5f} * x ({c1 / c1_max:.2f}x relative)")


if __name__ == "__main__":
    METRICS = ["time", "peak_memory"]
    MEASUREMENT_COLUMNS = [
        "peakmem non-differentiable [GiB]",
        "peakmem [GiB]",
        "mean [s]",
        "std [s]",
        "best [s]",
    ]

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
            with plt.rc_context(bundles.neurips2024(rel_width=1.0, ncols=2)):
                fig, axs = plt.subplots(ncols=2)
                # fix specific values, leaving only the data to be plotted
                df_fix = fix_columns(df, fix)
                for idx, (ax, metric) in enumerate(zip(axs, METRICS)):
                    plot_metric(df_fix, metric, x, lines, ax)
                    # set ymin to 0
                    ax.set_ylim(bottom=0)
                    if idx == 0:
                        ax.legend()
                filename = savepath(name=name, **fix)
                print(f"Saving plot for experiment {name} to {filename}.")
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)

            report_relative_performance(df_fix, x, lines)
