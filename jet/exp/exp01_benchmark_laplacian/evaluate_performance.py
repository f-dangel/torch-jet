"""Analyze the performance (absolute and relative) between implementations."""

from itertools import product
from os import makedirs, path

from numpy import polyfit
from pandas import read_csv

from jet.exp.exp01_benchmark_laplacian.plot import MEASUREMENT_COLUMNS, fix_columns
from jet.exp.exp01_benchmark_laplacian.run import EXPERIMENTS, GATHERDIR
from jet.exp.exp01_benchmark_laplacian.run import savepath as savepath_gathered
from jet.exp.utils import to_string

HEREDIR = path.dirname(path.abspath(__file__))
PERFDIR = path.join(HEREDIR, "performance")
makedirs(PERFDIR, exist_ok=True)

REFERENCE = "hessian_trace"


def savepath(
    name: str, implementation: str, metric: str, perfdir: str = PERFDIR, **kwargs
) -> str:
    """Generate a file path for saving a plot.

    Args:
        name: The name of the experiment.
        implementation: The implementation whose performance is stored.
        metric: The metric whose performance is reported.
        perfdir: The directory where the performance report will be stored. Default is
            the performance directory of the PyTorch benchmark.
        **kwargs: Other parameters of the experiment.

    Returns:
        A string representing the file path where the plot will be saved.
    """
    subdir = path.join(perfdir, to_string(name=name, **kwargs))
    makedirs(subdir, exist_ok=True)
    return path.join(subdir, f"{implementation}_{metric}.txt")


def report_relative_performance(
    name: str,
    x: str,
    lines: str,
    ref_line: str,
    gatherdir: str = GATHERDIR,
    perfdir: str = PERFDIR,
):
    """Report the relative performance between different lines.

    Fits a linear function to each line and reports the differences in slope.

    Args:
        name: The experiment's name.
        x: The column of the values used as x-axis.
        lines: The column of the values used to distinguish lines in the plot.
        ref_line: The reference line to compare against.
        gatherdir: The directory where the gathered data is stored. Default is the
            gathered data directory of the PyTorch benchmark.
        perfdir: The directory where the performance report will be stored. Default is
            the performance directory of the PyTorch benchmark.
    """
    # tuple of metric and its savename
    metrics = [
        ("best [s]", "best"),
        ("peakmem [GiB]", "peakmem"),
        ("peakmem non-differentiable [GiB]", "peakmem_nondifferentiable"),
    ]
    df = read_csv(savepath_gathered(name, gatherdir=gatherdir))

    # find all columns of df that are not x and lines
    columns = [
        c for c in df.columns.tolist() if c not in [x, lines, *MEASUREMENT_COLUMNS]
    ]
    # find out for which combinations we have to generate performance reports
    combinations = [
        dict(zip(columns, combination))
        for combination in product(*[df[col].unique().tolist() for col in columns])
    ]

    print(f"Performancd report for {name}")
    # go over all combinations and report the performance
    for fix in combinations:
        print(f"Processing combination: {fix}")
        df_fix = fix_columns(df, fix)

        line_vals = df_fix[lines].unique().tolist()

        # fit a linear function to each line
        offsets_and_slopes = {m: {val: {}} for m, _ in metrics for val in line_vals}

        for line in line_vals:
            sub_df_fix = df_fix[df_fix[lines] == line]
            xs = sub_df_fix[x].tolist()

            for metric, _ in metrics:
                ys = sub_df_fix[metric].tolist()

                # use ms instead of s and MiB instead of GiB
                if "[s]" in metric:
                    ys = [y * 1000 for y in ys]
                if "[GiB]" in metric:
                    ys = [y * 2**10 for y in ys]

                c1, c0 = polyfit(xs, ys, deg=1)
                offsets_and_slopes[metric][line] = (c1, c0)

        # report the numbers and save them to files
        for metric, metric_store in metrics:
            # use ms instead of s and MiB instead of GiB
            metric_adapted = metric.replace("[s]", "[ms]").replace("[GiB]", "[MiB]")
            print(f"Linear fit of {metric_adapted} w.r.t. x={x}:")
            c1_ref, _ = offsets_and_slopes[metric][ref_line]
            for line in line_vals:
                c1, c0 = offsets_and_slopes[metric][line]
                relative = c1 / c1_ref
                print(f"\t{line}:\t{c0:.5f} + {c1:.5f} * x ({relative:.3f}x relative)")

                # save to file
                path = savepath(name, line, metric_store, perfdir=perfdir, **fix)
                with open(path, "w") as f:
                    content = r"\num{" + str(c1) + r"} (\num{" + str(relative) + "}x)"
                    f.write(content)


if __name__ == "__main__":
    for name, _, (x, lines) in EXPERIMENTS:
        report_relative_performance(name, x, lines, REFERENCE)
