"""Visualize the JAX benchmark results."""

from itertools import product
from os import makedirs, path

from matplotlib import pyplot as plt
from pandas import read_csv
from tueplots import bundles

from jet.exp.exp01_benchmark_laplacian.plot import (
    MEASUREMENT_COLUMNS,
    fix_columns,
    plot_metric,
    savepath,
    savepath_gathered,
)
from jet.exp.exp04_jax_benchmark.run import EXPERIMENTS, GATHERDIR

HEREDIR = path.dirname(path.abspath(__file__))
PLOTDIR = path.join(HEREDIR, "figures")
makedirs(PLOTDIR, exist_ok=True)

if __name__ == "__main__":
    METRICS = ["time", "peak_memory"]

    for name, _, (x, lines) in EXPERIMENTS:
        df = read_csv(savepath_gathered(name, gatherdir=GATHERDIR))

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
                filename = savepath(name=name, plotdir=PLOTDIR, **fix)
                print(f"Saving plot for experiment {name} to {filename}.")
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)
