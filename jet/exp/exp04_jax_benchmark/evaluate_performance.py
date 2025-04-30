"""Analyze the performance (absolute and relative) between implementations."""

from os import makedirs, path

from jet.exp.exp01_benchmark_laplacian.evaluate_performance import (
    REFERENCE,
    report_relative_performance,
)
from jet.exp.exp04_jax_benchmark.run import EXPERIMENTS, GATHERDIR

HEREDIR = path.dirname(path.abspath(__file__))
PERFDIR = path.join(HEREDIR, "performance")
makedirs(PERFDIR, exist_ok=True)

if __name__ == "__main__":
    for name, _, (x, lines) in EXPERIMENTS:
        report_relative_performance(
            name, x, lines, REFERENCE, gatherdir=GATHERDIR, perfdir=PERFDIR
        )
