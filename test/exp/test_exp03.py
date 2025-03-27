"""Execute the demo script in the `exp03` folder."""

from os.path import abspath, dirname, join

import jet.exp.exp03_faa_di_bruno as exp03
from jet.exp.exp01_benchmark_laplacian.run import run_verbose

EXP03_DIR = abspath(dirname(exp03.__file__))


def test_run_exp03():
    """Execute the demo script in the `exp03` folder."""
    cmd = ["python", join(EXP03_DIR, "run.py")]
    run_verbose(cmd)
