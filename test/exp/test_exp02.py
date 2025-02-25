"""Execute the demo script in the `exp02` folder."""

from os.path import abspath, dirname, join

import jet.exp.exp02_explain_simplifications as exp02
from jet.exp.exp01_benchmark_laplacian.run import run_verbose

EXP02_DIR = abspath(dirname(exp02.__file__))


def test_run_exp02():
    """Execute the demo script in the `exp02` folder."""
    cmd = ["python", join(EXP02_DIR, "run.py")]
    run_verbose(cmd)
