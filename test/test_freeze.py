"""Test `jet.freeze`: a jet function traced once on fixed arguments."""

from pytest import mark, raises
from torch import rand
from torch.testing import assert_close

from jet import freeze, jet
from test.utils import SCALAR_OUTPUT_CASES, make_jet_args, setup_case, tolerances_for

MODES = [(1, False), (2, False), (2, True)]
MODE_IDS = [f"K={K}-{'collapsed' if c else 'standard'}" for K, c in MODES]


@mark.parametrize("K, collapsed", MODES, ids=MODE_IDS)
def test_freeze_matches_jet(K: int, collapsed: bool, device: str = "cpu"):
    """The frozen jet function matches the plain jet callable."""
    f, args = setup_case(SCALAR_OUTPUT_CASES[0], device)
    jet_fn = jet(f, args, collapsed=collapsed)
    jet_args = make_jet_args(args, K, collapsed=collapsed)
    frozen = freeze(jet_fn, jet_args)
    assert_close(frozen(*jet_args), jet_fn(*jet_args), **tolerances_for(device))


def test_freeze_rejects_other_arguments():
    """A different order or leaf shape raises ``ValueError``."""
    f, args = setup_case(SCALAR_OUTPUT_CASES[0])
    jet_args = make_jet_args(args, K=1)
    frozen = freeze(jet(f, args), jet_args)
    with raises(ValueError, match="leaves"):
        frozen(*make_jet_args(args, K=2))
    with raises(ValueError, match="shape"):
        frozen(*make_jet_args((rand(6, dtype=args[0].dtype),), K=1))
