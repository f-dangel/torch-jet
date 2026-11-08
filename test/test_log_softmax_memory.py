"""Peak memory of the ``log_softmax`` rule, counted deterministically.

The rule streams the exp jet through the row sums, so it allocates no more
than its output jet, on logits of any size.
"""

from pytest import mark
from torch import rand

from jet.compositions import log_softmax
from jet.primitives import JetTuple
from test.utils import StorageTracker, make_jet_args

ROWS, VOCAB = 512, 4096


@mark.parametrize("K", [1, 2, 3, 4], ids=lambda k: f"K{k}")
def test_log_softmax_peak_memory(K: int, collapsed: bool, device: str):
    """The rule allocates no more than its output jet.

    No ATen call inside the rule sees fewer rows than the input or a cut
    vocabulary, so the bound comes from lifetimes alone, not from a loop over
    chunks.
    """
    (x,) = make_jet_args((rand(ROWS, VOCAB, device=device),), K, collapsed)
    x = JetTuple(x, collapsed=collapsed)
    logits = x[0].untyped_storage().nbytes()
    outputs = sum(c.untyped_storage().nbytes() for c in x)
    slack = logits // 64

    with StorageTracker() as tracker:
        log_softmax(x, 1)

    assert tracker.peak <= outputs + slack
    assert {s[-2:] for s in tracker.arg_shapes} <= {(ROWS, VOCAB), (ROWS, 1)}
