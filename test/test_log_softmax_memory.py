"""Peak memory of the ``log_softmax`` rule, counted deterministically.

The rule streams the exp jet through the row sums, so it allocates no more
than its output jet, on logits of any size. The former rule composed ``sub``,
``exp``, ``sum``, ``log``, ``sub``, and held the jets of every intermediate at
once: the output jet plus a logits-sized tensor per order, and more.
"""

from weakref import finalize

from pytest import mark
from torch import Tensor, float32, ops, rand
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from jet._rules import RULES
from jet.compositions import log_softmax
from jet.primitives import JetTuple
from test.utils import make_jet_args

ROWS, VOCAB = 512, 4096


class StorageTracker(TorchDispatchMode):
    """Count the bytes every ATen call allocates and the running maximum.

    Views (``t._base is not None``) share their storage and are not counted;
    the bytes of a new storage are released when its tensor dies. Also records
    the trailing two dims of every tensor argument an ATen call receives.
    """

    def __init__(self) -> None:
        """Start with nothing allocated."""
        super().__init__()
        self.live, self.peak, self.arg_shapes = 0, 0, set()

    def _release(self, nbytes: int) -> None:
        self.live -= nbytes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Run ``func``, note its argument shapes, and track its new storages."""
        for t in tree_leaves((args, kwargs)):
            if isinstance(t, Tensor) and t.ndim >= 2:
                self.arg_shapes.add(tuple(t.shape[-2:]))
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if isinstance(t, Tensor) and t._base is None:
                nbytes = t.untyped_storage().nbytes()
                self.live += nbytes
                self.peak = max(self.peak, self.live)
                finalize(t, self._release, nbytes)
        return out


def composed_log_softmax(x: JetTuple, dim: int) -> JetTuple:
    """The former rule: the jet of every intermediate exists at once."""
    sub, exp = RULES[ops.aten.sub.Tensor], RULES[ops.aten.exp.default]
    sum, log = RULES[ops.aten.sum.dim_IntList], RULES[ops.aten.log.default]
    shifted = sub(x, x[0].amax(dim, keepdim=True))
    return sub(shifted, log(sum(exp(shifted), dim, keepdim=True)))


@mark.parametrize("K", [1, 2, 3, 4], ids=lambda k: f"K{k}")
def test_log_softmax_peak_memory(K: int, collapsed: bool, device: str):
    """The rule allocates no more than its output jet.

    The former composed rule held, on top of its output jet, at least one
    more logits-sized tensor per order. No ATen call inside the rule sees fewer
    rows than the input or a cut vocabulary, so the gain comes from lifetimes
    alone, not from a loop over chunks.
    """
    (x,) = make_jet_args(
        (rand(ROWS, VOCAB, dtype=float32, device=device),), K, collapsed
    )
    x = JetTuple(x, collapsed=collapsed)
    logits = x[0].untyped_storage().nbytes()
    outputs = sum(c.untyped_storage().nbytes() for c in x)
    slack = logits // 64

    with StorageTracker() as new:
        log_softmax(x, 1)
    with StorageTracker() as old:
        composed_log_softmax(x, 1)

    assert new.peak <= outputs + slack
    assert old.peak >= outputs + K * logits
    assert new.arg_shapes <= {(ROWS, VOCAB), (ROWS, 1)}
