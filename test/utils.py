"""Utility functions for testing."""

from typing import Any, Callable

from pytest import raises
from torch import (
    Tensor,
    dtype,
    float32,
    float64,
    manual_seed,
    ones,
    rand,
    rand_like,
    randint,
    sigmoid,
    softmax,
    stack,
    where,
    zeros_like,
)
from torch.func import vjp
from torch.nn import Linear, Sequential, Tanh, functional
from torch.nn.functional import (
    embedding,
    hardsigmoid,
    layer_norm,
    max_pool2d,
    scaled_dot_product_attention,
)
from torch.overrides import TorchFunctionMode
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from jet._jet import _rev_jet, jet
from jet.utils import _is_jet_leaf


def dtype_for_device(device: str) -> dtype:
    """Default floating dtype for ``device``.

    MPS doesn't support ``float64``, so MPS gets ``float32``; CPU and CUDA
    use ``float64`` (the test suite's preferred precision).
    """
    return float32 if device == "mps" else float64


def device_kw(device: str) -> dict[str, Any]:
    """Tensor-factory kwargs (``dtype``, ``device``) for ``device``."""
    return {"dtype": dtype_for_device(device), "device": device}


def _hardsigmoid(x: Tensor, inplace: bool = False) -> Tensor:
    """``hardsigmoid`` as a twice-differentiable ``where`` with the same kinks.

    ``relu6(x + 3) / 6`` would do, but ``relu6``'s backward puts slope ``1/6``
    at the kinks on MPS, where ``hardsigmoid``'s puts ``0`` on every device.
    """
    out = where((x > -3) & (x < 3), x / 6 + 0.5, (x >= 3).to(x.dtype))
    return x.copy_(out) if inplace else out


def _layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor:
    """``layer_norm`` from its definition (pytorch/pytorch#175754)."""
    dims = tuple(range(-len(normalized_shape), 0))
    centered = input - input.mean(dims, keepdim=True)
    out = centered * (centered.square().mean(dims, keepdim=True) + eps).rsqrt()
    out = out if weight is None else out * weight
    return out if bias is None else out + bias


def _sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """``scaled_dot_product_attention`` from its definition (``dropout_p == 0``)."""
    if enable_gqa:
        groups = q.shape[-3] // k.shape[-3]
        k, v = (x.repeat_interleave(groups, dim=-3) for x in (k, v))
    scores = (q * (q.shape[-1] ** -0.5 if scale is None else scale)) @ k.mT
    if is_causal:
        attn_mask = ones(q.shape[-2], k.shape[-2], device=q.device).tril().bool()
    if attn_mask is not None:
        if attn_mask.is_floating_point():
            scores = scores + attn_mask
        else:
            scores = scores.masked_fill(~attn_mask, -float("inf"))
    return softmax(scores, -1) @ v


class DifferentiableSubstitutes(TorchFunctionMode):
    """Swap ops PyTorch cannot differentiate (correctly) to high order for equivalent calls it can.

    ``SUBSTITUTES`` maps each such op (on some device) to its substitute. The
    oracle runs under this mode; :func:`assert_jet_matches_oracle` checks on
    every case that the swap leaves the forward pass unchanged, and
    ``test/test_substitutes.py`` asserts per entry the PyTorch limitation
    behind it, so that a PyTorch fix shows up as a failure to remove the entry.

    Intercepts the Python-level call (``F.max_pool2d``, which ``nn.MaxPool2d``
    also goes through), so the substitute's ops are what autograd records.
    While the mode handles one call, PyTorch switches it off, so calls made
    inside that call are not intercepted. ``F.multi_head_attention_forward``
    is such a call: it runs with the mode off and would reach the original
    ``scaled_dot_product_attention``. The mode therefore also points the names
    on ``torch.nn.functional`` at the substitutes while it is active, so inner
    calls find them by name.
    """

    SUBSTITUTES: dict[Callable, Callable] = {
        # On MPS, ``max_pool2d`` followed by a nonlinearity is not twice
        # differentiable unless called with ``return_indices=True``.
        max_pool2d: lambda *args, **kwargs: max_pool2d(
            *args, **{**kwargs, "return_indices": True}
        )[0],
        # ``hardsigmoid_backward`` has no derivative on any device.
        hardsigmoid: lambda x, inplace=False: _hardsigmoid(x, inplace),
        # ``native_layer_norm``'s third derivative is wrong on every device
        # (pytorch/pytorch#175754; the pending fix raises instead).
        layer_norm: _layer_norm,
        # ``embedding``'s backward zeroes the ``padding_idx`` row; the true
        # derivative of ``weight[input]`` does not depend on that flag.
        embedding: lambda input, weight, *args, **kwargs: weight[input],
        # The fused CPU/CUDA ``scaled_dot_product_attention`` kernels have no
        # double backward.
        scaled_dot_product_attention: _sdpa,
    }

    def __enter__(self) -> "DifferentiableSubstitutes":
        """Activate the mode and rebind the substituted ``functional`` attributes."""
        for func, substitute in self.SUBSTITUTES.items():
            setattr(functional, func.__name__, substitute)
        return super().__enter__()

    def __exit__(self, *exc: Any) -> None:
        """Deactivate the mode and restore the ``functional`` attributes."""
        super().__exit__(*exc)
        for func in self.SUBSTITUTES:
            setattr(functional, func.__name__, func)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Dispatch ``func`` to its substitute if it has one."""
        kwargs = kwargs or {}
        if func in self.SUBSTITUTES:
            return self.SUBSTITUTES[func](*args, **kwargs)
        return func(*args, **kwargs)


def _stateless(f: Callable) -> Callable[[str], Callable]:
    """Wrap a device-independent test function as a device-aware builder."""
    return lambda device: f


def class_index_loss(
    loss_fn: Callable, reduction: str, N: int, C: int, weighted: bool = False
) -> Callable[[str], Callable]:
    """Build ``loss_fn(input, target, weight, reduction)`` against class indices.

    Shared by the ``nll_loss`` and ``cross_entropy`` test cases. The integer
    ``target`` (and, when ``weighted``, the positive per-class ``weight``) are
    drawn under ``manual_seed(1)`` (distinct from the ``setup_case`` input) and
    live on ``device``; ``setup_case`` migrates only the float input, not
    closed-over constants. ``weight`` exercises the per-class weighting and (for
    ``mean``) the ``total_weight`` normalization.
    """

    def build(device: str) -> Callable:
        manual_seed(1)
        target = randint(0, C, (N,), device=device)
        weight = rand(C, **device_kw(device)) + 0.5 if weighted else None
        return lambda x: loss_fn(x, target, weight=weight, reduction=reduction)

    return build


def tolerances_for(device: str) -> dict[str, float]:
    """Relaxed ``assert_close`` tolerances for float32 devices."""
    return {"rtol": 5e-4, "atol": 1e-4} if dtype_for_device(device) == float32 else {}


def mlp(device: str) -> Sequential:
    """Build a two-layer tanh-activated MLP on ``device``.

    Sequential's Linear weights are concrete tensors at trace time, which FX
    requires. ``manual_seed(0)`` keeps the weights deterministic across calls.
    """
    manual_seed(0)
    kw = device_kw(device)
    return Sequential(
        Linear(5, 4, bias=False, **kw), Tanh(), Linear(4, 1, bias=True, **kw), Tanh()
    )


#: Scalar-output cases shared by the laplacian + bilaplacian consumer tests.
#: ``f`` is a builder ``device -> Callable``; ``args_fn`` is a zero-arg
#: closure returning the positional-args pytree (CPU tensors).
SCALAR_OUTPUT_CASES = [
    {"f": mlp, "args_fn": lambda: (rand(5),), "id": "two-layer-tanh-mlp"},
    {
        "f": _stateless(lambda x: sigmoid(sigmoid(x))),
        "args_fn": lambda: (rand(3),),
        "id": "sigmoid-sigmoid",
    },
]


def setup_case(
    config: dict[str, Any], device: str = "cpu"
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Instantiate the function and migrate its arguments to ``device``.

    Each case dict carries:

    - ``"f"``: a builder ``device -> Callable``.
    - ``"args_fn"``: a zero-arg closure returning the positional-args
      pytree (CPU tensors).
    - ``"tolerances"`` (optional): ``assert_close`` tolerances overriding the
      float64 defaults (see :func:`assert_jet_matches_oracle`).

    Returns:
        Tuple ``(f, args)`` where ``args`` is the migrated positional
        argument tuple, preserving the input pytree structure.
    """
    f = config["f"](device)
    # Seed AFTER ``f`` -- builders like ``mlp`` / ``_consts`` consume RNG to
    # construct their state; re-seeding here keeps ``args_fn`` deterministic.
    manual_seed(0)
    kw = device_kw(device)
    return f, tree_map(lambda t: t.to(**kw), config["args_fn"]())


def vjp_case(config: dict[str, Any]) -> dict[str, Any]:
    """Turn a case into the case of its vector-Jacobian product.

    The new function takes ``(*args, v)``, one cotangent per output, and returns
    the gradients w.r.t. the floating tensor inputs (integer tensors stay
    constants) via ``torch.func.vjp``: ``autograd.grad`` cannot run under the
    ``functionalize`` transform that :func:`jet.jet` traces with.
    """

    def build(device: str) -> Callable:
        f = config["f"](device)

        def vjp_f(*args_v: Any) -> tuple[Tensor, ...]:
            *args, v = args_v
            leaves, spec = tree_flatten(args)
            idx = [i for i, x in enumerate(leaves) if x.is_floating_point()]

            def f_floats(*floats: Tensor) -> Any:
                merged = dict(enumerate(leaves)) | dict(zip(idx, floats))
                return f(*tree_unflatten(list(merged.values()), spec))

            return vjp(f_floats, *(leaves[i] for i in idx))[1](v)

        return vjp_f

    def args_fn() -> tuple[Any, ...]:
        f, args = setup_case(config)
        manual_seed(1)
        return (*args, tree_map(rand_like, f(*args)))

    return {**config, "f": build, "args_fn": args_fn}


def make_jet_args(
    args: tuple[Any, ...], K: int, collapsed: bool = False, R: int = 2
) -> tuple[Any, ...]:
    """Build jet args from ``args`` by attaching Taylor coefficients per leaf.

    Each tensor leaf ``t`` in ``args`` becomes a jet tuple
    ``(t, c_1, ..., c_K)`` of tensors -- ``t`` itself is the primal. Shape
    contract by mode:

    - **standard** (``collapsed=False``): every ``c_k`` has shape ``t.shape``.
      ``R`` is ignored.
    - **collapsed** (``collapsed=True``): ``c_1..c_{K-1}`` have shape
      ``(R, *t.shape)`` (each carries the ``R`` directions); ``c_K`` has
      shape ``t.shape``.

    All coefficients are independently randomly drawn on the same device and
    dtype as the primal.
    """
    manual_seed(42)

    def make_leaf(t: Tensor) -> tuple[Tensor, ...]:
        if K == 0:
            return (t,)
        if not collapsed:
            return (t, *(rand_like(t) for _ in range(K)))
        batched = [
            rand(R, *t.shape, dtype=t.dtype, device=t.device) for _ in range(K - 1)
        ]
        return (t, *batched, rand_like(t))

    return tree_map(make_leaf, args)


def rev_collapsed_jet(f: Callable[..., Any], detach: bool = True) -> Callable[..., Any]:
    """Reference implementation for collapsed Taylor mode via :func:`jet._rev_jet`.

    ``detach`` is passed on to :func:`jet._rev_jet`.

    Built on :func:`jet._rev_jet` (nested reverse-mode AD), so independent of
    the FX-trace and interpreter machinery. See :func:`jet.jet` for the
    collapsed-mode shape contract.

    Run a standard ``_rev_jet`` per direction (full ``c_K`` into direction 0,
    zeros into the rest, exploiting that ``o_K`` is linear in ``c_K``), then
    combine: orders 1..K-1 stack across ``R``, order K sums. (``_rev_jet``
    uses :func:`torch.autograd.grad` which does not compose with
    :func:`torch.func.vmap`, hence the explicit Python loop over ``R``.)
    ``detach`` is passed on to :func:`jet._rev_jet`.
    """
    std_jet = _rev_jet(f, detach)

    def cjet_f(*args: Any) -> Any:
        leaves, _ = tree_flatten(args, is_leaf=_is_jet_leaf)
        K = len(leaves[0]) - 1
        R = leaves[0][1].shape[0]
        for leaf in leaves:
            if len(leaf) - 1 != K:
                raise ValueError(f"K mismatch across leaves: {K} vs {len(leaf) - 1}.")
            if leaf[1].shape[0] != R:
                raise ValueError(
                    f"R mismatch across leaves: {R} vs {leaf[1].shape[0]}."
                )

        def direction(leaf: tuple[Tensor, ...], r: int) -> tuple[Tensor, ...]:
            c_K = leaf[K] if r == 0 else zeros_like(leaf[K])
            return (leaf[0], *(leaf[k][r] for k in range(1, K)), c_K)

        per_dir = [
            std_jet(
                *tree_map(
                    lambda leaf, r=r: direction(leaf, r), args, is_leaf=_is_jet_leaf
                )
            )
            for r in range(R)
        ]

        def combine(*jets: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            return (
                jets[0][0],
                *(stack([j[k] for j in jets]) for k in range(1, K)),
                sum(j[K] for j in jets),
            )

        return tree_map(combine, *per_dir, is_leaf=_is_jet_leaf)

    return cjet_f


def assert_jet_matches_oracle(
    config: dict[str, Any], K: int, collapsed: bool, device: str = "cpu"
) -> None:
    """Assert ``jet(f, mock_args, collapsed)`` matches its mode-specific oracle.

    The oracle is :func:`jet._rev_jet` (standard) or :func:`rev_collapsed_jet`
    (collapsed). Both are built on nested reverse-mode AD and are independent
    of the FX-trace + interpreter machinery under test.

    Collapsed mode requires ``K >= 2``; for the invalid ``collapsed and K < 2``
    cells, assert the jet call raises the documented guard and return early.

    The oracle runs under :class:`DifferentiableSubstitutes`, so ``f`` may
    contain ops PyTorch cannot differentiate (correctly) to high order.
    """
    f, mock_args = setup_case(config, device)
    if collapsed and K < 2:
        with raises(ValueError, match=f"collapsed mode requires K >= 2, got K={K}"):
            jet(f, mock_args, collapsed=collapsed)(
                *make_jet_args(mock_args, K, collapsed=collapsed)
            )
        return
    jet_args = make_jet_args(mock_args, K, collapsed=collapsed)
    plain = f(*mock_args)
    oracle = rev_collapsed_jet(f) if collapsed else _rev_jet(f)
    actual = jet(f, mock_args, collapsed=collapsed)(*jet_args)
    # A case may relax the float64 tolerances (``"tolerances"``), e.g. a model
    # that computes parts in float32 whatever its dtype.
    tol = {**config.get("tolerances", {}), **tolerances_for(device)}
    with DifferentiableSubstitutes():
        # The substitutes must not change the function the oracle differentiates.
        assert_close(f(*mock_args), plain, **tol)
        expected = oracle(*jet_args)
    assert_close(actual, expected, **tol)
