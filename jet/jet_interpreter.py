"""Interpreter-based Taylor mode automatic differentiation (jets).

This module defines the `JetInterpreter`, which extends `torch.fx.Interpreter`
to execute a traced PyTorch computation graph while substituting ATen operations
with their Taylor mode (jet) equivalents on-the-fly. Unlike the graph-rewriting
approach of a `Transformer`, the interpreter propagates real values (or proxy
tensors under `make_fx`) and uses `isinstance` against the chosen jet type to
distinguish Taylor-expanded arguments from constants.

The same interpreter handles both standard and collapsed Taylor mode via the
``collapsed`` constructor flag, which wraps placeholders as
``JetTuple(..., collapsed=...)`` and, per op, selects the standard or collapsed
rule from the shared ``RULES`` registry (:mod:`jet._rules`).
"""

from typing import Any

from torch import Tensor, zeros_like
from torch.fx import GraphModule, Interpreter
from torch.fx.node import Argument, Target
from torch.utils._pytree import tree_leaves, tree_map

from jet._rules import RULES
from jet.operations import JetTuple
from jet.utils import Jet, PyTree


class JetInterpreter(Interpreter):
    """Interpreter that swaps in jet operations during execution.

    For each ``call_function`` node, the interpreter checks whether any
    positional argument is a jet tuple (a Taylor-expanded value). If so, it
    dispatches to the corresponding jet operation from the configured mapping;
    otherwise it falls through to the original ATen operation. The Taylor-
    expansion order ``K`` is inferred inside each jet op from its arguments (a
    jet tuple is ``(primal, c_1, ..., c_K)``).

    Args:
        module: The traced computation graph module to interpret.
        collapsed: If ``True``, propagate collapsed ``JetTuple`` values and
            dispatch to each op's collapsed rule (collapsed Taylor mode, where
            the highest-order coefficient is already summed over directions).
            If ``False`` (default), propagate standard ``JetTuple`` values and
            dispatch to each op's standard rule.
    """

    def __init__(self, module: GraphModule, collapsed: bool = False) -> None:
        """Initialize the JetInterpreter."""
        super().__init__(module)
        self.collapsed: bool = collapsed

    def run(
        self,
        derivative_order: int,
        collapsed_directions: int | None,
        *args: Tensor,
        **kwargs: Any,
    ) -> PyTree[Jet]:
        """Run the graph, then unwrap interpreter-internal jet types.

        ``derivative_order`` (``K``) and ``collapsed_directions`` (``R``)
        come from the validator that already inspected ``args``; passing them
        explicitly avoids re-deriving them from ``args[0]`` here. Used by
        :meth:`_normalize` to expand constant outputs to the right shapes.
        """
        result = super().run(*args, **kwargs)
        return self._normalize(result, derivative_order, collapsed_directions)

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Wrap each placeholder value in a ``JetTuple``."""
        value = super().placeholder(target, args, kwargs)
        return JetTuple(value, collapsed=self.collapsed)

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Dispatch to ``RULES[target][self.collapsed]`` when any arg is a jet tuple.

        Args:
            target: The function or callable to execute.
            args: Positional arguments of the node.
            kwargs: Keyword arguments of the node.

        Returns:
            The result of the jet operation (a jet tuple) or the original op.

        Raises:
            NotImplementedError: If a Taylor-dependent op has no jet rule.
        """

        # Jets may appear at the top level or nested one level inside a ``list``
        # arg (e.g. ``aten.cat([j1, j2], dim)``). Only ``list`` args are scanned,
        # not ``tuple``: multi-output ops (e.g. ``max_pool2d_with_indices``)
        # return a ``tuple`` whose element 0 is a jet, and the following
        # ``getitem`` must fall through to the default op, not dispatch here.
        def _jet_in(a: Argument) -> bool:
            if isinstance(a, JetTuple):
                return True
            if isinstance(a, list):
                return any(isinstance(e, JetTuple) for e in a)
            return False

        has_jet_arg = any(_jet_in(a) for a in args)
        if has_jet_arg:
            rule = RULES.get(target)
            if rule is None:
                raise NotImplementedError(
                    f"No jet rule for {target}. Please file an issue."
                )
            fn = rule.get(self.collapsed)
            if fn is None:
                raise NotImplementedError(
                    f"{target} has a standard jet rule but no collapsed one. "
                    "Call with collapsed=False."
                )
            result = fn(*args, **kwargs)
            self._check_collapsed(result, target)
            return result
        return super().call_function(target, args, kwargs)

    def _check_collapsed(self, result: Any, target: Target) -> None:
        """Assert every ``JetTuple`` a rule returns matches the run's mode.

        A single, central guard: if a rule builds its output with the wrong
        constructor (standard ``JetTuple`` vs. collapsed ``_cjet``), the
        mismatched ``.collapsed`` flag is caught here -- at the dispatch site,
        naming the op -- instead of surfacing later as an opaque shape error.
        """
        for leaf in tree_leaves(result, is_leaf=lambda x: isinstance(x, JetTuple)):
            if isinstance(leaf, JetTuple) and leaf.collapsed != self.collapsed:
                raise RuntimeError(
                    f"the jet rule for {target} returned a JetTuple with "
                    f"collapsed={leaf.collapsed}, but the interpreter is running "
                    f"in collapsed={self.collapsed} mode"
                )

    def _normalize(
        self,
        result: PyTree[JetTuple | Tensor],
        derivative_order: int,
        collapsed_directions: int | None,
    ) -> PyTree[Jet]:
        """Convert the pytree-of-jets into a pytree of plain tuples.

        Each ``JetTuple`` leaf becomes a plain ``(f_0, ..., f_K)`` tuple.
        Constant tensor leaves (outputs that do not depend on the inputs) are
        expanded to zero-coefficient jets matching the mode's shape contract:
        standard returns ``K`` zeros of the primal's shape; collapsed returns
        ``K - 1`` zeros of shape ``(R, *S)`` plus one zero of ``S``.
        """

        def _normalize_leaf(node: Any) -> Jet:
            if isinstance(node, JetTuple):
                return tuple(node)
            return (
                node,
                *self._zero_coeffs(node, derivative_order, collapsed_directions),
            )

        return tree_map(
            _normalize_leaf,
            result,
            is_leaf=lambda x: isinstance(x, (JetTuple, Tensor)),
        )

    def _zero_coeffs(
        self,
        primal: Tensor,
        derivative_order: int,
        collapsed_directions: int | None,
    ) -> list[Tensor]:
        """Build ``derivative_order`` zero-coefficients for a constant output leaf.

        Each returned tensor is a distinct allocation; sharing one
        ``zeros_like`` across coefficient slots would make in-place mutation
        of one slot mutate all the others.

        Raises:
            ValueError: If ``self.collapsed`` and ``collapsed_directions`` is
                ``None``. The validator guarantees ``R`` is set for any
                collapsed call (``K >= 2`` forces at least one batched
                coefficient), so this should be unreachable from the public
                API; an explicit raise (rather than ``assert``) keeps the
                contract visible under ``python -O``.
        """
        if not self.collapsed:
            return [zeros_like(primal) for _ in range(derivative_order)]
        if collapsed_directions is None:
            raise ValueError(
                "Constant output in collapsed mode requires R; the caller "
                "should have derived it from a jet tuple input."
            )
        return [
            primal.new_zeros(collapsed_directions, *primal.shape)
            for _ in range(derivative_order - 1)
        ] + [zeros_like(primal)]
