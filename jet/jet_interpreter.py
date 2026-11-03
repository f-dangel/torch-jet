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
from jet.primitives import (
    JetTuple,
    _has_jet_arg,
    _has_symbolic_zero,
    _is_batched,
    _zero_coeff,
)
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
        #: ``K`` and ``R`` of the current run (set by :meth:`run`).
        self._order: int = 0
        self._directions: int | None = None

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
        self._order, self._directions = derivative_order, collapsed_directions
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
        """Dispatch to ``RULES[target]`` when any arg is a jet tuple.

        Args:
            target: The function or callable to execute.
            args: Positional arguments of the node.
            kwargs: Keyword arguments of the node.

        A rule that has not opted in to symbolic zeros (see
        :class:`JetTuple`) receives its jet arguments with every ``None``
        coefficient materialized.

        Returns:
            The result of the jet operation (a jet tuple) or the original op.

        Raises:
            NotImplementedError: If a Taylor-dependent op has no jet rule.
        """
        if _has_jet_arg(args):
            rule = RULES.get(target)
            if rule is None:
                raise NotImplementedError(
                    f"No jet rule for {target}. Please file an issue."
                )
            if not getattr(rule, "symbolic_zeros", False) and _has_symbolic_zero(
                (args, kwargs)
            ):
                args, kwargs = self._materialize((args, kwargs))
            result = rule(*args, **kwargs)
            self._check_collapsed(result, target)
            return result
        return super().call_function(target, args, kwargs)

    def _materialize(self, tree: PyTree[Any]) -> PyTree[Any]:
        """Replace the ``None`` coefficients of every jet in ``tree`` by zeros.

        Each zero is a stride-0 view of one element in the slot's shape:
        ``(R, *S)`` for a batched collapsed slot, ``S`` otherwise.
        """

        def dense(jet: JetTuple) -> JetTuple:
            if not any(c is None for c in jet):
                return jet
            K, shape = len(jet) - 1, jet[0].shape
            return JetTuple(
                (
                    _zero_coeff(
                        jet[0],
                        (self._directions, *shape)
                        if _is_batched(k, K, self.collapsed)
                        else shape,
                    )
                    if c is None
                    else c
                    for k, c in enumerate(jet)
                ),
                collapsed=jet.collapsed,
            )

        return tree_map(
            lambda x: dense(x) if isinstance(x, JetTuple) else x,
            tree,
            is_leaf=lambda x: isinstance(x, JetTuple),
        )

    def _check_collapsed(self, result: Any, target: Target) -> None:
        """Assert every ``JetTuple`` a rule returns matches the run's mode.

        A single, central guard: if a rule builds its output with the wrong
        ``collapsed`` flag, the mismatch is caught here -- at the dispatch site,
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

        Each ``JetTuple`` leaf becomes a plain ``(f_0, ..., f_K)`` tuple; its
        symbolic-zero (``None``) coefficients are materialized as dense zeros.
        Constant tensor leaves (outputs that do not depend on the inputs) are
        expanded to zero-coefficient jets. Both follow the mode's shape
        contract: standard returns ``K`` zeros of the primal's shape; collapsed
        returns ``K - 1`` zeros of shape ``(R, *S)`` plus one zero of ``S``.
        """

        def _normalize_leaf(node: Any) -> Jet:
            if isinstance(node, JetTuple):
                return (
                    node[0],
                    *(
                        self._dense_zero(
                            node[0], k, derivative_order, collapsed_directions
                        )
                        if c is None
                        else c
                        for k, c in enumerate(node[1:], start=1)
                    ),
                )
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
        return [
            self._dense_zero(primal, k, derivative_order, collapsed_directions)
            for k in range(1, derivative_order + 1)
        ]

    def _dense_zero(
        self,
        primal: Tensor,
        k: int,
        derivative_order: int,
        collapsed_directions: int | None,
    ) -> Tensor:
        """A dense zero for the order-``k`` coefficient of a leaf with ``primal``.

        Used for a constant leaf and for a symbolic-zero coefficient of a jet
        leaf: shape ``(R, *S)`` for a collapsed ``c_1..c_{K-1}``, ``S``
        otherwise.

        Raises:
            ValueError: If ``self.collapsed`` and ``collapsed_directions`` is
                ``None`` (see :meth:`_zero_coeffs`).
        """
        if not self.collapsed or k == derivative_order:
            return zeros_like(primal)
        if collapsed_directions is None:
            raise ValueError(
                "Constant output in collapsed mode requires R; the caller "
                "should have derived it from a jet tuple input."
            )
        return primal.new_zeros(collapsed_directions, *primal.shape)
