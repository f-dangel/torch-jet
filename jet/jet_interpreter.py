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

from torch import Tensor
from torch.fx import GraphModule, Interpreter
from torch.fx.node import Argument, Target
from torch.utils._pytree import tree_leaves, tree_map

from jet._rules import RULES
from jet.primitives import JetTuple, _has_jet_arg, _slot_shape
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
        K: int,
        R: int | None,
        *args: Tensor,
        **kwargs: Any,
    ) -> PyTree[Jet]:
        """Run the graph, then unwrap interpreter-internal jet types.

        ``K`` and ``R``
        come from the validator that already inspected ``args``; passing them
        explicitly avoids re-deriving them from ``args[0]`` here. Used by
        :meth:`_normalize` to expand constant outputs to the right shapes.
        """
        result = super().run(*args, **kwargs)
        return self._normalize(result, K, R)

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
            if not getattr(rule, "symbolic_zeros", False):
                args = self._materialize(args)
            result = rule(*args, **kwargs)
            self._check_collapsed(result, target)
            return result
        return super().call_function(target, args, kwargs)

    def _materialize(self, args: tuple[Argument, ...]) -> tuple[Argument, ...]:
        """Replace the ``None`` coefficients of every jet in ``args`` by zeros.

        Scans where :func:`_has_jet_arg` does (top level and one level inside a
        ``list``); a jet without ``None`` is returned as is, so this is a no-op
        on dense jets (see :meth:`JetTuple.materialized`).
        """

        def dense(a: Argument) -> Argument:
            return a.materialized() if isinstance(a, JetTuple) else a

        return tuple(
            [dense(e) for e in a] if isinstance(a, list) else dense(a) for a in args
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
        K: int,
        R: int | None,
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
                        self._dense_zero(node[0], k, K, R) if c is None else c
                        for k, c in enumerate(node[1:], start=1)
                    ),
                )
            return (
                node,
                *self._zero_coeffs(node, K, R),
            )

        return tree_map(
            _normalize_leaf,
            result,
            is_leaf=lambda x: isinstance(x, (JetTuple, Tensor)),
        )

    def _zero_coeffs(
        self,
        primal: Tensor,
        K: int,
        R: int | None,
    ) -> list[Tensor]:
        """Build ``K`` zero coefficients for a constant output leaf.

        Each returned tensor is a distinct allocation; sharing one
        ``zeros_like`` across coefficient slots would make in-place mutation
        of one slot mutate all the others.

        Raises:
            ValueError: If ``self.collapsed`` and ``R`` is
                ``None`` (see :func:`_slot_shape`).
        """
        return [self._dense_zero(primal, k, K, R) for k in range(1, K + 1)]

    def _dense_zero(
        self,
        primal: Tensor,
        k: int,
        K: int,
        R: int | None,
    ) -> Tensor:
        """A dense zero for the order-``k`` coefficient of a leaf with ``primal``.

        Used for a constant leaf and for a symbolic-zero coefficient of a jet
        leaf (see :func:`_slot_shape`).
        """
        return primal.new_zeros(_slot_shape(primal, k, K, self.collapsed, R))
