"""Interpreter-based Taylor mode automatic differentiation (jets).

This module defines the `JetInterpreter`, which extends `torch.fx.Interpreter`
to execute a traced PyTorch computation graph while substituting ATen operations
with their Taylor mode (jet) equivalents on-the-fly. Unlike the graph-rewriting
approach of a `Transformer`, the interpreter propagates real values (or proxy
tensors under `make_fx`) and uses `isinstance` against the chosen jet type to
distinguish Taylor-expanded arguments from constants.

The same interpreter handles both standard and collapsed Taylor mode via the
``collapsed`` constructor flag, which selects the dispatch table
(``MAPPING`` vs ``COLLAPSED_MAPPING``) and the placeholder wrapper
(``JetTuple`` vs ``CollapsedJetTuple``).
"""

from typing import Any

from torch import Tensor, zeros_like
from torch.fx import GraphModule, Interpreter
from torch.fx.node import Argument, Target
from torch.utils._pytree import tree_flatten, tree_unflatten

from jet.collapsed_operations import COLLAPSED_MAPPING, CollapsedJetTuple
from jet.operations import MAPPING, JetTuple

_JetTypes = (JetTuple, CollapsedJetTuple)


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
        collapsed: If ``True``, propagate ``CollapsedJetTuple`` values and
            dispatch via ``COLLAPSED_MAPPING`` (collapsed Taylor mode, where
            the highest-order coefficient is already summed over directions).
            If ``False`` (default), propagate ``JetTuple`` values and dispatch
            via ``MAPPING`` (standard Taylor mode).
    """

    def __init__(self, module: GraphModule, collapsed: bool = False) -> None:
        """Initialize the JetInterpreter."""
        super().__init__(module)
        self.collapsed: bool = collapsed
        self.jet_type: type = CollapsedJetTuple if collapsed else JetTuple
        self.mapping: dict = COLLAPSED_MAPPING if collapsed else MAPPING
        self.label: str = "collapsed jet" if collapsed else "jet"

    def run(
        self,
        *args: Any,
        derivative_order: int,
        num_collapsed_directions: int | None,
        **kwargs: Any,
    ) -> Any:
        """Run the graph, then unwrap interpreter-internal jet types.

        ``derivative_order`` (``K``) and ``num_collapsed_directions`` (``R``)
        come from the validator that already inspected ``args``; passing them
        explicitly avoids re-deriving them from ``args[0]`` here. Used by
        :meth:`_normalize` to expand constant outputs to the right shapes.
        """
        result = super().run(*args, **kwargs)
        return self._normalize(result, derivative_order, num_collapsed_directions)

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Wrap each placeholder value in ``self.jet_type``."""
        value = super().placeholder(target, args, kwargs)
        return self.jet_type(value)

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Dispatch to ``self.mapping[target]`` when any arg is a jet tuple.

        Args:
            target: The function or callable to execute.
            args: Positional arguments of the node.
            kwargs: Keyword arguments of the node.

        Returns:
            The result of the jet operation (a jet tuple) or the original op.

        Raises:
            NotImplementedError: If a Taylor-dependent op has no jet rule, or
                if the node has non-empty ``kwargs``.
        """
        # TODO Only checks top-level args. Jets nested inside tuple/list/dict
        # args (e.g. for aten.stack, aten.cat) will be missed, causing a
        # TypeError instead of a clear NotImplementedError.
        has_jet_arg = any(isinstance(a, self.jet_type) for a in args)
        if has_jet_arg:
            if target not in self.mapping:
                raise NotImplementedError(
                    f"No {self.label} rule for {target}. "
                    "Please file an issue or add a rule."
                )
            if kwargs:
                raise NotImplementedError(
                    f"{self.label.capitalize()} dispatch does not support kwargs "
                    f"for {target} (got {kwargs})."
                )
            return self.mapping[target](*args)
        return super().call_function(target, args, kwargs)

    def _normalize(
        self,
        result: Any,
        derivative_order: int,
        num_collapsed_directions: int | None,
    ) -> Any:
        """Convert the pytree-of-jets into a pytree of plain tuples.

        Each ``self.jet_type`` leaf becomes a plain ``(f_0, ..., f_K)`` tuple.
        Constant tensor leaves (outputs that do not depend on the inputs) are
        expanded to zero-coefficient jets matching the mode's shape contract:
        standard returns ``K`` zeros of the primal's shape; collapsed returns
        ``K - 1`` zeros of shape ``(R, *S)`` plus one zero of ``S``.
        """
        flat, spec = tree_flatten(result, is_leaf=_is_jet_or_tensor)
        leaves = [
            tuple(node)
            if isinstance(node, _JetTypes)
            else (
                node,
                *self._zero_coeffs(node, derivative_order, num_collapsed_directions),
            )
            for node in flat
        ]
        return tree_unflatten(leaves, spec)

    def _zero_coeffs(
        self,
        primal: Tensor,
        derivative_order: int,
        num_collapsed_directions: int | None,
    ) -> list[Tensor]:
        """Build ``derivative_order`` zero-coefficients for a constant output leaf.

        Each returned tensor is a distinct allocation; sharing one
        ``zeros_like`` across coefficient slots would make in-place mutation
        of one slot mutate all the others.
        """
        if not self.collapsed:
            return [zeros_like(primal) for _ in range(derivative_order)]
        if num_collapsed_directions is None:
            raise ValueError(
                "Constant output in collapsed mode requires R; the caller "
                "should have derived it from a jet tuple input."
            )
        return [
            primal.new_zeros(num_collapsed_directions, *primal.shape)
            for _ in range(derivative_order - 1)
        ] + [zeros_like(primal)]


def _is_jet_or_tensor(x: Any) -> bool:
    """Return True for ``JetTuple``/``CollapsedJetTuple`` and plain tensors."""
    return isinstance(x, (*_JetTypes, Tensor))
