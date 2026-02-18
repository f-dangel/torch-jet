"""Implements individual simplification rules."""

from abc import ABC, abstractmethod
from math import prod
from typing import Any, Callable

import torch
from torch import Tensor
from torch.fx import Graph, Node

# ATen op references used by rules
_aten = torch.ops.aten

# Op targets for pattern matching
_sum_target = _aten.sum.dim_IntList


def is_sum(arg: Any) -> bool:
    """Check if an argument is a ``sum`` node.

    Args:
        arg: An entry from a ``Node.arg`` tuple.

    Returns:
        Whether the argument is a sum node.
    """
    return (
        isinstance(arg, Node)
        and arg.op == "call_function"
        and arg.target == _sum_target
    )


def _get_sum_pos(node: Node) -> int:
    """Extract the summed dimension from an ``aten.sum.dim_IntList`` node.

    With make_fx, ``x.sum(pos)`` traces to ``aten.sum.dim_IntList(x, [pos])``,
    so the dimension is always at ``node.args[1][0]``.

    Args:
        node: A sum node.

    Returns:
        The position along which the sum is performed.
    """
    return node.args[1][0]


def _make_sum_args(x: Node, pos: int) -> tuple[tuple, dict]:
    """Create args and kwargs for a new ``aten.sum.dim_IntList`` node.

    Args:
        x: The input tensor node.
        pos: The position to sum along.

    Returns:
        Tuple of (args, kwargs) for graph.call_function.
    """
    return (x, [pos]), {}


class Rule(ABC):
    """Base class for PullSum simplification rules.

    All rules match ``sum(op(...), pos)`` where ``op`` is in ``OPERATIONS``,
    then rewrite the graph to pull the sum through the operation.

    Subclasses must define ``OPERATIONS`` and implement ``_rewrite``.
    They may override ``_extra_match`` for additional matching criteria.

    Attributes:
        OPERATIONS: ATen op targets that this rule can pull a sum through.
    """

    OPERATIONS: list[Callable]

    def match(self, node: Node) -> bool:
        """Detect ``sum(op(...), pos)`` where ``op`` is in ``OPERATIONS``.

        Args:
            node: A node in a computation graph.

        Returns:
            True if the node matches the pattern, False otherwise.
        """
        if not is_sum(node) or not node.users:
            return False

        (in_node,) = node.all_input_nodes
        if in_node.op != "call_function" or in_node.target not in self.OPERATIONS:
            return False

        return self._extra_match(node, in_node)

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Additional matching criteria beyond the base pattern.

        Override in subclasses to add rule-specific checks.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The operation node consumed by the ``sum``.

        Returns:
            True if the extra criteria are met. Default: True.
        """
        return True

    def apply(self, sum_node: Node, graph: Graph) -> None:
        """Pull the ``sum`` through the inner operation.

        Args:
            sum_node: A ``sum`` node matching this rule's pattern.
            graph: The computation graph to which the rule is applied.
        """
        (inner_node,) = sum_node.all_input_nodes
        pos = _get_sum_pos(sum_node)
        result = self._rewrite(sum_node, inner_node, pos, graph)
        if result is not None:
            sum_node.replace_all_uses_with(result)

    @abstractmethod
    def _rewrite(
        self, sum_node: Node, inner_node: Node, pos: int, graph: Graph
    ) -> Node:
        """Create the rewritten subgraph with the sum pulled through.

        Args:
            sum_node: The ``sum`` node being simplified.
            inner_node: The operation node consumed by the ``sum``.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The replacement node for ``sum_node``, or ``None`` if
            ``replace_all_uses_with`` was already called internally.
        """
        ...


class PullSumScalarMultiplication(Rule):
    """Rule for simplifying ``sum(x * y)`` with one scalar argument.

    The following two cases simplify to the same result:

    1. ``x`` scalar: ``sum(x * y)`` -> ``x * sum(y)``.
    2. ``y`` scalar: ``sum(x * y)`` -> ``sum(x) * y``.

    Attributes:
        OPERATIONS: List of operations that can be simplified.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        _aten.mul.Tensor,
    ]

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Check that the multiplication has exactly one scalar argument.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The multiplication node.

        Returns:
            True if the multiplication has 2 args with exactly 1 scalar.
        """
        return (
            len(inner_node.args) == 2
            and inner_node.kwargs == {}
            and sum(isinstance(a, (float, int)) for a in inner_node.args) == 1
        )

    def _rewrite(self, sum_node: Node, mul_node: Node, pos: int, graph: Graph) -> Node:
        """Swap sum and scalar multiplication: ``sum(x * s)`` → ``sum(x) * s``.

        Args:
            sum_node: The ``sum`` node.
            mul_node: The multiplication node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The new multiplication node.
        """
        (x,) = mul_node.all_input_nodes
        x_pos = mul_node.args.index(x)

        with graph.inserting_after(sum_node):
            sum_args, sum_kwargs = _make_sum_args(x, pos)
            new_sum_node = graph.call_function(
                _sum_target, args=sum_args, kwargs=sum_kwargs
            )
        with graph.inserting_after(new_sum_node):
            new_args = tuple(
                new_sum_node if idx == x_pos else arg
                for idx, arg in enumerate(mul_node.args)
            )
            new_mul_node = graph.call_function(mul_node.target, args=new_args)

        return new_mul_node


class PullSumBroadcastedMultiplication(Rule):
    """Rule for simplifying ``sum(x * y, d)`` when one factor is broadcasted.

    When one multiplicand doesn't vary along the summed dimension (its shape
    has fewer leading dimensions, so the summed dim falls outside its range),
    the sum can be pulled through to the other factor:

    ``sum(x * y, d)`` → ``sum(x, d) * y``  (when ``y`` is invariant along ``d``).

    This arises naturally from curried + vmapped jet computations where a
    matrix result ``(K, M)`` is element-wise multiplied by a vector ``(M,)``
    and then summed over the leading vmap dimension.

    Attributes:
        OPERATIONS: List of operations that can be simplified.
    """

    OPERATIONS: list[Callable[[Tensor, Tensor], Tensor]] = [_aten.mul.Tensor]

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Check that both args are tensors and one is broadcasted along the sum dim.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The multiplication node.

        Returns:
            True if the multiplication has 2 tensor args and one is invariant
            along the summed dimension.
        """
        if len(inner_node.args) != 2 or inner_node.kwargs != {}:
            return False
        if not all(isinstance(a, Node) for a in inner_node.args):
            return False

        pos = _get_sum_pos(sum_node)
        out_ndim = len(inner_node.meta["tensor_meta"].shape)

        for arg in inner_node.args:
            arg_ndim = len(arg.meta["tensor_meta"].shape)
            if pos < out_ndim - arg_ndim:
                return True

        return False

    def _rewrite(self, sum_node: Node, mul_node: Node, pos: int, graph: Graph) -> Node:
        """Pull sum through to the non-broadcasted factor.

        ``sum(x * y, d)`` → ``sum(x, d) * y`` when ``y`` is invariant along ``d``.

        Args:
            sum_node: The ``sum`` node.
            mul_node: The multiplication node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The new multiplication node.
        """
        out_ndim = len(mul_node.meta["tensor_meta"].shape)
        arg0, arg1 = mul_node.args

        _ = len(arg0.meta["tensor_meta"].shape)
        arg1_ndim = len(arg1.meta["tensor_meta"].shape)

        if pos < out_ndim - arg1_ndim:
            varying_idx, invariant_idx = 0, 1
        else:
            varying_idx, invariant_idx = 1, 0

        varying = mul_node.args[varying_idx]
        invariant = mul_node.args[invariant_idx]

        with graph.inserting_after(sum_node):
            sum_args, sum_kwargs = _make_sum_args(varying, pos)
            new_sum = graph.call_function(_sum_target, args=sum_args, kwargs=sum_kwargs)
        with graph.inserting_after(new_sum):
            new_args = [None, None]
            new_args[varying_idx] = new_sum
            new_args[invariant_idx] = invariant
            new_mul = graph.call_function(mul_node.target, args=tuple(new_args))

        return new_mul


class PullSumTensorAddition(Rule):
    """Rule for simplifying ``sum(x + y)`` where x and y are tensors.

    The simplified result is ``sum(x) + sum(y)``.
    Same for subtraction.

    Warning:
        This rule assumes no broadcasting, i.e. ``x`` and ``y`` must have the same shape.

    Attributes:
        OPERATIONS: List of operations that can be simplified.
            Includes addition and subtraction.
    """

    OPERATIONS: list[Callable[[Tensor | float | int, Tensor | float | int], Tensor]] = [
        _aten.add.Tensor,
        _aten.sub.Tensor,
    ]

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Check that both arguments are tensors with the same shape (no broadcasting).

        Args:
            sum_node: The ``sum`` node.
            inner_node: The addition/subtraction node.

        Returns:
            True if the operation has 2 tensor arguments with matching shapes.
        """
        if (
            len(inner_node.args) != 2
            or inner_node.kwargs != {}
            or not all(isinstance(a, Node) for a in inner_node.args)
        ):
            return False
        a, b = inner_node.args
        return a.meta["tensor_meta"].shape == b.meta["tensor_meta"].shape

    def _rewrite(self, sum_node: Node, add_node: Node, pos: int, graph: Graph) -> Node:
        """Swap sum and addition: ``sum(x + y)`` → ``sum(x) + sum(y)``.

        Args:
            sum_node: The ``sum`` node.
            add_node: The addition/subtraction node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The new addition/subtraction node.
        """
        mapping = {}
        for x in add_node.all_input_nodes:
            with graph.inserting_after(x):
                sum_args, sum_kwargs = _make_sum_args(x, pos)
                new_sum_node = graph.call_function(
                    _sum_target, args=sum_args, kwargs=sum_kwargs
                )
            mapping[x] = new_sum_node

        with graph.inserting_after(add_node):
            new_args = tuple(mapping[x] for x in add_node.args)
            new_add_node = graph.call_function(add_node.target, args=new_args)

        return new_add_node


class PullSumMM(Rule):
    """Pull ``sum`` through ``mm``.

    Matches ``sum(mm(X, W), pos)`` where ``pos`` is a non-contracted dimension
    of the output.  For ``mm(X, W)`` with ``X = (M, K)`` and ``W = (K, N)``,
    the contracted dimension ``K`` (dim 1 of ``X``, dim 0 of ``W``) never
    appears in the ``(M, N)`` output.

    - ``pos == 0``: → ``squeeze(mm(unsqueeze(sum(X, 0), 0), W), 0)``
    - ``pos == 1``: → ``squeeze(mm(X, unsqueeze(sum(W, 1), 1)), 1)``

    Attributes:
        OPERATIONS: Supported ATen operations.
    """

    OPERATIONS = [_aten.mm.default]

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Check that the sum dimension is non-contracted.

        For ``mm(X, W)`` the output is always 2D, so ``pos`` is 0 or 1.
        The contracted dimension ``K`` (dim 1 of ``X``, dim 0 of ``W``) does
        not appear in the output, so neither position is contracted:

        - ``pos == 0`` targets ``X`` (arg 0); its contracted dim is 1 → valid.
        - ``pos == 1`` targets ``W`` (arg 1); its contracted dim is 0 → valid.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The ``mm`` node.

        Returns:
            True if ``pos`` is a non-contracted output dimension.
        """
        pos = _get_sum_pos(sum_node)
        contracted_dim = 1 - pos
        return pos != contracted_dim

    def _rewrite(self, sum_node: Node, mm_node: Node, pos: int, graph: Graph) -> Node:
        """Pull sum through mm.

        Replaces ``sum(mm(X, W), pos)`` with
        ``squeeze(mm(…, unsqueeze(sum(arg, pos), pos), …), pos)``.

        Args:
            sum_node: The ``sum`` node.
            mm_node: The ``mm`` node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The replacement node.
        """
        # pos=0 → pull sum into mat1 (arg 0), pos=1 → into mat2 (arg 1)
        target_idx = 0 if pos == 0 else 1
        target_arg = mm_node.args[target_idx]

        with graph.inserting_before(mm_node):
            sv_args, sv_kwargs = _make_sum_args(target_arg, pos)
            new_sv = graph.call_function(_sum_target, args=sv_args, kwargs=sv_kwargs)

        with graph.inserting_after(new_sv):
            unsqueeze_node = graph.call_function(
                _aten.unsqueeze.default, args=(new_sv, pos)
            )

        with graph.inserting_after(unsqueeze_node):
            mm_args = tuple(
                unsqueeze_node if i == target_idx else mm_node.args[i] for i in (0, 1)
            )
            new_mm = graph.call_function(_aten.mm.default, args=mm_args)

        with graph.inserting_after(new_mm):
            squeeze_node = graph.call_function(_aten.squeeze.dim, args=(new_mm, pos))

        return squeeze_node


class PullSumAddMM(Rule):
    """Pull ``sum`` through ``addmm``.

    Matches ``sum(addmm(b, X, W), pos)`` where ``pos`` is a non-contracted
    dimension.  The matrix part is rewritten using ``mm`` (not ``addmm``),
    and the bias ``b`` is corrected separately:

    - ``pos == 0``: each of ``M`` rows had ``b`` added; summing gives ``M * b``.
    - ``pos == 1``: each row had ``b`` element-wise; summing cols gives ``sum(b)``.

    Attributes:
        OPERATIONS: Supported ATen operations.
    """

    OPERATIONS = [_aten.addmm.default]

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Check that the sum dimension is non-contracted.

        For ``addmm(b, X, W)`` the output is always 2D, so ``pos`` is 0 or 1.
        The contracted dimension ``K`` (dim 1 of ``X``, dim 0 of ``W``) does
        not appear in the output, so neither position is contracted:

        - ``pos == 0`` targets ``X`` (arg 1); its contracted dim is 1 → valid.
        - ``pos == 1`` targets ``W`` (arg 2); its contracted dim is 0 → valid.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The ``addmm`` node.

        Returns:
            True if ``pos`` is a non-contracted output dimension.
        """
        pos = _get_sum_pos(sum_node)
        contracted_dim = 1 - pos
        return pos != contracted_dim

    def _rewrite(
        self, sum_node: Node, addmm_node: Node, pos: int, graph: Graph
    ) -> Node:
        """Pull sum through addmm.

        The matrix part is rewritten using ``mm``, and the bias ``b`` is
        corrected: ``V * b`` when ``pos == 0``, or ``sum(b)`` when ``pos == 1``.

        Args:
            sum_node: The ``sum`` node.
            addmm_node: The ``addmm`` node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The replacement node.
        """
        # addmm args: (bias, mat1, mat2) → matrix args at indices 1, 2
        target_idx = 1 if pos == 0 else 2
        target_arg = addmm_node.args[target_idx]

        with graph.inserting_before(addmm_node):
            sv_args, sv_kwargs = _make_sum_args(target_arg, pos)
            new_sv = graph.call_function(_sum_target, args=sv_args, kwargs=sv_kwargs)

        with graph.inserting_after(new_sv):
            unsqueeze_node = graph.call_function(
                _aten.unsqueeze.default, args=(new_sv, pos)
            )

        with graph.inserting_after(unsqueeze_node):
            mm_args = tuple(
                unsqueeze_node if i == target_idx else addmm_node.args[i]
                for i in (1, 2)
            )
            new_mm = graph.call_function(_aten.mm.default, args=mm_args)

        with graph.inserting_after(new_mm):
            squeeze_node = graph.call_function(_aten.squeeze.dim, args=(new_mm, pos))

        bias = addmm_node.args[0]
        if pos == 0:
            # Bias b was broadcast from (n,) to (V, n); summing V rows
            # gives V * b.
            V = target_arg.meta["tensor_meta"].shape[0]
            with graph.inserting_after(squeeze_node):
                scaled_b = graph.call_function(_aten.mul.Tensor, args=(bias, V))
            with graph.inserting_after(scaled_b):
                result = graph.call_function(
                    _aten.add.Tensor, args=(squeeze_node, scaled_b)
                )
        else:
            # pos == 1: each row had b added; summing over n columns
            # contributes sum(b) to every row.
            with graph.inserting_after(squeeze_node):
                sum_b = graph.call_function(_aten.sum.default, args=(bias,))
            with graph.inserting_after(sum_b):
                result = graph.call_function(
                    _aten.add.Tensor, args=(squeeze_node, sum_b)
                )
        return result


class PullSumSqueeze(Rule):
    """Pull ``sum`` through ``squeeze``.

    Rewrites ``sum(squeeze(x, sq_d), [s_d])`` into
    ``squeeze(sum(x, [adj_s_d]), adj_sq_d)``, adjusting dimensions so the
    result is numerically identical.

    Handles both ``squeeze.dim`` (scalar) and ``squeeze.dims`` (list with a
    single element).

    Attributes:
        OPERATIONS: Supported squeeze operations.
    """

    OPERATIONS = [_aten.squeeze.dim, _aten.squeeze.dims]

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Only handle single-dim squeezes.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The ``squeeze`` node.

        Returns:
            True if the squeeze operates on a single dimension.
        """
        if inner_node.target == _aten.squeeze.dims:
            dims = inner_node.args[1]
            return isinstance(dims, (list, tuple)) and len(dims) == 1
        return True

    def _rewrite(
        self, sum_node: Node, squeeze_node: Node, pos: int, graph: Graph
    ) -> Node:
        """Swap sum and squeeze, adjusting dimensions.

        Args:
            sum_node: The ``sum`` node.
            squeeze_node: The ``squeeze`` node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The new squeeze node.
        """
        squeeze_input = squeeze_node.args[0]

        # Normalise squeeze dim to a scalar
        if squeeze_node.target == _aten.squeeze.dims:
            sq_d = squeeze_node.args[1][0]
        else:
            sq_d = squeeze_node.args[1]

        # Compute adjusted dims in the original (unsqueezed) tensor
        new_s_d = pos + 1 if sq_d <= pos else pos
        # After sum removes new_s_d, adjust squeeze dim
        new_sq_d = sq_d - 1 if new_s_d < sq_d else sq_d

        # Build: squeeze(sum(x, [new_s_d]), new_sq_d)
        with graph.inserting_before(sum_node):
            sum_args, sum_kwargs = _make_sum_args(squeeze_input, new_s_d)
            new_sum = graph.call_function(_sum_target, args=sum_args, kwargs=sum_kwargs)
        with graph.inserting_after(new_sum):
            new_squeeze = graph.call_function(
                _aten.squeeze.dim, args=(new_sum, new_sq_d)
            )

        return new_squeeze


class PullSumUnsqueeze(Rule):
    """Pull ``sum`` through ``unsqueeze``.

    Two cases:

    1. ``sum_dim == unsqueeze_dim``: the sum collapses the freshly inserted
       size-1 dimension → the unsqueeze/sum pair is a no-op on ``x``.
    2. Otherwise: rewrite ``sum(unsqueeze(x, uq_d), [s_d])`` into
       ``unsqueeze(sum(x, [adj_s_d]), adj_uq_d)``.

    Attributes:
        OPERATIONS: Supported unsqueeze operations.
    """

    OPERATIONS = [_aten.unsqueeze.default]

    def _rewrite(
        self, sum_node: Node, unsqueeze_node: Node, pos: int, graph: Graph
    ) -> Node | None:
        """Swap sum and unsqueeze, adjusting dimensions.

        Args:
            sum_node: The ``sum`` node.
            unsqueeze_node: The ``unsqueeze`` node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The replacement node, or ``None`` if the pair is a no-op
            (handled via direct ``replace_all_uses_with``).
        """
        unsqueeze_input = unsqueeze_node.args[0]
        uq_d = unsqueeze_node.args[1]

        if pos == uq_d:
            # Summing over the freshly inserted size-1 dim is a no-op
            sum_node.replace_all_uses_with(unsqueeze_input)
            return None

        # Compute adjusted dims in the original (unsqueezed) tensor
        new_s_d = pos - 1 if uq_d <= pos else pos
        # After sum removes new_s_d, adjust unsqueeze dim
        new_uq_d = uq_d - 1 if new_s_d < uq_d else uq_d

        # Build: unsqueeze(sum(x, [new_s_d]), new_uq_d)
        with graph.inserting_before(sum_node):
            sum_args, sum_kwargs = _make_sum_args(unsqueeze_input, new_s_d)
            new_sum = graph.call_function(_sum_target, args=sum_args, kwargs=sum_kwargs)
        with graph.inserting_after(new_sum):
            new_unsqueeze = graph.call_function(
                _aten.unsqueeze.default, args=(new_sum, new_uq_d)
            )

        return new_unsqueeze


class PullSumView(Rule):
    """Pull ``sum`` through ``view`` and ``_unsafe_view``.

    Rewrites ``sum(view(x, shape), [s_d])`` into
    ``view(sum(x, [d']), shape')``, where ``d'`` is the original dimension
    that ``s_d`` maps to and ``shape'`` is ``shape`` with element ``s_d``
    removed.

    This only fires when dimension ``s_d`` in the viewed shape maps cleanly
    to a single dimension ``d'`` in the original shape (no splits or merges
    across the summed dimension boundary).

    Attributes:
        OPERATIONS: Supported view operations.
    """

    OPERATIONS = [_aten.view.default, _aten._unsafe_view.default]

    @staticmethod
    def _find_orig_dim(
        orig_shape: tuple[int, ...], new_shape: list[int], s_d: int
    ) -> int | None:
        """Map ``s_d`` in ``new_shape`` back to a dimension in ``orig_shape``.

        Returns the index ``d'`` if the dimension boundaries align, or
        ``None`` if the sum dimension spans a split/merge.

        Args:
            orig_shape: Shape of the tensor before the view.
            new_shape: Shape after the view.
            s_d: The dimension being summed in ``new_shape``.

        Returns:
            The corresponding dimension index in ``orig_shape``, or ``None``.
        """
        prefix_new = prod(new_shape[:s_d]) if s_d > 0 else 1
        prefix_orig = 1
        for d_prime in range(len(orig_shape)):
            if prefix_orig == prefix_new:
                if new_shape[s_d] == orig_shape[d_prime]:
                    return d_prime
                return None
            prefix_orig *= orig_shape[d_prime]
        return None

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Check that the summed dimension maps cleanly to an original dim.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The ``view`` / ``_unsafe_view`` node.

        Returns:
            True if the dimension mapping is clean.
        """
        view_input = inner_node.args[0]
        orig_shape = view_input.meta["tensor_meta"].shape
        new_shape = inner_node.args[1]
        s_d = _get_sum_pos(sum_node)
        return self._find_orig_dim(orig_shape, new_shape, s_d) is not None

    def _rewrite(self, sum_node: Node, view_node: Node, pos: int, graph: Graph) -> Node:
        """Swap the sum and view, adjusting the shape.

        Args:
            sum_node: The ``sum`` node.
            view_node: The ``view`` / ``_unsafe_view`` node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The new view node.
        """
        view_input = view_node.args[0]
        orig_shape = view_input.meta["tensor_meta"].shape
        new_shape = list(view_node.args[1])
        d_prime = self._find_orig_dim(orig_shape, new_shape, pos)

        # Shape after removing the summed dimension
        result_shape = new_shape[:pos] + new_shape[pos + 1 :]

        with graph.inserting_before(sum_node):
            sum_args, sum_kwargs = _make_sum_args(view_input, d_prime)
            new_sum = graph.call_function(_sum_target, args=sum_args, kwargs=sum_kwargs)
        with graph.inserting_after(new_sum):
            new_view = graph.call_function(
                view_node.target, args=(new_sum, result_shape)
            )

        return new_view
