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


def _make_sum_args(x: Any, pos: int) -> tuple[tuple, dict]:
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

    def _rewrite(
        self, sum_node: Node, mul_node: Node, pos: int, graph: Graph
    ) -> Node:
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
        out_val = inner_node.meta.get("val")
        if out_val is None or not hasattr(out_val, "shape"):
            return False
        out_ndim = len(out_val.shape)

        for arg in inner_node.args:
            arg_val = arg.meta.get("val")
            if arg_val is None or not hasattr(arg_val, "shape"):
                continue
            if pos < out_ndim - len(arg_val.shape):
                return True

        return False

    def _rewrite(
        self, sum_node: Node, mul_node: Node, pos: int, graph: Graph
    ) -> Node:
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
        out_ndim = len(mul_node.meta["val"].shape)
        arg0, arg1 = mul_node.args

        arg0_ndim = len(arg0.meta["val"].shape)
        arg1_ndim = len(arg1.meta["val"].shape)

        if pos < out_ndim - arg1_ndim:
            varying_idx, invariant_idx = 0, 1
        else:
            varying_idx, invariant_idx = 1, 0

        varying = mul_node.args[varying_idx]
        invariant = mul_node.args[invariant_idx]

        with graph.inserting_after(sum_node):
            sum_args, sum_kwargs = _make_sum_args(varying, pos)
            new_sum = graph.call_function(
                _sum_target, args=sum_args, kwargs=sum_kwargs
            )
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
        """Check that both arguments are tensors (no broadcasting).

        Args:
            sum_node: The ``sum`` node.
            inner_node: The addition/subtraction node.

        Returns:
            True if the operation has 2 tensor arguments.
        """
        return (
            len(inner_node.args) == 2
            and inner_node.kwargs == {}
            and sum(isinstance(a, Node) for a in inner_node.args) == 2
        )

    def _rewrite(
        self, sum_node: Node, add_node: Node, pos: int, graph: Graph
    ) -> Node:
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


class PullSumLinear(Rule):
    """Pull ``sum`` through ``mm`` and ``addmm``.

    Matches ``sum(op(…), pos)`` where ``op`` is ``mm`` or ``addmm``
    and ``pos`` identifies a non-contracted dimension. Moves the summation to
    the corresponding matrix argument:

    For ``mm(X, W)``:

    - ``pos == 0``: → ``squeeze(mm(unsqueeze(sum(X, 0), 0), W), 0)``
    - ``pos == 1``: → ``squeeze(mm(X, unsqueeze(sum(W, 1), 1)), 1)``

    For ``addmm(b, X, W)`` the matrix part is handled identically, plus a
    bias correction (``V * b`` for pos=0, ``sum(b)`` for pos=1).

    Attributes:
        OPERATIONS: Supported ATen matrix-multiply operations.
    """

    OPERATIONS = [_aten.mm.default, _aten.addmm.default]

    @staticmethod
    def _matrix_arg_indices(node: Node) -> tuple[int, int]:
        """Return the argument indices of the two matrix operands.

        Args:
            node: An ``mm`` or ``addmm`` node.

        Returns:
            Tuple of ``(mat1_index, mat2_index)``.
        """
        if node.target == _aten.mm.default:
            return (0, 1)
        return (1, 2)

    def _extra_match(self, sum_node: Node, inner_node: Node) -> bool:
        """Check that the sum dimension is 0 or 1.

        Args:
            sum_node: The ``sum`` node.
            inner_node: The ``mm``/``addmm`` node.

        Returns:
            True if ``pos`` is 0 (mat1's row dim) or 1 (mat2's col dim).
        """
        return _get_sum_pos(sum_node) in (0, 1)

    def _rewrite(
        self, sum_node: Node, matmul_node: Node, pos: int, graph: Graph
    ) -> Node:
        """Pull sum through mm/addmm.

        For ``mm``, replaces ``sum(mm(…), pos)`` with
        ``squeeze(mm(…, unsqueeze(sum(arg, pos), pos), …), pos)``.

        For ``addmm``, the same matrix rewrite is applied (using ``mm``),
        and the bias is corrected: ``V * b`` when ``pos == 0``, or
        ``sum(b)`` when ``pos == 1``.

        Args:
            sum_node: The ``sum`` node.
            matmul_node: The ``mm``/``addmm`` node.
            pos: The dimension along which the sum is performed.
            graph: The computation graph.

        Returns:
            The replacement node.
        """
        i1, i2 = self._matrix_arg_indices(matmul_node)
        # pos=0 → pull sum to mat1, pos=1 → pull to mat2
        target_idx = i1 if pos == 0 else i2
        target_arg = matmul_node.args[target_idx]

        # sum on the target argument
        with graph.inserting_before(matmul_node):
            sv_args, sv_kwargs = _make_sum_args(target_arg, pos)
            new_sv = graph.call_function(_sum_target, args=sv_args, kwargs=sv_kwargs)

        # unsqueeze to restore 2D shape for mm
        with graph.inserting_after(new_sv):
            unsqueeze_node = graph.call_function(
                _aten.unsqueeze.default, args=(new_sv, pos)
            )

        # new mm with sum+unsqueeze replacing the original arg
        with graph.inserting_after(unsqueeze_node):
            mm_args = tuple(
                unsqueeze_node if i == target_idx else matmul_node.args[i]
                for i in (i1, i2)
            )
            new_mm = graph.call_function(_aten.mm.default, args=mm_args)

        # squeeze the summed dimension from the result
        with graph.inserting_after(new_mm):
            squeeze_node = graph.call_function(_aten.squeeze.dim, args=(new_mm, pos))

        if matmul_node.target != _aten.addmm.default:
            return squeeze_node

        bias = matmul_node.args[0]
        if pos == 0:
            # Bias b was broadcast from (n,) to (V, n); summing V rows
            # gives V * b. Extract V from the traced tensor metadata.
            V = target_arg.meta["val"].shape[0]
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
        if not hasattr(view_input, "meta") or "val" not in view_input.meta:
            return False

        orig_shape = view_input.meta["val"].shape
        new_shape = inner_node.args[1]
        s_d = _get_sum_pos(sum_node)
        return self._find_orig_dim(orig_shape, new_shape, s_d) is not None

    def _rewrite(
        self, sum_node: Node, view_node: Node, pos: int, graph: Graph
    ) -> Node:
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
        orig_shape = view_input.meta["val"].shape
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
