"""Implementation of AD primitives in Taylor-mode arithmetic."""

from scipy.special import comb, factorial, stirling2
from torch import Tensor, addmm, cos, mm, mul, ops, relu, sigmoid, sin, tanh, zeros_like
from torch.utils._pytree import register_pytree_node

from jet.utils import (
    Primal,
    Value,
    integer_partitions,
    multiplicity,
)


class JetTuple(tuple):
    """A tuple subclass marking Taylor-expanded values (primal + coefficients).

    Using a distinct type instead of plain ``tuple`` prevents false positives
    from ATen ops that take tuple arguments (e.g. padding, stride).
    """


# Register with PyTorch's pytree so that vmap, make_fx, etc. can flatten/unflatten
# JetTuple the same way they handle plain tuples.
register_pytree_node(
    JetTuple,
    flatten_fn=lambda x: (list(x), None),
    unflatten_fn=lambda values, context: JetTuple(values),
)


def _partition_term(
    vs: tuple[Primal, ...], sigma: tuple[int, ...], dn: dict[int, Primal]
) -> Value | None:
    """Compute one term of the Faà di Bruno sum for a given partition.

    Returns ``None`` when ``dn[len(sigma)]`` is ``None``.
    """
    if dn[len(sigma)] is None:
        return None
    vs_count = {i: sigma.count(i) for i in sigma}
    vs_contract = [
        vs[i - 1] ** count if count > 1 else vs[i - 1] for i, count in vs_count.items()
    ]
    term = vs_contract[0]
    for v in vs_contract[1:]:
        term = mul(term, v)
    term = mul(term, dn[len(sigma)])
    nu = multiplicity(sigma)
    return nu * term if nu != 1.0 else term


def _faa_di_bruno(
    vs: tuple[Primal, ...],
    derivative_order: int,
    dn: dict[int, Primal],
    collapsed: bool = False,
) -> list[Value]:
    """Apply Faà di Bruno's formula for elementwise functions.

    Args:
        vs: The incoming Taylor coefficients.
        derivative_order: The order of the Taylor expansion.
        dn: A dictionary mapping the degree to the function's derivative.
        collapsed: If ``True``, treat ``vs[-1]`` as a collapsed (already
            summed) coefficient and ``vs[0:-1]`` as batched with a leading
            direction dimension *R*.  The last output coefficient is then
            computed by separating the linear contribution (which multiplies
            the collapsed input) from the nonlinear contributions (which are
            summed over *R*).

    Returns:
        The outgoing Taylor coefficients.
    """
    K = derivative_order
    vs_out = []
    for k in range(K):
        order = k + 1
        if order == K and collapsed:
            linear_term = mul(dn[1], vs[K - 1]) if dn[1] is not None else None
            nonlinear_term = None
            for sigma in integer_partitions(K):
                if sigma == (K,):
                    continue
                term = _partition_term(vs, sigma, dn)
                if term is not None:
                    nonlinear_term = (
                        term if nonlinear_term is None else nonlinear_term + term
                    )
            if nonlinear_term is not None and linear_term is not None:
                vs_out.append(linear_term + nonlinear_term.sum(0))
            elif nonlinear_term is not None:
                vs_out.append(nonlinear_term.sum(0))
            elif linear_term is not None:
                vs_out.append(linear_term)
            else:
                vs_out.append(zeros_like(dn[0]))
        else:
            result = None
            for sigma in integer_partitions(order):
                term = _partition_term(vs, sigma, dn)
                if term is not None:
                    result = term if result is None else result + term
            vs_out.append(result)
    return vs_out


# --- Derivative helpers (shared with collapsed mode) ---


def _sin_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``sin(x0)`` and its derivatives up to order *K*."""
    sin_x0 = sin(x0)
    d = {0: sin_x0}
    for k in range(1, K + 1):
        if k == 1:
            d[k] = cos(x0)
        elif k in {2, 3}:
            d[k] = -1 * d[k - 2]
        else:
            d[k] = d[k - 4]
    return sin_x0, d


def _cos_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``cos(x0)`` and its derivatives up to order *K*."""
    cos_x0 = cos(x0)
    d = {0: cos_x0}
    for k in range(1, K + 1):
        if k == 1:
            d[k] = -1 * sin(x0)
        elif k in {2, 3}:
            d[k] = -1 * d[k - 2]
        else:
            d[k] = d[k - 4]
    return cos_x0, d


def _tanh_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``tanh(x0)`` and its derivatives up to order *K*."""
    tanh_x0 = tanh(x0)
    d = {0: tanh_x0}
    if K >= 1:
        tanh_inc = tanh_x0 + 1
        tanh_dec = tanh_x0 - 1
        tanh_dec_powers = {1: tanh_dec}
        if K >= 2:
            for k in range(2, K + 1):
                tanh_dec_powers[k] = tanh_dec**k
        for m in range(1, K + 1):
            term = None
            for k in range(1, m + 1):
                scale = factorial(k, exact=True) / 2**k * stirling2(m, k, exact=True)
                term_k = (
                    (scale * tanh_dec_powers[k]) if scale != 1.0 else tanh_dec_powers[k]
                )
                term = term_k if term is None else term + term_k
            d[m] = (-2) ** m * tanh_inc * term
    return tanh_x0, d


def _sigmoid_derivatives(x0: Primal, K: int) -> tuple[Primal, dict[int, Primal]]:
    """Compute ``sigmoid(x0)`` and its derivatives up to order *K*."""
    sigmoid_x0 = sigmoid(x0)
    d = {0: sigmoid_x0}
    if K >= 1:
        sigmoid_powers = {1: sigmoid_x0}
        for n in range(2, K + 2):
            sigmoid_powers[n] = sigmoid_x0**n
        for n in range(1, K + 1):
            term = None
            for k in range(1, n + 2):
                scale = (
                    (-1) ** (k - 1)
                    * factorial(k - 1, exact=True)
                    * stirling2(n + 1, k, exact=True)
                )
                term_k = (
                    scale * sigmoid_powers[k] if scale != 1.0 else sigmoid_powers[k]
                )
                term = term_k if term is None else term + term_k
            d[n] = term
    return sigmoid_x0, d


def _pow_derivatives(
    x0: Primal, exponent: float | int, K: int
) -> tuple[Primal, dict[int, Primal | None]]:
    """Compute ``x0 ** exponent`` and its derivatives up to order *K*."""
    pow_x0 = x0**exponent
    d = {0: pow_x0}
    for k in range(1, K + 1):
        if exponent - k < 0 and int(exponent) == exponent:
            d[k] = None
        elif exponent == k:
            d[k] = factorial(exponent, exact=True)
        else:
            scale = 1
            for i in range(1, k + 1):
                scale *= exponent + 1 - i
            d[k] = scale * x0 if exponent - k == 1 else scale * x0 ** (exponent - k)
    return pow_x0, d


# --- Elementwise unary ---


def jet_sin(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sin(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]
    sin_self0, dsin = _sin_derivatives(self0, derivative_order)
    vs_out = _faa_di_bruno(vs, derivative_order, dsin)
    return JetTuple((sin_self0, *vs_out))


def jet_cos(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.cos(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]
    cos_self0, dcos = _cos_derivatives(self0, derivative_order)
    vs_out = _faa_di_bruno(vs, derivative_order, dcos)
    return JetTuple((cos_self0, *vs_out))


def jet_tanh(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.tanh(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]
    tanh_self0, dtanh = _tanh_derivatives(self0, derivative_order)
    vs_out = _faa_di_bruno(vs, derivative_order, dtanh)
    return JetTuple((tanh_self0, *vs_out))


def jet_sigmoid(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sigmoid(self)``.

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self0, vs = self[0], self[1:]
    sigmoid_self0, dsigmoid = _sigmoid_derivatives(self0, derivative_order)
    vs_out = _faa_di_bruno(vs, derivative_order, dsigmoid)
    return JetTuple((sigmoid_self0, *vs_out))


# --- Power ---


def jet_pow(
    self: JetTuple, exponent: float | int, *, derivative_order: int
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.pow(self, exponent)``.

    Args:
        self: The primal and its Taylor coefficients.
        exponent: The scalar exponent.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    assert isinstance(exponent, (float, int))
    self0, vs = self[0], self[1:]
    pow_self0, dpow = _pow_derivatives(self0, exponent, derivative_order)
    vs_out = _faa_di_bruno(vs, derivative_order, dpow)
    return JetTuple((pow_self0, *vs_out))


# --- Arithmetic ---


def jet_add(
    self: Primal | JetTuple | float | int,
    other: Primal | JetTuple | float | int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.add(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients, or a scalar.
        other: The second operand and its Taylor coefficients, or a scalar.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        return JetTuple(self[k] + other[k] for k in range(derivative_order + 1))
    elif self_is_jet:
        return JetTuple(
            (self[0] + other,) + tuple(self[k] for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (other[0] + self,) + tuple(other[k] for k in range(1, derivative_order + 1))
        )


def jet_sub(
    self: Primal | JetTuple | float | int,
    other: Primal | JetTuple | float | int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sub(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients, or a scalar.
        other: The second operand and its Taylor coefficients, or a scalar.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        return JetTuple(self[k] - other[k] for k in range(derivative_order + 1))
    elif self_is_jet:
        return JetTuple(
            (self[0] - other,) + tuple(self[k] for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (self - other[0],)
            + tuple(-other[k] for k in range(1, derivative_order + 1))
        )


def jet_mul(
    self: Primal | JetTuple,
    other: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mul(self, other)``.

    Args:
        self: The first operand and its Taylor coefficients.
        other: The second operand and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    other_is_jet = isinstance(other, JetTuple)

    if self_is_jet and other_is_jet:
        s_out = ()
        for k in range(derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * self[j] * other[k - j]
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif self_is_jet:
        return JetTuple(other * self[k] for k in range(derivative_order + 1))
    else:
        return JetTuple(self * other[k] for k in range(derivative_order + 1))


# --- Linear decomposition ---


def jet_mm(
    self: Primal | JetTuple,
    mat2: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mm(self, mat2)``.

    Args:
        self: The first matrix and its Taylor coefficients.
        mat2: The second matrix and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    self_is_jet = isinstance(self, JetTuple)
    mat2_is_jet = isinstance(mat2, JetTuple)

    if self_is_jet and mat2_is_jet:
        s_out = ()
        for k in range(derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * mm(self[j], mat2[k - j])
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif self_is_jet:
        return JetTuple(mm(self[k], mat2) for k in range(derivative_order + 1))
    else:
        return JetTuple(mm(self, mat2[k]) for k in range(derivative_order + 1))


def jet_addmm(
    self: Primal,
    mat1: Primal | JetTuple,
    mat2: Primal | JetTuple,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.addmm(self, mat1, mat2)``.

    Args:
        self: The bias tensor (constant).
        mat1: The first matrix and its Taylor coefficients.
        mat2: The second matrix and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    mat1_is_jet = isinstance(mat1, JetTuple)
    mat2_is_jet = isinstance(mat2, JetTuple)

    if mat1_is_jet and mat2_is_jet:
        s_out = (addmm(self, mat1[0], mat2[0]),)
        for k in range(1, derivative_order + 1):
            term = None
            for j in range(k + 1):
                term_j = comb(k, j, exact=True) * mm(mat1[j], mat2[k - j])
                term = term_j if term is None else term + term_j
            s_out = s_out + (term,)
        return JetTuple(s_out)

    elif mat1_is_jet:
        return JetTuple(
            (addmm(self, mat1[0], mat2),)
            + tuple(mm(mat1[k], mat2) for k in range(1, derivative_order + 1))
        )
    else:
        return JetTuple(
            (addmm(self, mat1, mat2[0]),)
            + tuple(mm(mat1, mat2[k]) for k in range(1, derivative_order + 1))
        )


def jet_view(self: JetTuple, size: list[int], *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.view(self, size)``.

    Args:
        self: The primal and its Taylor coefficients.
        size: The target shape.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each reshaped.
    """
    return JetTuple(
        ops.aten.view.default(self[k], size) for k in range(derivative_order + 1)
    )


def jet_unsqueeze(self: JetTuple, dim: int, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.unsqueeze(self, dim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimension to unsqueeze.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each unsqueezed.
    """
    return JetTuple(
        ops.aten.unsqueeze.default(self[k], dim) for k in range(derivative_order + 1)
    )


def jet_squeeze(self: JetTuple, dim: int, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.squeeze(self, dim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimension to squeeze.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients, each squeezed.
    """
    return JetTuple(
        ops.aten.squeeze.dim(self[k], dim) for k in range(derivative_order + 1)
    )


# --- Sum (dim reduction) ---


def jet_sum(
    self: JetTuple,
    dim: list[int],
    keepdim: bool = False,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.sum(self, dim, keepdim)``.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimensions along which to sum (list of ints).
        keepdim: Whether to keep the reduced dimension. Default: ``False``.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.

    Raises:
        NotImplementedError: If keepdim is True.
    """
    if keepdim:
        raise NotImplementedError("keepdim=True is not supported.")
    pos = dim[0] if isinstance(dim, list) else dim
    return JetTuple(self[k].sum(pos) for k in range(derivative_order + 1))


# --- Convolution (linear in input) ---


def jet_convolution(
    self: JetTuple,
    weight: Primal,
    bias: Primal | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.convolution(self, weight, bias, ...)``.

    Convolution is linear in its input when the weight is constant, so Taylor
    coefficients pass through the convolution independently (with bias only
    applied to the primal).

    Args:
        self: The input and its Taylor coefficients.
        weight: The convolution kernel (constant).
        bias: The bias tensor (constant), or ``None``.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        transposed: Whether to use transposed convolution.
        output_padding: Output padding for transposed convolution.
        groups: Number of groups.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    primal = ops.aten.convolution.default(
        self[0],
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    coeffs = tuple(
        ops.aten.convolution.default(
            self[k],
            weight,
            None,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        for k in range(1, derivative_order + 1)
    )
    return JetTuple((primal, *coeffs))


# --- Batch normalization (eval mode, affine in input) ---


def jet_native_batch_norm(
    self: JetTuple,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
    *,
    derivative_order: int,
) -> tuple:
    """Taylor-mode arithmetic for ``aten.native_batch_norm(self, ...)``.

    Only eval mode is supported, where batch normalization is affine in the
    input: ``y = (x - mean) / sqrt(var + eps) * weight + bias``.

    Args:
        self: The input and its Taylor coefficients.
        weight: Scale parameter (constant), or ``None``.
        bias: Shift parameter (constant), or ``None``.
        running_mean: Running mean (constant).
        running_var: Running variance (constant).
        training: Must be ``False``.
        momentum: Momentum (unused in eval mode).
        eps: Epsilon for numerical stability.
        derivative_order: The order of the Taylor expansion.

    Returns:
        A plain 3-tuple ``(JetTuple_output, save_mean, save_invstd)`` so that
        ``operator.getitem`` in the interpreter falls through correctly.

    Raises:
        NotImplementedError: If ``training`` is ``True``.
    """
    if training:
        raise NotImplementedError("Only eval-mode BatchNorm is supported in jet.")

    bn_result = ops.aten.native_batch_norm.default(
        self[0],
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    primal_out = bn_result[0]

    # Scale factor: weight / sqrt(running_var + eps)
    # For affine=False (weight is None), scale is 1/sqrt(running_var + eps)
    invstd = 1.0 / (running_var + eps).sqrt()
    scale = weight * invstd if weight is not None else invstd
    # Reshape for broadcasting over (N, C, *spatial)
    shape = [1] * self[0].ndim
    shape[1] = -1
    scale = scale.reshape(shape)

    coeffs = tuple(self[k] * scale for k in range(1, derivative_order + 1))

    return (JetTuple((primal_out, *coeffs)), bn_result[1], bn_result[2])


# --- ReLU (piecewise linear) ---


def jet_relu(self: JetTuple, *, derivative_order: int) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.relu(self)``.

    ReLU is piecewise linear with derivative equal to the Heaviside step
    function.  Higher-order derivatives are zero (treating the non-differentiable
    point at zero as having derivative zero).

    Args:
        self: The primal and its Taylor coefficients.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    primal = relu(self[0])
    mask = (self[0] > 0).to(self[0].dtype)
    coeffs = tuple(self[k] * mask for k in range(1, derivative_order + 1))
    return JetTuple((primal, *coeffs))


# --- Mean (linear, for AdaptiveAvgPool2d) ---


def jet_mean(
    self: JetTuple,
    dim: list[int],
    keepdim: bool = False,
    *,
    derivative_order: int,
) -> JetTuple:
    """Taylor-mode arithmetic for ``aten.mean(self, dim, keepdim)``.

    Mean is a linear operation, so it applies identically to each Taylor
    coefficient.

    Args:
        self: The primal and its Taylor coefficients.
        dim: The dimensions along which to compute the mean.
        keepdim: Whether to keep the reduced dimensions. Default: ``False``.
        derivative_order: The order of the Taylor expansion.

    Returns:
        The value and its Taylor coefficients.
    """
    return JetTuple(
        ops.aten.mean.dim(self[k], dim, keepdim) for k in range(derivative_order + 1)
    )


# --- MaxPool2d (piecewise linear, index-based) ---


def jet_max_pool2d_with_indices(
    self: JetTuple,
    kernel_size: list[int],
    stride: list[int] = (),
    padding: list[int] = (0, 0),
    dilation: list[int] = (1, 1),
    ceil_mode: bool = False,
    *,
    derivative_order: int,
) -> tuple:
    """Taylor-mode arithmetic for ``aten.max_pool2d_with_indices(self, ...)``.

    MaxPool2d selects one element per pooling window.  The indices from the
    primal computation are reused to gather the corresponding Taylor
    coefficients.

    Args:
        self: The input and its Taylor coefficients.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Padding added to the input.
        dilation: Spacing between kernel elements.
        ceil_mode: Whether to use ceil instead of floor for output shape.
        derivative_order: The order of the Taylor expansion.

    Returns:
        A plain 2-tuple ``(JetTuple_values, indices)`` so that
        ``operator.getitem`` in the interpreter falls through correctly.
    """
    values, indices = ops.aten.max_pool2d_with_indices.default(
        self[0],
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )

    flat_indices = indices.flatten(2)
    coeffs = tuple(
        self[k].flatten(2).gather(2, flat_indices).view_as(values)
        for k in range(1, derivative_order + 1)
    )

    return (JetTuple((values, *coeffs)), indices)


MAPPING = {
    # Elementwise unary
    ops.aten.sin.default: jet_sin,
    ops.aten.cos.default: jet_cos,
    ops.aten.tanh.default: jet_tanh,
    ops.aten.sigmoid.default: jet_sigmoid,
    # Power
    ops.aten.pow.Tensor_Scalar: jet_pow,
    # Arithmetic
    ops.aten.add.Tensor: jet_add,
    ops.aten.sub.Tensor: jet_sub,
    ops.aten.mul.Tensor: jet_mul,
    # Linear decomposition
    ops.aten.mm.default: jet_mm,
    ops.aten.addmm.default: jet_addmm,
    ops.aten.view.default: jet_view,
    ops.aten.unsqueeze.default: jet_unsqueeze,
    ops.aten.squeeze.dim: jet_squeeze,
    # Sum (dim reduction)
    ops.aten.sum.dim_IntList: jet_sum,
    # Convolution (linear in input)
    ops.aten.convolution.default: jet_convolution,
    # Batch normalization (eval mode, affine in input)
    ops.aten.native_batch_norm.default: jet_native_batch_norm,
    # ReLU (piecewise linear)
    ops.aten.relu.default: jet_relu,
    # Mean (linear, for AdaptiveAvgPool2d)
    ops.aten.mean.dim: jet_mean,
    # MaxPool2d (piecewise linear, index-based)
    ops.aten.max_pool2d_with_indices.default: jet_max_pool2d_with_indices,
}
