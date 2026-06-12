"""Implements a function transform that computes the Bi-Laplacian via jets."""

from typing import Callable

from torch import Tensor, eye, triu_indices, zeros, zeros_like

from jet._jet import _uncollapsed_via_vmap, jet
from jet.ttc_coefficients import compute_all_gammas
from jet.utils import (
    PyTree,
    require_single_tensor_input,
    require_single_tensor_output,
    sample,
    validate_randomization,
)

SUPPORTED_DISTRIBUTIONS = ["normal"]


def _set_up_taylor_coefficients(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Create the first Taylor coefficients for the Bi-Laplacian computation.

    Args:
        x: Input tensor.

    Returns:
        A tuple of three tensors (C1, C2, C3), one per 4-jet term.
    """
    in_shape = x.shape
    D = x.numel()
    E = eye(D, dtype=x.dtype, device=x.device)

    C1 = (4 * E).reshape(D, *in_shape)

    mask = ~eye(D, dtype=bool, device=x.device)
    i_idx, j_idx = mask.nonzero(as_tuple=True)
    C2 = (3 * E[i_idx] + E[j_idx]).reshape(D * (D - 1), *in_shape)

    i_idx, j_idx = triu_indices(D, D, offset=1, device=x.device)
    C3 = (2 * E[i_idx] + 2 * E[j_idx]).reshape(D * (D - 1) // 2, *in_shape)

    return C1, C2, C3


def bilaplacian(
    f: Callable[[Tensor], Tensor],
    mock_args: tuple[PyTree[Tensor], ...],
    collapsed: bool = True,
    randomization: tuple[str, int] | None = None,
    scale_coeffs: bool = False,
) -> Callable[[*tuple[PyTree[Tensor], ...]], Tensor]:
    r"""Transform f into a function that computes the Bi-Laplacian.

    The Bi-Laplacian of a function $f(\mathbf{x}) \in \mathbb{R}$ with
    $\mathbf{x} \in \mathbb{R}^D$ is defined as the Laplacian of the Laplacian, or

    $$
    \Delta^2 f(\mathbf{x})
    =
    \sum_{i=1}^D \sum_{j=1}^D
    \frac{\partial^4 f(\mathbf{x})}{\partial x_i^2 \partial x_j^2} \in \mathbb{R}\,.
    $$

    For functions that produce vectors or tensors, the Bi-Laplacian
    is defined per output component and has the same shape as $f(\mathbf{x})$.

    Only single-tensor functions (one tensor in, one tensor out) are supported.

    Args:
        f: The function whose Bi-Laplacian is computed. Must consume and return
            a single tensor.
        mock_args: Mock positional arguments for tracing ``f``, provided as a
            tuple matching ``f``'s positional arguments. Only shapes and dtypes
            matter, not the values. Currently must be a one-tuple of a single
            tensor.
        collapsed: Whether to use collapsed Taylor mode. If ``True``
            (default), uses the collapsed dispatch path
            (``JetInterpreter(..., collapsed=True)``) that directly propagates
            the summed fourth-order coefficient. If ``False``, propagates full
            4-jets over all directions via ``vmap`` and sums afterward.
            Collapsed mode is the more efficient default: propagating the
            summed coefficient moves smaller tensors through the graph.
        randomization: Optional tuple containing the distribution type and number
            of samples for randomized Bi-Laplacian. If provided, the Bi-Laplacian
            will be computed using Monte-Carlo sampling. The first element is the
            distribution type (must be 'normal'), and the second is the number of
            samples to use. Default is `None`.
        scale_coeffs: Whether to internally propagate the scaled polynomial
            coefficients ``x_tilde_k = x_k / k!`` instead of the default
            derivative-coefficient basis while preserving the public outputs.

    Returns:
        A plain Python callable ``bilap_f(*args)`` that maps ``x → bilap(f(x))``.
        To bake the operator into an FX ``GraphModule`` (for graph passes,
        ``torch.compile``, etc.), apply :func:`capture_graph` yourself.

    Examples:
        >>> from torch import manual_seed, rand, zeros
        >>> from torch.func import hessian
        >>> from torch.nn import Linear, Tanh, Sequential
        >>> from jet import bilaplacian
        >>> _ = manual_seed(0) # make deterministic
        >>> f = Sequential(Linear(3, 1), Tanh())
        >>> x0 = rand(3)
        >>> # Compute the Bilaplacian via Taylor mode
        >>> bilap = bilaplacian(f, (zeros(3),))(x0)
        >>> assert bilap.shape == f(x0).shape
        >>> # Compute the Bilaplacian with PyTorch's autodiff
        >>> laplacian_pt = lambda x: hessian(f)(x).squeeze(0).trace().unsqueeze(0)
        >>> bilaplacian_pt = hessian(laplacian_pt)(x0).squeeze(0).trace().unsqueeze(0)
        >>> assert bilap.shape == bilaplacian_pt.shape
        >>> assert bilaplacian_pt.allclose(bilap)
    """
    mock_x = require_single_tensor_input(mock_args, "bilaplacian")
    in_shape = mock_x.shape
    in_dim = mock_x.numel()

    validate_randomization(randomization, SUPPORTED_DISTRIBUTIONS)

    cjet_f = (
        jet(f, mock_args, collapsed=True, scale_coeffs=scale_coeffs)
        if collapsed
        else _uncollapsed_via_vmap(
            f, mock_args, randomization, scale_coeffs=scale_coeffs
        )
    )

    def _eval_4jet(x: Tensor, X1: Tensor) -> Tensor:
        """Evaluate the 4-jet for directions X1 and return the 4th coefficient.

        Args:
            x: Input tensor.
            X1: First-order directions, shape (R, *in_shape).

        Returns:
            The (collapsed or summed) 4th-order coefficient.
        """
        z = zeros_like(x)
        R = X1.shape[0]
        Z = zeros(R, *in_shape, dtype=x.dtype, device=x.device)
        result = require_single_tensor_output(cjet_f((x, X1, Z, Z, z)), "bilaplacian")
        _, _, _, _, F4 = result
        return F4

    def bilap_f(*args: PyTree[Tensor]) -> Tensor:
        """Compute the Bi-Laplacian of the function at the input tensor.

        Args:
            *args: Positional arguments for ``f`` (currently a single tensor
                matching the mock input's shape).

        Returns:
            The Bi-Laplacian. Has the same shape as f(x).

        Raises:
            ValueError: If the input shape does not match the expected shape.
        """
        (x,) = args
        if x.shape != in_shape:
            raise ValueError(f"Expected input shape {in_shape}, got {x.shape}.")

        if randomization is not None:
            distribution, num_samples = randomization
            X1 = sample(x, distribution, (num_samples, *in_shape))
            F4 = _eval_4jet(x, X1)
            return F4 / (3 * num_samples)

        C1, C2, C3 = _set_up_taylor_coefficients(x)
        D = in_dim

        gamma_4_4 = float(compute_all_gammas((4,))[(4,)])
        gammas = compute_all_gammas((2, 2))
        gamma_4_0 = float(gammas[(4, 0)])
        F4_1 = _eval_4jet(x, C1)
        factor1 = (gamma_4_4 + 2 * (D - 1) * gamma_4_0) / 24
        term1 = factor1 * F4_1

        if D == 1:
            return term1

        gamma_3_1 = float(gammas[(3, 1)])
        F4_2 = _eval_4jet(x, C2)
        term2 = 2 * gamma_3_1 / 24 * F4_2

        gamma_2_2 = float(gammas[(2, 2)])
        F4_3 = _eval_4jet(x, C3)
        term3 = 2 * gamma_2_2 / 24 * F4_3

        return term1 + term2 + term3

    return bilap_f
