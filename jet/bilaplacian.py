"""Implements a function transform that computes the Bi-Laplacian via jets."""

from typing import Callable

from torch import Tensor, eye, triu_indices, zeros_like
from torch.func import vmap

import jet
import jet.utils
from jet.ttc_coefficients import compute_all_gammas

SUPPORTED_DISTRIBUTIONS = ["normal"]


def bilaplacian(
    f: Callable[[Tensor], Tensor],
    mock_x: Tensor,
    randomization: tuple[str, int] | None = None,
) -> Callable[[Tensor], Tensor]:
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

    Args:
        f: The function whose Bi-Laplacian is computed.
        mock_x: A mock input tensor for tracing. Only the shape matters, not
            the actual values.
        randomization: Optional tuple containing the distribution type and number
            of samples for randomized Bi-Laplacian. If provided, the Bi-Laplacian
            will be computed using Monte-Carlo sampling. The first element is the
            distribution type (must be 'normal'), and the second is the number of
            samples to use. Default is `None`.

    Returns:
        A function `bilap_f(x)` that returns the Bi-Laplacian of f at x.

    Raises:
        ValueError: If the provided distribution is not supported or if the number
            of samples is not positive.

    Examples:
        >>> from torch import manual_seed, rand, zeros
        >>> from torch.func import hessian
        >>> from torch.nn import Linear, Tanh, Sequential
        >>> from jet.bilaplacian import bilaplacian
        >>> _ = manual_seed(0) # make deterministic
        >>> f = Sequential(Linear(3, 1), Tanh())
        >>> x0 = rand(3)
        >>> # Compute the Bilaplacian via Taylor mode
        >>> bilap = bilaplacian(f, zeros(3))(x0)
        >>> assert bilap.shape == f(x0).shape
        >>> # Compute the Bilaplacian with PyTorch's autodiff
        >>> laplacian_pt = lambda x: hessian(f)(x).squeeze(0).trace().unsqueeze(0)
        >>> bilaplacian_pt = hessian(laplacian_pt)(x0).squeeze(0).trace().unsqueeze(0)
        >>> assert bilap.shape == bilaplacian_pt.shape
        >>> assert bilaplacian_pt.allclose(bilap)
    """
    in_shape = mock_x.shape
    in_dim = mock_x.numel()

    if randomization is not None:
        (distribution, num_samples) = randomization
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported {distribution=} ({SUPPORTED_DISTRIBUTIONS=})."
            )
        if num_samples <= 0:
            raise ValueError(f"{num_samples=} must be positive.")

    derivative_order = 4
    jet_f = jet.jet(f, derivative_order, mock_x)

    def _set_up_taylor_coefficients(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Create the first Taylor coefficients for the Bi-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as mock_x.

        Returns:
            A tuple of three tensors (C1, C2, C3), one per 4-jet term.
        """
        D = in_dim
        in_meta = {"dtype": x.dtype, "device": x.device}
        E = eye(D, **in_meta)

        # first 4-jet: one direction per basis vector, X1 = 4*e_i
        C1 = (4 * E).reshape(D, *in_shape)

        # second 4-jet: all ordered pairs (i, j) with i != j.
        # Each row is 3*e_i + e_j, giving D*(D-1) directions.
        mask = ~eye(D, dtype=bool, device=x.device)
        i_idx, j_idx = mask.nonzero(as_tuple=True)
        C2 = (3 * E[i_idx] + E[j_idx]).reshape(D * (D - 1), *in_shape)

        # third 4-jet: all unordered pairs (i, j) with i < j.
        # Each row is 2*e_i + 2*e_j, giving D*(D-1)/2 directions.
        i_idx, j_idx = triu_indices(D, D, offset=1)
        C3 = (2 * E[i_idx] + 2 * E[j_idx]).reshape(D * (D - 1) // 2, *in_shape)

        return C1, C2, C3

    def bilap_f(x: Tensor) -> Tensor:
        """Compute the Bi-Laplacian of the function at the input tensor.

        Args:
            x: Input tensor. Must have same shape as mock_x.

        Returns:
            The Bi-Laplacian. Has the same shape as f(x).

        Raises:
            ValueError: If the input shape does not match the expected shape.
        """
        if x.shape != in_shape:
            raise ValueError(f"Expected input shape {in_shape}, got {x.shape}.")
        vmapped = vmap(
            lambda x1: jet_f(x, x1, zeros_like(x), zeros_like(x), zeros_like(x)),
            randomness="different",
            out_dims=(None, 0, 0, 0, 0),
        )

        if randomization is not None:
            distribution, num_samples = randomization
            X1 = jet.utils.sample(x, distribution, (num_samples, *in_shape))

            _, _, _, _, F4 = vmapped(X1)
            # need to divide the Laplacian by number of MC samples
            return F4.sum(0) / (3 * num_samples)

        # three lists of 4-jet coefficients, one for each term
        C1, C2, C3 = _set_up_taylor_coefficients(x)
        D = in_dim

        gamma_4_4 = float(compute_all_gammas((4,))[(4,)])
        gammas = compute_all_gammas((2, 2))
        gamma_4_0 = float(gammas[(4, 0)])
        # first summand
        _, _, _, _, F4_1 = vmapped(C1)
        factor1 = (gamma_4_4 + 2 * (D - 1) * gamma_4_0) / 24
        term1 = factor1 * F4_1.sum(0)

        # there are no off-diagonal terms if the dimension is 1
        if D == 1:
            return term1

        # second summand
        gamma_3_1 = float(gammas[(3, 1)])
        _, _, _, _, F4_2 = vmapped(C2)
        factor2 = 2 * gamma_3_1 / 24
        term2 = factor2 * F4_2.sum(0)

        # third term
        gamma_2_2 = float(gammas[(2, 2)])
        _, _, _, _, F4_3 = vmapped(C3)
        factor3 = 2 * gamma_2_2 / 24
        term3 = factor3 * F4_3.sum(0)

        return term1 + term2 + term3

    return bilap_f
