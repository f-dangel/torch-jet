"""Implements computing the Bi-Laplacian operator with Taylor mode."""

from typing import Callable

from torch import Tensor, eye, zeros, zeros_like
from torch.func import vmap
from torch.nn import Module

import jet
import jet.utils
from jet.ttc_coefficients import compute_all_gammas


class Bilaplacian(Module):
    r"""Module that computes the Bi-Laplacian of a function using jets.

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

    Attributes:
        SUPPORTED_DISTRIBUTIONS: List of supported distributions for the random vectors.
    """

    SUPPORTED_DISTRIBUTIONS = ["normal"]

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        mock_x: Tensor,
        randomization: tuple[str, int] | None = None,
    ):
        """Initialize the Bi-Laplacian module.

        Args:
            f: The function whose Bi-Laplacian is computed.
            mock_x: A mock input tensor for tracing. Only the shape matters, not
                the actual values.
            randomization: Optional tuple containing the distribution type and number
                of samples for randomized Bi-Laplacian. If provided, the Bi-Laplacian
                will be computed using Monte-Carlo sampling. The first element is the
                distribution type (must be 'normal'), and the second is the number of
                samples to use. Default is `None`.

        Raises:
            ValueError: If the provided distribution is not supported or if the number
                of samples is not positive.

        Examples:
            >>> from torch import manual_seed, rand, zeros
            >>> from torch.func import hessian
            >>> from torch.nn import Linear, Tanh, Sequential
            >>> from jet.bilaplacian import Bilaplacian
            >>> _ = manual_seed(0) # make deterministic
            >>> f = Sequential(Linear(3, 1), Tanh())
            >>> x0 = rand(3)
            >>> # Compute the Bilaplacian via Taylor mode
            >>> bilaplacian = Bilaplacian(f, zeros(3))(x0)
            >>> assert bilaplacian.shape == f(x0).shape
            >>> # Compute the Bilaplacian with PyTorch's autodiff
            >>> laplacian_pt = lambda x: hessian(f)(x).squeeze(0).trace().unsqueeze(0)
            >>> bilaplacian_pt = hessian(laplacian_pt)(x0).squeeze(0).trace().unsqueeze(0)
            >>> assert bilaplacian.shape == bilaplacian_pt.shape
            >>> assert bilaplacian_pt.allclose(bilaplacian)
        """
        super().__init__()

        self.in_shape = mock_x.shape
        self.in_dim = mock_x.numel()

        if randomization is not None:
            (distribution, num_samples) = randomization
            if distribution not in self.SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Unsupported {distribution=} ({self.SUPPORTED_DISTRIBUTIONS=})."
                )
            if num_samples <= 0:
                raise ValueError(f"{num_samples=} must be positive.")
        self.randomization = randomization

        derivative_order = 4
        self.jet_f = jet.jet(f, derivative_order, mock_x)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the Bi-Laplacian of the function at the input tensor.

        Args:
            x: Input tensor. Must have same shape as the mock input that was
                passed in the constructor.

        Returns:
            The Bi-Laplacian. Has the same shape as f(x).

        Raises:
            ValueError: If the input shape does not match the expected shape.
        """
        if x.shape != self.in_shape:
            raise ValueError(f"Expected input shape {self.in_shape}, got {x.shape}.")
        vmapped = vmap(
            lambda x1: self.jet_f(
                x, x1, zeros_like(x), zeros_like(x), zeros_like(x)
            ),
            randomness="different",
        )

        if self.randomization is not None:
            distribution, num_samples = self.randomization
            X1 = jet.utils.sample(x, distribution, (num_samples, *self.in_shape))

            _, _, _, _, F4 = vmapped(X1)
            # need to divide the Laplacian by number of MC samples
            return F4.sum(0) / (3 * num_samples)

        # three lists of 4-jet coefficients, one for each term
        C1, C2, C3 = self._set_up_taylor_coefficients(x)
        D = self.in_dim

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

    def _set_up_taylor_coefficients(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Create the first Taylor coefficients for the Bi-Laplacian computation.

        The higher coefficients (X2, X3, X4) are always zero and curried into
        the vmapped lambda via ``zeros_like``.

        Args:
            x: Input tensor. Must have same shape as the mock input that was
                passed in the constructor.

        Returns:
            A tuple of three tensors (C1, C2, C3), one per 4-jet term.
            The primal x and zero higher coefficients are curried via vmap.
        """
        D = self.in_dim
        in_meta = {"dtype": x.dtype, "device": x.device}

        # first 4-jet
        C1 = 4 * eye(D, **in_meta).reshape(D, *self.in_shape)

        # second 4-jet
        X2_1 = zeros(D, D - 1, D, **in_meta)
        for i in range(D):
            not_i = [j for j in range(D) if i != j]
            for j_idx, j in enumerate(not_i):
                X2_1[i, j_idx, i] = 3
                X2_1[i, j_idx, j] = 1
        C2 = X2_1.reshape(D * (D - 1), *self.in_shape)

        # third 4-jet
        X3_1 = zeros(D * (D - 1) // 2, D, **in_meta)
        counter = 0
        for i in range(D - 1):
            for j in range(i + 1, D):
                X3_1[counter, i] = 2
                X3_1[counter, j] = 2
                counter += 1
        assert counter == D * (D - 1) // 2
        C3 = X3_1.reshape(D * (D - 1) // 2, *self.in_shape)

        return C1, C2, C3
