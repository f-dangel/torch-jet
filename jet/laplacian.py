"""Implements a module that computes the Laplacian via jets and can be simplified."""

from typing import Callable

from torch import Tensor, eye, zeros_like
from torch.func import vmap
from torch.nn import Module

import jet


class Laplacian(Module):
    r"""Module that computes the exact or randomized Laplacian of a function using jets.

    The Laplacian of a function $f(\mathbf{x}) \in \mathbb{R}$ with
    $\mathbf{x} \in \mathbb{R}^D$ is defined as the Hessian trace, or

    $$
    \Delta f(\mathbf{x})
    =
    \sum_{i=d}^D
    \frac{\partial^2 f(\mathbf{x})}{\partial x_d^2} \in \mathbb{R}\,.
    $$

    For functions that produce vectors or tensors, the Laplacian
    is defined per output component and has the same shape as $f(\mathbf{x})$.

    Attributes:
        SUPPORTED_DISTRIBUTIONS: List of supported distributions for the random vectors.
    """

    SUPPORTED_DISTRIBUTIONS = ["normal", "rademacher"]

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        mock_x: Tensor,
        randomization: tuple[str, int] | None = None,
        weighting: tuple[Callable[[Tensor, Tensor], Tensor], int] | None = None,
    ):
        """Initialize the Laplacian module.

        Args:
            f: The function whose Laplacian is computed.
            mock_x: A mock input tensor for tracing. Does not need to be the actual
                input; only the shape matters.
            randomization: Optional tuple containing the distribution type and number
                of samples for randomized Laplacian. If provided, the Laplacian will
                be computed using Monte-Carlo sampling. The first element is the
                distribution type (e.g., 'normal', 'rademacher'), and the second is the
                number of samples to use.
            weighting: A tuple specifying how the second-order derivatives should be
                weighted. This is described by a coefficient tensor C(x) of shape
                `[*D, *D]`. The first entry is a function (x, V) ↦ V @ S(x).T that
                applies the symmetric factorization S(x) of the weights
                C(x) = S(x) @ S(x).T at the input x to the matrix V. S(x) has shape
                `[*D, rank_C]` while V is `[K, rank_C]` with arbitrary `K`. The second
                entry specifies `rank_C`. If `None`, then the weightings correspond to
                the identity matrix (i.e. computing the standard Laplacian).

        Raises:
            ValueError: If the provided distribution is not supported or if the number
                of samples is not positive.

        Examples:
            >>> from torch import manual_seed, rand, zeros
            >>> from torch.func import hessian
            >>> from torch.nn import Linear, Tanh, Sequential
            >>> from jet.laplacian import Laplacian
            >>> _ = manual_seed(0) # make deterministic
            >>> f = Sequential(Linear(3, 1), Tanh())
            >>> x0 = rand(3)
            >>> # Compute the Laplacian via Taylor mode
            >>> _, _, laplacian = Laplacian(f, zeros(3))(x0)
            >>> assert laplacian.shape == f(x0).shape
            >>> # Compute the Laplacian with PyTorch's autodiff (Hessian trace)
            >>> laplacian_pt = hessian(f)(x0).squeeze(0).trace().unsqueeze(0)
            >>> assert laplacian.shape == laplacian_pt.shape
            >>> assert laplacian_pt.allclose(laplacian)
        """
        super().__init__()

        self.in_shape = mock_x.shape
        self.in_dim = mock_x.numel()

        (self.apply_weightings, self.rank_weightings) = (
            (lambda x, V: V.reshape(self.num_jets, *self.in_shape), self.in_dim)
            if weighting is None
            else weighting
        )

        # Optional: Use randomization instead of deterministic computation
        if randomization is not None:
            (distribution, num_samples) = randomization
            if distribution not in self.SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Unsupported {distribution=} ({self.SUPPORTED_DISTRIBUTIONS=})."
                )
            if num_samples <= 0:
                raise ValueError(f"{num_samples=} must be positive.")
        self.randomization = randomization

        self.num_jets = (
            self.rank_weightings if randomization is None else randomization[1]
        )
        self.jet_f = jet.jet(f, 2, mock_x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the (weighted and/or randomized) Laplacian of f at x.

        Args:
            x: Input tensor. Must have same shape as the mock input that was
                passed in the constructor.

        Returns:
            Tuple containing the function value, the weighted and/or
                randomized Jacobian, and the Laplacian.

        Raises:
            ValueError: If the input shape does not match the mock input shape
                passed in the constructor.
        """
        if x.shape != self.in_shape:
            raise ValueError(f"Expected input shape {self.in_shape}, got {x.shape}.")
        X1 = self._set_up_first_taylor_coefficient(x)
        vmapped = vmap(
            lambda x1: self.jet_f(x, x1, zeros_like(x)), randomness="different"
        )
        F0, F1, F2 = vmapped(X1)
        if self.randomization is not None:
            # Monte Carlo averaging: scale by 1 / number of samples
            monte_carlo_scaling = 1.0 / self.randomization[1]
            F2 = F2 * monte_carlo_scaling
        return F0, F1, F2.sum(0)

    def _set_up_first_taylor_coefficient(self, x: Tensor) -> Tensor:
        """Create the first Taylor coefficients for the Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the mock input that was
                passed in the constructor.

        Returns:
            The first Taylor coefficient for computing the Laplacian.
        """
        shape = (self.num_jets, self.rank_weightings)
        in_meta = {"dtype": x.dtype, "device": x.device}
        V = (
            eye(self.rank_weightings, **in_meta)
            if self.randomization is None
            else jet.utils.sample(x, self.randomization[0], shape)
        )
        return self.apply_weightings(x, V)
