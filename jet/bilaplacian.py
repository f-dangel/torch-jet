"""Implements computing the Bi-Laplacian operator with Taylor mode."""

from typing import Callable, Tuple

from torch import Tensor, eye, randn, zeros
from torch.fx import wrap
from torch.nn import Module

from jet import jet
from jet.utils import replicate, sum_vmapped

# tell `torch.fx` to trace `replicate` as one node (required for simplification)
wrap(replicate)
# tell `torch.fx` to trace `sum_vmapped` as one node (required for simplification)
wrap(sum_vmapped)


class Bilaplacian(Module):
    """Module that computes the Bi-Laplacian of a function using jets.

    The Bi-Laplacian of a function f(x) ∈ R with x ∈ Rⁿ
    is defined as the Laplacian of the Laplacian, or
    Δf(x) = ∑ᵢ ∑ⱼ ∂⁴f(x) / ∂xᵢ²∂xⱼ² ∈ R.
    For functions that produce vectors or tensors, the Bi-Laplacian
    is defined per output component and has the same shape as f(x).
    """

    def __init__(
        self, f: Callable[[Tensor], Tensor], dummy_x: Tensor, is_batched: bool
    ):
        """Initialize the Bi-Laplacian module.

        Args:
            f: The function whose Bi-Laplacian is computed.
            dummy_x: The input on which the Bi-Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
        """
        super().__init__()
        self.jet_f = jet(f, 6, vmap=True)
        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.x_shape = dummy_x.shape
        self.x_kwargs = {"dtype": dummy_x.dtype, "device": dummy_x.device}

        self.unbatched_dim = (self.x_shape[1:] if is_batched else self.x_shape).numel()
        self.batched_dim = self.x_shape[0] if is_batched else 1
        self.is_batched = is_batched

    def forward(self, x: Tensor) -> Tensor:
        """Compute the Bi-Laplacian of the function at the input tensor.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The Bi-Laplacian. Has the same shape as f(x).
        """
        X0, X1, X2, X3, X4, X5, X6 = self.set_up_taylor_coefficients(x)
        _, _, _, _, _, _, F6_1 = self.jet_f(X0, X1, X2, X3, X4, X5, X6)
        _, _, _, _, _, _, F6_2 = self.jet_f(X0, X1, -1 * X2, X3, X4, X5, X6)
        _, _, _, _, _, _, F6_3 = self.jet_f(X0, X1, 0 * X2, X3, X4, X5, X6)

        return sum_vmapped((F6_1 + F6_2 - 2 * F6_3) / 90)

    def set_up_taylor_coefficients(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the Bi-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The six input tensors to the 6-jet that computes the Bi-Laplacian.
        """
        X0 = replicate(x, self.unbatched_dim**2)
        X3 = zeros(self.unbatched_dim**2, *self.x_shape, **self.x_kwargs)
        X4 = zeros(self.unbatched_dim**2, *self.x_shape, **self.x_kwargs)
        X5 = zeros(self.unbatched_dim**2, *self.x_shape, **self.x_kwargs)
        X6 = zeros(self.unbatched_dim**2, *self.x_shape, **self.x_kwargs)

        X1 = (
            eye(self.unbatched_dim, **self.x_kwargs)
            .unsqueeze(0)
            .repeat(self.unbatched_dim, 1, 1)
        )
        if self.is_batched:
            X1 = X1.reshape(self.unbatched_dim**2, 1, *self.x_shape[1:])
            # copy without using more memory
            X1 = X1.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X1 = X1.reshape(self.unbatched_dim**2, *self.x_shape)

        # In comparison to X1, note how for X2 we have to change the order of the
        # canonical basis vectors so that we loop over all possible combinations (i, j).
        # This is done by repeating the identity matrix along the middle, instead of the
        # leading, axis.
        X2 = (
            eye(self.unbatched_dim, **self.x_kwargs)
            .unsqueeze(1)
            .repeat(1, self.unbatched_dim, 1)
        )
        if self.is_batched:
            X2 = X2.reshape(self.unbatched_dim**2, 1, *self.x_shape[1:])
            # copy without using more memory
            X2 = X2.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X2 = X2.reshape(self.unbatched_dim**2, *self.x_shape)

        return X0, X1, X2, X3, X4, X5, X6


class RandomizedBiLaplacian(Bilaplacian):
    """Computes a Monte-Carlo estimate of the Bi-Laplacian using jets.

    Attributes:
        SUPPORTED_DISTRIBUTIONS: List of supported distributions for the random vectors.
    """

    SUPPORTED_DISTRIBUTIONS = ["normal"]

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        dummy_x: Tensor,
        is_batched: bool,
        num_samples: int,
        distribution: str,
    ):
        """Initialize the Monte-Carlo Bi-Laplacian module.

        Args:
            f: The function whose Bi-Laplacian is approximated.
            dummy_x: The input on which the Bi-Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
            num_samples: How many Monte-Carlo samples should be used by the estimation.
                Must be a positive integer.
            distribution: From which distribution to draw the random vectors.
                Possible values is `'normal'`.

        Raises:
            ValueError: If the distribution is not supported or the number of samples
                is not positive.
        """
        super().__init__(f, dummy_x, is_batched)

        if distribution not in self.SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {distribution!r}. "
                f"Supported distributions are {self.SUPPORTED_DISTRIBUTIONS}."
            )
        if num_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {num_samples}.")

        self.distribution = distribution
        self.sample_func = {"normal": randn}[distribution]
        self.num_samples = num_samples
        # TODO Remove this once the exact Laplacian is implemented with 4-jets
        # rather than 6-jets
        self.jet_f = jet(f, 4, vmap=True)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the MC-Bi-Laplacian of the function at the input tensor.

        Replicates the input tensor, then evaluates the 4-jet of f using
        random vectors for v1 and zero vectors for v2, v3, v4.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The randomized Bi-Laplacian.
        """
        X0, X1, X2, X3, X4 = self.set_up_taylor_coefficients(x)
        _, _, _, _, F4 = self.jet_f(X0, X1, X2, X3, X4)

        # need to divide the Laplacian by number of MC samples
        return sum_vmapped(F4) / (3 * self.num_samples)

    def set_up_taylor_coefficients(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the MC-Bi-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The five input tensors to the 4-jet that computes the MC-Bi-Laplacian.
        """
        X0 = replicate(x, self.num_samples)
        X2 = zeros(self.num_samples, *self.x_shape, **self.x_kwargs)
        X3 = zeros(self.num_samples, *self.x_shape, **self.x_kwargs)
        X4 = zeros(self.num_samples, *self.x_shape, **self.x_kwargs)

        # sample the random vectors
        shape = (
            (self.num_samples, self.batched_dim, *self.x_shape[1:])
            if self.is_batched
            else (self.num_samples, *self.x_shape)
        )
        X1 = self.sample_func(*shape, **self.x_kwargs)

        return X0, X1, X2, X3, X4
