"""Implements a module that computes the Laplacian via jets and can be simplified."""

from typing import Callable, Tuple

from torch import Tensor, eye, randn, zeros
from torch.nn import Module

import jet
from jet.utils import rademacher
from jet.vmap import traceable_vmap


class Laplacian(Module):
    """Module that computes the Laplacian of a function using jets."""

    def __init__(
        self, f: Callable[[Tensor], Tensor], dummy_x: Tensor, is_batched: bool
    ):
        """Initialize the Laplacian module.

        Args:
            f: The function whose Laplacian is computed.
            dummy_x: The input on which the Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
        """
        super().__init__()
        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.x_shape = dummy_x.shape
        self.x_kwargs = {"dtype": dummy_x.dtype, "device": dummy_x.device}

        self.unbatched_dim = (self.x_shape[1:] if is_batched else self.x_shape).numel()
        self.batched_dim = self.x_shape[0] if is_batched else 1
        self.is_batched = is_batched

        jet_f = jet.jet(f, 2)
        self.jet_f = traceable_vmap(jet_f, self.unbatched_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the Laplacian of the function at the input tensor.

        Replicates the input tensor, then evaluates the 2-jet of f using
        canonical basis vectors for v1 and zero vectors for v2.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            Tuple containing the jet.utils.replicated function value, the Jacobian, and the
            Laplacian.
        """
        X0, X1, X2 = self.set_up_taylor_coefficients(x)
        F0, F1, F2 = self.jet_f(X0, X1, X2)
        return F0, F1, jet.utils.sum_vmapped(F2)

    def set_up_taylor_coefficients(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The three input tensors to the 2-jet that computes the Laplacian.
        """
        X0 = jet.utils.replicate(x, self.unbatched_dim)
        X2 = zeros(self.unbatched_dim, *self.x_shape, **self.x_kwargs)

        X1 = eye(self.unbatched_dim, **self.x_kwargs)
        if self.is_batched:
            X1 = X1.reshape(self.unbatched_dim, 1, *self.x_shape[1:])
            # copy without using more memory
            X1 = X1.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X1 = X1.reshape(self.unbatched_dim, *self.x_shape)

        return X0, X1, X2


class RandomizedLaplacian(Laplacian):
    """Computes a Monte-Carlo estimate of the Laplacian using jets.

    Attributes:
        SUPPORTED_DISTRIBUTIONS: List of supported distributions for the random vectors.
    """

    SUPPORTED_DISTRIBUTIONS = ["normal", "rademacher"]

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        dummy_x: Tensor,
        is_batched: bool,
        num_samples: int,
        distribution: str,
    ):
        """Initialize the Monte-Carlo Laplacian module.

        Args:
            f: The function whose Laplacian is computed.
            dummy_x: The input on which the Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
            num_samples: How many Monte-Carlo samples should be used by the estimation.
                Must be a positive integer.
            distribution: From which distribution to draw the random vectors.
                Possible values are `'normal'` or `'rademacher'`.

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
        self.sample_func = {"normal": randn, "rademacher": rademacher}[distribution]
        self.num_samples = num_samples

        jet_f = jet.jet(f, 2)
        self.jet_f = traceable_vmap(jet_f, self.num_samples)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the MC-Laplacian of the function at the input tensor.

        Replicates the input tensor, then evaluates the 2-jet of f using
        random vectors for v1 and zero vectors for v2.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            Tuple containing the jet.utils.replicated function value, the randomized Jacobian,
            and the randomized Laplacian.
        """
        F0, F1, F2 = super().forward(x)

        # need to divide the Laplacian by number of MC samples
        return F0, F1, F2 / self.num_samples

    def set_up_taylor_coefficients(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the MC-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The three input tensors to the 2-jet that computes the MC-Laplacian.
        """
        X0 = jet.utils.replicate(x, self.num_samples)
        X2 = zeros(self.num_samples, *self.x_shape, **self.x_kwargs)

        # sample the random vectors
        shape = (
            (self.num_samples, self.batched_dim, *self.x_shape[1:])
            if self.is_batched
            else (self.num_samples, *self.x_shape)
        )
        X1 = self.sample_func(*shape, **self.x_kwargs)

        return X0, X1, X2
