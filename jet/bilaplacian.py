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
        self.jet_f = jet(f, 4, vmap=True)
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
        # three lists of 4-jet coefficients, one for each term
        C1, C2, C3 = self.set_up_taylor_coefficients(x)

        gamma_4_4 = 3 / 32
        gamma_4_0 = 13 / 192
        # first summand
        _, _, _, _, F4_1 = self.jet_f(*C1)
        # NOTE Needs a 1/24 here to match the current write-up in the paper
        factor1 = (gamma_4_4 + 2 * (self.unbatched_dim - 1) * gamma_4_0) / 24
        term1 = factor1 * sum_vmapped(F4_1)

        # there are no off-diagonal terms if the dimension is 1
        if self.unbatched_dim == 1:
            return term1

        # second summand
        gamma_3_1 = -1 / 3
        _, _, _, _, F4_2 = self.jet_f(*C2)
        # NOTE Needs a 1/24 here to match the current write-up in the paper
        factor2 = 2 * gamma_3_1 / 24
        term2 = factor2 * sum_vmapped(F4_2)

        # third term
        gamma_2_2 = 5 / 8
        _, _, _, _, F4_3 = self.jet_f(*C3)
        # NOTE Needs a 1/24 here to match the current write-up in the paper
        factor3 = 2 * gamma_2_2 / 24
        term3 = factor3 * sum_vmapped(F4_3)

        return term1 + term2 + term3

    def set_up_taylor_coefficients(self, x: Tensor) -> Tuple[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ]:
        """Create the Taylor coefficients for the Bi-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            A tuple containing the inputs to the three 4-jets.
        """
        D = self.unbatched_dim

        Z = zeros(*self.x_shape, **self.x_kwargs)

        # first 4-jet
        X1_0 = replicate(x, D)
        X1_2 = replicate(Z, D)
        X1_3 = replicate(Z, D)
        X1_4 = replicate(Z, D)

        X1_1 = 4 * eye(D, **self.x_kwargs)
        if self.is_batched:
            X1_1 = X1_1.reshape(D, 1, *self.x_shape[1:])
            # copy without using more memory
            X1_1 = X1_1.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X1_1 = X1_1.reshape(D, *self.x_shape)

        C1 = (X1_0, X1_1, X1_2, X1_3, X1_4)

        # second 4-jet
        X2_0 = replicate(x, D * (D - 1))
        X2_2 = replicate(Z, D * (D - 1))
        X2_3 = replicate(Z, D * (D - 1))
        X2_4 = replicate(Z, D * (D - 1))

        X2_1 = zeros(D, D - 1, D, **self.x_kwargs)
        for i in range(D):
            not_i = [j for j in range(D) if i != j]
            for j_idx, j in enumerate(not_i):
                X2_1[i, j_idx, i] = 3
                X2_1[i, j_idx, j] = 1

        if self.is_batched:
            X2_1 = X2_1.reshape(D * (D - 1), 1, *self.x_shape[1:])
            # copy without using more memory
            X2_1 = X2_1.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X2_1 = X2_1.reshape(D * (D - 1), *self.x_shape)

        C2 = (X2_0, X2_1, X2_2, X2_3, X2_4)

        # third 4-jet
        X3_0 = replicate(x, D * (D - 1) // 2)
        X3_2 = replicate(Z, D * (D - 1) // 2)
        X3_3 = replicate(Z, D * (D - 1) // 2)
        X3_4 = replicate(Z, D * (D - 1) // 2)

        X3_1 = zeros(D * (D - 1) // 2, D, **self.x_kwargs)
        counter = 0
        for i in range(D - 1):
            for j in range(i + 1, D):
                X3_1[counter, i] = 2
                X3_1[counter, j] = 2
                counter += 1
        assert counter == D * (D - 1) // 2

        if self.is_batched:
            X3_1 = X3_1.reshape(D * (D - 1) // 2, 1, *self.x_shape[1:])
            # copy without using more memory
            X3_1 = X3_1.expand(-1, self.batched_dim, *(-1 for _ in self.x_shape[1:]))
        else:
            X3_1 = X3_1.reshape(D * (D - 1) // 2, *self.x_shape)

        C3 = (X3_0, X3_1, X3_2, X3_3, X3_4)

        return C1, C2, C3


class RandomizedBilaplacian(Bilaplacian):
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
