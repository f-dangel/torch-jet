"""Module that computes the weighted Laplacian via jets and can be simplified."""

from typing import Callable

from torch import Tensor, arange, cat, einsum, randn, zeros

import jet
from jet.laplacian import Laplacian
from jet.utils import rademacher
from jet.vmap import traceable_vmap


class WeightedLaplacian(Laplacian):
    """Class for computing dot products of the Hessian with a PSD matrix.

    Given a function f, computes < ∂²f(x), C(x) > where C(x) is the PSD coefficient
    tensor for the dot product which may depend on x. C is supplied by a function
    S_func that computes a matrix S(x) such that C(x) = S(x) @ S(x).T.
    """

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        dummy_x: Tensor,
        is_batched: bool,
        weighting: str,
        rank_ratio: float = 1.0,
    ):
        """Initialize the WeightedLaplacian module.

        Args:
            f: The function whose Laplacian is computed.
            dummy_x: The input on which the Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
            weighting: The type of weighting to use. Currently only
                "diagonal_increments" is supported.
            rank_ratio: The ratio of the rank of the coefficient tensor C(x). `1.0`
                means that the rank is equal to the number of dimensions of the input
                tensor. `0.5` means that the rank is half of the number of dimensions,
                etc. Default is `1.0` (full rank).
        """
        super().__init__(f, dummy_x, is_batched)
        self.rank_ratio = rank_ratio
        self.S_func = {"diagonal_increments": self.S_func_diagonal_increments}[
            weighting
        ]

        # determine how many columns S has, i.e. the rank of C, which determines the
        # number of vectors we have to feed into the jet
        self.num_vectors = self.S_func(dummy_x).shape[-1]

        jet_f = jet.jet(f, 2)
        self.jet_f = traceable_vmap(jet_f, self.num_vectors)

    def set_up_taylor_coefficients(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the MC-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The three input tensors to the 2-jet that computes the MC-Laplacian.
        """
        X0 = jet.utils.replicate(x, self.num_vectors)
        X2 = zeros(self.num_vectors, *self.x_shape, **self.x_kwargs)

        # compute the coefficient's factorization C = S @ S.T
        S = self.S_func(x)
        X1 = S.movedim(-1, 0)

        return X0, X1, X2

    # Definitions of synthetic coefficient functions for illustration purposes

    def S_func_diagonal_increments(self, x: Tensor) -> Tensor:
        """Compute a synthetic coefficient factor S(x) for weighting the Laplacian.

        The factor S(x) relates to the coefficient tensor C(x) via C(x) = S(x) @ S(x).T.

        Args:
            x: Argument at which the weighted Laplacian is evaluated.

        Returns:
            The coefficient factor as a tensor of shape `(batch_size, *x.shape[1:],
                rank_C)` if `is_batched` is True, otherwise `(*x.shape, rank_C)`.

        Raises:
            ValueError: If the coefficient tensor rank is not positive.
        """
        unbatched = self.x_shape[1:] if self.is_batched else self.x_shape
        D = (self.x_shape[1:] if self.is_batched else self.x_shape).numel()
        rank_C = int(self.rank_ratio * D)
        if rank_C <= 0:
            raise ValueError(f"Coefficient tensor rank must be positive. Got {rank_C}.")

        S = zeros(D, rank_C, **self.x_kwargs)
        idx = arange(rank_C, device=self.x_kwargs["device"])
        S[idx, idx] = (arange(rank_C, **self.x_kwargs) + 1).sqrt()
        S = S.reshape(*unbatched, rank_C)

        if self.is_batched:
            S = S.unsqueeze(0).expand(self.x_shape[0], *unbatched, rank_C)

        return S


class RandomizedWeightedLaplacian(WeightedLaplacian):
    """Computes a Monte-Carlo estimate of the weighted Laplacian using jets.

    Given a function f, approximates < ∂²f(x), C(x) > where C(x) is the PSD coefficient
    tensor for the dot product which may depend on x. C is supplied by a function
    apply_S_func that applies a matrix S(x) such that C(x) = S(x) @ S(x).T to another
    matrix V.

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
        weighting: str,
        rank_ratio: float = 1.0,
    ):
        """Initialize the RandomizedWeightedLaplacian module.

        Args:
            f: The function whose Laplacian is approximated.
            dummy_x: The input on which the Laplacian is approximated. It is only used
                to infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
            num_samples: How many Monte-Carlo samples should be used by the estimation.
                Must be a positive integer.
            distribution: From which distribution to draw the random vectors.
                Possible values are `'normal'` or `'rademacher'`.
            weighting: The type of weighting to use. Currently only
                "diagonal_increments" is supported.
            rank_ratio: The ratio of the rank of the coefficient tensor C(x). `1.0` means
                that the rank is equal to the number of dimensions of the input tensor.
                `0.5` means that the rank is half of the number of dimensions, etc.
                Default is `1.0` (full rank).

        Raises:
            ValueError: If the distribution is unsupported, or the number of samples is
                non-positive.
        """
        super().__init__(f, dummy_x, is_batched, weighting, rank_ratio=rank_ratio)

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

        self.apply_S_func = {
            "diagonal_increments": self.apply_S_func_diagonal_increments
        }[weighting]
        self.rank_C = {"diagonal_increments": int(rank_ratio * self.unbatched_dim)}[
            weighting
        ]
        if self.rank_C <= 0:
            raise ValueError(
                f"Coefficient tensor rank must be positive. Got {self.rank_C}."
            )

        jet_f = jet.jet(f, 2)
        self.jet_f = traceable_vmap(jet_f, self.num_samples)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the MC weighted Laplacian of the function at the input tensor.

        Replicates the input tensor, then evaluates the 2-jet of f using
        random vectors for v1 and zero vectors for v2.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            Tuple containing the replicated function value, the randomized Jacobian,
            and the randomized Laplacian.
        """
        F0, F1, F2 = super().forward(x)

        # need to divide the weighted Laplacian by number of MC samples
        return F0, F1, F2 / self.num_samples

    def set_up_taylor_coefficients(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the weighted MC-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The three input tensors to the 2-jet that computes the weighted
            MC-Laplacian.
        """
        X0 = jet.utils.replicate(x, self.num_samples)
        X2 = zeros(self.num_samples, *self.x_shape, **self.x_kwargs)

        # sample the random vectors
        shape = (
            (self.num_samples, self.batched_dim, self.rank_C)
            if self.is_batched
            else (self.num_samples, self.rank_C)
        )
        V = self.sample_func(*shape, **self.x_kwargs)
        X1 = self.apply_S_func(x, V)

        return X0, X1, X2

    def apply_S_func_diagonal_increments(self, x: Tensor, V: Tensor) -> Tensor:
        """Apply a synthetic coefficient factor S(x) for weighting the Laplacian to V.

        The factor S(x) relates to the coefficient tensor C(x) via C(x) = S(x) @ S(x).T.

        Args:
            x: Argument at which the weighted Laplacian is evaluated.
            V: The matrix onto which S(x) is applied. Has shape
                `(K, batch_size, rank_C)` where `K` is the number of columns if
                `is_batched` is True, otherwise `(K, rank_C)`.

        Returns:
            The coefficient factor S(x) applied to V. Has shape `(K, *x.shape)`.
        """
        S = (arange(self.rank_C, **self.x_kwargs) + 1).sqrt()
        SV = einsum("c,...c->...c", S, V)

        # if rank_C < D, we have to add zero padding to satisfy the output dimension
        D = (self.x_shape[1:] if self.is_batched else self.x_shape).numel()
        padding_shape = (
            (self.num_samples, self.x_shape[0], D - self.rank_C)
            if self.is_batched
            else (self.num_samples, D - self.rank_C)
        )
        P = zeros(*padding_shape, **self.x_kwargs)
        SV_padded = cat([SV, P], dim=-1)

        return SV_padded.reshape(self.num_samples, *self.x_shape)


# Definitions of synthetic coefficient functions for illustration purposes
def S_func_diagonal_increments(
    x: Tensor, is_batched: bool, rank_ratio: float = 1.0
) -> Tensor:
    """Compute a synthetic coefficient factor S(x) for weighting the Laplacian.

    The factor S(x) relates to the coefficient tensor C(x) via C(x) = S(x) @ S(x).T.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.
        is_batched: Whether `x` is batched.
        rank_ratio: The ratio of the rank of the coefficient tensor C(x). `1.0` means
            that the rank is equal to the number of dimensions of the input tensor.
            `0.5` means that the rank is half of the number of dimensions, etc.
            Default is `1.0` (full rank).

    Returns:
        The coefficient factor as a tensor of shape `(rank_C, batch_size, *x.shape[1:])`
        if `is_batched` is True, otherwise `(rank_C, *x.shape)`.

    Raises:
        ValueError: If the coefficient tensor rank is not positive.
    """
    unbatched = x.shape[1:] if is_batched else x.shape
    D = (x.shape[1:] if is_batched else x.shape).numel()
    rank_C = int(rank_ratio * D)
    if rank_C <= 0:
        raise ValueError(f"Coefficient tensor rank must be positive. Got {rank_C}.")

    S = zeros(D, rank_C, dtype=x.dtype, device=x.device)
    idx = arange(rank_C, device=x.device)
    S[idx, idx] = (arange(rank_C, dtype=x.dtype, device=x.device) + 1).sqrt()
    S = S.reshape(*unbatched, rank_C)

    if is_batched:
        batch_size = x.shape[0]
        S = S.unsqueeze(0).expand(batch_size, *unbatched, rank_C)

    return einsum("...c->c...", S)


def C_func_diagonal_increments(
    x: Tensor, is_batched: bool, rank_ratio: float = 1.0
) -> Tensor:
    """Compute a synthetic coefficient tensor C(x) for weighting the Laplacian.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.
        is_batched: Whether `x` is batched.
        rank_ratio: The ratio of the rank of the coefficient tensor C(x). `1.0` means
            that the rank is equal to the number of dimensions of the input tensor.
            `0.5` means that the rank is half of the number of dimensions, etc.
            Default is `1.0` (full rank).

    Returns:
        The coefficient tensor as a tensor of shape `(batch_size, *x.shape[1:],
            *x.shape[1:])` if `is_batched` is True, otherwise of shape
            `(*x.shape, *x.shape)`.

    Raises:
        ValueError: If the coefficient tensor rank is not positive.
    """
    unbatched = x.shape[1:] if is_batched else x.shape
    D = (x.shape[1:] if is_batched else x.shape).numel()
    rank_C = int(rank_ratio * D)
    if rank_C <= 0:
        raise ValueError(f"Coefficient tensor rank must be positive. Got {rank_C}.")

    C = zeros(D, dtype=x.dtype, device=x.device)
    C[:rank_C] = arange(rank_C, dtype=x.dtype, device=x.device) + 1
    C = C.diag().reshape(*unbatched, *unbatched)

    if is_batched:
        batch_size = x.shape[0]
        C = C.unsqueeze(0).expand(batch_size, *unbatched, *unbatched)

    return C
