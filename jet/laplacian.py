"""Implements a module that computes the Laplacian via jets and can be simplified."""

from typing import Callable

from torch import Tensor, eye, zeros
from torch.nn import Module

import jet
from jet.vmap import traceable_vmap


class Laplacian(Module):
    """Module that computes the exact or randomized Laplacian of a function using jets.

    Attributes:
        SUPPORTED_DISTRIBUTIONS: List of supported distributions for the random vectors.
    """

    SUPPORTED_DISTRIBUTIONS = ["normal", "rademacher"]

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        dummy_x: Tensor,
        randomization: tuple[str, int] | None = None,
        weighting: tuple[Callable[[Tensor, Tensor], Tensor], int] | None = None,
    ):
        """Initialize the Laplacian module.

        Args:
            f: The function whose Laplacian is computed.
            dummy_x: The input on which the Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
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
        """
        super().__init__()

        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.in_shape = dummy_x.shape
        self.in_meta = {"dtype": dummy_x.dtype, "device": dummy_x.device}
        self.in_dim = dummy_x.numel()

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
                    f"Unsupported distribution {distribution!r}. "
                    f"Supported distributions are {self.SUPPORTED_DISTRIBUTIONS}."
                )
            if num_samples <= 0:
                raise ValueError(
                    f"Number of samples must be positive, got {num_samples}."
                )
        self.randomization = randomization

        jet_f = jet.jet(f, 2)
        self.num_jets = (
            self.rank_weightings if randomization is None else self.randomization[1]
        )
        self.jet_f = traceable_vmap(jet_f, self.num_jets)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the Laplacian of the function at the input tensor.

        Replicates the input tensor, then evaluates the 2-jet of f using
        canonical basis vectors for v1 and zero vectors for v2.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            Tuple containing the replicated function value, the exact/randomized
            Jacobian, and the exact/randomized Laplacian.
        """
        X0 = jet.utils.replicate(x, self.num_jets)
        X1 = self.set_up_first_taylor_coefficient(x)
        X2 = zeros(self.num_jets, *self.in_shape, **self.in_meta)
        F0, F1, F2 = self.jet_f(X0, X1, X2)
        if self.randomization is not None:
            F2 = F2 * (1.0 / self.randomization[1])
        return F0, F1, jet.utils.sum_vmapped(F2)

    def set_up_first_taylor_coefficient(self, x: Tensor) -> Tensor:
        """Create the first Taylor coefficients for the Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The first Taylor coefficient for computing the Laplacian.
        """
        shape = self.num_jets, self.rank_weightings
        V = (
            eye(self.rank_weightings, **self.in_meta)
            if self.randomization is None
            else jet.utils.sample(x, self.randomization[0], shape)
        )
        return self.apply_weightings(x, V)
