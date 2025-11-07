"""Implements computing the Bi-Laplacian operator with Taylor mode."""

from typing import Callable

from torch import Tensor, eye, zeros
from torch.nn import Module

import jet
from jet.ttc_coefficients import compute_all_gammas
from jet.vmap import traceable_vmap


class Bilaplacian(Module):
    """Module that computes the Bi-Laplacian of a function using jets.

    The Bi-Laplacian of a function f(x) ∈ R with x ∈ Rⁿ
    is defined as the Laplacian of the Laplacian, or
    Δ²f(x) = ∑ᵢ ∑ⱼ ∂⁴f(x) / ∂xᵢ²∂xⱼ² ∈ R.
    For functions that produce vectors or tensors, the Bi-Laplacian
    is defined per output component and has the same shape as f(x).

    Attributes:
        SUPPORTED_DISTRIBUTIONS: List of supported distributions for the random vectors.
    """

    SUPPORTED_DISTRIBUTIONS = ["normal"]

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        dummy_x: Tensor,
        randomization: tuple[str, int] | None = None,
    ):
        """Initialize the Bi-Laplacian module.

        Args:
            f: The function whose Bi-Laplacian is computed.
            dummy_x: The input on which the Bi-Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            randomization: Optional tuple containing the distribution type and number
                of samples for randomized Bi-Laplacian. If provided, the Bi-Laplacian
                will be computed using Monte-Carlo sampling. The first element is the
                distribution type (must be 'normal'), and the second is the number of
                samples to use. Default is `None`.

        Raises:
            ValueError: If the provided distribution is not supported or if the number
                of samples is not positive.
        """
        super().__init__()

        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.in_shape = dummy_x.shape
        self.in_meta = {"dtype": dummy_x.dtype, "device": dummy_x.device}
        self.in_dim = dummy_x.numel()

        if randomization is not None:
            (distribution, num_samples) = randomization
            if distribution not in self.SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Unsupported {distribution=} ({self.SUPPORTED_DISTRIBUTIONS=})."
                )
            if num_samples <= 0:
                raise ValueError(f"{num_samples=} must be positive.")
        self.randomization = randomization

        jet_f = jet.jet(f, 4)
        D = self.in_dim
        num_jets = (
            {self.randomization[1]}
            if self.randomization
            else {D, D * (D - 1), D * (D - 1) // 2}
        )
        num_jets = {n for n in num_jets if n > 0}
        for n in num_jets:
            multijet_f = traceable_vmap(jet_f, n)
            setattr(self, f"jets_f_{n}", multijet_f)

    def get_multijet(
        self, multi: int
    ) -> Callable[
        [Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ]:
        """Get the multi-jet function for a given number of jets.

        Args:
            multi: The number of jets to retrieve.

        Returns:
            The multi-jet function for the specified number of jets.
        """
        return getattr(self, f"jets_f_{multi}")

    def forward(self, x: Tensor) -> Tensor:
        """Compute the Bi-Laplacian of the function at the input tensor.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The Bi-Laplacian. Has the same shape as f(x).
        """
        if self.randomization is not None:
            distribution, num_samples = self.randomization
            X0 = jet.utils.replicate(x, num_samples)
            X1 = jet.utils.sample(x, distribution, (num_samples, *self.in_shape))
            X2 = zeros(num_samples, *self.in_shape, **self.in_meta)
            X3 = zeros(num_samples, *self.in_shape, **self.in_meta)
            X4 = zeros(num_samples, *self.in_shape, **self.in_meta)

            jet_f = self.get_multijet(num_samples)
            _, _, _, _, F4 = jet_f(X0, X1, X2, X3, X4)
            # need to divide the Laplacian by number of MC samples
            return jet.utils.sum_vmapped(F4) / (3 * num_samples)

        # three lists of 4-jet coefficients, one for each term
        C1, C2, C3 = self.set_up_taylor_coefficients(x)
        D = self.in_dim

        gamma_4_4 = float(compute_all_gammas((4,))[(4,)])
        gammas = compute_all_gammas((2, 2))
        gamma_4_0 = float(gammas[(4, 0)])
        # first summand
        jet_f = self.get_multijet(D)
        _, _, _, _, F4_1 = jet_f(*C1)
        factor1 = (gamma_4_4 + 2 * (D - 1) * gamma_4_0) / 24
        term1 = factor1 * jet.utils.sum_vmapped(F4_1)

        # there are no off-diagonal terms if the dimension is 1
        if D == 1:
            return term1

        # second summand
        gamma_3_1 = float(gammas[(3, 1)])
        jet_f = self.get_multijet(D * (D - 1))
        _, _, _, _, F4_2 = jet_f(*C2)
        factor2 = 2 * gamma_3_1 / 24
        term2 = factor2 * jet.utils.sum_vmapped(F4_2)

        # third term
        gamma_2_2 = float(gammas[(2, 2)])
        jet_f = self.get_multijet(D * (D - 1) // 2)
        _, _, _, _, F4_3 = jet_f(*C3)
        factor3 = 2 * gamma_2_2 / 24
        term3 = factor3 * jet.utils.sum_vmapped(F4_3)

        return term1 + term2 + term3

    def set_up_taylor_coefficients(
        self, x: Tensor
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ]:
        """Create the Taylor coefficients for the Bi-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            A tuple containing the inputs to the three 4-jets.
        """
        D = self.in_dim

        # first 4-jet
        X1_0 = jet.utils.replicate(x, D)
        X1_2 = zeros(D, *self.in_shape, **self.in_meta)
        X1_3 = zeros(D, *self.in_shape, **self.in_meta)
        X1_4 = zeros(D, *self.in_shape, **self.in_meta)

        X1_1 = 4 * eye(D, **self.in_meta).reshape(D, *self.in_shape)

        C1 = (X1_0, X1_1, X1_2, X1_3, X1_4)

        # second 4-jet
        X2_0 = jet.utils.replicate(x, D * (D - 1))
        X2_2 = zeros(D * (D - 1), *self.in_shape, **self.in_meta)
        X2_3 = zeros(D * (D - 1), *self.in_shape, **self.in_meta)
        X2_4 = zeros(D * (D - 1), *self.in_shape, **self.in_meta)

        X2_1 = zeros(D, D - 1, D, **self.in_meta)
        for i in range(D):
            not_i = [j for j in range(D) if i != j]
            for j_idx, j in enumerate(not_i):
                X2_1[i, j_idx, i] = 3
                X2_1[i, j_idx, j] = 1
        X2_1 = X2_1.reshape(D * (D - 1), *self.in_shape)

        C2 = (X2_0, X2_1, X2_2, X2_3, X2_4)

        # third 4-jet
        X3_0 = jet.utils.replicate(x, D * (D - 1) // 2)
        X3_2 = zeros(D * (D - 1) // 2, *self.in_shape, **self.in_meta)
        X3_3 = zeros(D * (D - 1) // 2, *self.in_shape, **self.in_meta)
        X3_4 = zeros(D * (D - 1) // 2, *self.in_shape, **self.in_meta)

        X3_1 = zeros(D * (D - 1) // 2, D, **self.in_meta)
        counter = 0
        for i in range(D - 1):
            for j in range(i + 1, D):
                X3_1[counter, i] = 2
                X3_1[counter, j] = 2
                counter += 1
        assert counter == D * (D - 1) // 2
        X3_1 = X3_1.reshape(D * (D - 1) // 2, *self.in_shape)

        C3 = (X3_0, X3_1, X3_2, X3_3, X3_4)

        return C1, C2, C3
