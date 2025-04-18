"""Module that computes the weighted Laplacian via jets and can be simplified."""

from typing import Callable, Tuple

from torch import Tensor, arange, zeros
from torch.fx import wrap

from jet.laplacian import Laplacian
from jet.utils import replicate


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
        S_func: Callable[[Tensor], Tensor],
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
            S_func: Function that computes the matrix S(x) such that the coefficient
                matrix C(x) = S(x) @ S(x).T. If is_batched is True, then S(x) must
                return a tensor of shape (batch_size, *x.shape[1:], rank_C), otherwise
                it must return a tensor of shape (*x.shape, rank_C).
        """
        super().__init__(f, dummy_x, is_batched)
        self.S_func = S_func
        # determine how many columns S has, i.e. the rank of C, which determines the
        # number of vectors we have to feed into the jet
        self.num_vectors = S_func(dummy_x).shape[-1]

    def set_up_taylor_coefficients(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Create the Taylor coefficients for the MC-Laplacian computation.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
                passed in the constructor.

        Returns:
            The three input tensors to the 2-jet that computes the MC-Laplacian.
        """
        X0 = replicate(x, self.num_vectors)
        X2 = zeros(self.num_vectors, *self.x_shape, **self.x_kwargs)

        # compute the coefficient's factorization C = S @ S.T
        S = self.S_func(x)
        X1 = S.movedim(-1, 0)

        return X0, X1, X2


# Definitions of synthetic coefficient functions for illustration purposes


def S_func_diagonal_increments(x: Tensor, is_batched: bool) -> Tensor:
    """Compute a synthetic coefficient factor S(x) for weighting the Laplacian.

    The factor S(x) relates to the coefficient tensor C(x) via C(x) = S(x) @ S(x).T.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.
        is_batched: Whether `x` is batched.

    Returns:
        The coefficient factor as a tensor of shape `(batch_size, *x.shape[1:],
            rank_C)` if `is_batched` is True, otherwise of shape `(*x.shape, rank_C)`.
    """
    unbatched = x.shape[1:] if is_batched else x.shape
    D = (x.shape[1:] if is_batched else x.shape).numel()
    rank_C = D

    S = zeros(D, rank_C, dtype=x.dtype, device=x.device)
    idx = arange(rank_C, device=x.device)
    S[idx, idx] = (arange(rank_C, dtype=x.dtype, device=x.device) + 1).sqrt()
    S = S.reshape(*unbatched, rank_C)

    if is_batched:
        batch_size = x.shape[0]
        S = S.unsqueeze(0).expand(batch_size, *unbatched, rank_C)

    return S


def C_func_diagonal_increments(x: Tensor, is_batched: bool) -> Tensor:
    """Compute a synthetic coefficient tensor C(x) for weighting the Laplacian.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.
        is_batched: Whether `x` is batched.

    Returns:
        The coefficient tensor as a tensor of shape `(batch_size, *x.shape[1:],
            *x.shape[1:])` if `is_batched` is True, otherwise of shape
            `(*x.shape, *x.shape)`.
    """
    unbatched = x.shape[1:] if is_batched else x.shape
    D = (x.shape[1:] if is_batched else x.shape).numel()
    rank_C = D

    C = zeros(D, dtype=x.dtype, device=x.device)
    C[:rank_C] = arange(rank_C, dtype=x.dtype, device=x.device) + 1
    C = C.diag().reshape(*unbatched, *unbatched)

    if is_batched:
        batch_size = x.shape[0]
        C = C.unsqueeze(0).expand(batch_size, *unbatched, *unbatched)

    return C


# tell `torch.fx` to trace `S_func_diagonal_increments as one` node
# (required for simplification)
wrap(S_func_diagonal_increments)
