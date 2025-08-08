"""Definitions of synthetic coefficient functions for illustration purposes."""

from torch import Tensor, arange, einsum, zeros
from torch.nn.functional import pad


def apply_S_func_diagonal_increments(x: Tensor, V: Tensor) -> Tensor:
    """Apply a synthetic coefficient factor S(x).T for weighting the Laplacian to V.

    The factor S(x) relates to the coefficient tensor C(x) via C(x) = S(x) @ S(x).T.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.
        V: The matrix onto which S(x) is applied. Has shape `(K, rank_C)` where `K`
            is the number of columns.

    Returns:
        The coefficient factor S(x).T applied to V. Has shape `(K, *x.shape)`.
    """
    rank_C = x.numel()
    S = (arange(rank_C, device=x.device, dtype=x.dtype) + 1).sqrt()
    SV = einsum("c,kc->kc", S, V)

    # if rank_C < D, we have to add zero padding to satisfy the output dimension
    D = x.numel()
    if rank_C < D:
        padding = (0, 0, 0, D - rank_C)
        SV = pad(SV, padding)

    return SV.reshape(V.shape[0], *x.shape)


def C_func_diagonal_increments(x: Tensor) -> Tensor:
    """Compute a synthetic coefficient tensor C(x) for weighting the Laplacian.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.

    Returns:
        The coefficient tensor as a tensor of shape `(*x.shape, *x.shape)`.
    """
    D = rank_C = x.numel()

    C = zeros(D, dtype=x.dtype, device=x.device)
    C[:rank_C] = arange(rank_C, dtype=x.dtype, device=x.device) + 1
    C = C.diag().reshape(*x.shape, *x.shape)

    return C
