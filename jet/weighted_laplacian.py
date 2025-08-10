"""Definitions of synthetic coefficient functions for illustration purposes."""

from functools import partial
from typing import Callable

from torch import Tensor, arange, zeros
from torch.nn.functional import pad

import jet.utils


def apply_S_func_diagonal_increments(x: Tensor, V: Tensor, fx_info: dict) -> Tensor:
    """Apply a synthetic coefficient factor S(x).T for weighting the Laplacian to V.

    The factor S(x) relates to the coefficient tensor C(x) via C(x) = S(x) @ S(x).T.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.
        V: The matrix onto which S(x) is applied. Has shape `(K, rank_C)` where `K`
            is the number of columns.
        fx_info: A dictionary that contains all information `torch.fx` cannot infer
            while tracing. This serves to make the function trace-able.

    Returns:
        The coefficient factor S(x).T applied to V. Has shape `(K, *x.shape)`.
    """
    rank_C = fx_info["rank_C"]
    S = (arange(rank_C, device=fx_info["device"], dtype=fx_info["dtype"]) + 1).sqrt()
    S = jet.utils.replicate(S, fx_info["V_rows"])
    print(S.shape, V.shape)
    SV = S * V

    # if rank_C < D, we have to add zero padding to satisfy the output dimension
    D = fx_info["in_shape"].numel()
    if rank_C < D:
        padding = (0, 0, 0, D - rank_C)
        SV = pad(SV, padding)

    return SV.reshape(fx_info["V_rows"], *fx_info["in_shape"])


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


def get_weighting(
    x: Tensor, weights: str | None, randomization: tuple[str, int] | None = None
) -> tuple[Callable[[Tensor, Tensor], Tensor], int] | None:
    # determine the Laplacian's weighting
    if weights == "diagonal_increments":
        fx_info = {
            "in_shape": x.shape,
            "device": x.device,
            "dtype": x.dtype,
            "rank_C": x.numel(),
            "V_rows": x.numel() if randomization is None else randomization[1],
        }
        apply_weighting = partial(apply_S_func_diagonal_increments, fx_info=fx_info)
        rank_weighting = x.numel()
        return apply_weighting, rank_weighting
    else:
        assert weights is None
        return None
