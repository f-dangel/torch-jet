"""Definitions of synthetic coefficient functions for illustration purposes."""

from functools import partial
from typing import Callable

from torch import Tensor, arange, zeros
from torch.nn.functional import pad


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
    SV = S.unsqueeze(0) * V

    # if rank_C < D, we have to add zero padding to satisfy the output dimension
    D = fx_info["in_shape"].numel()
    if rank_C < D:
        padding = (0, D - rank_C, 0, 0)
        SV = pad(SV, padding)

    return SV.reshape(fx_info["num_jets"], *fx_info["in_shape"])


def C_func_diagonal_increments(x: Tensor, rank_ratio: float = 1.0) -> Tensor:
    """Compute a synthetic coefficient tensor C(x) for weighting the Laplacian.

    Args:
        x: Argument at which the weighted Laplacian is evaluated.
        rank_ratio: The ratio of the rank of the coefficient tensor to the number of
            elements in `x`. If `rank_ratio` is 1.0, the coefficient tensor is full
            rank, i.e. `rank_C = x.numel()`. If `rank_ratio` is less than 1.0, the
            coefficient tensor is low-rank, i.e. `rank_C = rank_ratio * x.numel()`.

    Returns:
        The coefficient tensor as a tensor of shape `(*x.shape, *x.shape)`.
    """
    D = x.numel()
    rank_C = max(int(rank_ratio * D), 1)

    C = zeros(D, dtype=x.dtype, device=x.device)
    C[:rank_C] = arange(rank_C, dtype=x.dtype, device=x.device) + 1
    C = C.diag().reshape(*x.shape, *x.shape)

    return C


def get_weighting(
    dummy_x: Tensor,
    weights: str | None | tuple[str, float],
    randomization: tuple[str, int] | None = None,
) -> tuple[Callable[[Tensor, Tensor], Tensor], int] | None:
    """Set up the `weighting` argument.

    Args:
        dummy_x: A dummy input tensor to infer the shape and device.
        weights: A string specifying the type of weighting to use. If `None`, the
            standard Laplacian is computed.
            Can also be a tuple of (type, rank_ratio).
        randomization: A tuple specifying the randomization distribution and number
            of samples, e.g. `("normal", 100)`. If `None`, no randomization is applied.

    Returns:
        A tuple containing the function that applies the weighting and the rank of
        the coefficient tensor, or `None` if no weighting is applied.

    Raises:
        ValueError: If the provided weighting option is not supported.
    """
    if weights == "diagonal_increments":
        weights = ("diagonal_increments", 1.0)

    # determine the Laplacian's weighting
    if (
        isinstance(weights, tuple)
        and len(weights) == 2
        and weights[0] == "diagonal_increments"
    ):
        _, rank_ratio = weights
        rank_weighting = max(int(rank_ratio * dummy_x.numel()), 1)
        fx_info = {
            "in_shape": dummy_x.shape,
            "device": dummy_x.device,
            "dtype": dummy_x.dtype,
            "rank_C": rank_weighting,
            "num_jets": rank_weighting if randomization is None else randomization[1],
        }
        apply_weighting = partial(apply_S_func_diagonal_increments, fx_info=fx_info)
        return apply_weighting, rank_weighting

    elif weights is None:
        return None

    raise ValueError(f"Unknown weights option {weights=}.")
