"""Implements a function transform that computes the Laplacian via jets."""

from typing import Callable

from torch import Tensor, eye, zeros_like

from jet._jet import _uncollapsed_via_vmap, jet
from jet.utils import (
    PyTree,
    require_single_tensor_input,
    require_single_tensor_output,
    sample,
    validate_randomization,
)

SUPPORTED_DISTRIBUTIONS = ["normal", "rademacher"]


def laplacian(
    f: Callable[[Tensor], Tensor],
    mock_args: tuple[PyTree[Tensor], ...],
    randomization: tuple[str, int] | None = None,
    weighting: tuple[Callable[[Tensor, Tensor], Tensor], int] | None = None,
    collapsed: bool = True,
) -> Callable[[*tuple[PyTree[Tensor], ...]], Tensor]:
    r"""Transform f into a function that computes lap(f(x)).

    The Laplacian of a function $f(\mathbf{x}) \in \mathbb{R}$ with
    $\mathbf{x} \in \mathbb{R}^D$ is defined as the Hessian trace, or

    $$
    \Delta f(\mathbf{x})
    =
    \sum_{d=1}^D
    \frac{\partial^2 f(\mathbf{x})}{\partial x_d^2} \in \mathbb{R}\,.
    $$

    For functions that produce vectors or tensors, the Laplacian
    is defined per output component and has the same shape as $f(\mathbf{x})$.

    Only single-tensor functions (one tensor in, one tensor out) are supported.

    Args:
        f: The function whose Laplacian is computed. Must consume and return a
            single tensor.
        mock_args: Mock positional arguments for tracing ``f``, provided as a
            tuple matching ``f``'s positional arguments. Does not need to be the
            actual input; only shapes and dtypes matter. Currently must be a
            one-tuple of a single tensor.
        randomization: Optional tuple containing the distribution type and number
            of samples for randomized Laplacian. If provided, the Laplacian will
            be computed using Monte-Carlo sampling. The first element is the
            distribution type (e.g., 'normal', 'rademacher'), and the second is the
            number of samples to use.
        weighting: A tuple specifying how the second-order derivatives should be
            weighted. This is described by a coefficient tensor C(x) of shape
            `[*D, *D]`. The first entry is a function (x, V) -> V @ S(x).T that
            applies the symmetric factorization S(x) of the weights
            C(x) = S(x) @ S(x).T at the input x to the matrix V. S(x) has shape
            `[*D, rank_C]` while V is `[K, rank_C]` with arbitrary `K`. The second
            entry specifies `rank_C`. If `None`, then the weightings correspond to
            the identity matrix (i.e. computing the standard Laplacian).
        collapsed: Whether to use collapsed Taylor mode. If ``True``
            (default), uses the collapsed dispatch path
            (``JetInterpreter(..., collapsed=True)``) that directly propagates
            the summed second-order coefficient. If ``False``, propagates full
            2-jets over all directions via ``vmap`` and sums afterward.
            Collapsed mode is the more efficient default: propagating the
            summed coefficient moves smaller tensors through the graph.

    Returns:
        A plain Python callable ``lap_f(*args)`` that maps ``x → lap(f(x))``.
        To bake the operator into an FX ``GraphModule`` (for graph passes,
        ``torch.compile``, etc.), apply :func:`capture_graph` yourself.

    Examples:
        >>> from torch import manual_seed, rand, zeros
        >>> from torch.func import hessian
        >>> from torch.nn import Linear, Tanh, Sequential
        >>> from jet import laplacian
        >>> _ = manual_seed(0) # make deterministic
        >>> f = Sequential(Linear(3, 1), Tanh())
        >>> x0 = rand(3)
        >>> # Compute the Laplacian via Taylor mode
        >>> lap = laplacian(f, (zeros(3),))(x0)
        >>> assert lap.shape == f(x0).shape
        >>> # Compute the Laplacian with PyTorch's autodiff (Hessian trace)
        >>> lap_pt = hessian(f)(x0).squeeze(0).trace().unsqueeze(0)
        >>> assert lap.shape == lap_pt.shape
        >>> assert lap_pt.allclose(lap)
    """
    mock_x = require_single_tensor_input(mock_args, "laplacian")
    in_shape = mock_x.shape
    in_dim = mock_x.numel()

    rank_weightings = in_dim if weighting is None else weighting[1]

    validate_randomization(randomization, SUPPORTED_DISTRIBUTIONS)

    num_jets = rank_weightings if randomization is None else randomization[1]
    apply_weightings = (
        (lambda x, V: V.reshape(num_jets, *in_shape))
        if weighting is None
        else weighting[0]
    )

    cjet_f = (
        jet(f, mock_args, collapsed=True)
        if collapsed
        else _uncollapsed_via_vmap(f, mock_args, randomization)
    )

    def lap_f(*args: PyTree[Tensor]) -> Tensor:
        """Compute the (weighted and/or randomized) Laplacian of f at x.

        Args:
            *args: Positional arguments for ``f`` (currently a single tensor
                matching the mock input's shape).

        Returns:
            The (weighted and/or randomized) Laplacian. Has the same shape as
                ``f(x)``.

        Raises:
            ValueError: If the input shape does not match the mock input shape.
        """
        (x,) = args
        if x.shape != in_shape:
            raise ValueError(f"Expected input shape {in_shape}, got {x.shape}.")

        # Set up first Taylor coefficients
        shape = (num_jets, rank_weightings)
        in_meta = {"dtype": x.dtype, "device": x.device}
        V = (
            eye(rank_weightings, **in_meta)
            if randomization is None
            else sample(x, randomization[0], shape)
        )
        X1 = apply_weightings(x, V)
        z = zeros_like(x)

        _, _, F2 = require_single_tensor_output(cjet_f((x, X1, z)), "laplacian")

        if randomization is not None:
            monte_carlo_scaling = 1.0 / randomization[1]
            F2 = F2 * monte_carlo_scaling

        return F2

    return lap_f
