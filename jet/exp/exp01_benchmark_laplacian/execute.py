"""Script that carries out a measurement of peak memory and run time."""

from argparse import ArgumentParser, Namespace
from functools import partial
from os import makedirs, path
from sys import platform
from time import perf_counter
from typing import Callable

from einops import einsum
from torch import (
    Tensor,
    allclose,
    device,
    dtype,
    eye,
    float64,
    manual_seed,
    no_grad,
    rand,
)
from torch import compile as torch_compile
from torch.func import hessian, jacrev, jvp, vmap
from torch.nn import Linear, Sequential, Tanh

import jet
from jet.bilaplacian import bilaplacian as jet_bilaplacian
from jet.exp.utils import measure_peak_memory, measure_time, to_string
from jet.laplacian import laplacian as jet_laplacian
from jet.simplify import common_subexpression_elimination
from jet.tracing import capture_graph
from jet.ttc_coefficients import compute_all_gammas
from jet.utils import run_seeded, sample
from jet.weighted_laplacian import get_weighting

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
RAWDIR = path.join(HEREDIR, "raw")
makedirs(RAWDIR, exist_ok=True)

ON_MAC = platform == "darwin"

# Define supported PyTorch architectures
SUPPORTED_ARCHITECTURES = {
    "tanh_mlp_768_768_512_512_1": lambda dim: Sequential(
        Linear(dim, 768),
        Tanh(),
        Linear(768, 768),
        Tanh(),
        Linear(768, 512),
        Tanh(),
        Linear(512, 512),
        Tanh(),
        Linear(512, 1),
    )
}

# Define supported strategies
SUPPORTED_STRATEGIES = ["hessian_trace", "jet_naive", "jet_simplified"]
# Verify other implementations against the result of this baseline
BASELINE = "hessian_trace"


def hessian_trace_laplacian(
    f: Callable[[Tensor], Tensor], dummy_x: Tensor
) -> Callable[[Tensor], Tensor]:
    """Generate a function that computes the Laplacian of f by tracing the Hessian.

    Args:
        f: The function whose Laplacian we want to compute. The function should take
            the input tensor as arguments and return the output tensor.
        dummy_x: A dummy input tensor to determine the input dimensions.

    Returns:
        A function that computes the Laplacian of f at the input tensor X.
    """
    hess_f = hessian(f)

    # trace with einsum to support Laplacians of functions with non-scalar output
    dims = " ".join([f"d{i}" for i in range(dummy_x.ndim)])
    tr_equation = f"... {dims} {dims} -> ..."

    def laplacian(x: Tensor) -> Tensor:
        """Compute the Laplacian of f on an un-batched input.

        Args:
            x: The input tensor.

        Returns:
            The Laplacian of f at x. Has the same shape as f(x).
        """
        return einsum(hess_f(x), tr_equation)

    return laplacian


def vector_hessian_vector_product(
    f: Callable[[Tensor], Tensor], dummy_x: Tensor
) -> Callable[[Tensor, Tensor], Tensor]:
    """Generate function to compute the vector-Hessian-vector product of f at x with v.

    Args:
        f: The function whose vector-Hessian-vector product we want to compute.
            It should take the input tensor as argument and return the output tensor.
        dummy_x: An un-batched dummy input tensor to determine the input dimensions.

    Returns:
        A function that computes the vector-Hessian-vector product of f at the input
        tensor x given x and the vector. Has the same shape as f(x).
    """
    # perform the contraction of the HVP and the vector with einsum to support
    # functions with non-scalar output
    sum_dims = dummy_x.ndim
    in_dims = " ".join([f"d{i}" for i in range(sum_dims)])
    out_dims = "..."
    equation = f"{out_dims} {in_dims}, {in_dims} -> {out_dims}"

    def vhv(x: Tensor, v: Tensor) -> Tensor:
        """Compute the vector-Hessian-vector product of f with v evaluated at x.

        Args:
            x: The input to the function at which the vector-Hessian-vector product
                is computed.
            v: The vector to compute the vector-Hessian-vector product with.
                Has same shape as `x`.

        Returns:
            The vector-Hessian-vector product. Has the same shape as f(x).
        """
        grad_func = jacrev(f)
        _, hvp = jvp(grad_func, (x,), (v,))

        return einsum(hvp, v, equation)

    return vhv


def vector_hessian_vector_product_laplacian(
    f: Callable[[Tensor], Tensor],
    dummy_x: Tensor,
    randomization: tuple[str, int] | None,
    weighting: tuple[Callable[[Tensor, Tensor], Tensor], int] | None = None,
):
    """Create a function that computes the Laplacian of f via VHVPs.

    Args:
        f: The function whose Laplacian we want to compute. The function should take
            the input tensor as arguments and return the output tensor.
        dummy_x: A dummy input tensor to determine the input dimensions.
        randomization: If not `None`, a tuple containing the distribution type and
            number of samples for randomized Laplacian. The first element is the
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

    Returns:
        A function that computes the (weighted and/or randomized) Laplacian of f at
        the input tensor x.
    """
    D = dummy_x.numel()

    # determine the number of jets that need to be computed
    if randomization is None:
        num_jets = D if weighting is None else weighting[1]
    else:
        num_jets = randomization[1]

    # determines dimension of random vectors
    rank_weightings = D if weighting is None else weighting[1]

    apply_weightings = (
        (lambda x, V: V.reshape(num_jets, *dummy_x.shape))
        if weighting is None
        else weighting[0]
    )

    vhvp = vector_hessian_vector_product(f, dummy_x)
    # along multiple vectors in parallel
    VhVp = vmap(vhvp, in_dims=(None, 0))

    def laplacian(x: Tensor) -> Tensor:
        """Compute the (weighted and/or randomized) Laplacian of f at x.

        Args:
            x: The input tensor at which to compute the Laplacian.

        Returns:
            The Laplacian of f at x. Has the same shape as f(x).
        """
        V = (
            eye(rank_weightings, device=x.device, dtype=x.dtype)
            if randomization is None
            else sample(x, randomization[0], (num_jets, rank_weightings))
        )
        S = apply_weightings(x, V)
        SHS = VhVp(x, S)
        return SHS.sum(0) if randomization is None else SHS.mean(0)

    return laplacian


def naive_jet_laplacian(
    f: Callable[[Tensor], Tensor],
    mock_x: Tensor,
    randomization: tuple[str, int] | None = None,
    weighting: tuple[Callable[[Tensor, Tensor], Tensor], int] | None = None,
) -> Callable[[Tensor], Tensor]:
    """Compute the Laplacian using standard (non-collapsed) jets + vmap + sum.

    This is the naive approach: propagate R full 2-jets, then sum the 2nd
    coefficient over all directions.

    Args:
        f: The function whose Laplacian is computed.
        mock_x: A mock input tensor for tracing.
        randomization: Optional (distribution, num_samples) tuple.
        weighting: Optional weighting specification.

    Returns:
        A function that takes x and returns the Laplacian of f at x.
    """
    from torch import zeros_like

    in_shape = mock_x.shape
    in_dim = mock_x.numel()

    (apply_weightings, rank_weightings) = (
        (lambda x, V: V.reshape(num_jets, *in_shape), in_dim)
        if weighting is None
        else weighting
    )

    num_jets = rank_weightings if randomization is None else randomization[1]
    jet_f = jet.jet(f, 2, (mock_x,))

    def lap_f(x: Tensor) -> Tensor:
        shape = (num_jets, rank_weightings)
        in_meta = {"dtype": x.dtype, "device": x.device}
        V = (
            eye(rank_weightings, **in_meta)
            if randomization is None
            else sample(x, randomization[0], shape)
        )
        X1 = apply_weightings(x, V)

        vmapped = vmap(
            lambda x1: jet_f((x,), ((x1,), (zeros_like(x),))),
            randomness="different",
            out_dims=(None, (0, 0)),
        )
        _, (_, F2) = vmapped(X1)
        if randomization is not None:
            return F2.sum(0) / randomization[1]
        return F2.sum(0)

    return lap_f


def naive_jet_bilaplacian(
    f: Callable[[Tensor], Tensor],
    mock_x: Tensor,
    randomization: tuple[str, int] | None = None,
) -> Callable[[Tensor], Tensor]:
    """Compute the Bi-Laplacian using standard (non-collapsed) jets + vmap + sum.

    Args:
        f: The function whose Bi-Laplacian is computed.
        mock_x: A mock input tensor for tracing.
        randomization: Optional (distribution, num_samples) tuple.

    Returns:
        A function that takes x and returns the Bi-Laplacian of f at x.
    """
    from torch import triu_indices, zeros_like

    in_shape = mock_x.shape
    in_dim = mock_x.numel()
    jet_f = jet.jet(f, 4, (mock_x,))

    def _set_up_taylor_coefficients(x: Tensor):
        D = in_dim
        in_meta = {"dtype": x.dtype, "device": x.device}
        E = eye(D, **in_meta)
        C1 = (4 * E).reshape(D, *in_shape)
        mask = ~eye(D, dtype=bool, device=x.device)
        i_idx, j_idx = mask.nonzero(as_tuple=True)
        C2 = (3 * E[i_idx] + E[j_idx]).reshape(D * (D - 1), *in_shape)
        i_idx, j_idx = triu_indices(D, D, offset=1)
        C3 = (2 * E[i_idx] + 2 * E[j_idx]).reshape(D * (D - 1) // 2, *in_shape)
        return C1, C2, C3

    def bilap_f(x: Tensor) -> Tensor:
        z = zeros_like(x)
        vmapped = vmap(
            lambda x1: jet_f((x,), ((x1,), (z,), (z,), (z,))),
            randomness="different",
            out_dims=(None, (0, 0, 0, 0)),
        )

        if randomization is not None:
            distribution, num_samples = randomization
            X1 = sample(x, distribution, (num_samples, *in_shape))
            _, (_, _, _, F4) = vmapped(X1)
            return F4.sum(0) / (3 * num_samples)

        C1, C2, C3 = _set_up_taylor_coefficients(x)
        D = in_dim

        gamma_4_4 = float(compute_all_gammas((4,))[(4,)])
        gammas = compute_all_gammas((2, 2))
        gamma_4_0 = float(gammas[(4, 0)])

        _, (_, _, _, F4_1) = vmapped(C1)
        factor1 = (gamma_4_4 + 2 * (D - 1) * gamma_4_0) / 24
        term1 = factor1 * F4_1.sum(0)

        if D == 1:
            return term1

        gamma_3_1 = float(gammas[(3, 1)])
        _, (_, _, _, F4_2) = vmapped(C2)
        factor2 = 2 * gamma_3_1 / 24
        term2 = factor2 * F4_2.sum(0)

        gamma_2_2 = float(gammas[(2, 2)])
        _, (_, _, _, F4_3) = vmapped(C3)
        factor3 = 2 * gamma_2_2 / 24
        term3 = factor3 * F4_3.sum(0)

        return term1 + term2 + term3

    return bilap_f


def laplacian_function(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    strategy: str,
    randomization: tuple[str, int] | None = None,
    weighting: tuple[Callable[[Tensor, Tensor], Tensor], int] | None = None,
) -> Callable[[], Tensor]:
    """Construct a function to compute the Laplacian using different strategies.

    Args:
        f: The function to compute the Laplacian of. Processes an un-batched tensor.
        X: The input tensor at which to compute the Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The Laplacian is computed by tracing the Hessian.
              The Hessian is computed via forward-over-reverse mode autodiff.
            - `'jet_naive'`: The Laplacian is computed using standard (non-collapsed)
              Taylor mode with vmap + sum.
            - `'jet_simplified'`: The Laplacian is computed using collapsed Taylor mode
              with CSE and dead code elimination.
        randomization: If not `None`, a tuple containing the distribution type and
            number of samples for randomized Laplacian. The first element is the
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

    Returns:
        A function that computes the Laplacian of the function f at the input tensor X.
        The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy is not supported.
    """
    # Set up the function that computes the Laplacian on an un-batched datum
    dummy_x = X[0] if is_batched else X

    if strategy == "hessian_trace":
        if weighting is None and randomization is None:
            laplacian = hessian_trace_laplacian(f, dummy_x)
        else:
            laplacian = vector_hessian_vector_product_laplacian(
                f, dummy_x, randomization, weighting
            )

    elif strategy == "jet_naive":
        lap_fn = naive_jet_laplacian(
            f, dummy_x, randomization=randomization, weighting=weighting
        )
        lap_fn = capture_graph(lap_fn, dummy_x)
        common_subexpression_elimination(lap_fn.graph)
        lap_fn.recompile()
        laplacian = lap_fn

    elif strategy == "jet_simplified":
        lap_fn = jet_laplacian(
            f, dummy_x, randomization=randomization, weighting=weighting
        )
        common_subexpression_elimination(lap_fn.graph)
        lap_fn.recompile()
        laplacian = lambda x: lap_fn(x)[2]  # noqa: E731

    else:
        raise ValueError(f"Unsupported {strategy=}. {SUPPORTED_STRATEGIES=}.")

    if is_batched:
        laplacian = vmap(
            laplacian, randomness="error" if randomization is None else "different"
        )
    return partial(laplacian, X)


def vector_hessian_vector_product_bilaplacian(
    f: Callable[[Tensor], Tensor], dummy_x: Tensor, randomization: tuple[str, int]
) -> Callable[[Tensor], Tensor]:
    """Create a function that computes the Bi-Laplacian of f via VHVPs.

    Args:
        f: The function whose Bi-Laplacian we want to compute. The function should take
            the input tensor as arguments and return the output tensor.
        dummy_x: A dummy input tensor to determine the input dimensions.
        randomization: A tuple containing the distribution type and number of samples
            for randomized Bi-Laplacian. The first element is the distribution type
            (must be 'normal'), and the second is the number of samples to use.

    Returns:
        A function that computes the Bi-Laplacian of f at the input tensor x.
    """
    d2f_vv = vector_hessian_vector_product(f, dummy_x)
    d4f_vvvv = lambda x, v: vector_hessian_vector_product(  # noqa: E731
        lambda x: d2f_vv(x, v), dummy_x
    )(x, v)

    # vmap over vectors
    d4f_VVVV = vmap(d4f_vvvv, in_dims=(None, 0))

    (distribution, num_samples) = randomization
    shape = (num_samples, *dummy_x.shape)

    def bilaplacian(x: Tensor) -> Tensor:
        """Compute the exact/randomized Bi-Laplacian of f at x.

        Args:
            x: The input tensor at which to compute the Bi-Laplacian.

        Returns:
            The Bi-Laplacian of f at x. Has the same shape as f(x).
        """
        V = sample(x, distribution, shape)
        return d4f_VVVV(x, V).mean(0) / 3

    return bilaplacian


def bilaplacian_function(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    strategy: str,
    randomization: tuple[str, int] | None = None,
) -> Callable[[], Tensor]:
    """Construct a function to compute the Bi-Laplacian using different strategies.

    Args:
        f: The function to compute the Bi-Laplacian of. Processes an un-batched tensor.
        X: The input tensor at which to compute the Bi-Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the Bi-Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The Bi-Laplacian is computed by computing the tensor
              of fourth-order derivatives, then summing the necessary entries. The
              derivative tensor is computed as Hessian of the Hessian with PyTorch.
            - `'jet_naive'`: The Bi-Laplacian is computed using standard (non-collapsed)
              Taylor mode with vmap + sum.
            - `'jet_simplified'`: The Bi-Laplacian is computed using collapsed Taylor
              mode with CSE and dead code elimination.
        randomization: If not `None`, a tuple containing the distribution type and
            number of samples for randomized Bi-Laplacian. The first element is the
            distribution type (e.g., 'normal'), and the second is the number of samples
            to use.

    Returns:
        A function that computes the Bi-Laplacian of the function f at the input
        tensor X. The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy is not supported.
    """
    dummy_x = X[0] if is_batched else X

    if strategy == "hessian_trace":
        if randomization is None:
            laplacian = hessian_trace_laplacian(f, dummy_x)
            bilap_fn = hessian_trace_laplacian(laplacian, dummy_x)
        else:
            bilap_fn = vector_hessian_vector_product_bilaplacian(
                f, dummy_x, randomization
            )

    elif strategy == "jet_naive":
        bilap_fn = naive_jet_bilaplacian(f, dummy_x, randomization=randomization)
        bilap_fn = capture_graph(bilap_fn, dummy_x)
        common_subexpression_elimination(bilap_fn.graph)
        bilap_fn.recompile()

    elif strategy == "jet_simplified":
        bilap_fn = jet_bilaplacian(f, dummy_x, randomization=randomization)
        common_subexpression_elimination(bilap_fn.graph)
        bilap_fn.recompile()

    else:
        raise ValueError(f"Unsupported strategy: {strategy}.")

    if is_batched:
        bilap_fn = vmap(
            bilap_fn, randomness="error" if randomization is None else "different"
        )

    return partial(bilap_fn, X)


def setup_architecture(
    architecture: str, dim: int, dev: device, dt: dtype, seed: int = 0
) -> Callable[[Tensor], Tensor]:
    """Set up a neural network architecture based on the specified configuration.

    Args:
        architecture: The architecture identifier.
        dim: The input dimension for the architecture.
        dev: The device to place the model on.
        dt: The data type to use.
        seed: The random seed for initialization. Default is `0`.

    Returns:
        A PyTorch model of the specified architecture.
    """
    manual_seed(seed)
    return SUPPORTED_ARCHITECTURES[architecture](dim).to(device=dev, dtype=dt)


def savepath(rawdir: str = RAWDIR, **kwargs: str | int) -> str:
    """Generate a file path for saving measurement results.

    Args:
        rawdir: The directory where the results will be saved. Default is the raw
            directory of the PyTorch benchmark.
        **kwargs: Key-value pairs representing the parameters of the measurement.

    Returns:
        A string representing the file path where the results will be saved.
    """
    filename = to_string(**kwargs)
    return path.join(rawdir, f"{filename}.csv")


def check_arg_conflict(args: Namespace):
    """Check if arguments are correctly specified.

    Args:
        args: The parsed arguments.

    Raises:
        ValueError: If the arguments are not mutually specified or unspecified.
        ValueError: If the `rank_ratio` argument's value is invalid.
    """
    distribution, num_samples = args.distribution, args.num_samples
    if (distribution is None) != (num_samples is None):
        raise ValueError(
            f"Arguments 'distribution' ({distribution}) and 'num_samples'"
            f" ({num_samples}) are mutually required."
        )

    if args.operator == "weighted-laplacian":
        if args.rank_ratio is None:
            raise ValueError(
                f"Argument 'rank_ratio' ({args.rank_ratio}) is required for "
                f"operator {args.operator!r}."
            )
        elif not 0 < args.rank_ratio <= 1:
            raise ValueError(
                f"Argument 'rank_ratio' ({args.rank_ratio}) must be in range (0,1]."
            )


def get_function_and_description(
    operator: str,
    strategy: str,
    distribution: str | None,
    num_samples: int | None,
    net: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    compiled: bool,
    rank_ratio: float | None = None,
) -> tuple[Callable[[], Tensor], Callable[[], Tensor], str]:
    """Determine the function and its description based on the operator and strategy.

    Args:
        operator: The operator to be used, either 'laplacian', 'weighted-laplacian',
            or 'bilaplacian'.
        strategy: The strategy to be used for computation.
        distribution: The distribution type, if any.
        num_samples: The number of samples, if any.
        net: The neural network model.
        X: The input tensor.
        is_batched: A flag indicating if the input is batched.
        compiled: A flag indicating if the function should be compiled.
        rank_ratio: Ratio of the rank to use for the coefficient matrix (∈ (0; 1]).
            Only used for weighted Laplacian.

    Returns:
        A tuple containing the function to compute the operator (differentiable),
        the function to compute the operator (non-differentiable), and a description
        string.
    """
    is_stochastic = distribution is not None and num_samples is not None

    description = f"{strategy}, {compiled=}"
    if is_stochastic:
        description += f", {distribution=}, {num_samples=}"
    if operator == "weighted-laplacian" and rank_ratio is not None:
        description += f", {rank_ratio=}"

    randomization = (
        (distribution, num_samples)
        if distribution is not None and num_samples is not None
        else None
    )

    # set up arguments
    args = (net, X, is_batched, strategy)
    kwargs = {"randomization": randomization}
    if operator == "weighted-laplacian":
        kwargs["weighting"] = get_weighting(
            X[0] if is_batched else X,
            ("diagonal_increments", rank_ratio),
            randomization=randomization,
        )

    factory = {
        "laplacian": laplacian_function,
        "weighted-laplacian": laplacian_function,
        "bilaplacian": bilaplacian_function,
    }[operator]

    func = factory(*args, **kwargs)

    @no_grad()
    def func_no() -> Tensor:
        """Non-differentiable computation.

        Returns:
            Value of the differentiable operator
        """
        return func()

    if compiled:
        compile_error = operator == "bilaplacian" and strategy == "hessian_trace"
        if ON_MAC:
            print("Skipping torch.compile due to MAC-incompatibility.")
        elif compile_error:
            print("Skipping torch.compile due to bug in torch.compile error.")
        else:
            print("Using torch.compile")
            func, func_no = torch_compile(func), torch_compile(func_no)
    else:
        print("Not using torch.compile")

    return func, func_no, description


def setup_input(
    batch_size: int, dim: int, dev: device, dt: dtype, seed: int = 1
) -> Tensor:
    """Set up the seeded input tensor for the neural network.

    Args:
        batch_size: The number of samples in the batch.
        dim: The dimensionality of the input tensor.
        dev: The device to place the tensor on.
        dt: The data type of the tensor.
        seed: The random seed for initialization. Default is `1`.

    Returns:
        A PyTorch tensor of shape (batch_size, dim) and specified data type and device.
    """
    manual_seed(seed)
    shape = (batch_size, dim)
    return rand(*shape, dtype=dt, device=dev)


def parse_args() -> Namespace:
    """Parse the benchmark script's command line arguments.

    Returns:
        The benchmark script's arguments.
    """
    parser = ArgumentParser("Parse arguments of measurement.")
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=set(SUPPORTED_ARCHITECTURES.keys()),
    )
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--strategy", type=str, required=True, choices=set(SUPPORTED_STRATEGIES)
    )
    parser.add_argument(
        "--distribution", required=False, choices={"normal", "rademacher"}
    )
    parser.add_argument("--num_samples", required=False, type=int)
    parser.add_argument("--device", type=str, choices={"cpu", "cuda"}, required=True)
    parser.add_argument(
        "--operator",
        type=str,
        choices={"laplacian", "weighted-laplacian", "bilaplacian"},
        required=True,
    )
    parser.add_argument(
        "--compiled",
        action="store_true",
        default=False,
        help="Whether to use torch.compile for the functions",
    )
    parser.add_argument(
        "--rank_ratio",
        type=float,
        required=False,
        help="Rank ratio for weighted Laplacian (between 0 and 1)",
    )

    # parse and check validity
    args = parser.parse_args()
    check_arg_conflict(args)

    return args


if __name__ == "__main__":
    args = parse_args()

    # set up the function that will be measured
    dev = device(args.device)
    dt = float64
    net = setup_architecture(args.architecture, args.dim, dev, dt)
    is_batched = True
    X = setup_input(args.batch_size, args.dim, dev, dt)

    start = perf_counter()
    func, func_no, description = get_function_and_description(
        args.operator,
        args.strategy,
        args.distribution,
        args.num_samples,
        net,
        X,
        is_batched,
        args.compiled,
        args.rank_ratio,
    )

    print(f"Setting up functions took: {perf_counter() - start:.3f} s.")

    is_cuda = args.device == "cuda"
    is_stochastic = args.distribution is not None and args.num_samples is not None
    op = args.operator.capitalize()

    # Carry out the measurements

    # 1) Peak memory with non-differentiable result
    mem_no = measure_peak_memory(
        func_no, f"{op} non-differentiable ({description})", is_cuda
    )

    # 2) Peak memory with differentiable result
    mem = measure_peak_memory(func, f"{op} ({description})", is_cuda)

    # 3) Run time
    mu, sigma, best = measure_time(func, f"{op} ({description})", is_cuda)

    # Sanity check: make sure that the results correspond to the baseline implementation
    #
    # NOTE Compilation may affect randomness in the baseline implementation differently
    #      than in the to-be-tested implementation. Therefore we cannot expect that
    #      their output matches.
    #
    # Check is carried out for deterministic, and un-compiled stochastic computations.
    if not is_stochastic or not args.compiled:
        print("Checking correctness against baseline.")
        with no_grad():
            result = run_seeded(func, 3)

        _, baseline_func_no, _ = get_function_and_description(
            args.operator,
            BASELINE,
            args.distribution,
            args.num_samples,
            net,
            X,
            is_batched,
            False,  # do not use compilation for ground truth
            args.rank_ratio,
        )
        baseline_result = run_seeded(baseline_func_no, 3)

        assert baseline_result.shape == result.shape, (
            f"Shapes do not match: {baseline_result.shape} != {result.shape}."
        )
        same = allclose(baseline_result, result)
        assert same, f"Results do not match: {result} != {baseline_result}."
        print("Results match.")

    # Write measurements to a file
    data = ", ".join([str(val) for val in [mem_no, mem, mu, sigma, best]])
    filename = savepath(**vars(args))
    with open(filename, "w") as f:
        print(f"Writing to {filename}.")
        f.write(data)
