"""Script that carries out a measurement of peak memory and run time."""

from argparse import ArgumentParser, Namespace
from functools import partial
from os import makedirs, path
from sys import platform
from time import perf_counter
from typing import Callable, Union

from einops import einsum
from torch import (
    Tensor,
    allclose,
)
from torch import compile as torch_compile
from torch import (
    device,
    dtype,
    float64,
    manual_seed,
    no_grad,
    rand,
    randn,
)
from torch.func import hessian, jacrev, jvp, vmap
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import Bilaplacian, RandomizedBilaplacian
from jet.exp.utils import measure_peak_memory, measure_time, to_string
from jet.laplacian import Laplacian, RandomizedLaplacian
from jet.simplify import simplify
from jet.utils import rademacher
from jet.weighted_laplacian import (
    C_func_diagonal_increments,
    RandomizedWeightedLaplacian,
    WeightedLaplacian,
)

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


def laplacian_function(
    f: Callable[[Tensor], Tensor], X: Tensor, is_batched: bool, strategy: str
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
            - `'jet_naive'`: The Laplacian is computed using jets. The computation graph
                is simplified by propagating replication nodes.
            - `'jet_simplified'`: The Laplacian is computed using Taylor mode. The
              computation graph is simplified by propagating replications down, and
              summations up, the computation graph.

    Returns:
        A function that computes the Laplacian of the function f at the input tensor X.
        The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy is not supported.
    """
    if strategy == "hessian_trace":
        dummy_X = X[0] if is_batched else X
        laplacian = hessian_trace_laplacian(f, dummy_X)

        if is_batched:
            laplacian = vmap(laplacian)

        return lambda: laplacian(X)

    elif strategy in {"jet_naive", "jet_simplified"}:
        laplacian = Laplacian(f, X, is_batched)
        pull_sum_vmapped = strategy == "jet_simplified"
        laplacian = simplify(
            symbolic_trace(laplacian), pull_sum_vmapped=pull_sum_vmapped
        )

        return lambda: laplacian(X)[2]

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Supported: {SUPPORTED_STRATEGIES}."
        )


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
    dims = " ".join([f"d{i}" for i in range(sum_dims)])
    equation = f"... {dims}, {dims} -> ..."

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


def randomized_laplacian_function(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    strategy: str,
    distribution: str,
    num_samples: int,
) -> Callable[[], Tensor]:
    """Construct a function to compute the MC-Laplacian using different strategies.

    Args:
        f: The function to compute the Laplacian of. Processes an un-batched tensor.
        X: The input tensor at which to compute the Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The Laplacian is computed by tracing the Hessian.
              The Hessian is computed via forward-over-reverse mode autodiff.
            - `'jet_naive'`: The Laplacian is computed using jets. The computation graph
                is simplified by propagating replication nodes.
            - `'jet_simplified'`: The Laplacian is computed using Taylor mode. The
              computation graph is simplified by propagating replications down, and
              summations up, the computation graph.
        distribution: From which distribution to draw the random vectors. Supported
            values are `'normal'` and `'rademacher'`.
        num_samples: How many Monte-Carlo samples should be used by the estimation.

    Returns:
        A function that computes the randomized Laplacian of the function f at the input
        tensor X. The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy or distribution are not supported.
    """
    if distribution not in {"normal", "rademacher"}:
        raise ValueError(f"Unsupported distribution: {distribution!r}.")

    if strategy == "hessian_trace":

        dummy_x = X[0] if is_batched else X
        vhv = vector_hessian_vector_product(f, dummy_x)

        # vmap over data points and fix data
        if is_batched:
            vhv = vmap(vhv)
        vhv_fix_X = partial(vhv, X)

        # vmap over HVP
        VhV_vmap = vmap(vhv_fix_X)

        sample_func = {"normal": randn, "rademacher": rademacher}[distribution]
        V = sample_func(num_samples, *X.shape, device=X.device, dtype=X.dtype)

        return lambda: VhV_vmap(V).mean(0)

    if strategy in {"jet_naive", "jet_simplified"}:
        laplacian = RandomizedLaplacian(f, X, is_batched, num_samples, distribution)
        pull_sum_vmapped = strategy == "jet_simplified"
        laplacian = simplify(
            symbolic_trace(laplacian), pull_sum_vmapped=pull_sum_vmapped
        )
        return lambda: laplacian(X)[2]

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Supported: {SUPPORTED_STRATEGIES}."
        )


def weighted_laplacian_function(
    f: Callable[[Tensor], Tensor], X: Tensor, is_batched: bool, strategy: str
) -> Callable[[], Tensor]:
    """Construct a function to compute a weighted Laplacian using different strategies.

    Args:
        f: The function to compute the weighted Laplacian of.
            Processes an un-batched tensor.
        X: The input tensor at which to compute the weighted Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the weighted Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The Laplacian is computed by computing the Hessian,
                then weighting it. The Hessian is computed via forward-over-reverse
                mode autodiff.
            - `'jet_naive'`: The weighted Laplacian is computed using jets. The
                computation graph is simplified by propagating replication nodes.
            - `'jet_simplified'`: The weighted Laplacian is computed using Taylor mode.
                The computation graph is simplified by propagating replications down,
                and summations up, the computation graph.

    Returns:
        A function that computes the weighted Laplacian of the function f at the input
        tensor X. The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy is not supported.
    """
    if strategy == "hessian_trace":
        hess_f = hessian(f)
        C = C_func_diagonal_increments(X, is_batched)

        # weight with einsum to support Laplacians of functions with non-scalar output
        unbatched = X.ndim - 1 if is_batched else X.ndim
        dims1 = " ".join([f"i{i}" for i in range(unbatched)])
        dims2 = " ".join([f"j{j}" for j in range(unbatched)])
        tr_equation = f"... {dims1} {dims2}, {dims1} {dims2} -> ..."

        def weighted_laplacian(x: Tensor, c: Tensor) -> Tensor:
            """Compute the weighted Laplacian of f on an un-batched input.

            Args:
                x: The input tensor.
                c: The coefficient tensor. Must have shape (*x.shape, *x.shape).

            Returns:
                The weighted Laplacian of f at x. Has the same shape as f(x).
            """
            return einsum(hess_f(x), c, tr_equation)

        if is_batched:
            weighted_laplacian = vmap(weighted_laplacian)

        return lambda: weighted_laplacian(X, C)

    elif strategy in {"jet_naive", "jet_simplified"}:
        weighted_laplacian = WeightedLaplacian(f, X, is_batched, "diagonal_increments")
        pull_sum_vmapped = strategy == "jet_simplified"
        weighted_laplacian = simplify(
            symbolic_trace(weighted_laplacian), pull_sum_vmapped=pull_sum_vmapped
        )

        return lambda: weighted_laplacian(X)[2]

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Supported: {SUPPORTED_STRATEGIES}."
        )


def randomized_weighted_laplacian_function(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    strategy: str,
    distribution: str,
    num_samples: int,
) -> Callable[[], Tensor]:
    """Build function to compute the weighted MC-Laplacian with different strategies.

    Args:
        f: The function to compute the weighted MC-Laplacian of. Processes an
            un-batched tensor.
        X: The input tensor at which to compute the weighted MC-Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the weighted MC-Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The weighted MC-Laplacian is computed by vector-
                Hessian-vector products with random vectors via forward-over-reverse
                mode autodiff.
            - `'jet_naive'`: The weighted MC-Laplacian is computed using jets.
                The computation graph is simplified by propagating replication nodes.
            - `'jet_simplified'`: The weighted MC-Laplacian is computed using Taylor
                mode. The computation graph is simplified by propagating replications
                down, and summations up, the computation graph.
        distribution: From which distribution to draw the random vectors. Supported
            values are `'normal'` and `'rademacher'`.
        num_samples: How many Monte-Carlo samples should be used by the estimation.

    Returns:
        A function that computes the randomized weighted Laplacian of the function f
        at the input tensor X. The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy or distribution are not supported.
    """
    if distribution not in {"normal", "rademacher"}:
        raise ValueError(f"Unsupported distribution: {distribution!r}.")

    weighting = "diagonal_increments"
    H_dot_C = RandomizedWeightedLaplacian(
        f, X, is_batched, num_samples, distribution, weighting
    )

    if strategy == "hessian_trace":
        dummy_x = X[0] if is_batched else X
        vhv = vector_hessian_vector_product(f, dummy_x)

        # vmap over data points and fix data
        if is_batched:
            vhv = vmap(vhv)
        vhv_fix_X = partial(vhv, X)

        # vmap over VHVP
        VhV_vmap = vmap(vhv_fix_X)

        sample_func = {"normal": randn, "rademacher": rademacher}[distribution]
        rank_C = {
            "diagonal_increments": (X.shape[1:] if is_batched else X.shape).numel()
        }[weighting]
        sample_shape = (
            (num_samples, X.shape[0], rank_C) if is_batched else (num_samples, rank_C)
        )
        V = sample_func(*sample_shape, device=X.device, dtype=X.dtype)
        SV = H_dot_C.apply_S_func(X, V)

        return lambda: VhV_vmap(SV).mean(0)

    if strategy in {"jet_naive", "jet_simplified"}:
        pull_sum_vmapped = strategy == "jet_simplified"
        H_dot_C = simplify(symbolic_trace(H_dot_C), pull_sum_vmapped=pull_sum_vmapped)
        return lambda: H_dot_C(X)[2]

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Supported: {SUPPORTED_STRATEGIES}."
        )


def bilaplacian_function(
    f: Callable[[Tensor], Tensor], X: Tensor, is_batched: bool, strategy: str
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
            - `'jet_naive'`: The Bi-Laplacian is computed using jets. The computation
                graph is simplified by propagating replication nodes.

    Returns:
        A function that computes the Bi-Laplacian of the function f at the input
        tensor X. The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy is not supported.
    """
    if strategy == "hessian_trace":
        dummy_x = X[0] if is_batched else X
        laplacian = hessian_trace_laplacian(f, dummy_x)
        bilaplacian = hessian_trace_laplacian(laplacian, dummy_x)

        if is_batched:
            bilaplacian = vmap(bilaplacian)

    elif strategy in {"jet_naive", "jet_simplified"}:
        bilaplacian = Bilaplacian(f, X, is_batched)
        pull_sum_vmapped = strategy == "jet_simplified"
        bilaplacian = simplify(
            symbolic_trace(bilaplacian), pull_sum_vmapped=pull_sum_vmapped
        )

    else:
        raise ValueError(f"Unsupported strategy: {strategy}.")

    return lambda: bilaplacian(X)


def randomized_bilaplacian_function(
    f: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    strategy: str,
    distribution: str,
    num_samples: int,
) -> Callable[[], Tensor]:
    """Build function to compute the MC-Bi-Laplacian with different strategies.

    Args:
        f: The function to compute the MC-Bi-Laplacian of. Processes an
            un-batched tensor.
        X: The input tensor at which to compute the MC-Bi-Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the MC-Bi-Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The MC-Bi-Laplacian is computed by vector-
                tensor products between a random vector and the fourth-order
                derivative tensor via (3x)forward-over-reverse mode autodiff.
            - `'jet_naive'`: The MC-Bi-Laplacian is computed using jets.
                The computation graph is simplified by propagating replication nodes.
            - `'jet_simplified'`: The MC-Bi-Laplacian is computed using Taylor
                mode. The computation graph is simplified by propagating replications
                down, and summations up, the computation graph.
        distribution: From which distribution to draw the random vectors. Supported
            values are `'normal'`.
        num_samples: How many Monte-Carlo samples should be used by the estimation.

    Returns:
        A function that computes the randomized Bi-Laplacian of the function f at the
        input tensor X. The function is expected to be called with no arguments.

    Raises:
        ValueError: If the strategy or distribution are not supported.
    """
    if distribution != "normal":
        raise ValueError(f"Unsupported distribution: {distribution!r}.")

    if strategy == "hessian_trace":
        dummy_x = X[0] if is_batched else X
        d2f_vv = vector_hessian_vector_product(f, dummy_x)
        d4f_vvvv = lambda x, v: vector_hessian_vector_product(  # noqa: E731
            lambda x: d2f_vv(x, v), dummy_x
        )(x, v)

        # vmap over data points and fix data
        if is_batched:
            d4f_vvvv = vmap(d4f_vvvv)
        d4f_vvvv_fix_X = partial(d4f_vvvv, X)

        # vmap over vectors
        d4f_VVVV_vmap = vmap(d4f_vvvv_fix_X)

        # draw random vectors
        sample_func = {"normal": randn}[distribution]
        V = sample_func(num_samples, *X.shape, device=X.device, dtype=X.dtype)

        return lambda: d4f_VVVV_vmap(V).mean(0) / 3

    if strategy in {"jet_naive", "jet_simplified"}:
        pull_sum_vmapped = strategy == "jet_simplified"
        bilap = RandomizedBilaplacian(f, X, is_batched, num_samples, distribution)
        bilap = simplify(symbolic_trace(bilap), pull_sum_vmapped=pull_sum_vmapped)
        return lambda: bilap(X)

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Supported: {SUPPORTED_STRATEGIES}."
        )


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


def savepath(rawdir: str = RAWDIR, **kwargs: Union[str, int]) -> str:
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


def check_mutually_required(args: Namespace):
    """Check if mutually required arguments are specified or unspecified.

    Args:
        args: The parsed arguments.

    Raises:
        ValueError: If the arguments are not mutually specified or unspecified.
    """
    distribution, num_samples = args.distribution, args.num_samples
    if (distribution is None) != (num_samples is None):
        raise ValueError(
            f"Arguments 'distribution' ({distribution}) and 'num_samples'"
            f" ({num_samples}) are mutually required."
        )


def get_function_and_description(
    operator: str,
    strategy: str,
    distribution: Union[str, None],
    num_samples: Union[int, None],
    net: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    compiled: bool = False,
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
            Default: `False``.

    Returns:
        A tuple containing the function to compute the operator (differentiable),
        the function to compute the operator (non-differentiable), and a description
        string.

    Raises:
        ValueError: If an unsupported operator is specified.
    """
    is_stochastic = distribution is not None and num_samples is not None
    args = (
        (net, X, is_batched, strategy, distribution, num_samples)
        if is_stochastic
        else (net, X, is_batched, strategy)
    )
    description = (
        f"{strategy}, distribution={distribution}, " + f"num_samples={num_samples}"
        if is_stochastic
        else f"{strategy}"
    )

    if operator == "bilaplacian":
        func = (
            randomized_bilaplacian_function(*args)
            if is_stochastic
            else bilaplacian_function(*args)
        )
    elif operator == "laplacian":
        func = (
            randomized_laplacian_function(*args)
            if is_stochastic
            else laplacian_function(*args)
        )
    elif operator == "weighted-laplacian":
        func = (
            randomized_weighted_laplacian_function(*args)
            if is_stochastic
            else weighted_laplacian_function(*args)
        )
    else:
        raise ValueError(f"Unsupported operator: {operator}.")

    @no_grad()
    def func_no() -> Tensor:
        """Non-differentiable computation.

        Returns:
            Value of the differentiable operator
        """
        return func()

    compile_error = operator == "bilaplacian" and strategy == "hessian_trace"

    if compiled:
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

    # parse and check validity
    args = parser.parse_args()
    check_mutually_required(args)

    return args


if __name__ == "__main__":
    args = parse_args()

    # set up the function that will be measured
    dev = device(args.device)
    dt = float64
    net = setup_architecture(args.architecture, args.dim, dev, dt)
    is_batched = True
    X = setup_input(args.batch_size, args.dim, dev, dt)

    manual_seed(2)  # this allows making the randomized methods deterministic
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
    )

    print(f"Setting up functions took: {perf_counter() - start:.3f} s.")

    is_cuda = args.device == "cuda"
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
    if args.strategy != BASELINE or args.compiled:
        print("Checking correctness against un-compiled baseline.")
        with no_grad():
            result = func()

        manual_seed(2)  # make sure that the baseline is deterministic
        _, baseline_func_no, _ = get_function_and_description(
            args.operator,
            BASELINE,
            args.distribution,
            args.num_samples,
            net,
            X,
            is_batched,
            compiled=False,
        )
        baseline_result = baseline_func_no()

        assert (
            baseline_result.shape == result.shape
        ), f"Shapes do not match: {baseline_result.shape} != {result.shape}."
        same = allclose(baseline_result, result)
        assert same, f"Results do not match: {result} != {baseline_result}."
        print("Results match.")

    # Write measurements to a file
    data = ", ".join([str(val) for val in [mem_no, mem, mu, sigma, best]])
    filename = savepath(**vars(args))
    with open(filename, "w") as f:
        print(f"Writing to {filename}.")
        f.write(data)
