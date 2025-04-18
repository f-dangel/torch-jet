"""Script that carries out a measurement of peak memory and run time."""

from argparse import ArgumentParser, Namespace
from functools import partial
from os import makedirs, path
from time import perf_counter
from typing import Callable, Union

from einops import einsum
from torch import Tensor, device, manual_seed, no_grad, rand, randn
from torch.func import hessian, jacrev, jvp, vmap
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Tanh

from jet.bilaplacian import Bilaplacian
from jet.exp.utils import measure_peak_memory, measure_time, to_string
from jet.laplacian import Laplacian, RandomizedLaplacian
from jet.simplify import simplify
from jet.utils import rademacher
from jet.weighted_laplacian import C_func_diagonal_increments, WeightedLaplacian

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
RAWDIR = path.join(HEREDIR, "raw")
makedirs(RAWDIR, exist_ok=True)

# Define supported architectures
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
        hess_f = hessian(f)

        # trace with einsum to support Laplacians of functions with non-scalar output
        dims = " ".join([f"d{i}" for i in range(X.ndim - 1 if is_batched else X.ndim)])
        tr_equation = f"... {dims} {dims} -> ..."

        def laplacian(x: Tensor) -> Tensor:
            """Compute the Laplacian of f on an un-batched input.

            Args:
                x: The input tensor.

            Returns:
                The Laplacian of f at x. Has the same shape as f(x).
            """
            return einsum(hess_f(x), tr_equation)

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

        # perform the contraction of the HVP and the vector with einsum to support
        # functions with non-scalar output
        sum_dims = X.ndim - 1 if is_batched else X.ndim
        dims = " ".join([f"d{i}" for i in range(sum_dims)])
        equation = f"... {dims}, {dims} -> ..."

        def vhv(x: Tensor, v: Tensor) -> Tensor:
            """Compute vector-Hessian-vector products of f evaluated at x.

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
        weighted_laplacian = WeightedLaplacian(
            f, X, is_batched, weighting="diagonal_increments"
        )
        pull_sum_vmapped = strategy == "jet_simplified"
        weighted_laplacian = simplify(
            symbolic_trace(weighted_laplacian), pull_sum_vmapped=pull_sum_vmapped
        )

        return lambda: weighted_laplacian(X)[2]

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
        d4f = hessian(hessian(f))

        # trace it using einsum to support functions with non-scalar outputs
        num_summed_dims = X.ndim - 1 if is_batched else X.ndim
        dims1 = " ".join([f"i{i}" for i in range(num_summed_dims)])
        dims2 = " ".join([f"j{j}" for j in range(num_summed_dims)])
        # if x is a vector, this is just '... i i j j -> ...' where '...' corresponds
        # to the shape of f(x)
        equation = f"... {dims1} {dims1} {dims2} {dims2} -> ..."

        def bilaplacian(x: Tensor) -> Tensor:
            """Compute the Bi-Laplacian of f on an un-batched input x.

            Args:
                x: The input tensor.

            Returns:
                The Laplacian of f at x. Has the same shape as f(x).
            """
            return einsum(d4f(x), equation)

        if is_batched:
            bilaplacian = vmap(bilaplacian)

        return lambda: bilaplacian(X)

    elif strategy in {"jet_naive", "jet_simplified"}:
        bilaplacian = Bilaplacian(f, X, is_batched)
        pull_sum_vmapped = strategy == "jet_simplified"
        bilaplacian = simplify(
            symbolic_trace(bilaplacian), pull_sum_vmapped=pull_sum_vmapped
        )
        return lambda: bilaplacian(X)

    else:
        raise ValueError(f"Unsupported strategy: {strategy}.")


def setup_architecture(architecture: str, dim: int) -> Callable[[Tensor], Tensor]:
    """Set up a neural network architecture based on the specified configuration.

    Args:
        architecture: The architecture identifier.
        dim: The input dimension for the architecture.

    Returns:
        A PyTorch model configured with the specified architecture.
    """
    return SUPPORTED_ARCHITECTURES[architecture](dim)


def savepath(**kwargs: Union[str, int]) -> str:
    """Generate a file path for saving measurement results.

    Args:
        **kwargs: Key-value pairs representing the parameters of the measurement.

    Returns:
        A string representing the file path where the results will be saved.
    """
    filename = to_string(**kwargs)
    return path.join(RAWDIR, f"{filename}.csv")


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
) -> tuple[Callable[[], Tensor], str]:
    """Determine the function and its description based on the operator and strategy.

    Args:
        operator: The operator to be used, either 'laplacian' or 'bilaplacian'.
        strategy: The strategy to be used for computation.
        distribution: The distribution type, if any.
        num_samples: The number of samples, if any.
        net: The neural network model.
        X: The input tensor.
        is_batched: A flag indicating if the input is batched.

    Returns:
        A tuple containing the function to compute the operator and a description
        string.

    Raises:
        ValueError: If an unsupported operator is specified.
        NotImplementedError: If a randomized Bi-Laplacian or weighted Laplacian
            are requested.
    """
    if operator == "laplacian":
        if distribution is None and num_samples is None:
            func = laplacian_function(net, X, is_batched, strategy)
            description = f"{strategy}"
        else:
            func = randomized_laplacian_function(
                net, X, is_batched, strategy, distribution, num_samples
            )
            description = (
                f"{strategy}, distribution={distribution}, "
                + f"num_samples={num_samples}"
            )
    elif operator == "weighted-laplacian":
        if distribution is None and num_samples is None:
            func = weighted_laplacian_function(net, X, is_batched, strategy)
            description = f"{strategy}"
        else:
            raise NotImplementedError
    elif operator == "bilaplacian":
        if distribution is None and num_samples is None:
            func = bilaplacian_function(net, X, is_batched, strategy)
            description = f"{strategy}"
        else:
            raise NotImplementedError("Randomized Bi-Laplacian not implemented.")
    else:
        raise ValueError(f"Unsupported operator: {operator}.")

    return func, description


if __name__ == "__main__":
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
    args = parser.parse_args()

    check_mutually_required(args)

    # set up the function that will be measured
    dev = device(args.device)
    manual_seed(0)
    net = setup_architecture(args.architecture, args.dim).double().to(dev)
    X = rand(args.batch_size, args.dim).double().to(dev)
    is_batched = True

    manual_seed(1)  # this allows making the randomized methods deterministic
    start = perf_counter()
    func, description = get_function_and_description(
        args.operator,
        args.strategy,
        args.distribution,
        args.num_samples,
        net,
        X,
        is_batched,
    )
    print(f"Setting up function took: {perf_counter() - start:.3f} s.")

    is_cuda = args.device == "cuda"

    op = args.operator.capitalize()
    # carry out the measurements
    with no_grad():
        mem_no = measure_peak_memory(
            func, f"{op} non-differentiable ({description})", is_cuda
        )
    mem = measure_peak_memory(func, f"{op} ({description})", is_cuda)
    mu, sigma, best = measure_time(func, f"{op} ({description})", is_cuda)

    # sanity check: make sure that the results correspond to the baseline implementation
    if args.strategy != BASELINE:
        print("Checking correctness against baseline.")
        with no_grad():
            result = func()

        manual_seed(1)  # make sure that the baseline is deterministic
        baseline_func, _ = get_function_and_description(
            args.operator,
            BASELINE,
            args.distribution,
            args.num_samples,
            net,
            X,
            is_batched,
        )
        with no_grad():
            baseline_result = baseline_func()

        assert (
            baseline_result.shape == result.shape
        ), f"Shapes do not match: {baseline_result.shape} != {result.shape}."
        assert baseline_result.allclose(
            result
        ), f"Results do not match: {result} != {baseline_result}."
        print("Results match.")

    # write them to a file
    data = ", ".join([str(val) for val in [mem_no, mem, mu, sigma, best]])
    filename = savepath(**vars(args))
    with open(filename, "w") as f:
        print(f"Writing to {filename}.")
        f.write(data)
