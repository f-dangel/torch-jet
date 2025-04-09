"""Script that carries out a measurement of peak memory and run time."""

from argparse import ArgumentParser, Namespace
from functools import partial
from os import makedirs, path
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
        A function that computes the Bi-Laplacian of the function f at the input tensor X.
        The function is expected to be called with no arguments.

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

    elif strategy == "jet_naive":
        bilaplacian = Bilaplacian(f, X, is_batched)
        bilaplacian = simplify(symbolic_trace(bilaplacian), pull_sum_vmapped=False)
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
        "--operator", type=str, choices={"laplacian", "bilaplacian"}, required=True
    )
    args = parser.parse_args()

    check_mutually_required(args)

    # set up the function that will be measured
    dev = device(args.device)
    manual_seed(0)
    net = setup_architecture(args.architecture, args.dim).double().to(dev)
    X = rand(args.batch_size, args.dim).double().to(dev)
    is_batched = True

    if args.operator == "laplacian":
        if args.distribution is None and args.num_samples is None:
            func = laplacian_function(net, X, is_batched, args.strategy)
            description = f"{args.strategy}"
        else:
            func = randomized_laplacian_function(
                net, X, is_batched, args.strategy, args.distribution, args.num_samples
            )
            description = (
                f"{args.strategy}, distribution={args.distribution}, "
                + f"num_samples={args.num_samples}"
            )
    elif args.operator == "bilaplacian":
        if args.distribution is None and args.num_samples is None:
            func = bilaplacian_function(net, X, is_batched, args.strategy)
            description = f"{args.strategy}"
        else:
            raise NotImplementedError("Randomized Bi-Laplacian not implemented.")
    else:
        raise ValueError(f"Unsupported operator: {args.operator}.")
    is_cuda = args.device == "cuda"

    op = args.operator.capitalize()
    # carry out the measurements
    with no_grad():
        mem_no = measure_peak_memory(
            func, f"{op} non-differentiable ({description})", is_cuda
        )
    mem = measure_peak_memory(func, f"{op} ({description})", is_cuda)
    mu, sigma, best = measure_time(func, f"{op} ({description})", is_cuda)

    # write them to a file
    data = ", ".join([str(val) for val in [mem_no, mem, mu, sigma, best]])
    filename = savepath(**vars(args))
    with open(filename, "w") as f:
        print(f"Writing to {filename}.")
        f.write(data)
