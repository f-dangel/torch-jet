"""Script that carries out a measurement of peak memory and run time."""

from argparse import ArgumentParser, Namespace
from os import makedirs, path
from typing import Callable, Union

from torch import Tensor, device, manual_seed, no_grad, ones_like, rand, randn
from torch.func import hessian, jvp, vjp, vmap
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Tanh

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
        dim_x = (X.shape[1:] if is_batched else X.shape).numel()
        hess_f = hessian(f)

        def laplacian(x: Tensor) -> Tensor:
            """Compute the Laplacian of f on an un-batched input.

            Args:
                x: The input tensor.

            Returns:
                The Laplacian of f at x.
            """
            return hess_f(x).reshape(dim_x, dim_x).trace().unsqueeze(0)

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

        def vhv(v: Tensor) -> Tensor:
            """Compute vector-Hessian-vector products of f evaluated at X.

            Args:
                v: The vector to compute the vector-Hessian-vector product with.
                    Has same shape as `X`.

            Returns:
                The vector-Hessian-vector product. Has shape `(N, 1)` if `is_batched`
                is `True` (where `N` is the batch size), otherwise `(1,)`.
            """

            def grad_func(X: Tensor) -> Tensor:
                f_X, vjp_func = vjp(f, X)
                return vjp_func(ones_like(f_X))

            _, (hvp,) = jvp(grad_func, (X,), (v,))

            if is_batched:
                return (v * hvp).flatten(start_dim=1).sum(dim=1, keepdim=True)
            else:
                return (v * hvp).sum().unsqueeze(0)

        vhv_vmap = vmap(vhv)

        sample_func = {"normal": randn, "rademacher": rademacher}[distribution]
        V = sample_func(num_samples, *X.shape, device=X.device, dtype=X.dtype)

        return lambda: vhv_vmap(V).mean(0)

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
    if (
        distribution is None
        and num_samples is not None
        or distribution is not None
        and num_samples is None
    ):
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
    args = parser.parse_args()

    check_mutually_required(args)

    # set up the function that will be measured
    dev = device(args.device)
    manual_seed(0)
    net = setup_architecture(args.architecture, args.dim).double().to(dev)
    X = rand(args.batch_size, args.dim).double().to(dev)
    is_batched = True

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
    is_cuda = args.device == "cuda"

    # carry out the measurements
    with no_grad():
        mem_no = measure_peak_memory(
            func, f"Laplacian non-differentiable ({description})", is_cuda
        )
    mem = measure_peak_memory(func, f"Laplacian ({description})", is_cuda)
    mu, sigma, best = measure_time(func, f"Laplacian ({description})", is_cuda)

    # write them to a file
    data = ", ".join([str(val) for val in [mem_no, mem, mu, sigma, best]])
    filename = savepath(**vars(args))
    with open(filename, "w") as f:
        print(f"Writing to {filename}.")
        f.write(data)
