"""Script that carries out a measurement of peak memory and run time."""

from argparse import ArgumentParser
from os import makedirs, path
from typing import Callable, Union

from torch import Tensor, device, manual_seed, no_grad, rand
from torch.func import hessian, vmap
from torch.fx import symbolic_trace
from torch.nn import Linear, Sequential, Tanh

from jet.exp.utils import measure_peak_memory, measure_time, to_string
from jet.laplacian import Laplacian
from jet.simplify import simplify

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
    parser.add_argument("--device", type=str, choices={"cpu", "cuda"}, required=True)
    args = parser.parse_args()

    # set up the function that will be measured
    dev = device(args.device)
    manual_seed(0)
    net = setup_architecture(args.architecture, args.dim).double().to(dev)
    X = rand(args.batch_size, args.dim).double().to(dev)
    is_batched = True

    # carry out the measurements
    func = laplacian_function(net, X, is_batched, args.strategy)
    is_cuda = args.device == "cuda"

    with no_grad():
        mem_no = measure_peak_memory(
            func, f"Laplacian non-differentiable ({args.strategy})", is_cuda
        )
    mem = measure_peak_memory(func, f"Laplacian ({args.strategy})", is_cuda)
    mu, sigma, best = measure_time(func, f"Laplacian ({args.strategy})", is_cuda)

    # write them to a file
    data = ", ".join([str(val) for val in [mem_no, mem, mu, sigma, best]])
    filename = savepath(**vars(args))
    with open(filename, "w") as f:
        print(f"Writing to {filename}.")
        f.write(data)
