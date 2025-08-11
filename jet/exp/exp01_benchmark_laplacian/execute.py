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
)
from torch import compile as torch_compile
from torch import (
    device,
    dtype,
    eye,
    float64,
    manual_seed,
    no_grad,
    rand,
)
from torch.func import hessian, jacrev, jvp, vmap
from torch.nn import Linear, Sequential, Tanh
from torch.random import fork_rng

from jet.bilaplacian import Bilaplacian
from jet.exp.utils import measure_peak_memory, measure_time, to_string
from jet.laplacian import Laplacian
from jet.simplify import simplify
from jet.utils import sample
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
            else sample(x, randomization[0], (num_jets, *x.shape))
        )
        S = apply_weightings(x, V)
        SHS = VhVp(x, S)
        return SHS.sum(0) if randomization is None else SHS.mean(0)

    return laplacian


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
            - `'jet_naive'`: The Laplacian is computed using jets. The computation graph
                is simplified by propagating replication nodes.
            - `'jet_simplified'`: The Laplacian is computed using Taylor mode. The
              computation graph is simplified by propagating replications down, and
              summations up, the computation graph.
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

    elif strategy in {"jet_naive", "jet_simplified"}:
        lap_mod = Laplacian(
            f, dummy_x, randomization=randomization, weighting=weighting
        )
        pull_sum_vmapped = strategy == "jet_simplified"
        lap_mod = simplify(lap_mod, pull_sum_vmapped=pull_sum_vmapped)
        laplacian = lambda x: lap_mod(x)[2]  # noqa: E731

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
            - `'jet_naive'`: The Bi-Laplacian is computed using jets. The computation
                graph is simplified by propagating replication nodes.
            - `'jet_simplified'`: The Bi-Laplacian is computed using Taylor mode. The
                computation graph is simplified by propagating replications down, and
                summations up, the computation graph.
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
            bilaplacian = hessian_trace_laplacian(laplacian, dummy_x)
        else:
            bilaplacian = vector_hessian_vector_product_bilaplacian(
                f, dummy_x, randomization
            )

    elif strategy in {"jet_naive", "jet_simplified"}:
        bilaplacian = Bilaplacian(f, dummy_x, randomization=randomization)
        pull_sum_vmapped = strategy == "jet_simplified"
        bilaplacian = simplify(bilaplacian, pull_sum_vmapped=pull_sum_vmapped)

    else:
        raise ValueError(f"Unsupported strategy: {strategy}.")

    if is_batched:
        bilaplacian = vmap(
            bilaplacian, randomness="error" if randomization is None else "different"
        )

    return partial(bilaplacian, X)


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
    distribution: str | None,
    num_samples: int | None,
    net: Callable[[Tensor], Tensor],
    X: Tensor,
    is_batched: bool,
    compiled: bool,
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

    Returns:
        A tuple containing the function to compute the operator (differentiable),
        the function to compute the operator (non-differentiable), and a description
        string.

    Raises:
        ValueError: If an unsupported operator is specified.
    """
    is_stochastic = distribution is not None and num_samples is not None

    description = f"{strategy}, {compiled=}"
    if is_stochastic:
        description += f", {distribution=}, {num_samples=}"

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
            "diagonal_increments",
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
        with no_grad(), fork_rng():
            manual_seed(2)
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
            False,  # do not use compilation
        )
        with fork_rng():
            manual_seed(2)
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
