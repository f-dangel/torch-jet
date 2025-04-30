"""Script that carries out measurements of peak memory and run time in JAX."""

from os import makedirs, path
from sys import platform
from time import perf_counter
from typing import Callable

from einops import einsum
from folx import ForwardLaplacianOperator
from jax import (
    Device,
    block_until_ready,
    config,
    device_put,
    devices,
    grad,
    hessian,
    jit,
    vmap,
)
from jax.example_libraries import stax
from jax.experimental.jet import jet
from jax.numpy import allclose, array, eye, float64, size, zeros
from jax.random import PRNGKey, uniform
from jax.tree_util import tree_map
from jax.typing import ArrayLike, DTypeLike

from jet.exp.exp01_benchmark_laplacian.execute import (
    BASELINE,
    SUPPORTED_STRATEGIES,
    parse_args,
    savepath,
)
from jet.exp.utils import measure_peak_memory, measure_time

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
RAWDIR = path.join(HEREDIR, "raw")
makedirs(RAWDIR, exist_ok=True)


# Turning on double precision on MAC gives errors
ON_MAC = platform == "darwin"
if not ON_MAC:
    # Enable float64 computation in JAX. This has to be done at start-up!
    config.update("jax_enable_x64", True)

# Define supported PyTorch architectures
SUPPORTED_ARCHITECTURES = {
    "tanh_mlp_768_768_512_512_1": lambda: stax.serial(
        stax.Dense(768),
        stax.Tanh,
        stax.Dense(768),
        stax.Tanh,
        stax.Dense(512),
        stax.Tanh,
        stax.Dense(512),
        stax.Tanh,
        stax.Dense(1),  # No activation on the last layer
    )
}


def setup_architecture(
    architecture: str, dim: int, dev: Device, dt: DTypeLike, seed: int = 0
) -> tuple[list[ArrayLike], Callable[[list[ArrayLike], ArrayLike], ArrayLike]]:
    """Set up the architecture for the benchmark.

    Args:
        architecture: The architecture to use. Must be one of the supported
            architectures in `SUPPORTED_ARCHITECTURES`.
        dim: The dimension of the input.
        dev: The device to use.
        dt: The data type to use.
        seed: The random seed to use. Default is 0.

    Returns:
        A tuple containing the parameters of the architecture and the function to
        compute the output of the architecture given parameters and data.
    """
    init_fun, apply_fun = SUPPORTED_ARCHITECTURES[architecture]()
    key = PRNGKey(seed)
    _, params = init_fun(key, (dim,))
    # move to data type and device
    params = tree_map(lambda x: device_put(array(x, dtype=dt), device=dev), params)
    return params, apply_fun


def setup_input(
    batch_size: int, dim: int, dev: Device, dt: DTypeLike, seed: int = 1
) -> ArrayLike:
    """Set up the seeded input for the benchmark.

    Args:
        batch_size: The batch size to use.
        dim: The dimension of the input.
        dev: The device to use.
        dt: The data type to use.
        seed: The random seed to use. Default is 1.

    Returns:
        The seeded input tensor.
    """
    shape = (batch_size, dim)
    key = PRNGKey(seed)
    return device_put(uniform(key, shape=shape, dtype=dt), dev)


def laplacian_function(  # noqa: C901
    params_and_f: tuple[
        list[ArrayLike], Callable[[list[ArrayLike], ArrayLike], ArrayLike]
    ],
    X: ArrayLike,
    is_batched: bool,
    strategy: str,
) -> tuple[Callable[[], ArrayLike], Callable[[], list[ArrayLike]]]:
    """Construct a function to compute the Laplacian in JAX using different strategies.

    Args:
        params_and_f: The neural net's parameters and the unbatched forward function
            whose Laplacian we want to compute. The function should take the parameters
            and the input tensor as arguments and return the output tensor.
        X: The input tensor at which to compute the Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The Laplacian is computed by tracing the Hessian.
              The Hessian is computed via forward-over-reverse mode autodiff.
            - `'jet_naive'`: The Laplacian is computed using jets.
            - `'jet_simplified'`: The Laplacian is computed using the forward
              Laplacian library.

    Returns:
        A function that computes the Laplacian of the function f at the input tensor X.
        The function is expected to be called with no arguments and is jitted.

    Raises:
        ValueError: If the strategy is not supported.
    """
    params, f = params_and_f
    if strategy == "hessian_trace":
        hess_f = hessian(f, argnums=1)

        # trace with einsum to support Laplacians of functions with non-scalar output
        dims = " ".join([f"d{i}" for i in range(X.ndim - 1 if is_batched else X.ndim)])
        tr_equation = f"... {dims} {dims} -> ..."

        def laplacian(params: list[ArrayLike], x: ArrayLike) -> ArrayLike:
            """Compute the Laplacian of f on an un-batched input.

            Args:
                x: The input tensor.

            Returns:
                The Laplacian of f at x. Has the same shape as f(x).
            """
            return einsum(hess_f(params, x), tr_equation)

        if is_batched:
            laplacian = vmap(laplacian, in_axes=[None, 0])

        # function that computes the Laplacian
        func = lambda: laplacian(params, X)  # noqa: E731
        # function that computes the Laplacian's gradient used as proxy for the
        # computation graph's memory footprint
        summed_laplacian = lambda params, X: laplacian(params, X).sum()  # noqa: E731
        grad_func = lambda: grad(summed_laplacian, argnums=0)(params, X)  # noqa: E731

        # jit the functions
        func = jit(func)
        grad_func = jit(grad_func)

        # add a trailing statement to wait until the computations are done
        return lambda: block_until_ready(func()), lambda: block_until_ready(grad_func())

    elif strategy == "jet_naive":
        shape = X.shape[1:] if is_batched else X.shape
        D = size(X[0] if is_batched else X)

        f_fix_params = lambda x: f(params, x)  # noqa: E731

        def laplacian(x: ArrayLike) -> ArrayLike:
            """Compute the Laplacian of f on an un-batched input.

            Args:
                x: The un-batched input tensor.

            Returns:
                The Laplacian of f at x. Has the same shape as f(x).
            """
            v2 = zeros(shape, dtype=X.dtype, device=X.device)

            def d2(x, v1):
                f0, (f1, f2) = jet(f_fix_params, (x,), ((v1, v2),))
                return f2

            d2_vmap = vmap(lambda v1: d2(x, v1))
            v1 = eye(D, dtype=X.dtype, device=X.device).reshape(D, *shape)
            return d2_vmap(v1).sum(0)

        if is_batched:
            laplacian = vmap(laplacian)

        laplacian = jit(laplacian)
        func = lambda: block_until_ready(laplacian(X))  # noqa: E731

        # NOTE JAX's jet does not support the gradient of the Laplacian,
        # because we cannot compute Taylor-mode w.r.t. x, followed by the
        # gradient w.r.t. params (we could do the opposite order, but that
        # would not be a meaningful proxy for the compute graph size of the
        # Laplacian when using PyTorch with requires_grad=True). Hence we
        # simply return the same function twice and ignore one of the
        # measurements.
        return func, func

    elif strategy == "jet_simplified":
        f_fix_params = lambda x: f(params, x)  # noqa: E731
        # disable sparsity to remove its run time benefits
        lap_f = ForwardLaplacianOperator(0)(f_fix_params)

        def laplacian(x: ArrayLike) -> ArrayLike:
            """Compute the Laplacian of f on an un-batched input.

            Args:
                x: The un-batched input tensor.

            Returns:
                The Laplacian of f at x. Has the same shape as f(x).
            """
            return lap_f(x)[0]

        if is_batched:
            laplacian = vmap(laplacian)

        laplacian = jit(laplacian)
        func = lambda: block_until_ready(laplacian(X))  # noqa: E731

        # NOTE See the comment for `jet_naive` from above
        return func, func

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Supported: {SUPPORTED_STRATEGIES}."
        )


def get_function_and_description(
    operator: str,
    strategy: str,
    distribution: str,
    num_samples: int,
    params_and_net: tuple[
        list[ArrayLike], Callable[[list[ArrayLike], ArrayLike], ArrayLike]
    ],
    X: ArrayLike,
    is_batched: bool,
) -> tuple[Callable[[], ArrayLike], Callable[[], list[ArrayLike]], str]:
    """Determine the function and its description based on the operator and strategy.

    Args:
        operator: The operator to be used, either 'laplacian' or 'bilaplacian'.
        strategy: The strategy to be used for computation.
        distribution: The distribution type, if any.
        num_samples: The number of samples, if any.
        params_and_net: The parameters and neural network function.
        X: The input tensor.
        is_batched: A flag indicating if the input is batched.

    Returns:
        A tuple containing the functions to compute the operator w/o being
        able to differentiate through it, and a description string.

    Raises:
        ValueError: If an unsupported operator is specified.
        NotImplementedError: If the specified mode is stochastic.
    """
    is_stochastic = distribution is not None and num_samples is not None
    args = (
        (params_and_net, X, is_batched, strategy, distribution, num_samples)
        if is_stochastic
        else (params_and_net, X, is_batched, strategy)
    )
    description = (
        f"{strategy}, distribution={distribution}, " + f"num_samples={num_samples}"
        if is_stochastic
        else f"{strategy}"
    )

    if operator != "laplacian":
        raise ValueError(f"Unsupported operator: {operator}.")

    if is_stochastic:
        raise NotImplementedError("Stochastic operators are not implemented yet.")

    func_no, func = laplacian_function(*args)

    return func_no, func, description


if __name__ == "__main__":
    args = parse_args()

    # set up the function that will be measured
    dev = devices(args.device)[0]
    dt = float64
    params, net = setup_architecture(args.architecture, args.dim, dev, dt)
    is_batched = True
    X = setup_input(args.batch_size, args.dim, dev, dt)

    start = perf_counter()
    func_no, func, description = get_function_and_description(
        args.operator,
        args.strategy,
        args.distribution,
        args.num_samples,
        (params, net),
        X,
        is_batched,
    )
    print(f"Setting up function took: {perf_counter() - start:.3f} s.")
    is_cuda = args.device == "cuda"
    op = args.operator.capitalize()

    # Carry out the measurements
    # 1) Peak memory with non-differentiable result
    mem_no = measure_peak_memory(
        func_no,
        f"{op} non-differentiable ({description})",
        is_cuda,
        use_jax=True,
    )

    # 2) Peak memory with differentiable result (NOTE we can not always build a
    # good proxy function to measure the equivalent of PyTorch's peak memory
    # with requires_grad=True in JAX because it does not have an equivalent,
    # see the discussion in https://github.com/jax-ml/jax/issues/1937)
    mem = measure_peak_memory(func, f"{op} ({description})", is_cuda, use_jax=True)
    if args.strategy != "hessian_trace" and not is_cuda:
        mem = float("nan")

    # 3) Run time
    mu, sigma, best = measure_time(func_no, f"{op} ({description})", is_cuda)

    # Sanity check: make sure that the results correspond to the baseline implementation
    if args.strategy != BASELINE:
        print("Checking correctness against baseline.")
        result = func_no()

        baseline_func_no, _, _ = get_function_and_description(
            args.operator,
            BASELINE,
            args.distribution,
            args.num_samples,
            (params, net),
            X,
            is_batched,
        )
        baseline_result = baseline_func_no()

        assert (
            baseline_result.shape == result.shape
        ), f"Shapes do not match: {baseline_result.shape} != {result.shape}."
        # NOTE On MAC, we cannot force float64 computations without getting errors.
        # Therefore we need to increase the tolerance.
        tols = {"atol": 5e-6, "rtol": 5e-4} if ON_MAC else {}
        same = allclose(baseline_result, result, **tols)
        assert same, f"Results do not match: {result} != {baseline_result}."
        print("Results match.")

    # Write measurements to a file
    data = ", ".join([str(val) for val in [mem_no, mem, mu, sigma, best]])
    filename = savepath(rawdir=RAWDIR, **vars(args))
    with open(filename, "w") as f:
        print(f"Writing to {filename}.")
        f.write(data)
