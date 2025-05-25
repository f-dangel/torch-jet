"""Script that carries out measurements of peak memory and run time in JAX."""

from os import makedirs, path
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
    jacrev,
    jit,
    jvp,
    vmap,
)
from jax.example_libraries import stax
from jax.experimental.jet import jet
from jax.numpy import allclose, array, eye, float64, size, zeros
from jax.random import PRNGKey, normal, uniform
from jax.tree_util import tree_map
from jax.typing import ArrayLike, DTypeLike

from jet.exp.exp01_benchmark_laplacian.execute import (
    BASELINE,
    ON_MAC,
    parse_args,
    savepath,
)
from jet.exp.utils import measure_peak_memory, measure_time

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
RAWDIR = path.join(HEREDIR, "raw")
makedirs(RAWDIR, exist_ok=True)


# Turning on double precision on MAC gives errors
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


def vector_hessian_vector_product(
    f: Callable[[list[ArrayLike], ArrayLike], ArrayLike], dummy_x: ArrayLike
) -> Callable[[list[ArrayLike], ArrayLike, ArrayLike], ArrayLike]:
    """Generate function to compute the vector-Hessian-vector product of f at x with v.

    Args:
        f: The function whose vector-Hessian-vector product we want to compute.
            Takes the params and input tensor as argument and return the output tensor.
        dummy_x: An un-batched dummy input tensor to determine the input dimensions.

    Returns:
        A function that computes the vector-Hessian-vector product of f at the input
        tensor x given params, x and the vector. Has the same shape as f(params, x).
    """
    # perform the contraction of the HVP and the vector with einsum to support
    # functions with non-scalar output
    sum_dims = dummy_x.ndim
    in_dims = " ".join([f"d{i}" for i in range(sum_dims)])
    out_dims = "..."
    equation = f"{out_dims} {in_dims}, {in_dims} -> {out_dims}"

    def vhv(params: list[ArrayLike], x: ArrayLike, v: ArrayLike) -> ArrayLike:
        """Compute the vector-Hessian-vector product of f with v evaluated at x.

        Args:
            params: The parameters of the neural network.
            x: The input to the function at which the vector-Hessian-vector product
                is computed.
            v: The vector to compute the vector-Hessian-vector product with.
                Has same shape as `x`.

        Returns:
            The vector-Hessian-vector product. Has the same shape as f(params, x).
        """
        grad_func = jacrev(lambda x: f(params, x))
        _, hvp = jvp(grad_func, (x,), (v,))

        return einsum(hvp, v, equation)

    return vhv


def hessian_trace_laplacian(
    f: Callable[[list[ArrayLike], ArrayLike], ArrayLike], dummy_x: ArrayLike
) -> Callable[[list[ArrayLike], ArrayLike], ArrayLike]:
    """Generate a function that computes the Laplacian of f by tracing the Hessian.

    Args:
        f: The function whose Laplacian we want to compute.
            The function should take the parameters and the input tensor as arguments
            and return the output tensor.
        dummy_x: A dummy input tensor to determine the input dimensions.

    Returns:
        A function that computes the Laplacian of f at the input tensor X given the
        parameters and X.
    """
    hess_f = hessian(f, argnums=1)

    # trace with einsum to support Laplacians of functions with non-scalar output
    dims = " ".join([f"d{i}" for i in range(dummy_x.ndim)])
    tr_equation = f"... {dims} {dims} -> ..."

    def laplacian(params: list[ArrayLike], x: ArrayLike) -> ArrayLike:
        """Compute the Laplacian of f on an un-batched input.

        Args:
            params: The parameters of the neural network.
            x: The input tensor.

        Returns:
            The Laplacian of f at x. Has the same shape as f(x).
        """
        return einsum(hess_f(params, x), tr_equation)

    return laplacian


def jet_naive_laplacian(
    f: Callable[[list[ArrayLike], ArrayLike], ArrayLike], dummy_x: ArrayLike
) -> Callable[[list[ArrayLike], ArrayLike], ArrayLike]:
    """Generate a function that computes the Laplacian of f using jets.

    Args:
        f: The function whose Laplacian we want to compute.
            The function should take the parameters and the input tensor as arguments
            and return the output tensor.
        dummy_x: A dummy input tensor to determine the input dimensions.

    Returns:
        A function that computes the Laplacian of f at the input tensor X given the
        parameters and X.
    """
    D = size(dummy_x)

    def laplacian(params: list[ArrayLike], x: ArrayLike) -> ArrayLike:
        """Compute the Laplacian of f on an un-batched input.

        Args:
            params: The parameters of the neural network.
            x: The un-batched input tensor.

        Returns:
            The Laplacian of f at x. Has the same shape as f(x).
        """
        v2 = zeros(dummy_x.shape, dtype=dummy_x.dtype, device=dummy_x.device)

        def d2(x, v1):
            _, (_, f2) = jet(lambda x: f(params, x), (x,), ((v1, v2),))
            return f2

        d2_vmap = vmap(lambda v1: d2(x, v1))
        v1 = eye(D, dtype=dummy_x.dtype, device=dummy_x.device).reshape(
            D, *dummy_x.shape
        )
        return d2_vmap(v1).sum(0)

    return laplacian


def jet_simplified_laplacian(
    f: Callable[[list[ArrayLike], ArrayLike], ArrayLike], dummy_x: ArrayLike
) -> Callable[[list[ArrayLike], ArrayLike], ArrayLike]:
    """Generate a function that computes the Laplacian of forward Laplacian of f.

    Args:
        f: The function whose Laplacian we want to compute.
            The function should take the parameters and the input tensor as arguments
            and return the output tensor.
        dummy_x: A dummy input tensor to determine the input dimensions.

    Returns:
        A function that computes the Laplacian of f at the input tensor X given the
        parameters and X.
    """
    # disable sparsity to remove its run time benefits
    fwd_lap = ForwardLaplacianOperator(0)

    def laplacian(params: list[ArrayLike], x: ArrayLike) -> ArrayLike:
        """Compute the Laplacian of f on an un-batched input.

        Args:
            params: The parameters of the neural network.
            x: The un-batched input tensor.

        Returns:
            The Laplacian of f at x. Has the same shape as f(x).
        """
        lap_f = fwd_lap(lambda x: f(params, x))
        return lap_f(x)[0]

    return laplacian


def laplacian_function(
    params_and_f: tuple[
        list[ArrayLike], Callable[[list[ArrayLike], ArrayLike], ArrayLike]
    ],
    X: ArrayLike,
    is_batched: bool,
    strategy: str,
) -> Callable[[list[ArrayLike], ArrayLike], ArrayLike]:
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
        A function that computes the Laplacian of the function f given X and params.
    """
    _, f = params_and_f
    dummy_X = X[0] if is_batched else X

    transforms = {
        "hessian_trace": hessian_trace_laplacian,
        "jet_naive": jet_naive_laplacian,
        "jet_simplified": jet_simplified_laplacian,
    }
    laplacian = transforms[strategy](f, dummy_X)

    if is_batched:
        laplacian = vmap(laplacian, in_axes=[None, 0])

    return laplacian


def randomized_laplacian_function(
    params_and_f: tuple[
        list[ArrayLike], Callable[[list[ArrayLike], ArrayLike], ArrayLike]
    ],
    X: ArrayLike,
    is_batched: bool,
    strategy: str,
    distribution: str,
    num_samples: int,
) -> Callable[[list[ArrayLike], ArrayLike], ArrayLike]:
    """Construct function to compute the MC Laplacian in JAX with different strategies.

    The MC-Laplacian is computed with 2-jets or nested vector-Hessian-vector products.

    Args:
        params_and_f: The neural net's parameters and the unbatched forward function
            whose MC Laplacian we want to compute. The function should take the
            parameters and the input tensor as arguments and return the output tensor.
        X: The input tensor at which to compute the Bi-Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the MC Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The MC Laplacian is computed by multiplying the Hessian
              against random vectors, using VHVPs.
            - `'jet_naive'`: The Laplacian is approximated using 2-jets.
        distribution: The distribution to use for the random vectors. Can be `'normal'`.
        num_samples: The number of samples to use for the random vectors.

    Returns:
        A function that computes the MC Bi-Laplacian of f given X and params.
    """
    _, f = params_and_f
    dummy_X = X[0] if is_batched else X

    # draw random vectors
    key = PRNGKey(2)
    sample_func = {"normal": normal}[distribution]
    V = device_put(
        sample_func(key, shape=(num_samples, *X.shape), dtype=dummy_X.dtype),
        dummy_X.device,
    )

    if strategy == "hessian_trace":
        d2f_vv = vector_hessian_vector_product(f, dummy_X)

    elif strategy == "jet_naive":

        def d2f_vv(params: list[ArrayLike], x: ArrayLike, v: ArrayLike) -> ArrayLike:
            """Multiply the 2-nd order derivative tensor with v.

            Args:
                params: The parameters of the neural network.
                x: The un-batched input tensor.
                v: The random vector to multiply with.

            Returns:
                The 2-nd order derivative tensor of f at x multiplied with v.
                Has the same shape as f(params, x).
            """
            v2 = zeros(dummy_X.shape, dtype=dummy_X.dtype, device=dummy_X.device)
            _, (_, f2) = jet(lambda x: f(params, x), (x,), ((v, v2),))
            return f2

    else:
        raise ValueError(f"Unsupported strategy: {strategy}.")

    # vmap over data points
    if is_batched:
        d2f_vv = vmap(d2f_vv, in_axes=[None, 0, 0])

    # vmap over vectors and fix them
    d2f_VV = vmap(d2f_vv, in_axes=[None, None, 0])
    d2f_fix_V = lambda params, X: d2f_VV(params, X, V)  # noqa: E731

    return lambda params, X: d2f_fix_V(params, X).mean(0) / 3


def bilaplacian_function(
    params_and_f: tuple[
        list[ArrayLike], Callable[[list[ArrayLike], ArrayLike], ArrayLike]
    ],
    X: ArrayLike,
    is_batched: bool,
    strategy: str,
) -> Callable[[list[ArrayLike], ArrayLike], ArrayLike]:
    """Construct function to compute the Bi-Laplacian in JAX using different strategies.

    The Bi-Laplacian is computed by taking the Laplacian of the Laplacian.

    Args:
        params_and_f: The neural net's parameters and the unbatched forward function
            whose Bi-Laplacian we want to compute. The function should take the
            parameters and the input tensor as arguments and return the output tensor.
        X: The input tensor at which to compute the Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the Bi-Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The Bi-Laplacian is computed by tracing the Hessian,
              then again taking and tracing the Hessian of the result.
              The Hessian is computed via forward-over-reverse mode autodiff.
            - `'jet_naive'`: The Bi-Laplacian is computed using jets.
            - `'jet_simplified'`: The Bi-Laplacian is computed using the forward
              Laplacian library.

    Returns:
        A function that computes the Bi-Laplacian of the function f given X and params.
    """
    _, f = params_and_f
    dummy_X = X[0] if is_batched else X

    transform = {
        "hessian_trace": hessian_trace_laplacian,
        "jet_naive": jet_naive_laplacian,
        "jet_simplified": jet_simplified_laplacian,
    }[strategy]
    laplacian = transform(f, dummy_X)
    bilaplacian = transform(laplacian, dummy_X)

    if is_batched:
        bilaplacian = vmap(bilaplacian, in_axes=[None, 0])

    return bilaplacian


def randomized_bilaplacian_function(
    params_and_f: tuple[
        list[ArrayLike], Callable[[list[ArrayLike], ArrayLike], ArrayLike]
    ],
    X: ArrayLike,
    is_batched: bool,
    strategy: str,
    distribution: str,
    num_samples: int,
) -> Callable[[list[ArrayLike], ArrayLike], ArrayLike]:
    """Construct function to compute the MC Bi-Laplace in JAX with different strategies.

    The Bi-Laplacian is computed with 4-jets or nested vector-Hessian-vector products.

    Args:
        params_and_f: The neural net's parameters and the unbatched forward function
            whose MC Bi-Laplacian we want to compute. The function should take the
            parameters and the input tensor as arguments and return the output tensor.
        X: The input tensor at which to compute the Bi-Laplacian.
        is_batched: Whether the input is a batched tensor.
        strategy: Which strategy will be used by the returned function to compute
            the MC Bi-Laplacian. The following strategies are supported:
            - `'hessian_trace'`: The MC Bi-Laplacian is computed by multiplying the
              tensor of 4th-order derivatives against random vectors, using VHVPs
            - `'jet_naive'`: The Bi-Laplacian is approximated using 4-jets.
        distribution: The distribution to use for the random vectors. Can be `'normal'`.
        num_samples: The number of samples to use for the random vectors.

    Returns:
        A function that computes the MC Bi-Laplacian of f given X and params.
    """
    _, f = params_and_f
    dummy_X = X[0] if is_batched else X

    # draw random vectors
    key = PRNGKey(2)
    sample_func = {"normal": normal}[distribution]
    V = device_put(
        sample_func(key, shape=(num_samples, *X.shape), dtype=dummy_X.dtype),
        dummy_X.device,
    )

    if strategy == "hessian_trace":
        # nest vector-Hessian-vector products to multiply with 4th-order derivatives
        d2f_vv = vector_hessian_vector_product(f, dummy_X)
        d4f_vvvv = lambda params, x, v: vector_hessian_vector_product(  # noqa: E731
            lambda params, x: d2f_vv(params, x, v), dummy_X
        )(params, x, v)

    elif strategy == "jet_naive":

        def d4f_vvvv(params: list[ArrayLike], x: ArrayLike, v: ArrayLike) -> ArrayLike:
            """Multiply the 4-th order derivative tensor with v.

            Args:
                params: The parameters of the neural network.
                x: The un-batched input tensor.
                v: The random vector to multiply with.

            Returns:
                The 4-th order derivative tensor of f at x multiplied with v.
                Has the same shape as f(params, x).
            """
            v234 = zeros(dummy_X.shape, dtype=dummy_X.dtype, device=dummy_X.device)
            _, (_, _, _, f4) = jet(
                lambda x: f(params, x), (x,), ((v, v234, v234, v234),)
            )
            return f4

    else:
        raise ValueError(f"Unsupported strategy: {strategy}.")

    # vmap over data points
    if is_batched:
        d4f_vvvv = vmap(d4f_vvvv, in_axes=[None, 0, 0])

    # vmap over vectors and fix them
    d4f_VVVV = vmap(d4f_vvvv, in_axes=[None, None, 0])
    d4f_fix_V = lambda params, X: d4f_VVVV(params, X, V)  # noqa: E731

    return lambda params, X: d4f_fix_V(params, X).mean(0) / 3


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
        A tuple containing the jitted functions to compute the operator w/o being
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
    description = f"{strategy}, compiled=True"
    if is_stochastic:
        description += f", {distribution=}, {num_samples=}"

    if operator == "laplacian":
        op_func = (
            randomized_laplacian_function(*args)
            if is_stochastic
            else laplacian_function(*args)
        )
    elif operator == "bilaplacian":
        op_func = (
            randomized_bilaplacian_function(*args)
            if is_stochastic
            else bilaplacian_function(*args)
        )
    else:
        raise ValueError(f"Unsupported operator: {operator}.")

    op_func = jit(op_func)

    # function that computes the Laplacian
    func = lambda: op_func(params, X)  # noqa: E731
    # function that computes the operator's gradient used as proxy for the
    # computation graph's memory footprint
    summed_op_func = lambda params, X: op_func(params, X).sum()  # noqa: E731
    grad_func = lambda: grad(summed_op_func, argnums=0)(params, X)  # noqa: E731

    # jit the functions
    func, grad_func = jit(func), jit(grad_func)

    # add a trailing statement to wait until the computations are done
    return (
        lambda: block_until_ready(func()),
        lambda: block_until_ready(grad_func()),
        description,
    )


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

    # 2) On CPU, measuring peak memory does not seem to work
    mem = measure_peak_memory(func, f"{op} ({description})", is_cuda, use_jax=True)
    if not is_cuda:
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
