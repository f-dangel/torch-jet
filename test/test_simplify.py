"""Test simplification mechanism of compute graphs captured with `torch.fx`.

There are three kinds of tests:

1. [Replicating functions] Take a function f: x -> f(x).
   i. Construct the replicated function f_rep: x -> f(replicate(x))
   ii. Simplify the compute graph of f_rep. This should yield the compute graph of
       x -> replicate(f(x)).

2. [Replicating jets] Take a function f: x -> f(x).
    i. Construct the jet of f, jet_f: x, v1, v2, ... -> jet_f(x, v1, v2, ...)
    ii. Construct the replicated jet of f, jet_f_rep:
        x, v1, v2, ... -> jet_f(replicate(x), replicate(v1), replicate(v2), ...)
    iii. Simplify the compute graph of jet_f_rep. This should yield the compute graph of
        x, v1, v2, ... -> replicate(jet_f(x, v1, v2, ...)).

3. [Forward Laplacians] Take a function f: x -> f(x).
    i. Construct the vmapped 2-jet of f: X, V1, V2 -> jet_f(X, V1, V2)
    ii. Construct the Laplacian of f on x via the vmapped 2-jet.
    iii. Simplify the compute graph of the Laplacian. This should yield the compute
        graph of the forward Laplacian (https://arxiv.org/abs/2307.08214), i.e. the
        second Taylor component can be propagated in collapsed form, and the forward
        pass is only carried out for one x, and not their replicated version X.
"""

from functools import partial
from test.test___init__ import (
    ATOMIC_CASE_IDS,
    ATOMIC_CASES,
    CASE_IDS,
    CASES,
    CASES_COMPACT,
    CASES_COMPACT_IDS,
    K_MAX,
    compare_jet_results,
    report_nonclose,
    setup_case,
)
from test.test_bilaplacian import bilaplacian
from test.test_laplacian import DISTRIBUTION_IDS, DISTRIBUTIONS, laplacian
from test.test_weighted_laplacian import weighted_laplacian
from typing import Any, Callable, Dict, Optional, Tuple, Union

from pytest import mark, skip
from torch import Size, Tensor, arange, manual_seed, rand
from torch.fx import Graph, GraphModule, symbolic_trace, wrap
from torch.nn import Module

from jet import JetTracer, jet, rev_jet
from jet.bilaplacian import Bilaplacian, RandomizedBilaplacian
from jet.laplacian import Laplacian, RandomizedLaplacian
from jet.simplify import (
    RewriteReplicate,
    RewriteSumVmapped,
    common_subexpression_elimination,
    simplify,
)
from jet.utils import (
    PrimalAndCoefficients,
    ValueAndCoefficients,
    WrapperModule,
    integer_partitions,
    recursive_getattr,
    replicate,
    sum_vmapped,
)
from jet.weighted_laplacian import (
    C_func_diagonal_increments,
    RandomizedWeightedLaplacian,
    WeightedLaplacian,
)

# tell `torch.fx` to trace `replicate` as one node
wrap(replicate)
# tell `torch.fx` to trace `sum_vmapped` as one node
wrap(sum_vmapped)


def is_first_op_linear(config: Dict[str, Any]) -> bool:
    """Determine if the first operation in the test case configuration is linear.

    Args:
        config: The configuration of the test case.

    Returns:
        True if the first operation is linear, False otherwise.
    """
    return config["id"].endswith("mlp") or config["id"] == "linear"


class Replicate(Module):
    """Layer that replicates the forward pass of a function.

    This module is trace-able and the trace will correspond to the
    graph that the `jet` function transforms, which means it is close to
    the forward pass in the compute graph of a `jet`.
    """

    def __init__(self, f: Callable[[Tensor], Tensor], num_replica: int) -> None:
        """Initialize the `Replicate` module.

        Args:
            f: The function to replicate. Must process its input like `vmap` if
                the input is a batch of tensors.
            num_replica: The number of replicas to create.
        """
        super().__init__()
        # Wrap the function in a module if it is not already a module.
        # We want to always produce an executable `torch.fx.GraphModule`.
        if not isinstance(f, Module):
            f = WrapperModule(f)
        # Trace the function so we get the same representation as inside `jet`
        graph = JetTracer().trace(f)
        self.traced_f = GraphModule(f, graph)
        self.num_replica = num_replica

    def forward(self, x: Tensor) -> Tensor:
        """Replicate the input tensor, then compute a forward pass through the function.

        Args:
            x: The input tensor.

        Returns:
            The replicated output tensor.
        """
        X = replicate(x, self.num_replica)
        return self.traced_f(X)


def ensure_outputs_replicates(graph: Graph, num_outputs: int, num_replicates: int):
    """Make sure the compute graph outputs only `replicate` nodes.

    Args:
        graph: The compute graph to check.
        num_outputs: The number of nodes that should be returned.
        num_replicates: The number of `replicate` nodes that should be returned.
    """
    output = list(graph.nodes)[-1]  # -1 is the output node
    parents = [n for n in graph.nodes if n in output.all_input_nodes]
    assert len(parents) == num_outputs
    replicates = [n for n in parents if RewriteReplicate.is_replicate(n)]
    assert len(replicates) == num_replicates


def ensure_num_replicates(graph: Graph, num_replicates: int):
    """Make sure the compute graph has the specified number of `replicate` nodes.

    Args:
        graph: The compute graph to check.
        num_replicates: The number of `replicate` nodes to check for.
    """
    replicates = [n for n in graph.nodes if RewriteReplicate.is_replicate(n)]
    assert len(replicates) == num_replicates


def ensure_num_sum_vmapped(graph: Graph, num_sum_vmapped: int):
    """Make sure the compute graph has the specified number of `sum_vmapped` nodes.

    Args:
        graph: The compute graph to check.
        num_sum_vmapped: The number of `sum_vmapped` nodes to check for.
    """
    sum_nodes = [n for n in graph.nodes if RewriteSumVmapped.is_sum_vmapped(n)]
    assert len(sum_nodes) == num_sum_vmapped


def ensure_tensor_constants_collapsed(
    mod: GraphModule,
    collapsed_shape: Union[Size, Tuple[int, ...]],
    non_collapsed_shape: Union[Size, Tuple[int, ...]],
    other_shapes: Optional[list[Union[Size, tuple[int, ...]]]] = None,
    at_least: int = 1,
    strict: bool = True,
):
    """Make sure some tensor constants in the module are collapsed.

    Args:
        mod: The module to check.
        collapsed_shape: The shape of a collapsed tensor constant.
        non_collapsed_shape: The shape of a non-collapsed tensor constant.
        other_shapes: Other admissible shapes that will not lead to errors if
            encountered. Default is `None`, i.e. no other shapes are expected.
        at_least: The smallest number of tensor constants that should be detected as
            collapsed for the check to pass. Default: `1`.
        strict: Whether to raise an error if the number of collapsed tensor
            constants is not exactly `at_least`. Default: `False`.

    Raises:
        ValueError: If the number of collapsed tensor constants is not as expected,
            if there is a tensor constant with an unexpected shape, or if there is
            an overlap between the supplied `other_shapes` and the (non-)collapsed ones.
    """
    other_shapes = [] if other_shapes is None else other_shapes
    if any(s in [collapsed_shape, non_collapsed_shape] for s in other_shapes):
        raise ValueError(
            f"Shape in other_shapes ({other_shapes}) matches either collapsed"
            + f" ({collapsed_shape}) or non-collapsed ({non_collapsed_shape}) shape."
        )

    constants = {
        n.target
        for n in mod.graph.nodes
        if n.op == "get_attr" and n.target.startswith("_tensor_constant")
    }
    for c in constants:
        print(f"Tensor constant {c} has shape {recursive_getattr(mod, c).shape}.")

    num_collapsed = 0
    for c in constants:
        c_tensor = recursive_getattr(mod, c)
        shape = c_tensor.shape
        if shape == collapsed_shape:
            num_collapsed += 1
        elif shape != non_collapsed_shape and shape not in other_shapes:
            raise ValueError(
                f"Unexpected shape for {c}: {shape}. "
                + f"Should be {collapsed_shape} or {non_collapsed_shape}."
                + f" Other accepted shapes are {other_shapes}."
            )

    if num_collapsed < at_least or strict and num_collapsed != at_least:
        raise ValueError(
            f"Expected {'' if strict else '>'}={at_least} collapsed tensor constants. "
            + f" Found {num_collapsed}."
        )


@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
def test_propagate_replication(config: Dict[str, Any], num_replicas: int = 3):
    """Test the propagation of replication node through a compute graph.

    It is always better to compute then replicate, rather than carry out
    redundant computations on a replicated tensor.

    Args:
        config: The configuration of the test case.
        num_replicas: The number of replicas to create. Default: `3`.
    """
    f, x, _, _ = setup_case(config, taylor_coefficients=False)
    f_rep = Replicate(f, num_replicas)

    # check that the `Replicate` module works as expected
    ref = replicate(f(x), num_replicas)
    assert ref.allclose(f_rep(x), atol=1e-7)
    print("Replicate module works as expected.")

    # check that the `Replicate` module can be traced and simplified
    fast = symbolic_trace(f_rep)
    fast = simplify(fast, verbose=True)
    assert ref.allclose(fast(x))
    print("After simplifying, Replicate module still behaves the same.")

    # make sure the `replicate` node made it to the end
    ensure_outputs_replicates(fast.graph, 1, 1)
    # make sure there are no other `replicate` nodes in the graph
    ensure_num_replicates(fast.graph, 1)


class ReplicateJet(Module):
    """Layer that replicates the jet of a given function and is trace-able."""

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        num_replica: int,
        k: int,
        verbose: bool = False,
    ) -> None:
        """Initialize the `ReplicateJet` module.

        Args:
            f: The function whose jet to replicate. Must process its input like `vmap`
                if the input is a batch of tensors.
            num_replica: The number of replicas to create.
            k: The order of the Taylor expansion.
            verbose: Whether to print debug information when creating the jet.
                Default: `False`.
        """
        super().__init__()
        self.jet_f = jet(f, k, vmap=True, verbose=verbose)
        self.k = k
        self.num_replica = num_replica

    def forward(self, s: PrimalAndCoefficients) -> ValueAndCoefficients:
        """Replicate the input tensor and coefficients, then compute the jet.

        Args:
            s: The input tensor and Taylor coefficients.

        Returns:
            The replicated output tensor.
        """
        x, vs = s[0], s[1:]
        X = replicate(x, self.num_replica)
        VS = [replicate(vs[k], self.num_replica) for k in range(self.k)]
        return self.jet_f(X, *VS)


@mark.parametrize("config", CASES, ids=CASE_IDS)
def test_propagate_replication_jet(config: Dict[str, Any], num_replicas: int = 3):
    """Test the propagation of replication nodes through a compute graph of a jet.

    It is always better to compute then replicate, rather than carry out
    redundant computations on a replicated tensor.

    Args:
        config: The configuration of the test case.
        num_replicas: The number of replicas to create. Default: `3`.
    """
    f, x, vs, _ = setup_case(config)
    k = len(vs)

    # use a single jet, then replicate
    jet_f = rev_jet(f, order=k)
    jet_f_result = jet_f(x, *vs)
    jet_f_result = tuple(replicate(j, num_replicas) for j in jet_f_result)

    # # use a replicated jet
    mod = ReplicateJet(f, num_replicas, k)
    mod_result = mod((x, *vs))

    compare_jet_results(jet_f_result, mod_result)
    print("ReplicateJet module works as expected.")

    # simplify the traced module
    fast = symbolic_trace(mod)
    fast = simplify(fast, verbose=True)
    fast_result = fast((x, *vs))
    compare_jet_results(jet_f_result, fast_result)

    # make sure the `replicate` nodes made it to the end
    ensure_outputs_replicates(fast.graph, k + 1, k + 1)
    ensure_num_replicates(fast.graph, k + 1)


@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
@mark.parametrize(
    "distribution", [None] + DISTRIBUTIONS, ids=["exact"] + DISTRIBUTION_IDS
)
def test_simplify_laplacian(config: Dict[str, Any], distribution: Optional[str]):
    """Test the simplification of a Laplacian's compute graph.

    Replicate nodes should be propagated down the graph.
    Sum nodes should be propagated up.

    Args:
        config: The configuration of the test case.
        distribution: The distribution from which to draw random vectors.
            If `None`, the exact Laplacian is computed. Default: `None`.
    """
    randomized = distribution is not None
    num_samples, seed = 42, 1  # only relevant with randomization

    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)
    mod = (
        RandomizedLaplacian(f, x, is_batched, num_samples, distribution)
        if randomized
        else Laplacian(f, x, is_batched)
    )

    # we have to set the random seed to make sure the same random vectors are used
    if randomized:
        manual_seed(seed)
    mod_out = mod(x)

    if not randomized:
        lap = laplacian(f, x)
        assert lap.allclose(mod_out[2])
        print("Exact Laplacian in functorch and jet match.")

    # simplify the traced module

    # we have to set the random seed because tracing executes the functions that
    # draw random vectors and stores them as tensor constants
    if randomized:
        manual_seed(seed)
    fast = symbolic_trace(mod)
    fast = simplify(fast, verbose=True)

    # make sure the simplified module still behaves the same
    fast_out = fast(x)
    compare_jet_results(mod_out, fast_out)
    print("Laplacian via jet matches Laplacian via simplified module.")

    # make sure the `replicate` node from the 0th component made it to the end
    ensure_outputs_replicates(fast.graph, num_outputs=3, num_replicates=1)

    # make sure the module's tensor constant corresponding to the highest
    # Taylor coefficient was collapsed
    if randomized:
        num_vectors = num_samples
    else:
        num_vectors = x.shape[1:].numel() if is_batched else x.numel()
    non_collapsed_shape = (num_vectors, *x.shape)
    collapsed_shape = x.shape

    # NOTE if we have a linear layer at the beginning, or any operation whose second
    # derivative vanishes, the term sum_vmapped(x1 ** 2) will not show up. Therefore
    # the number of collapsed term will be smaller
    first_op_linear = is_first_op_linear(config)
    num_collapsed = 1 if first_op_linear else 2
    ensure_tensor_constants_collapsed(
        fast, collapsed_shape, non_collapsed_shape, at_least=num_collapsed
    )


@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
@mark.parametrize(
    "distribution", [None] + DISTRIBUTIONS, ids=["exact"] + DISTRIBUTION_IDS
)
def test_simplify_weighted_laplacian(
    config: Dict[str, Any], distribution: Optional[str]
):
    """Test the simplification of a weighted Laplacian's compute graph.

    Replicate nodes should be propagated down the graph.
    Sum nodes should be propagated up.

    Args:
        config: The configuration of the test case.
        distribution: The distribution from which to draw random vectors.
            If `None`, the exact Laplacian is computed. Default: `None`.
    """
    randomized = distribution is not None
    num_samples, seed = 42, 1  # only relevant with randomization
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)
    weighting = "diagonal_increments"

    mod = (
        RandomizedWeightedLaplacian(
            f, x, is_batched, num_samples, distribution, weighting
        )
        if randomized
        else WeightedLaplacian(f, x, is_batched, weighting)
    )

    # we have to set the random seed to make sure the same random vectors are used
    if randomized:
        manual_seed(seed)
    mod_out = mod(x)

    if not randomized:
        C_func = partial(C_func_diagonal_increments, is_batched=is_batched)
        H_dot_C = weighted_laplacian(f, x, is_batched, C_func)
        assert H_dot_C.allclose(mod_out[2])
        print("Exact weighted Laplacian in functorch and jet match.")

    # simplify the traced module

    # we have to set the random seed because tracing executes the functions that
    # draw random vectors and stores them as tensor constants
    if randomized:
        manual_seed(seed)
    fast = symbolic_trace(mod)
    fast = simplify(fast, verbose=True)

    # make sure the simplified module still behaves the same
    fast_out = fast(x)
    compare_jet_results(mod_out, fast_out)
    print(
        "Weighted Laplacian via jet matches weighted Laplacian via simplified module."
    )

    # make sure the `replicate` node from the 0th component made it to the end
    ensure_outputs_replicates(fast.graph, num_outputs=3, num_replicates=1)

    # make sure the module's tensor constant corresponding to the highest
    # Taylor coefficient was collapsed
    if randomized:
        num_vectors = num_samples
    else:
        num_vectors = (x.shape[1:] if is_batched else x.shape).numel()
    non_collapsed_shape = (num_vectors, *x.shape)
    collapsed_shape = x.shape

    # NOTE if we have a linear layer at the beginning, or any operation whose second
    # derivative vanishes, the term sum_vmapped(x1 ** 2) will not show up. Therefore
    # the number of collapsed term will be smaller
    first_op_linear = is_first_op_linear(config)
    num_collapsed = 1 if first_op_linear else 2
    ensure_tensor_constants_collapsed(
        fast, collapsed_shape, non_collapsed_shape, at_least=num_collapsed
    )


def test_simplify_remove_unused_nodes():
    """Test removal of unused nodes."""

    def f(x: Tensor) -> Tensor:
        unused1 = x + 1
        # Note how unused1 only becomes unused once we have removed unused2
        unused2 = unused1 + 2  # noqa: F841

        used = x + 3
        return used

    x = arange(10)

    f_traced = symbolic_trace(f)
    f_x = f_traced(x)
    # there should be 5 nodes: x, unused1, unused2, used, output
    assert len(list(f_traced.graph.nodes)) == 5

    f_simple = simplify(f_traced, remove_unused=True, verbose=True)
    # there should be 3 nodes: x, used, output
    assert len(list(f_simple.graph.nodes)) == 3

    report_nonclose(f_x, f_simple(x), name="f(x)")


def test_common_subexpression_elimination():
    """Test common subexpression elimination."""

    def f(x: Tensor) -> Tensor:
        # NOTE that instead of computing y1, y2, we could simply compute y1 and
        # return y1 + y1
        x1 = x + 1
        x2 = x + 1
        y1 = 2 * x1
        y2 = 2 * x2
        z = y1 + y2
        return z

    x = arange(10)

    f_traced = symbolic_trace(f)
    f_x = f_traced(x)
    # there should be 7 nodes: x, x1, x2, y1, y2, z, output
    assert len(list(f_traced.graph.nodes)) == 7

    common_subexpression_elimination(f_traced.graph, verbose=True)
    # there should be 5 nodes after CSE: x, v=x+1, w=2*v, z=w+w, output
    assert len(list(f_traced.graph.nodes)) == 5

    report_nonclose(f_x, f_traced(x), name="f(x)")


class Collapsed(Module):
    """Layer that computes a collapsed K-jet of a function along random directions."""

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        dummy_x: Tensor,
        is_batched: bool,
        k: int,
        num_vectors: int = 3,
    ) -> None:
        """Trace-able layer that computes a collapsed K-jet along random directions.

        Args:
            f: The function whose K-jet to compute.
            dummy_x: A dummy input tensor to determine the shape and dtype of the
                input tensor.
            is_batched: Whether the input tensor is batched.
            k: The order of the Taylor expansion.
            num_vectors: The number of vectors to use for the K-jet. Default: `3`.
        """
        super().__init__()
        self.jet_f = jet(f, k, vmap=is_batched)
        self.x_shape = dummy_x.shape
        self.x_kwargs = {"dtype": dummy_x.dtype, "device": dummy_x.device}
        self.k = k
        self.num_vectors = num_vectors

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Compute the collapsed K-jet along random directions.

        Args:
            x: The input tensor.

        Returns:
            A tuple containing the non-collapsed K-jet components and the collapsed
            K-jet component, which is the sum of all non-collapsed components.
        """
        vs = [
            rand(self.num_vectors, *self.x_shape, **self.x_kwargs)
            for _ in range(self.k)
        ]
        x_replicated = replicate(x, self.num_vectors)
        jet_out = self.jet_f(x_replicated, *vs)
        non_collapsed = jet_out[: self.k]
        collapsed = sum_vmapped(jet_out[self.k])
        return (*non_collapsed, collapsed)


@mark.parametrize("config", ATOMIC_CASES, ids=ATOMIC_CASE_IDS)
@mark.parametrize(
    "k", list(range(1, K_MAX + 1)), ids=[f"{k=}" for k in range(1, K_MAX + 1)]
)
def test_simplify_collapsed_K_jet(
    config: Dict[str, Any], k: int, num_vectors: int = 3
) -> None:
    """Sum backpropagation through a summed K-jet.

    Args:
        config: The configuration of the test case.
        k: The order of the Taylor expansion.
        num_vectors: The number of vectors to use for the K-jet. Default: `3`.
    """
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)
    if config["k_max"] < k:
        skip(f"Skipping {config['id']} for {k=} because k_max={config['k_max']}.")

    collapsed = Collapsed(f, x, is_batched, k)
    traced = symbolic_trace(collapsed)
    simple = simplify(traced, test_x=x, verbose=True)

    # figure out how many tensor constants were collapsed
    terms = list(integer_partitions(k))
    if config["id"] == "linear":
        terms = [t for t in terms if len(t) == 1]
    num_collapsed = len(terms)

    ensure_tensor_constants_collapsed(
        simple,
        collapsed_shape=x.shape,
        non_collapsed_shape=(num_vectors, *x.shape),
        at_least=num_collapsed,
    )


@mark.parametrize("config", CASES_COMPACT, ids=CASES_COMPACT_IDS)
@mark.parametrize(
    "distribution", [None] + ["normal"], ids=["exact", "distribution=normal"]
)
def test_simplify_bilaplacian(config: Dict[str, Any], distribution: Optional[str]):
    """Test the simplifications for the Bi-Laplacian module.

    Args:
        config: The configuration of the test case.
        distribution: The distribution from which to draw random vectors.
            If `None`, the exact Bi-Laplacian is computed.
    """
    randomized = distribution is not None
    num_samples, seed = 42, 1  # only relevant with randomization
    f, x, _, is_batched = setup_case(config, taylor_coefficients=False)

    bilap_mod = (
        RandomizedBilaplacian(f, x, is_batched, num_samples, distribution)
        if randomized
        else Bilaplacian(f, x, is_batched)
    )
    # we have to set the random seed to make sure the same random vectors are used
    if randomized:
        manual_seed(seed)
    bilap = bilap_mod(x)

    if not randomized:
        bilap_true = bilaplacian(f, x, is_batched)
        assert bilap_true.allclose(bilap)
        print("Exact Bi-Laplacian in functorch and jet match.")

    # simplify the traced module

    # we have to set the random seed because tracing executes the functions that
    # draw random vectors and stores them as tensor constants
    if randomized:
        manual_seed(seed)
    simple_mod = symbolic_trace(bilap_mod)
    simple_mod = simplify(simple_mod, verbose=True, test_x=x)

    # make sure the `replicate` node from the 0th component made it to the end
    ensure_outputs_replicates(simple_mod.graph, num_outputs=1, num_replicates=0)

    # NOTE The module creates x1, x2, x3, x4, but torch.fx traces creates tensor
    # constants for x4**4, x1**2 * x2, x1 * x3, x2 ** 2, and x4 for propagating the
    # highest component of each jet. If we have a linear layer at the beginning, meaning
    # all derivatives of degree larger than two disappear, only the term
    # sum_vmapped(x4) should show up. Otherwise, there will be summed constants for all
    # terms (some of them will be removed by common tensor constant elimination).
    first_op_linear = is_first_op_linear(config)

    # make sure that Taylor coefficients were collapsed
    D = (x.shape[1:] if is_batched else x).numel()

    if randomized:
        num_vectors = num_samples
        collapsed_shape = x.shape
        non_collapsed_shape = (num_vectors, *x.shape)
        num_collapsed = 1 if first_op_linear else 2
        ensure_tensor_constants_collapsed(
            simple_mod, collapsed_shape, non_collapsed_shape, at_least=num_collapsed
        )

    else:
        # we need to run three checks because we use D-dimensional 4-jets,
        # D*(D-1)-dimensional 4-jets, and D*(D-1)/2-dimensional 4-jets
        num_vectors1 = D
        non_collapsed_shape1 = (num_vectors1, *x.shape)

        num_vectors2 = D * (D - 1)
        non_collapsed_shape2 = (num_vectors2, *x.shape)

        num_vectors3 = D * (D - 1) // 2
        non_collapsed_shape3 = (num_vectors3, *x.shape)

        collapsed_shape = x.shape
        non_collapsed_shapes = {
            non_collapsed_shape1,
            non_collapsed_shape2,
            non_collapsed_shape3,
        }

        if D == 1:  # uses only one 4-jet
            num_collapsed = 1 if first_op_linear else 2
        elif D in {2, 3}:  # uses three 4-jets, but two of them have same num_vectors
            num_collapsed = 2 if first_op_linear else 4
        else:  # uses three 4-jets, all of them have different num_vectors
            num_collapsed = 1 if first_op_linear else 4

        for non_collapsed in non_collapsed_shapes:
            ensure_tensor_constants_collapsed(
                simple_mod,
                collapsed_shape,
                non_collapsed,
                other_shapes=list(non_collapsed_shapes - {non_collapsed}),
                at_least=num_collapsed,
            )

    # make sure the simplified module still behaves the same
    bilap_simple = simple_mod(x)
    report_nonclose(bilap, bilap_simple, name="Bi-Laplacians")

    # check for a bunch of configs that the number of nodes remains the same
    if not randomized:
        expected_nodes = {
            # NOTE The Bi-Laplacian for a 1d function does not evaluate off-diagonal
            # terms (there are none), hence the number of ops varies
            "sin": 20 if D == 1 else 32,
            "sin-sin": 139,
            "tanh-tanh": 185,
            "tanh-linear": 59,
            "two-layer-tanh-mlp": 255,
            "batched-two-layer-tanh-mlp": 255,
            "sigmoid-sigmoid": 181,
        }
        if config["id"] in expected_nodes:
            assert len(list(simple_mod.graph.nodes)) == expected_nodes[config["id"]]
