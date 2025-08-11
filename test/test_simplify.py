"""Test simplification mechanism on compute graphs of the (Bi-)Laplacian."""

from test.test___init__ import compare_jet_results, setup_case
from test.test_bilaplacian import bilaplacian
from test.test_laplacian import (
    DISTRIBUTION_IDS,
    DISTRIBUTIONS,
    WEIGHT_IDS,
    WEIGHTS,
    get_coefficients,
    get_weighting,
    laplacian,
)
from test.utils import report_nonclose
from typing import Any, Callable

from pytest import mark
from torch import Size, Tensor, arange, manual_seed, sigmoid, sin, tanh, tensor
from torch.fx import Graph, GraphModule
from torch.nn import Linear, Module, Sequential, Tanh
from torch.nn.functional import linear

from jet.bilaplacian import Bilaplacian
from jet.laplacian import Laplacian
from jet.rules import is_replicate
from jet.simplify import common_subexpression_elimination, simplify
from jet.tracing import capture_graph
from jet.utils import recursive_getattr

# make generation of test cases deterministic
manual_seed(0)

SIMPLIFY_CASES = [
    # 1d sine function
    {"f": sin, "shape": (1,), "id": "sin"},
    # 2d sine function
    {"f": sin, "shape": (2,), "id": "sin"},
    # 2d sin(sin) function
    {"f": lambda x: sin(sin(x)), "shape": (2,), "id": "sin-sin"},
    # 2d tanh(tanh) function
    {"f": lambda x: tanh(tanh(x)), "shape": (2,), "id": "tanh-tanh"},
    # 2d linear(tanh) function
    {
        "f": lambda x: linear(
            tanh(x),
            tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]).double(),
            bias=tensor([0.12, -0.34]).double(),
        ),
        "shape": (3,),
        "id": "tanh-linear",
    },
    # 5d tanh-activated two-layer MLP
    {
        "f": Sequential(
            Linear(5, 4, bias=False), Tanh(), Linear(4, 1, bias=True), Tanh()
        ),
        "shape": (5,),
        "id": "two-layer-tanh-mlp",
    },
    # 3d sigmoid(sigmoid) function
    {
        "f": lambda x: sigmoid(sigmoid(x)),
        "shape": (3,),
        "id": "sigmoid-sigmoid",
    },
]


def count_replicate_nodes(f: Callable | Module | GraphModule) -> int:
    """Count the number of `replicate` nodes in the compute graph of a function.

    Args:
        f: The function or module to analyze. If a `GraphModule`, it is used directly.
            If a `Module` or function, it is traced first.

    Returns:
        The number of `replicate` nodes in the compute graph of the function.
    """
    mod = capture_graph(f)
    return len([n for n in mod.graph.nodes if is_replicate(n)])


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
    replicates = [n for n in parents if is_replicate(n)]
    assert len(replicates) == num_replicates


def ensure_tensor_constants_collapsed(
    mod: GraphModule,
    collapsed_shape: Size | tuple[int, ...],
    non_collapsed_shape: Size | tuple[int, ...],
    other_shapes: list[Size | tuple[int, ...]] | None = None,
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


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("config", SIMPLIFY_CASES, ids=[c["id"] for c in SIMPLIFY_CASES])
@mark.parametrize(
    "distribution", [None] + DISTRIBUTIONS, ids=["exact"] + DISTRIBUTION_IDS
)
def test_simplify_laplacian(
    config: dict[str, Any], distribution: str | None, weights: str | None
):
    """Test the simplification of a Laplacian's compute graph.

    Replicate nodes should be propagated down the graph.
    Sum nodes should be propagated up.

    Args:
        config: The configuration of the test case.
        distribution: The distribution from which to draw random vectors.
            If `None`, the exact Laplacian is computed. Default: `None`.
        weights: The weighting to use for the Laplacian. If `None`, the Laplacian is
            unweighted. If `diagonal_increments`, a synthetic coefficient tensor is
            used that has diagonal elements that are increments of 1 starting from 1.
    """
    num_samples, seed = 42, 1  # only relevant with randomization
    randomization = None if distribution is None else (distribution, num_samples)

    f, x, _, _ = setup_case({**config, "is_batched": None})

    weighting = get_weighting(x, weights, randomization=randomization)
    mod = Laplacian(f, x, randomization=randomization, weighting=weighting)

    # we have to set the random seed to make sure the same random vectors are used
    if randomization is not None:
        manual_seed(seed)
    mod_out = mod(x)

    if randomization is None:
        C = get_coefficients(x, weights)
        lap = laplacian(f, x, C)
        assert lap.allclose(mod_out[2])
        print("Exact Laplacian in functorch and jet match.")

    # trace and simplify the module

    # we have to set the random seed because tracing executes the functions that
    # draw random vectors and stores them as tensor constants
    if randomization is not None:
        manual_seed(seed)
    fast = simplify(mod, verbose=True, test_x=x)

    # make sure the simplified module still behaves the same
    fast_out = fast(x)
    compare_jet_results(mod_out, fast_out)
    print("Laplacian via jet matches Laplacian via simplified module.")

    # make sure the `replicate` node from the 0th component made it to the end
    ensure_outputs_replicates(fast.graph, num_outputs=3, num_replicates=1)

    # make sure the module's tensor constant corresponding to the highest
    # Taylor coefficient was collapsed
    num_vectors = x.numel() if randomization is None else num_samples
    non_collapsed_shape = (num_vectors, *x.shape)
    collapsed_shape = x.shape
    ensure_tensor_constants_collapsed(
        fast, collapsed_shape, non_collapsed_shape, at_least=1, strict=False
    )


@mark.parametrize("config", SIMPLIFY_CASES, ids=[c["id"] for c in SIMPLIFY_CASES])
@mark.parametrize(
    "distribution", [None] + ["normal"], ids=["exact", "distribution=normal"]
)
def test_simplify_bilaplacian(config: dict[str, Any], distribution: str | None):
    """Test the simplifications for the Bi-Laplacian module.

    Args:
        config: The configuration of the test case.
        distribution: The distribution from which to draw random vectors.
            If `None`, the exact Bi-Laplacian is computed.
    """
    randomized = distribution is not None
    num_samples, seed = 42, 1  # only relevant with randomization
    f, x, _, _ = setup_case({**config, "is_batched": None})

    randomization = (distribution, num_samples) if randomized else None

    bilap_mod = Bilaplacian(f, x, randomization=randomization)

    # we have to set the random seed to make sure the same random vectors are used
    if randomized:
        manual_seed(seed)
    bilap = bilap_mod(x)

    if not randomized:
        bilap_true = bilaplacian(f, x)
        assert bilap_true.allclose(bilap)
        print("Exact Bi-Laplacian in functorch and jet match.")

    # simplify the traced module

    # we have to set the random seed because tracing executes the functions that
    # draw random vectors and stores them as tensor constants
    if randomized:
        manual_seed(seed)
    simple_mod = simplify(
        bilap_mod, verbose=True, eliminate_tensor_constants=False, test_x=x
    )

    # make sure the `replicate` node from the 0th component made it to the end
    ensure_outputs_replicates(simple_mod.graph, num_outputs=1, num_replicates=0)

    # make sure that Taylor coefficients were collapsed
    D = x.numel()
    collapsed_shape = x.shape

    if randomized:
        num_vectors = num_samples
        non_collapsed_shape = (num_vectors, *x.shape)
        ensure_tensor_constants_collapsed(
            simple_mod, collapsed_shape, non_collapsed_shape, at_least=1, strict=False
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

        non_collapsed_shapes = {
            non_collapsed_shape1,
            non_collapsed_shape2,
            non_collapsed_shape3,
        }

        # uses three 4-jets
        num_collapsed = 3 if D > 1 else 1

        for non_collapsed in non_collapsed_shapes:
            ensure_tensor_constants_collapsed(
                simple_mod,
                collapsed_shape,
                non_collapsed,
                other_shapes=list(non_collapsed_shapes - {non_collapsed}),
                at_least=num_collapsed,
                strict=False,
            )

    # make sure the simplified module still behaves the same
    bilap_simple = simple_mod(x)
    report_nonclose(bilap, bilap_simple, name="Bi-Laplacians")

    # also remove duplicate tensor_constants
    simpler_mod = simplify(
        simple_mod, verbose=True, eliminate_tensor_constants=True, test_x=x
    )

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
            "sigmoid-sigmoid": 181,
        }
        assert len(list(simpler_mod.graph.nodes)) == expected_nodes[config["id"]]


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

    f_traced = capture_graph(f)
    f_x = f_traced(x)
    # there should be 7 nodes: x, x1, x2, y1, y2, z, output
    assert len(list(f_traced.graph.nodes)) == 7

    common_subexpression_elimination(f_traced.graph, verbose=True)
    # there should be 5 nodes after CSE: x, v=x+1, w=2*v, z=w+w, output
    assert len(list(f_traced.graph.nodes)) == 5

    report_nonclose(f_x, f_traced(x), name="f(x)")
