"""Test simplification mechanism on compute graphs of the (Bi-)Laplacian."""

from typing import Any, Callable

import torch
from pytest import mark
from torch import Tensor, arange, float64, manual_seed, sigmoid, sin, tanh, tensor
from torch.fx import Graph, GraphModule, Node
from torch.nn import Linear, Sequential, Tanh
from torch.nn.functional import linear

from jet.bilaplacian import Bilaplacian
from jet.laplacian import Laplacian
from jet.simplify import common_subexpression_elimination, simplify
from jet.tracing import capture_graph
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

# make generation of test cases deterministic
manual_seed(0)

_TANH_LINEAR_W = tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=float64)
_TANH_LINEAR_B = tensor([0.12, -0.34], dtype=float64)

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
        "f": lambda x: linear(tanh(x), _TANH_LINEAR_W, bias=_TANH_LINEAR_B),
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


def count_nodes(graph: Graph, predicate: Callable[[Node], bool]) -> int:
    """Count nodes in an FX graph that satisfy a predicate.

    Args:
        graph: The FX graph to inspect.
        predicate: Function that returns True for nodes to count.

    Returns:
        The number of matching nodes.
    """
    return len([n for n in graph.nodes if predicate(n)])


def get_output_args(graph: Graph) -> tuple[Node, ...]:
    """Get the direct parent nodes of the output node.

    For graphs returning a tuple (e.g., jet functions returning (F0, F1, F2)),
    this returns the individual tensor nodes that form the output tuple.

    Args:
        graph: The FX graph to inspect.

    Returns:
        Tuple of nodes that are the direct arguments of the output node.
    """
    output_node = next(n for n in graph.nodes if n.op == "output")
    args = output_node.args[0]
    if isinstance(args, (tuple, list)):
        return tuple(args)
    return (args,)


def _assert_bilaplacian_structure(
    simple_mod: GraphModule,
    x: Tensor,
    config: dict[str, Any],
    randomized: bool,
) -> None:
    """Assert structural properties of bilaplacian simplification.

    Verifies that simplification rules actually fired and the resulting graph
    has the expected structure (node counts).
    """
    D = x.numel()

    # The bilaplacian output should be a single tensor (not a tuple)
    out_args_simple = get_output_args(simple_mod.graph)
    assert len(out_args_simple) == 1, (
        f"Bilaplacian should return 1 output, got {len(out_args_simple)}"
    )

    # Exact node counts: detect regressions if simplification rules stop firing
    if not randomized:
        expected_nodes = {
            "sin": 35 if D == 1 else 101,
            "sin-sin": 198,
            "tanh-tanh": 244,
            "tanh-linear": 147,
            "two-layer-tanh-mlp": 361,
            "sigmoid-sigmoid": 240,
        }
        n_nodes = len(list(simple_mod.graph.nodes))
        expected = expected_nodes[config["id"]]
        assert n_nodes == expected, (
            f"Expected {expected} nodes for {config['id']} (D={D}), got {n_nodes}"
        )


@mark.parametrize("weights", WEIGHTS, ids=WEIGHT_IDS)
@mark.parametrize("config", SIMPLIFY_CASES, ids=[c["id"] for c in SIMPLIFY_CASES])
@mark.parametrize(
    "distribution", [None] + DISTRIBUTIONS, ids=["exact"] + DISTRIBUTION_IDS
)
def test_simplify_laplacian(
    config: dict[str, Any],
    distribution: str | None,
    weights: str | None | tuple[str, float],
):
    """Test the simplification of a Laplacian's compute graph.

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

    f, x, _ = setup_case(config)

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
    fast = simplify(mod, x, verbose=True, test_x=x)

    # make sure the simplified module still behaves the same
    # With make_fx, random ops (aten.randn) are in the graph, so we must set the
    # seed before each evaluation to get deterministic results
    if randomization is not None:
        manual_seed(seed)
    fast_out = fast(x)
    compare_jet_results(mod_out, fast_out)
    print("Laplacian via jet matches Laplacian via simplified module.")


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
    f, x, _ = setup_case(config)

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
        bilap_mod,
        x,
        verbose=True,
        test_x=x,
    )

    # make sure the simplified module still behaves the same
    if randomized:
        manual_seed(seed)
    bilap_simple = simple_mod(x)
    report_nonclose(bilap, bilap_simple, name="Bi-Laplacians")

    # Structural assertions: verify simplification rules actually fired
    _assert_bilaplacian_structure(simple_mod, x, config, randomized)


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

    f_traced = capture_graph(f, x)
    f_x = f_traced(x)
    nodes_before = len(list(f_traced.graph.nodes))
    # make_fx produces ATen-level nodes; verify duplicate subexpressions exist
    assert nodes_before >= 7

    # Count specific duplicate ops before CSE
    _add_target = torch.ops.aten.add.Tensor
    adds_before = count_nodes(f_traced.graph, lambda n: n.target == _add_target)

    common_subexpression_elimination(f_traced.graph, verbose=True)
    nodes_after = len(list(f_traced.graph.nodes))
    # CSE should have removed at least some duplicate nodes
    assert nodes_after < nodes_before

    # CSE should have eliminated duplicate aten.add.Tensor nodes (x+1 appeared twice)
    adds_after = count_nodes(f_traced.graph, lambda n: n.target == _add_target)
    assert adds_after < adds_before, (
        f"CSE should reduce duplicate add nodes: {adds_before} -> {adds_after}"
    )

    report_nonclose(f_x, f_traced(x), name="f(x)")


@mark.parametrize("config", SIMPLIFY_CASES, ids=[c["id"] for c in SIMPLIFY_CASES])
def test_full_simplification_structural(config: dict[str, Any]):
    """Verify structural properties of full simplification.

    Checks that the node count matches expectations (regression detection).

    Args:
        config: The configuration of the test case.
    """
    f, x, _ = setup_case(config)
    mod = Laplacian(f, x)

    simplified = simplify(mod, x)

    # Exact node counts: detect regressions if simplification rules stop firing
    expected_nodes = {
        "sin": 16,
        "sin-sin": 25,
        "tanh-tanh": 41,
        "tanh-linear": 43,
        "two-layer-tanh-mlp": 73,
        "sigmoid-sigmoid": 37,
    }
    n_nodes = len(list(simplified.graph.nodes))
    expected = expected_nodes[config["id"]]
    assert n_nodes == expected, (
        f"Expected {expected} nodes for {config['id']}, got {n_nodes}"
    )

    # Verify output correctness
    mod_out = mod(x)
    simplified_out = simplified(x)
    compare_jet_results(mod_out, simplified_out)
