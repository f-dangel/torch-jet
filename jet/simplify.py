"""Functions to simplify a compute graph captured with `torch.fx`."""

import operator

from torch import add as torch_add
from torch import cos, cosh, div, einsum, mul
from torch import pow as torch_pow
from torch import sigmoid, sin
from torch import sub as torch_sub
from torch import tanh
from torch.fx import GraphModule, Node
from torch.nn.functional import linear

from jet.utils import replicate


class RewriteReplicate:
    def __init__(self, graph, verbose: bool = False):
        self.graph = graph
        self.verbose = verbose

    @staticmethod
    def is_replicate(arg):
        return (
            isinstance(arg, Node)
            and arg.op == "call_function"
            and arg.target == replicate
        )

    def parents(self, node):
        return [n for n in self.graph.nodes if n in node.all_input_nodes]

    def children(self, node):
        return [n for n in self.graph.nodes if node in n.all_input_nodes]

    def find_pattern(self):
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue

            pattern = None

            # operations that consume a single tensor and no other arguments
            if (
                node.target
                in {
                    cos,
                    tanh,
                    sigmoid,
                    cosh,
                    torch_pow,
                    sin,
                    operator.pow,
                }
                and len(self.parents(node)) == 1
                and all(self.is_replicate(p) for p in self.parents(node))
            ):
                pattern = [self.parents(node), node]

            # a linear layer that processes a replicated input tensor
            elif node.target == linear and self.is_replicate(node.args[0]):
                pattern = [[node.args[0]], node]

            # operations that consume a single tensor and a scalar
            elif (
                node.target
                in {
                    mul,
                    operator.mul,
                    div,
                    operator.truediv,
                    operator.add,
                    torch_add,
                    operator.sub,
                    torch_sub,
                }
                and len(self.parents(node)) == 1
                and all(self.is_replicate(p) for p in self.parents(node))
            ):
                pattern = [self.parents(node), node]

            # operations that consume two tensors tensor and nothing else
            elif (
                node.target
                in {
                    torch_add,
                    operator.add,
                    mul,
                    operator.mul,
                    torch_sub,
                    operator.sub,
                }
                and len(node.args) == 2
                and all(self.is_replicate(arg) for arg in node.args)
            ):
                pattern = [list(node.args), node]

            # operations that consume multiple tensors and nothing else
            elif (
                node.target == einsum
                and node.args[0] == ",".join(len(node.args[1:]) * ["..."]) + "->..."
                and all(self.is_replicate(arg) for arg in node.args[1:])
            ):
                pattern = [list(node.args[1:]), node]

            if pattern is not None:
                print(f"Can swap {pattern[0]} and {pattern[1]}")
                return pattern

    def maybe_erase(self, rep):
        children = self.children(rep)
        if len(children) == 0:
            print(f"Erasing {rep}.")
            self.graph.erase_node(rep)
        else:
            print(f"Not removing {rep} because it has children {children}.")

    def replace_pattern(self, pattern):
        replicates, op = pattern
        replicate_parents = {}
        for rep in replicates:
            (rep_parent,) = self.parents(rep)
            replicate_parents[rep] = rep_parent

        with self.graph.inserting_after(op):
            new_rep = self.graph.call_function(replicate, kwargs=replicates[0].kwargs)
        op.replace_all_uses_with(new_rep)

        op.args = tuple(replicate_parents.get(arg, arg) for arg in op.args)

        new_rep.args = (op,) + replicates[0].args[1:]

        for rep in replicates:
            self.maybe_erase(rep)

    def maybe_print(self, message: str):
        if self.verbose:
            print(message)


def simplify(mod: GraphModule, verbose: bool = False) -> GraphModule:
    """Simplify a compute graph.

    At the moment, the following simplifications are implemented:

    - Propagation of `replicate` nodes down the graph as much as possible.
      This avoids redundant computations on replicated tensors.

    Args:
        mod: A computation graph that will be simplified.
        verbose: Whether to print debug information. Default: `False`.

    Returns:
        The simplified computation graph.
    """
    if verbose:
        print(f"Traced graph before simplification:\n{mod.graph}")

    rewriter = RewriteReplicate(mod.graph, verbose=verbose)
    while pattern := rewriter.find_pattern():
        rewriter.replace_pattern(pattern)

    mod.graph.lint()
    mod.recompile()

    if verbose:
        print(f"Traced graph after simplification:\n{mod.graph}")

    return mod
