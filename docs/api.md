## PyTrees

Several functions below accept and return **pytrees**: arbitrarily nested
`tuple`, `list`, and `dict` containers (with string keys) whose leaves are
tensors. A bare tensor counts as a (trivial) pytree. Wherever a signature reads
`PyTree[Tensor]` you may pass any such nesting — the transform maps over the
tensor leaves and returns a result mirroring the input's structure, using
PyTorch's pytree machinery to flatten and rebuild it.

The jet transforms further use `PyTree[Jet]`, where each tensor leaf is replaced
by a *jet tuple* `(primal, c_1, ..., c_K)`: a tensor bundled with its `K` Taylor
coefficients.

## Function transformations

### ::: jet.jet

### ::: jet.laplacian

### ::: jet.bilaplacian

## Graph capture

### ::: jet.capture_graph

### ::: jet.common_subexpression_elimination

### ::: jet.visualize_graph
