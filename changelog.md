# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added/New

- **Backward-incompatible.** `jet()`, `collapsed_jet()`, `laplacian()`, and
  `bilaplacian()` now return plain Python callables (was `GraphModule`).
  `jet()` and `collapsed_jet()` also drop their `derivative_order` argument
  (now inferred per call from the input jet tuples). To bake any of these
  into a `GraphModule` for graph passes (CSE, `torch.compile`, etc.), apply
  `capture_graph` to it yourself — graph capture is now a single explicit
  step at the user's chosen point. `capture_graph(f, mock_args)` now takes
  one tuple of positional pytrees (was variadic tensors) and returns
  `(mod, in_spec)`; use `mod(*in_spec.flatten_up_to(args))` to call the
  captured graph
  ([PR](https://github.com/f-dangel/torch-jet/pull/134))

- **Backward-incompatible.** Rename the `use_collapsing` parameter on
  `laplacian()` and `bilaplacian()` to `collapsed` (defaults unchanged).
  Bundled with an internal cleanup of the jet op dispatch that merges the
  two jet interpreters into one
  ([PR](https://github.com/f-dangel/torch-jet/pull/132))

- **Backward-incompatible.** Bundle each argument's primal with its Taylor
  coefficients. The transforms `jet()`, `collapsed_jet()`, and `rev_jet()` now
  take one argument per argument of `f`, where each tensor leaf is a tuple
  `(x_0, x_1, ..., x_K)` (primal followed by Taylor coefficients), and return a
  pytree mirroring `f`'s output with each leaf a tuple `(f_0, f_1, ..., f_K)`.
  This replaces the previous arg-major `jet_f(primals, taylor_coeffs)` /
  `(primals_out, taylor_coeffs_out)` convention—a jet is now a single
  self-contained object. Inputs and outputs may be arbitrary
  `tuple`/`list`/`dict` pytrees. For a 3-jet of a two-argument function
  `f(x, y)`:

  ```python
  # before: arg-major (primals, taylor_coeffs) -> (primals_out, taylor_coeffs_out)
  f0, (f1, f2, f3) = jet_f((x, y), ((x1, x2, x3), (y1, y2, y3)))

  # after: one (primal, *coeffs) jet per argument -> one (f0, *coeffs) jet out
  f0, f1, f2, f3 = jet_f((x, x1, x2, x3), (y, y1, y2, y3))
  ```

  ([PR](https://github.com/f-dangel/torch-jet/pull/130))

- Add a `collapsed_jet()` transform for collapsed Taylor mode. It has the same
  calling convention as `jet()` but collapses (sums over directions) the
  highest-order coefficient as it propagates, so intermediate tensors—and the
  resulting compute graph—stay smaller. This replaces the old
  `simplify(jet(...))` workflow: collapsing now happens inside the interpreter,
  so you no longer need a separate simplification pass to shrink the graph.
  `laplacian()` and `bilaplacian()` now use it by default
  ([PR](https://github.com/f-dangel/torch-jet/pull/129))

- **Backward-incompatible.** Replace `Laplacian` and `Bilaplacian` `nn.Module`s
  with `laplacian()` and `bilaplacian()` function transforms that return
  plain callables
  ([PR](https://github.com/f-dangel/torch-jet/pull/123))

- **Backward-incompatible.** Switch FX tracing from `symbolic_trace` to `make_fx`.
  `jet()` and `simplify()` now require a `mock_x` tensor argument for concrete
  tracing. Laplacians use PyTorch's built-in `torch.func.vmap` instead of a custom
  batching implementation. Remove `replicate` and `sum_vmapped` from `jet.utils`
  ([PR](https://github.com/f-dangel/torch-jet/pull/122))

- **Backward-incompatible.** Support general functions with multiple inputs and
  pytree I/O in `jet()` and `rev_jet()`. `jet()` now accepts `mock_args` as a
  tuple and returns a `GraphModule` `jet_f(primals, series)` that returns
  `(primals_out, series_out)`
  ([PR](https://github.com/f-dangel/torch-jet/pull/126))

### Fixed/Removed

- **Backward-incompatible.** `laplacian()` now returns only the Laplacian
  instead of the `(value, Jacobian, Laplacian)` tuple. Collapsing is now handled
  inside the interpreter rather than by PullSum graph rewrites, so `simplify()`
  no longer accepts the `pull_sum` argument and only performs common-subexpression
  and dead-code elimination
  ([PR](https://github.com/f-dangel/torch-jet/pull/129))

### Internal

- Trace with fake tensors (`tracing_mode="fake"`, `_allow_non_fake_inputs=True`)
  via a shared `_make_fx` partial in `jet/tracing.py`. Fake mode skips kernel
  execution and just propagates shape/dtype
  ([PR](https://github.com/f-dangel/torch-jet/pull/133))

- **Backward-incompatible.** Remove the `verbose` argument from `jet()`.
  Replace `JetTransformer` (graph rewriting via `torch.fx.Transformer`) with
  `JetInterpreter` (execution-time dispatch via `torch.fx.Interpreter`).
  `jet()` still returns a `GraphModule` (the interpreter closure is traced
  with `make_fx`). Removes `analyze_dependencies`,
  `_replace_operations_with_taylor`, and `jet_transformer.py` (~250 lines).
  No changes to `laplacian()`, `bilaplacian()`, or `simplify()`
  ([PR](https://github.com/f-dangel/torch-jet/pull/125))

- **Backward-incompatible.** Rewrite tracing and simplification to operate on
  ATen-level ops. Remove `jet/vmap.py` (custom `traceable_vmap`),
  `jet/signature_parser.py`, and related utilities (`replicate`, `sum_vmapped`,
  `standardize_signature`). Simplification rules now match `aten.sum.dim_IntList`
  nodes and use `node.meta["tensor_meta"].shape` for shape reasoning
  ([PR](https://github.com/f-dangel/torch-jet/pull/122))

- Also benchmark compiled Laplacian functions in example 02
  ([PR](https://github.com/f-dangel/torch-jet/pull/123))

## [0.0.1] - 2025-11-14

Today, we are releasing a cleaned up version of the library used in the experiments for our NeurIPS 2025 paper.
The repository also hosts the LaTeX source for the paper and poster.

[Unreleased]: https://github.com/f-dangel/torch-jet/compare/0.0.1...HEAD
[0.0.1]: https://github.com/f-dangel/torch-jet/releases/tag/0.0.1
