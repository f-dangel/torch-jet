# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

`jet` is now in Beta: the public API is stable and more operators are
supported, though coverage is still growing.

### Added/New

- **Backward-incompatible.** Flatten the public API: all public functions are
  now importable from the top level (e.g. `from jet import laplacian`)
  ([PR](https://github.com/f-dangel/torch-jet/pull/172)).

- Support `aten.t.default` (matrix transpose) in Taylor mode (both standard and
  collapsed modes)
  ([PR](https://github.com/f-dangel/torch-jet/pull/168)).

- Support a Taylor-expanded bias in `aten.convolution` (both standard and
  collapsed modes), removing the restriction that the bias must be a constant
  `Tensor` or `None`
  ([PR](https://github.com/f-dangel/torch-jet/pull/160)).

- Support classic torchvision CNNs in Taylor mode by adding jet rules for the
  ATen ops they need:
  - `aten.relu.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/144)).
  - `aten.convolution.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/145)).
  - `aten.max_pool2d_with_indices.default` and `aten.max_pool2d.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/159)).
  - `aten._adaptive_avg_pool2d.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/149)).
  - `aten.mean.dim` and `aten.mean.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/150)).
  - `aten.avg_pool2d.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/151)).
  - `aten.cat.default` (dispatch now also detects jets nested in `list` args)
    ([PR](https://github.com/f-dangel/torch-jet/pull/154)).
  - `aten.native_batch_norm.default` in eval mode (affine per channel);
    training mode is deferred until PyTorch fixes its fused op's incorrect
    higher-order autograd in training ([pytorch/pytorch#186256](https://github.com/pytorch/pytorch/issues/186256))
    ([PR](https://github.com/f-dangel/torch-jet/pull/157)).

- Support loss functions in Taylor mode by adding jet rules for the ATen ops
  they need:
  - `aten.neg.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/155)).
  - `aten.mse_loss.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/146)).
  - `aten.div.Scalar`
    ([PR](https://github.com/f-dangel/torch-jet/pull/156)).
  - `aten.exp.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/162)).
  - `aten.log.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/163)).
  - `aten._log_softmax.default`
    ([PR](https://github.com/f-dangel/torch-jet/pull/152)).
  - `aten.nll_loss_forward.default`, together with `_log_softmax` this enables
    `nn.CrossEntropyLoss`
    ([PR](https://github.com/f-dangel/torch-jet/pull/153)).

- Support a Taylor-expanded bias in `aten.addmm` (both standard and collapsed
  modes), removing the restriction that the bias must be a constant `Tensor`
  ([PR](https://github.com/f-dangel/torch-jet/pull/147)).

- **Backward-incompatible.** Replace the `Laplacian` / `Bilaplacian`
  `nn.Module`s with `laplacian()` / `bilaplacian()` function transforms that
  return plain Python callables. They take `mock_args` as a tuple matching
  `f`'s positional arguments (mirroring `jet()`) and return a callable taking
  one positional argument per argument of `f`; the propagation regime is
  selected via a `collapsed` flag. Only single-tensor functions (one tensor
  in, one tensor out) are supported for now; a non-single-tensor input or
  output raises `NotImplementedError`
  ([PR](https://github.com/f-dangel/torch-jet/pull/143)).
  Earlier in this release the `nn.Module`s were replaced by the function
  transforms ([PR #123](https://github.com/f-dangel/torch-jet/pull/123)), and
  the `use_collapsing` flag was renamed to `collapsed` (bundled with merging
  the two jet interpreters into one;
  [PR #132](https://github.com/f-dangel/torch-jet/pull/132)).

- Add jet rules for `aten.zeros_like.default`, `aten._unsafe_view.default`,
  and `aten.squeeze.dims`. Op dispatch now forwards kwargs to the
  registered rule. Together with the collapsed Leibniz fix in PR #141
  this unblocks `laplacian(laplacian(f))`
  ([PR](https://github.com/f-dangel/torch-jet/pull/140)).

- **Backward-incompatible.** Merge `collapsed_jet` into `jet` as a
  ``collapsed: bool = False`` flag (``jet(f, mock_args, collapsed=True)``
  replaces ``collapsed_jet(f, mock_args)``). The single transform now
  presents both propagation regimes — standard and collapsed — matching
  the paper's framing of collapsed mode as a *mode of* Taylor mode rather
  than a separate algorithm
  ([PR](https://github.com/f-dangel/torch-jet/pull/135)).
  Earlier in this release `collapsed_jet` was introduced as a separate
  transform ([PR](https://github.com/f-dangel/torch-jet/pull/129)) and
  then evolved alongside `jet()` (callable-return + drop `derivative_order`
  in [PR #134](https://github.com/f-dangel/torch-jet/pull/134); per-arg
  primal+coeffs bundling in [PR #130](https://github.com/f-dangel/torch-jet/pull/130)).

- **Backward-incompatible.** `jet()`, `laplacian()`, and `bilaplacian()`
  now return plain Python callables (was `GraphModule`). `jet()` also
  drops its `derivative_order` argument (now inferred per call from the
  input jet tuples). To bake any of these into a `GraphModule` for graph
  passes (CSE, `torch.compile`, etc.), apply `capture_graph` to it
  yourself — graph capture is now a single explicit step at the user's
  chosen point. `capture_graph(f, mock_args)` now takes one tuple of
  positional pytrees (was variadic tensors) and returns `(mod, in_spec)`;
  use `mod(*in_spec.flatten_up_to(args))` to call the captured graph
  ([PR](https://github.com/f-dangel/torch-jet/pull/134))

- **Backward-incompatible.** Bundle each argument's primal with its Taylor
  coefficients. The transforms `jet()` and `rev_jet()` now
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

- Correctly handle broadcasting in `add` / `sub` between operands of different
  rank (similar to #141). In collapsed mode, two Taylor-expanded operands of
  different primal rank collided the direction dim `R` (e.g. `x + y` with shapes
  `(4,)` and `(3, 4)`); and, in both standard and collapsed mode, a jet combined
  with a larger constant did not broadcast its coefficients up to the result
  shape. A broadcasting stress matrix now covers every dispatch branch
  (`JJ` / `JC` / `CJ`) of `add` / `sub` / `mul`
  ([PR](https://github.com/f-dangel/torch-jet/pull/166),
  [PR](https://github.com/f-dangel/torch-jet/pull/167)).

- Fix Taylor-mode coefficients for power functions with non-positive exponents.
  `_pow_derivatives` no longer truncates the derivatives of negative integer
  exponents (e.g. `x ** -2`), and Faà di Bruno now materializes a structurally
  zero coefficient as an explicit zero tensor instead of leaking ``None`` --
  which previously broke constant elementwise outputs such as `x ** 0`
  ([PR](https://github.com/f-dangel/torch-jet/pull/164)).

- Fix the failing Read the Docs build by removing the gallery example's
  "Unsupported Operations" subsection
  ([PR](https://github.com/f-dangel/torch-jet/pull/158)).

- Fix collapsed-mode Leibniz misaligning the leading direction dim `R` when
  product operands have different primal ranks (`cjet_mul` / `cjet_mm` /
  `cjet_addmm`). Right-aligned PyTorch broadcasting collided one operand's
  `R` against a middle primal dim of the other; `vmap(binary_op,
  in_dims=...)` now aligns `R` per-direction explicitly
  ([PR](https://github.com/f-dangel/torch-jet/pull/141)).

- Constant output leaves in collapsed mode are now wrapped with the correct
  shape: coefficients `c_1..c_{K-1}` are `(R, *S)` and `c_K` is `S`. Previously
  all zero coefficients were `S`-shaped, silently producing wrong shapes for
  functions whose outputs include constants (e.g., `f(x) = (sin(x), ones(4))`)
  ([PR](https://github.com/f-dangel/torch-jet/pull/134))

- **Backward-incompatible.** `laplacian()` now returns only the Laplacian
  instead of the `(value, Jacobian, Laplacian)` tuple. Collapsing is now handled
  inside the interpreter rather than by PullSum graph rewrites, so `simplify()`
  no longer accepts the `pull_sum` argument and only performs common-subexpression
  and dead-code elimination
  ([PR](https://github.com/f-dangel/torch-jet/pull/129))

### Internal

- De-duplicate every remaining standard/collapsed primitive rule into a single
  mode-agnostic body
  ([PR](https://github.com/f-dangel/torch-jet/pull/177)).

- De-duplicate the composite op rules and reorganize the jet rules by role into
  `primitives` and `compositions`
  ([PR](https://github.com/f-dangel/torch-jet/pull/176)).

- Replace the two op-dispatch tables with a single `RULES` registry and
  de-duplicate rules shared across standard and collapsed modes
  ([PR](https://github.com/f-dangel/torch-jet/pull/174)).

- Unify the two internal jet tuple types into a single `JetTuple`
  ([PR](https://github.com/f-dangel/torch-jet/pull/175)).

- Improve docstrings
  ([PR](https://github.com/f-dangel/torch-jet/pull/173)).

- Make `jet_add` / `jet_sub` (and collapsed) total over constant operands:
  `add` / `sub` on two constants now return the plain result instead of
  indexing a non-jet operand, completing the `{jet, constant}` totality that
  #170 gave the product ops
  ([PR](https://github.com/f-dangel/torch-jet/pull/171)).

- Extract an `_apply_bilinear(op, a, b)` combinator (standard and collapsed
  modes) for product-like jet ops
  ([PR](https://github.com/f-dangel/torch-jet/pull/170)).

- Drop the redundant primal from the `_*_derivatives` helpers' return value.
  Each helper already stored the primal at `dn[0]` and also returned it as a
  separate first element; the helpers now return just the derivative dict and
  `_jet_elementwise` / `_cjet_elementwise` / `jet_pow` / `cjet_pow` read the
  primal from `dn[0]`
  ([PR](https://github.com/f-dangel/torch-jet/pull/165)).

- API doc cleanups: rename `rev_jet` to `_rev_jet` to mark it internal (a
  reference implementation used only for testing `jet`, absent from the public
  API docs), document the previously-undocumented `visualize_graph` helper in
  `api.md`, and fix the Laplacian docstring formula (the Hessian trace was
  written `\sum_{i=d}^D` but sums over `x_d`; corrected to `\sum_{d=1}^D`)
  ([PR](https://github.com/f-dangel/torch-jet/pull/161)).

- Add JAX-style `deflinear(prim)` / `defzero(prim)` helpers in
  `jet/operations.py` and `jet/collapsed_operations.py` to bulk-register
  linear and constant-output ops. Drops the per-op `jet_view` /
  `jet_unsqueeze` / `jet_squeeze` / `jet_squeeze_dims` / `jet_zeros_like`
  functions and their `cjet_*` mirrors; adding a new op in either category
  is now a one-line edit
  ([PR](https://github.com/f-dangel/torch-jet/pull/142)).

- Annotate pytree-shaped signatures with a new `PyTree[Leaf]` recursive alias
  and a `Jet = tuple[Tensor, ...]` alias in `jet/utils.py`
  ([PR](https://github.com/f-dangel/torch-jet/pull/138)).

- Slim the test suite into per-concern layers
  ([PR](https://github.com/f-dangel/torch-jet/pull/136)).

- Parametrize the test suite over available devices (CPU + CUDA + MPS,
  auto-detected). MPS uses `float32` with relaxed `assert_close`
  tolerances since the framework doesn't support `float64`
  ([PR](https://github.com/f-dangel/torch-jet/pull/139)).

- Drop the `is_batched` parameter from the experiment harness functions
  (`laplacian_function` / `bilaplacian_function` in
  `jet/exp/exp01_benchmark_laplacian/execute.py` and the four matching JAX
  functions in `jet/exp/exp04_jax_benchmark/execute.py`)
  ([PR](https://github.com/f-dangel/torch-jet/pull/137))

- Extract input validation into `jet/validation.py` (entry point:
  `validate_input_jet`); the `JetInterpreter` now owns both ends of the
  type boundary (wrapping inputs in `placeholder()`, unwrapping outputs in
  `run()`) so `jet/__init__.py` no longer imports the internal
  `JetTuple`/`CollapsedJetTuple` dispatch types
  ([PR](https://github.com/f-dangel/torch-jet/pull/134))

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
