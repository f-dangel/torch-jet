# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added/New

- **Backward-incompatible.** Replace `Laplacian` and `Bilaplacian` `nn.Module`
  classes with `laplacian()` and `bilaplacian()` function transforms that return
  plain callables
  ([PR](https://github.com/f-dangel/torch-jet/pull/123))

- **Backward-incompatible.** Switch FX tracing from `symbolic_trace` to `make_fx`.
  `jet()` and `simplify()` now require a `mock_x` tensor argument for concrete
  tracing. Laplacians use PyTorch's built-in `torch.func.vmap` instead of a custom
  batching implementation. Remove `replicate` and `sum_vmapped` from `jet.utils`
  ([PR](https://github.com/f-dangel/torch-jet/pull/122))

### Fixed/Removed

### Internal

- Replace `JetTransformer` (graph rewriting via `torch.fx.Transformer`) with
  `JetInterpreter` (execution-time dispatch via `torch.fx.Interpreter`).
  `jet()` now returns a plain closure instead of a `GraphModule`. Removes
  `analyze_dependencies`, `_replace_operations_with_taylor`, and
  `jet_transformer.py` (~250 lines). No changes to `laplacian()`,
  `bilaplacian()`, or `simplify()`

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
