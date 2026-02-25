# Collapsed Taylor Mode: Design Rationale

## Background

Collapsed Taylor mode is an optimization for computing differential operators
(Laplacians, Bi-Laplacians) via Taylor-mode automatic differentiation. Instead
of propagating `R` full `K`-jets and then summing the `K`-th coefficient
afterward, collapsed Taylor mode directly propagates the sum — reducing both
memory and computation.

## Two Implementation Approaches

### 1. PullSum Graph Rewrites (removed)

The original implementation used FX graph rewriting rules (`jet/rules.py`) to
pattern-match `sum(op(...))` subgraphs and rewrite them into `op(sum(...))`.
Seven rule classes (~685 lines) handled different operation types (`PullSumMul`,
`PullSumAddOrSub`, `PullSumMM`, `PullSumAddMM`, `PullSumSqueeze`,
`PullSumUnsqueeze`, `PullSumView`).

**Drawbacks:**

- **High per-operation cost.** Each new operation required a 30–80 line rule
  class with `match()` / `apply()` / `_rewrite()` methods and graph surgery.
- **Rule interaction bugs.** Rules had to be applied in a fixed-point loop, and
  new rules could invalidate assumptions of existing ones.
- **Fragile shape reasoning.** Rules inspected `node.meta["tensor_meta"].shape`
  and manually adjusted dimension indices — any off-by-one broke silently.
- **Large code surface.** ~685 lines in `rules.py` + ~480 lines in
  `test_rules.py` + the `apply_all` / `pull_sum` machinery in `simplify.py`.

### 2. Collapsed Jet Interpreter (current)

The replacement uses a `CollapsedJetInterpreter` (subclass of
`torch.fx.Interpreter`) that propagates `CollapsedJetTuples` — tuples where:

- Coefficient 0 (primal): shape `(...)`
- Coefficients 1..K-1: shape `(R, ...)` — batched over `R` directions
- Coefficient K: shape `(...)` — already collapsed (summed over directions)

At each operation, the interpreter computes the K-th output coefficient by
splitting contributions into a linear term (uses collapsed input) and nonlinear
terms (uses batched inputs, then `.sum(0)`). The mathematical identity behind
this is that the highest-order input always appears linearly in Faà di Bruno's
formula and the Leibniz rule.

**Advantages:**

- **2-line operations.** Adding a new elementwise op is a one-line wrapper
  calling `_cjet_elementwise`. Linear ops use `_apply_linear` with `vmap`.
- **No graph surgery.** Collapsing happens at execution time — no pattern
  matching, no node insertion/deletion, no shape metadata propagation.
- **Automatic batch handling.** `vmap` manages the leading batch dimension for
  batched coefficients, eliminating manual `dim+1` arithmetic.
- **Shared code.** Derivative helpers (`_sin_derivatives`, etc.) and
  `_faa_di_bruno` are shared between the standard `JetInterpreter` and the
  `CollapsedJetInterpreter`, avoiding duplication.
- **Composable.** The collapsed interpreter's output can be traced with
  `capture_graph` to produce a static `GraphModule`, then further optimized
  with CSE/DCE via `simplify()`, and compiled with `torch.compile`.

## Kept Infrastructure

A lightweight `simplify()` is retained for common subexpression elimination
(CSE) and dead code elimination (DCE). These are graph-level optimizations
orthogonal to collapsing and remain useful for reducing redundant computation in
traced graphs.
