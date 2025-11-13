# Taylor Mode Autodiff in PyTorch

This library provides a PyTorch implementation Taylor mode automatic differentiation, a generalization of forward mode to higher-order derivatives.
It is similar to JAX's Taylor mode ([`jax.experimental.jet`](https://docs.jax.dev/en/latest/jax.experimental.jet.html)).

The repository also hosts the Python functionality+experiments and LaTeX source for our NeurIPS 2025 paper ["Collapsing Taylor Mode Automatic Differentiation"](https://openreview.net/forum?id=XgQVL1uP34), which allows to further accelerate Taylor mode for many practical differential operators.

!!! warning "🔪 **Expect rough edges.** 🔪"

    This is a research prototype with various limitations (e.g. operator coverage).
    We highly recommend double-checking your results with PyTorch's autodiff.
    Please help us improve the package by providing feedback, filing issues, and opening pull requests.

## Getting Started

### Installation

```bash
pip install jet-for-pytorch
```

### Examples

TODO Coming soon, once we host the documentation on readthedocs.

## Citing

If you find the `jet` package useful for your research, consider citing

```bibtex

@inproceedings{dangel2025collapsing,
  title =        {Collapsing Taylor Mode Automatic Differentiation},
  author =       {Felix Dangel and Tim Siebert and Marius Zeinhofer and Andrea
                  Walther},
  year =         2025,
  journal =      {Advances in Neural Information Processing Systems (NeurIPS)},
}

```
