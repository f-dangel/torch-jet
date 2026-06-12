# Taylor Mode Autodiff in PyTorch

This library provides a PyTorch implementation of Taylor mode automatic differentiation, a generalization of forward mode to higher-order derivatives.
It is similar to JAX's Taylor mode ([`jax.experimental.jet`](https://docs.jax.dev/en/latest/jax.experimental.jet.html)).

The repository also hosts the Python functionality+experiments and LaTeX source for our NeurIPS 2025 paper ["Collapsing Taylor Mode Automatic Differentiation"](https://openreview.net/forum?id=XgQVL1uP34), which allows to further accelerate Taylor mode for many practical differential operators.

Operator coverage is still growing, so please help us improve the package by providing feedback, filing issues, and opening pull requests.

![Taylor mode propagates the jet of an input curve through a function to obtain the jet of the output curve](https://raw.githubusercontent.com/f-dangel/torch-jet/master/docs/examples/01_taylor_mode.png)


## Getting Started

### Installation

```bash
pip install jet-for-pytorch
```

### Quickstart

Compute the third-order derivative of a scalar function with `jet`:

<!-- BEGIN quickstart -->
```python
from torch import tensor, ones_like, zeros_like
from jet import jet

f = lambda x: x**4  # f'''(x) = 24 * x

x = tensor(2.0)
f_jet = jet(f, (x,))  # (x₀, x₁, x₂, x₃) ↦ (f₀, f₁, f₂, f₃)

# x₀ = x, x₁ = 1, x₂ = x₃ = 0 ⇒ f₃ = f'''(x)
f0, f1, f2, f3 = f_jet((x, ones_like(x), zeros_like(x), zeros_like(x)))

assert f3.allclose(24 * x)  # third derivative matches by hand
```
<!-- END quickstart -->

For the full list of supported operators, see the
[supported operations](https://torch-jet.readthedocs.io/en/latest/generated/gallery/01_taylor_mode/)
section of the introductory tutorial.

### Examples

See the [documentation](https://torch-jet.readthedocs.io/en/latest/generated/gallery/).

## Citing

If you find the `jet` package useful for your research, consider citing

```bibtex

@inproceedings{dangel2025collapsing,
  title =        {Collapsing Taylor Mode Automatic Differentiation},
  author =       {Felix Dangel and Tim Siebert and Marius Zeinhofer and Andrea
                  Walther},
  year =         2025,
  booktitle =    {Advances in Neural Information Processing Systems (NeurIPS)},
}

```
