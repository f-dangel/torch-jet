# Developer guide

This guide describes principles and workflows for developers.

## Setup

We recommend programming in a fresh virtual environment. You can set up the
`conda` environment and activate it

```bash
make conda-env
conda activate jet
```

If you don't use `conda`, set up your preferred environment and run

```bash
pip install -e ."[lint,test,exp,doc]"
```
to install the package in editable mode, along with all required development dependencies
(the quotes are for OS compatibility, see
[here](https://github.com/mu-editor/mu/issues/852#issuecomment-498759372)).

## Continuous integration

To standardize code style and enforce high quality, checks are carried out with
Github actions when you push. You can also run them locally, as they are managed
via `make`:

- Run tests with `make test` and doctest with `make doctest`

- Run all linters with `make lint`, or separately with:

    - Run auto-formatting and import sorting with `make ruff-format`

    - Run linting with `make ruff`

## Documentation

We use the [Google docstring
convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
and `mkdocs` which allows using markdown syntax in a docstring to achieve
formatting.

To build the docs, run `mkdocs build` in the repository root.
