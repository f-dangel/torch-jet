.DEFAULT: help

help:
	@echo "install"
	@echo "        Install jet and dependencies"
	@echo "uninstall"
	@echo "        Uninstall jet"
	@echo "lint"
	@echo "        Run all linting commands"
	@echo "install-dev"
	@echo "        Install all development tools"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
	@echo "ruff-format"
	@echo "        Run ruff format on the project"
	@echo "ruff-format-check"
	@echo "        Check if ruff format would change files"
	@echo "ruff"
	@echo "        Run ruff on the project and fix errors"
	@echo "ruff-check"
	@echo "        Run ruff check on the project without fixing errors"
	@echo "conda-env"
	@echo "        Create conda environment 'jet' with dev setup"
	@echo "arxiv"
	@echo "        Run arxiv-collector to prepare a submission to arXiv (requires latexmk)"

.PHONY: install

install:
	@pip install -e .

.PHONY: uninstall

uninstall:
	@pip uninstall jet

.PHONY: install-dev

install-dev:
	@pip install -e ."[test,lint,exp]"

.PHONY: install-test

install-test:
	@pip install -e ."[test,exp]"

.PHONY: test

test:
	@pytest -vx --cov=jet test

.PHONY: doctest

doctest:
	@pytest -vx --doctest-modules jet

.PHONY: lint

lint:
	make ruff-check
	make ruff-format-check

.PHONY: install-lint

install-lint:
	@pip install -e ."[lint]"

.PHONY: ruff ruff-check

ruff:
	@ruff check . --fix

ruff-check:
	@ruff check .

.PHONY: ruff-format ruff-format-check

ruff-format:
	@ruff format .

ruff-format-check:
	@ruff format --check .

.PHONY: conda-env

conda-env:
	@conda env create --file .conda_env.yml

.PHONY: arxiv

arxiv:
	@cd paper && arxiv-collector main.tex
	@echo "!!! IMPORTANT: The .tar.gz does not work. !!!"
	@echo "!!! IMPORTANT: You need to re-compress into .zip. !!!"
