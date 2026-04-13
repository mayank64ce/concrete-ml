# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Concrete ML is an open-source privacy-preserving machine learning library by Zama. It enables data scientists to use fully homomorphic encryption (FHE) with familiar scikit-learn/PyTorch APIs, converting ML models into FHE-compatible equivalents without requiring cryptography expertise.

**Package**: `concrete-ml` (v1.9.0)
**Python**: 3.8 - 3.12
**Build system**: Poetry (1.8.4)
**Source**: `src/concrete/ml/`

## Common Commands

All commands use `make` and run through Poetry. Prefix with `poetry run` when running tools directly.

### Environment Setup
```bash
make setup_env          # Initial install (Poetry + deps)
make sync_env           # Sync env after pulling changes
```

### Running Tests
```bash
make pytest                              # Full suite with 100% coverage enforcement (parallel)
make pytest_one TEST=tests/sklearn/      # Run specific test file or directory
make pytest_one TEST=tests/sklearn/test_linear_models.py
make pytest_one_single_cpu RANDOMLY_SEED=12345 TEST=tests/sklearn/  # Reproduce with seed
make pytest_run_last_failed              # Re-run previously failed tests
make pytest_no_flaky                     # Skip known flaky tests (no coverage)
```

Tests use `pytest` with plugins: `pytest-xdist` (parallel), `pytest-randomly` (random ordering), `pytest-cov` (coverage), `pytest-repeat`.

**100% code coverage is required** on `src/` — the CI and `make pytest` will fail otherwise.

### Formatting & Linting
```bash
make python_format       # Auto-format (isort + black + ruff)
make conformance         # Auto-fix formatting, notebooks, licenses, markdown
make pcc                 # Full pre-commit checks (must pass before PR)
make spcc                # Faster subset of pre-commit checks for code-only changes
```

Individual linters:
```bash
make pylint_src          # Pylint on source
make pylint_tests        # Pylint on tests
make ruff                # Ruff linter
make flake8              # Flake8 + darglint
make mypy                # Mypy on source
make mypy_test           # Mypy on tests
make pydocstyle          # Docstring validation (Google convention)
```

### Documentation
```bash
make apidocs             # Generate API docs
make docs_no_links       # Build docs without link checking
```

## Architecture

### Source Layout (`src/concrete/ml/`)
- **`sklearn/`** — scikit-learn compatible models (linear, tree-based, KNN, etc.) — the primary API surface
- **`torch/`** — PyTorch/Brevitas model conversion to FHE circuits
- **`quantization/`** — Quantization-aware training, post-training quantization, quantized modules and operations
- **`onnx/`** — ONNX model import, graph manipulation, and operator conversion
- **`deployment/`** — Client/server deployment for FHE inference (serialization, key management)
- **`pandas/`** — Pandas DataFrame-based model wrappers
- **`common/`** — Shared utilities, debugging tools, check functions, serialization
- **`search_parameters/`** — Hyperparameter search utilities for FHE parameters

### Key Flow
1. User trains a model with scikit-learn/PyTorch/XGBoost API
2. Model is exported to ONNX (`onnx/`)
3. ONNX graph is quantized (`quantization/`)
4. Quantized model is compiled to an FHE circuit via `concrete-python`
5. Model can predict in clear, simulated FHE, or actual FHE mode

### Test Layout (`tests/`)
Mirrors source structure. Tests are in corresponding subdirectories (e.g., `tests/sklearn/`, `tests/torch/`). The root `conftest.py` provides custom pytest options (`--no-flaky`, `--use_gpu`, `--weekly`) and dataset/metric fixtures.

## Code Style

- **Line length**: 100 characters (black, ruff, flake8 all enforce this)
- **Docstrings**: Google convention (enforced by pydocstyle and darglint)
- **Imports**: sorted by isort (black-compatible profile)
- **Type hints**: checked by mypy with `--ignore-missing-imports --implicit-optional --check-untyped-defs`
- `__init__.py` files may have unused imports (F401 is suppressed there)

## Git Conventions

**Conventional commits** format:
- Types: `feat`, `fix`, `docs`, `chore`
- Branch naming: `{feat|fix|docs|chore}/short_description_$(issue_id)`
- Example: `feat/add_avgpool_operator_470`

## Key Dependencies

- **concrete-python** (2.10.0) — FHE compiler backend
- **brevitas** (0.10.2) — quantization-aware training for PyTorch
- **onnx** (1.17.0) / **onnxruntime** (1.18) / **onnxoptimizer** (0.3.13) — model conversion pipeline
- **scikit-learn** (1.1.3–1.5.0) / **xgboost** (1.6.2) — ML model APIs
- **torch** (2.2.2 on macOS Intel, 2.3.1 elsewhere) — neural network support
