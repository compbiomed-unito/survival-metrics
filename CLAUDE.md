# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**survmetr** — a Python library for advanced predictive metrics in survival analysis, focused on Antolini's time-dependent concordance index. Package name is `survmetr`, repo is `survival-metrics`.

## Build & Install

```bash
flit build           # Build the package
pip install -e .     # Development install
```

Build system: Flit. Python 3.8+. Dependencies: numpy, scikit-learn, scikit-survival. Optional: numba (for JIT-accelerated PyCox implementation).

## Architecture

The package provides three independent implementations of Antolini's C-index plus time-dependent classification scorers:

- **`antolini.py`** — Main vectorized (pure NumPy) implementation. `c_index_antolini_vector()` is the primary API, exported as `c_index_antolini` in `__init__.py`. Supports configurable time-tie handling and tie weights.
- **`antolini_sksurv.py`** — Adaptation using scikit-survival internals. Called via `c_index_antolini_sksurv()` wrapper in antolini.py.
- **`antolini_pycox.py`** — Numba-JIT optimized implementation with parallel computation. Supports 'antolini' and 'adj_antolini' methods. Called via `c_index_antolini_pycox()` wrapper in antolini.py.
- **`classification.py`** — `make_survival_scorer()` factory that wraps any sklearn classification metric (ROC-AUC, Brier, log loss) into a time-dependent survival scorer. Evaluates at event times, quantiles, or absolute times.
- **`others.py`** — Harrel's standard C-index wrapper.
- **`util.py`** — Helpers: `split_y()` extracts event/time from structured arrays.

## Data Conventions

- Survival outcomes use scikit-survival structured numpy arrays with `'event'` (bool) and `'time'` fields.
- Predictions for Antolini vector: shape `(n_samples, n_events)` — failure probabilities at event times.
- Predictions for PyCox: shape `(n_times, n_samples)` — survival function values.
- All metric functions support `return_all=True` to get detailed stats (concordant pairs, comparable pairs, tied risk, etc.).

## Current State

Early research stage (v0.1.0). No formal test suite yet. Classification metrics not yet exported from `__init__.py`. See README.md TODO for planned work (flexible outcome passing, model API standardization, IPCW metrics, tutorial).
