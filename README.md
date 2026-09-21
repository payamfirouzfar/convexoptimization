# Convex Quadratic Optimization with Block-Simplex Constraints

This repository explores a specialised convex quadratic problem in which groups of non-negative variables must each sum to one. I implemented a dual deflected-subgradient method, then used CVXPY as a reference for checking the solution and objective gap.

The problem has the form

```text
minimize    0.5 xᵀQx + qᵀx
subject to  Σ xᵢ = 1  for each variable group
            x ≥ 0
```

`Q` is generated as a positive-semidefinite matrix. The code creates synthetic problem instances, solves the reference problem, runs the custom dual method, records convergence information, and can search over algorithm settings.

## Main files

- `main.py` — runnable example and result summary
- `optimization_core.py` — coordinates problem generation, reference solving, and the custom algorithm
- `problem_data.py` — creates reproducible synthetic instances
- `primal_solver.py` — CVXPY reference solution
- `dual_solver.py` — dual and deflected-subgradient logic
- `hyperparameter_search.py` — random search over solver settings
- `visualization.py` — convergence plots
- `fin.py` and `fin_modularized.py` — earlier combined versions of the experiment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

See [`INSTALLATION.md`](INSTALLATION.md) for more detail.

## What to expect

The default example uses a small synthetic problem but allows many iterations and enables a random hyperparameter search, so it may take longer than a quick smoke test. You can reduce `max_iter`, set `run_hyperparameter_search = False`, or disable plots in `main.py` when you only want to inspect the pipeline.

## Project status

This is a numerical research and learning project. It compares a custom method with established solvers, but it does not claim to outperform general-purpose optimization libraries. Performance depends on the problem size, conditioning, tolerances, and chosen hyperparameters.
