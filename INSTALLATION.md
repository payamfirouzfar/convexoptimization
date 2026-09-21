# Installation

This project is a Python experiment in convex quadratic optimization. The steps below create an isolated environment, install the numerical libraries, and run the default example.

## Requirements

- Python 3.8 or newer
- NumPy
- SciPy
- Matplotlib
- CVXPY and an available CVXPY solver

## Set up an environment

```bash
git clone https://github.com/payamfirouzfar/convexoptimization.git
cd convexoptimization
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Check the installation

```bash
python -c "import numpy, scipy, matplotlib, cvxpy; print('Dependencies are available')"
```

## Run the example

```bash
python main.py
```

The default script generates a synthetic problem, computes a CVXPY reference solution, runs the custom dual method, prints the objective gap and timing information, and saves convergence plots.

It also enables a random hyperparameter search. For a shorter first run, open `main.py` and set:

```python
run_hyperparameter_search = False
```

You can also reduce `max_iter` or set `enable_plots=False` while checking that the environment works.

## Common problems

### A module cannot be imported

Confirm that the virtual environment is active, then reinstall the requirements:

```bash
python -m pip install -r requirements.txt
```

### CVXPY cannot find a solver

Check the solvers visible to CVXPY:

```bash
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

Install a solver supported by your Python and CVXPY versions, or update the solver choice in the code.

### The run takes longer than expected

The custom method may use many iterations, and the optional search repeats the optimization with different settings. Start with the search disabled and a smaller `max_iter`, then increase the workload once the basic run succeeds.
