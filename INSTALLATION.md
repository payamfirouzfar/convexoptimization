# Installation Guide

## Required Packages

This project requires the following Python packages:

- **numpy** (≥1.21.0) - Numerical computing
- **matplotlib** (≥3.4.0) - Plotting and visualization
- **cvxpy** (≥1.2.0) - Convex optimization
- **scipy** (≥1.7.0) - Scientific computing

## Installation Methods

### Method 1: Using requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

### Method 2: Manual Installation

```bash
pip install numpy matplotlib cvxpy scipy
```

### Method 3: Using Python 3.13 (Your Setup)

```bash
C:/Users/Payam/AppData/Local/Programs/Python/Python313/python.exe -m pip install numpy matplotlib cvxpy scipy
```

## Verify Installation

After installation, verify all packages are installed correctly:

```bash
python -c "import numpy, matplotlib, cvxpy, scipy; print('All packages installed successfully!')"
```

## Run the Program

Once all packages are installed, you can run the optimization:

```bash
python main.py
```

Or:

```bash
python fin_modularized.py
```

## Package Details

### CVXPY Dependencies
CVXPY automatically installs its solver dependencies:
- **OSQP** - Operator Splitting QP solver (primary solver)
- **ECOS** - Embedded Conic Solver (fallback)
- **SCS** - Splitting Conic Solver (fallback)
- **Clarabel** - Interior point solver

These solvers are used automatically by CVXPY to solve the optimization problems.

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Install the missing package using pip

```bash
pip install <package_name>
```

### Issue: Permission Denied
**Solution**: Use --user flag

```bash
pip install --user numpy matplotlib cvxpy scipy
```

### Issue: Outdated pip
**Solution**: Upgrade pip first

```bash
python -m pip install --upgrade pip
```

## System Requirements

- Python 3.8 or higher (tested with Python 3.13)
- Windows, macOS, or Linux
- ~500MB disk space for packages and dependencies

## Installation Complete ✅

All packages have been successfully installed. You can now run:

```bash
python main.py
```

The program will:
1. Generate optimization problem data
2. Solve the primal problem with CVXPY
3. Solve the dual problem with deflected subgradient method
4. Create convergence plots
5. Optionally run hyperparameter search
6. Display comprehensive results

Enjoy your modularized optimization code! 🎉
