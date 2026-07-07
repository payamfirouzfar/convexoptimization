"""
Modularized Optimization Package

A complete optimization framework for solving quadratic programs with simplex constraints
using deflected subgradient methods.

Main Components:
- run_optimization: Main optimization function
- random_search_hyperparameters: Hyperparameter tuning
- solve_primal_problem: CVXPY-based primal solver
- deflected_subgradient_method: Custom dual solver
- create_plots: Visualization utilities

Example:
    from optimization_core import run_optimization
    
    results = run_optimization(n=100, k=10, seed=0)
    print(f"Optimal value: {results['primal_opt']}")
"""

__version__ = "1.0.0"
__author__ = "Optimization Team"

# Import main functions for easy access
from optimization_core import run_optimization
from hyperparameter_search import random_search_hyperparameters
from primal_solver import solve_primal_problem, solve_dual_with_cvxpy
from dual_solver import deflected_subgradient_method
from visualization import create_plots
from problem_data import generate_problem_data
from matrix_operations import factorize_matrix, solve_linear_system

__all__ = [
    'run_optimization',
    'random_search_hyperparameters',
    'solve_primal_problem',
    'solve_dual_with_cvxpy',
    'deflected_subgradient_method',
    'create_plots',
    'generate_problem_data',
    'factorize_matrix',
    'solve_linear_system',
]
