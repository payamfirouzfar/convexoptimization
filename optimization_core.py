"""
Core optimization module.
Main optimization function that orchestrates all components.
"""

import numpy as np
import time
from problem_data import generate_problem_data
from primal_solver import solve_primal_problem
from dual_solver import deflected_subgradient_method
from visualization import create_plots


def run_optimization(n=10, k=7, seed=0, regularization=1e-6, 
                    alpha=0.05, beta=0.1, max_iter=1000, rel_gap_tol=1e-6,
                    alpha_min=1e-7, step_reduction_factor=0.9, non_decrease_threshold=5,
                    early_phase_iters=30, enable_plots=True, save_plots=True):
    """
    Main function to run the optimization algorithm with configurable parameters.
    
    Parameters:
    - n: Number of variables (default: 10)
    - k: Number of simplex constraints (default: 7)
    - seed: Random seed for reproducibility (default: 0)
    - regularization: Value added to diagonal of Q to ensure positive definiteness (default: 1e-6)
    - alpha: Initial step size (default: 0.05)
    - beta: Deflection parameter (default: 0.1)
    - max_iter: Maximum number of iterations (default: 1000)
    - rel_gap_tol: Relative gap tolerance for convergence (default: 1e-6)
    - alpha_min: Minimum step size (default: 1e-7)
    - step_reduction_factor: Factor to reduce step size (default: 0.9)
    - non_decrease_threshold: Number of iterations without improvement before reducing step size (default: 5)
    - early_phase_iters: Number of iterations to use fixed step size (default: 30)
    - enable_plots: Whether to generate plots (default: True)
    - save_plots: Whether to save plots to files (default: True)
    
    Returns:
    - results: Dictionary containing optimization results
    """
    # Generate problem data
    Q_psd, q, A, b, Iks, eigvals, eigvecs = generate_problem_data(n, k, seed, regularization)
    
    # Solve the primal problem to get the optimal primal value
    x_primal_opt, primal_opt, cvxpy_solve_time = solve_primal_problem(n, Q_psd, q, Iks)
    print(f"Benchmark primal optimal value: {primal_opt:.6e}")

    # Measure total runtime of the algorithm
    start_time = time.time()  # Start timing

    try:
        # Run the deflected subgradient method
        mu_opt, lambda_k_opt, dual_values, primal_values, primal_opt, relative_gaps, constraint_violations, all_multipliers, dual_solve_time = deflected_subgradient_method(
            n, Q_psd, q, Iks, eigvals, primal_opt,
            alpha, beta, max_iter, rel_gap_tol,
            alpha_min, step_reduction_factor,
            non_decrease_threshold, early_phase_iters,
            regularization
        )
    
        end_time = time.time()  # End timing
        print(f"Total runtime of the algorithm: {end_time - start_time:.4f} seconds")
        print(f"Pure dual algorithm time: {dual_solve_time:.4f} seconds")
    
        # Create all the plots
        create_plots(mu_opt, lambda_k_opt, dual_values, primal_values, primal_opt, 
                    relative_gaps, constraint_violations, all_multipliers,
                    enable_plots, save_plots)
    
        # Print detailed final results
        print("\nFinal Results:")
        print(f"Optimal Primal Value (CVXPY): {primal_opt:.6e}")
        print(f"Final Dual Value: {dual_values[-1]:.6e}")
        print(f"Final Relative Gap: {relative_gaps[-1]:.2e}")
        print(f"Final Max Constraint Violation: {constraint_violations[-1]:.2e}")
        print(f"Total Runtime: {end_time - start_time:.2f} seconds")
        print(f"Pure Dual Algorithm Time: {dual_solve_time:.2f} seconds")
        print(f"Number of Iterations: {len(dual_values)}")
    
        # Return the results dictionary
        results = {
            'mu_opt': mu_opt,
            'lambda_k_opt': lambda_k_opt,
            'primal_opt': primal_opt,
            'dual_value': dual_values[-1],
            'relative_gap': relative_gaps[-1],
            'iterations': len(dual_values),
            'runtime': end_time - start_time,
            'dual_solve_time': dual_solve_time,  # Add pure dual solve time
            'cvxpy_solve_time': cvxpy_solve_time
        }
        return results
    
    except Exception as e:
        print(f"Error encountered during algorithm execution: {e}")
        import traceback
        traceback.print_exc()
        return None
