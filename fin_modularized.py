## the third one - MODULARIZED VERSION

"""
This is the modularized version of fin.py that imports from separate modules.
All functionality remains exactly the same, but code is now organized into logical modules.

Modules:
- problem_data.py: Problem data generation
- matrix_operations.py: Matrix factorization and linear system solving
- primal_solver.py: Primal problem solving with CVXPY
- dual_solver.py: Dual problem solving with deflected subgradient method
- visualization.py: Plotting and visualization
- hyperparameter_search.py: Hyperparameter optimization
- optimization_core.py: Main optimization orchestration
- main.py: Entry point for execution

To run the optimization, you can either:
1. Import and use run_optimization from optimization_core
2. Run main.py directly
"""

# Import all the main functions from the modules
from optimization_core import run_optimization
from hyperparameter_search import random_search_hyperparameters
from primal_solver import solve_primal_problem, solve_dual_with_cvxpy
from dual_solver import deflected_subgradient_method
from visualization import create_plots
from problem_data import generate_problem_data
from matrix_operations import factorize_matrix, solve_linear_system

# Main execution block
if __name__ == "__main__":
    n=10
    k=3

    # Example call with larger problem size
    results = run_optimization(
        n=n,              # Larger number of variables
        k=k,               # More simplex constraints
        seed=0,             # Random seed
        regularization=1e-0, # Stronger regularization for stability with large problems
        alpha=0.999,          # Moderate initial step size
        beta=0.00001,           # Less deflection for large problems
        rel_gap_tol=1e-4,   # More relaxed convergence tolerance
        alpha_min=1e-5,
        max_iter=52000,     # Minimum step size
        step_reduction_factor=0.99,  # Gentler step size reduction
        non_decrease_threshold=100,   # Allow more iterations without improvement
        early_phase_iters=3000,       # Longer initial phase
        enable_plots=True,           # Generate plots
        save_plots=True              # Save plots to files
    )
    
    print("\nFirst Algorithm Results (Manual Implementation):")
    print(f"Algorithm Runtime: {results['runtime']:.6f} seconds")
    print(f"Final Objective Value: {results['primal_opt']:.6e}")
    print(f"Relative Gap Achieved: {results['relative_gap']:.2e}")
    print(f"Number of Iterations: {results['iterations']}")
    
    # Example of running the random search for hyperparameters
    # You might want to use smaller n_rs, k_rs for faster search, e.g., n_rs=30, k_rs=5
    # And fewer trials for a quick test, e.g., num_trials=20
    run_hyperparameter_search = True # Set to False to skip
    if run_hyperparameter_search:
        print("\n--- Starting Hyperparameter Random Search ---")
        best_params, best_gap = random_search_hyperparameters(
            run_optimization_func=run_optimization,
            num_trials=20, # Number of trials for random search
            n_rs=n,        # Number of variables for search runs
            k_rs=k,        # Number of constraints for search runs
            seed_rs=0     # Seed for the search process itself
        )
        if best_params:
            print("\n--- Hyperparameter Random Search Finished ---")
            print("Best parameters found by random search:")
            for key, val in best_params.items():
                 if isinstance(val, float): print(f"  {key}: {val:.4e}")
                 else: print(f"  {key}: {val}")
            print(f"Achieved Relative Gap: {best_gap:.4e}")
        else:
            print("\n--- Hyperparameter Random Search Finished ---")
            print("No best parameters found or search was not successful.")
    
    print(f"\nComplete Problem Summary:")
    print(f"CVXPY Primal Solve Time: {results['cvxpy_solve_time']:.6f} seconds")
    print(f"Total Algorithm Runtime: {results['runtime']:.4f} seconds")
    print(f"Pure Dual Algorithm Time: {results['dual_solve_time']:.4f} seconds")
    print(f"Final Objective Value: {results['primal_opt']:.6e}")
    print(f"Best Achieved Gap: {best_gap:.2e}")
    
    # Add comparison with direct CVXPY dual solution
    print("\nComparing with direct CVXPY dual solution:")
    # Use same problem size and seed as the main run
    dual_results = solve_dual_with_cvxpy(n=n, k=k, seed=0)
    if dual_results:
        print(f"\nSolving time comparison:")
        print(f"Your Dual Algorithm: {results['dual_solve_time']:.6f} seconds")
        print(f"CVXPY Primal: {results['cvxpy_solve_time']:.6f} seconds")
        print(f"CVXPY Dual: {dual_results['solve_time']:.6f} seconds")
        print(f"\nObjective value comparison:")
        print(f"Your Dual Value: {results['dual_value']:.6e}")
        print(f"CVXPY Primal Value: {results['primal_opt']:.6e}")
        print(f"CVXPY Dual Value: {dual_results['optimal_value']:.6e}")
