"""
Hyperparameter search module.
Handles random search for optimal hyperparameters.
"""

import numpy as np


def random_search_hyperparameters(run_optimization_func, num_trials=10, n_rs=50, k_rs=10, seed_rs=42):
    """
    Performs a random search to find good hyperparameters for run_optimization.

    Parameters:
    - run_optimization_func: Function to run optimization with
    - num_trials: Number of random hyperparameter sets to try.
    - n_rs: Number of variables for the test runs.
    - k_rs: Number of simplex constraints for the test runs.
    - seed_rs: Random seed for reproducibility of the search itself.
    
    Returns:
    - best_hyperparameters: Best hyperparameters found
    - best_result_metric: Best result metric (relative gap)
    """
    # Save current random state
    random_state = np.random.get_state()
    
    # Set seed for hyperparameter generation
    np.random.seed(seed_rs)

    best_hyperparameters = None
    best_result_metric = float('inf')  # We want to minimize relative_gap

    # Define search spaces for each hyperparameter
    search_spaces = {
        'regularization': [1e-3, 1e-2, 1e-1, 1e-0],
        'alpha': np.random.uniform(0.001, 1.0, size=num_trials),
        'beta': np.random.uniform(0.0, 0.99, size=num_trials),
        'max_iter': np.random.randint(500, 10001, size=num_trials),
        'rel_gap_tol': [1e-3, 1e-4, 1e-5, 1e-6],
        'alpha_min': [1e-8, 1e-7, 1e-6, 1e-5],
        'step_reduction_factor': np.random.uniform(0.5, 0.99, size=num_trials),
        'non_decrease_threshold': np.random.randint(3, 31, size=num_trials),
        'early_phase_iters': np.random.randint(10, 501, size=num_trials)
    }

    print(f"\nStarting random hyperparameter search for {num_trials} trials...")
    print(f"Using n={n_rs}, k={k_rs} for evaluation runs.")

    for i in range(num_trials):
        current_hyperparameters = {}
        print(f"\nTrial {i+1}/{num_trials}")
        for param, space in search_spaces.items():
            if isinstance(space, list):
                current_hyperparameters[param] = np.random.choice(space)
            elif isinstance(space, np.ndarray):
                current_hyperparameters[param] = space[i]
            else:
                current_hyperparameters[param] = space
        
        run_seed = 0  # Use same seed as main run for consistent problem data
        print(f"Testing with: {current_hyperparameters}, seed={run_seed}")

        try:
            results = run_optimization_func(
                n=n_rs,
                k=k_rs,
                seed=run_seed,  # Use same seed for consistent problem data
                regularization=current_hyperparameters['regularization'],
                alpha=current_hyperparameters['alpha'],
                beta=current_hyperparameters['beta'],
                max_iter=current_hyperparameters['max_iter'],
                rel_gap_tol=current_hyperparameters['rel_gap_tol'],
                alpha_min=current_hyperparameters['alpha_min'],
                step_reduction_factor=current_hyperparameters['step_reduction_factor'],
                non_decrease_threshold=current_hyperparameters['non_decrease_threshold'],
                early_phase_iters=current_hyperparameters['early_phase_iters'],
                enable_plots=False,
                save_plots=False
            )

            if results and 'relative_gap' in results and results['relative_gap'] is not None:
                current_metric = results['relative_gap']
                print(f"Trial {i+1} result: Relative Gap = {current_metric:.4e}, Runtime = {results['runtime']:.2f}s, Iterations = {results['iterations']}")
                if current_metric < best_result_metric:
                    best_result_metric = current_metric
                    best_hyperparameters = current_hyperparameters.copy()
                    best_hyperparameters['seed_for_run'] = run_seed # Store seed used for this best run
                    print(f"*** New best hyperparameters found! Relative Gap: {best_result_metric:.4e} ***")
            else:
                print(f"Trial {i+1} did not return a valid result or relative_gap.")

        except Exception as e:
            print(f"Trial {i+1} failed with error: {e}")
            import traceback
            traceback.print_exc()
            
    if best_hyperparameters:
        print("\nRandom Search Complete!")
        print("Best hyperparameters found:")
        for param, value in best_hyperparameters.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4e}")
            else:
                print(f"  {param}: {value}")
        print(f"Best Relative Gap: {best_result_metric:.4e}")
        
        # Optionally, run one final time with the best hyperparameters and plots enabled
        print("\nRunning final optimization with best hyperparameters and plots enabled...")
        final_results = run_optimization_func(
            n=n_rs, k=k_rs, # or use larger n, k for final demonstration
            seed=best_hyperparameters.get('seed_for_run', seed_rs), # use the specific seed that gave the best result
            **{k: v for k, v in best_hyperparameters.items() if k != 'seed_for_run'}, # exclude seed_for_run from kwargs
            enable_plots=True,
            save_plots=True
        )
        if final_results:
            run_seed = 0
            print(f" seed: {run_seed}")
            print("\nFinal run with best hyperparameters:")
            print(f"  Optimal Primal Value (CVXPY): {final_results['primal_opt']:.6e}")
            print(f"  Final Dual Value: {final_results['dual_value']:.6e}")
            print(f"  Final Relative Gap: {final_results['relative_gap']:.2e}")
            print(f"  Total Runtime: {final_results['runtime']:.2f} seconds")
            print(f"  Number of Iterations: {final_results['iterations']}")
    else:
        print("\nRandom Search Complete! No successful runs or valid hyperparameters found.")

    return best_hyperparameters, best_result_metric
