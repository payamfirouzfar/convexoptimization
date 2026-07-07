"""
Visualization module.
Handles plotting and visualization of optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_plots(mu_opt, lambda_k_opt, dual_values, primal_values, primal_opt, 
                relative_gaps, constraint_violations, all_multipliers, 
                enable_plots=True, save_plots=True):
    """
    Create comprehensive plots of the optimization results.
    
    Parameters:
    - mu_opt: Optimal dual variables for non-negativity constraints
    - lambda_k_opt: Optimal dual variables for simplex constraints
    - dual_values: History of dual function values
    - primal_values: History of primal function values
    - primal_opt: Optimal primal value
    - relative_gaps: History of relative gaps
    - constraint_violations: History of constraint violations
    - all_multipliers: History of all multipliers
    - enable_plots: Whether to generate plots
    - save_plots: Whether to save plots to files
    """
    print("\nCreating plots...")
    print(f"Number of iterations to plot: {len(dual_values)}")
    
    if not enable_plots:
        print("Plotting disabled (enable_plots=False)")
        return
        
    # Create a comprehensive plot of the results - show full iteration history
    plt.figure(figsize=(15, 10))
    iterations = np.arange(len(dual_values))

    # Plot 1: Objective values convergence
    plt.subplot(2, 2, 1)
    plt.plot(iterations, dual_values, label="Dual Function Value", linewidth=1.5)
    valid_primal_indices = ~np.isnan(primal_values)
    valid_iterations = np.arange(len(primal_values))[valid_primal_indices]
    valid_values = np.array(primal_values)[valid_primal_indices]
    plt.plot(valid_iterations, valid_values, 'o', markersize=3, label="Feasible Primal Value", linestyle='--')
    plt.axhline(y=primal_opt, color='r', linestyle='--', label="Optimal Primal Value (CVXPY)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title(f"Convergence of Objective Values (Total {len(dual_values)} Iterations)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Relative gap (log scale)
    plt.subplot(2, 2, 2)
    plt.semilogy(iterations, relative_gaps, label="Relative Optimality Gap", linewidth=1.5)
    plt.axhline(y=1e-6, color='r', linestyle='--', label="Convergence Threshold (1e-6)")
    plt.xlabel("Iteration")
    plt.ylabel("Relative Gap (log scale)")
    plt.title("Convergence of Relative Gap")
    plt.legend()
    plt.grid(True)

    # Plot 3: Constraint violations (log scale)
    plt.subplot(2, 2, 3)
    if np.any(np.array(constraint_violations) > 0):
        plt.semilogy(iterations, constraint_violations, label="Max Constraint Violation", linewidth=1.5)
        plt.axhline(y=1e-4, color='r', linestyle='--', label="Feasibility Threshold (1e-4)")
    else:
        plt.text(0.5, 0.5, "No constraint violations", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
    plt.xlabel("Iteration")
    plt.ylabel("Constraint Violation (log scale)")
    plt.title("Constraint Feasibility")
    plt.legend()
    plt.grid(True)

    # Plot 4: Dual multipliers evolution
    plt.subplot(2, 2, 4)
    # Extract mu values from multipliers history
    all_mu_array = np.array([mult['mu'] for mult in all_multipliers])
    # Find the most significant mu multipliers
    mu_changes = np.ptp(all_mu_array, axis=0)
    significant_mu_indices = np.argsort(-mu_changes)[:min(3, all_mu_array.shape[1])]
    
    # Extract lambda values from multipliers history
    all_lambda_array = np.array([mult['lambda'] for mult in all_multipliers])
    # Find the most significant lambda multipliers
    lambda_changes = np.ptp(all_lambda_array, axis=0)
    significant_lambda_indices = np.argsort(-lambda_changes)[:min(2, all_lambda_array.shape[1])]
    
    # Fix the dimensions mismatch
    mult_iterations = np.arange(len(all_multipliers))
    
    # Plot most significant mu values
    for i in significant_mu_indices:
        plt.plot(mult_iterations, all_mu_array[:, i], label=f"μ_{i}", linestyle='-')
    
    # Plot most significant lambda values
    for i in significant_lambda_indices:
        plt.plot(mult_iterations, all_lambda_array[:, i], label=f"λ_{i}", linestyle='--')
    
    plt.xlabel("Iteration")
    plt.ylabel("Multiplier Value")
    plt.title("Evolution of Most Significant Dual Multipliers")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save the figure before showing it
    if save_plots:
        plt.savefig("thirdalg_convergence.png")

    if enable_plots:
        plt.show()
        
    print("Plotting completed. Files saved: thirdalg_convergence.png")
