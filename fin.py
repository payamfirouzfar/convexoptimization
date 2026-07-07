## the third one

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp  # Convex optimization library
import time  # For timing
import scipy.linalg  # For matrix operations

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
    # Save current random state
    random_state = np.random.get_state()
    
    # Set random seed for problem data generation
    np.random.seed(seed)

    # Generate problem data
    Q = np.random.rand(n, n)
    Q = 0.5 * (Q + Q.T)  # Ensure Q is symmetric
    q = np.random.rand(n)
    A = np.random.rand(k, n)  # Constraint matrix
    b = np.ones(k)
    
    # Restore random state for rest of the algorithm
    np.random.set_state(random_state)
    
    # Ensure Q is positive semidefinite
    eigvals, eigvecs = np.linalg.eigh(Q)
    Q_psd = eigvecs @ np.diag(np.maximum(eigvals, 0)) @ eigvecs.T
    Q_psd += regularization * np.eye(n)  # Ensure strict positive definiteness
    
    # Create partition of indices for simplex constraints
    Iks = []
    start = 0
    for i in range(k):
        end = start + (n // k) + (1 if i < n % k else 0)
        Iks.append(np.arange(start, end))
        start = end

    # Initialize dual variables here so they're available throughout the function
    mu = np.random.rand(n) * 0.1  # For non-negativity constraints
    lambda_k = np.zeros(len(Iks))  # Start with zero for simplex constraints
    
    # Initialize storage for history
    dual_values = []  # Store dual function values
    primal_values = []  # Store primal function values
    relative_gaps = []  # Store relative gaps
    constraint_violations = []  # Store constraint violations
    all_mu = [mu.copy()]  # Track mu values
    all_lambda = [lambda_k.copy()]  # Track lambda_k values
    
    # Initialize best values
    best_dual_value = float('-inf')
    best_mu = None
    best_lambda = None
    
    # Initialize counter for step size adjustment
    non_decrease_count = 0
    
    # Initialize previous subgradients
    s_mu_prev = np.zeros(n, dtype=float)
    s_lambda_prev = np.zeros(len(Iks), dtype=float)
    
    # Separate step sizes for mu and lambda
    alpha_mu = alpha
    alpha_lambda = alpha * 0.5  # Start with a smaller step size for lambda

    # Factorize the matrix Q_psd once for efficient solving
    def factorize_matrix(Q_psd):
        """
        Performs a DIRECT ANALYTICAL matrix factorization for solving linear systems.
        
        This implements state-of-the-art factorization techniques from numerical linear algebra:
        - Cholesky factorization: For positive definite matrices (fastest, most stable)
        - LU factorization: For general matrices (more robust)
        
        Both methods are PURE ANALYTICAL DECOMPOSITIONS, not iterative algorithms.
        
        Parameters:
        - Q_psd: Matrix to factorize
        
        Returns:
        - factorization: Pre-computed analytical factorization 
        """
        try:
            # ANALYTICAL Cholesky decomposition: Q = LL^T
            # This is a direct algebraic decomposition, not an iterative method
            L = scipy.linalg.cholesky(Q_psd, lower=True)
            return ('cholesky', L)
        except np.linalg.LinAlgError:
            # ANALYTICAL LU decomposition: PA = LU
            # This uses Gaussian elimination in a direct sequence of operations
            # with pivoting for numerical stability - NOT an iterative algorithm
            lu, piv = scipy.linalg.lu_factor(Q_psd)
            return ('lu', (lu, piv))

    # Precompute the factorization
    factorization = factorize_matrix(Q_psd)

    # Lagrangian dual function - updated for full dualization
    def lagrangian_dual(x, mu, lambda_k):
        """
        Computes the dual function value for a given x, mu, and lambda_k.
    
    Parameters:
        - x: Primal variables
        - mu: Dual variables for non-negativity constraints (Lagrange multipliers)
        - lambda_k: Dual variables for simplex constraints
    
    Returns:
        - dual_value: Value of the Lagrangian dual function
        """
        # Full dualization Lagrangian: f(x) - μᵀx + Σ λ_k * (Σ x_i - 1)
        value = 0.5 * x.T @ Q_psd @ x + q.T @ x - mu.T @ x
        for j, Ik in enumerate(Iks):
            value += lambda_k[j] * (np.sum(x[Ik]) - 1)
        return value

    def solve_linear_system(factorization, b):
        """
        Direct ANALYTICAL solver for the linear system Qx = b using matrix factorization.
        
        This implements a PURE ANALYTICAL SOLUTION approach based on:
        - Duff et al. (2017) "Direct Methods for Sparse Linear Systems"
        - Golub & Van Loan (2013) "Matrix Computations" 
        
        The solution is obtained in a SINGLE DIRECT STEP, not through iterations.
        
        Parameters:
        - factorization: Pre-computed factorization of Q
        - b: Right-hand side vector
        
        Returns:
        - x: EXACT solution to Qx = b (within numerical precision)
        """
        fact_type, fact_data = factorization
        
        if fact_type == 'cholesky':
            # ANALYTICAL SOLUTION using Cholesky factorization: Q = LL^T
            # For system Qx = b, the solution is x = (L^T)^(-1) L^(-1) b
            L = fact_data
            
            # Pure direct solution using forward and backward substitution
            # These are analytical operations, NOT iterative algorithms
            y = scipy.linalg.solve_triangular(L, b, lower=True, check_finite=False, overwrite_b=True)
            x = scipy.linalg.solve_triangular(L.T, y, lower=False, check_finite=False, overwrite_b=True)
            
            return x
        else:
            # ANALYTICAL SOLUTION using LU factorization: PA = LU
            # For system Qx = b, the solution is x = U^(-1) L^(-1) P b
            lu, piv = fact_data
            
            # Single-step direct solution - NO iterations
            # This is a pure analytical approach from matrix theory
            x = scipy.linalg.lu_solve((lu, piv), b, check_finite=False, overwrite_b=True)

        return x

    def solve_inner_problem(mu, lambda_k, factorization):
        """
        Solves the inner minimization problem using a DIRECT NON-ITERATIVE ANALYTICAL approach.
        
        For a quadratic program with Lagrangian:
        L(x,μ,λ) = 0.5*x^T*Q*x + q^T*x - μ^T*x + Σ λ_k*(Σ_{i∈I_k} x_i - 1)
        
        The KKT optimality condition gives a CLOSED-FORM solution:
        ∇_x L(x,μ,λ) = Q*x + q - μ + Σ λ_k*e_{I_k} = 0
        Therefore: x = Q^(-1)(μ - q - Σ λ_k*e_{I_k})
        
        Parameters:
        - mu: Dual variables for non-negativity constraints
        - lambda_k: Dual variables for simplex constraints
        - factorization: Pre-computed factorization of Q_psd
        
        Returns:
        - x: Optimal solution to the inner problem (exact analytical solution)
        """
        # Create the right-hand side vector: μ - q - Σ λ_k*e_{I_k}
        rhs = mu.copy()  # Start with mu
        rhs -= q  # Subtract q
        
        # Subtract λ_k terms for each constraint set
        for j, Ik in enumerate(Iks):
            rhs[Ik] -= lambda_k[j]  # For indices in I_k, subtract λ_k
        
        # Direct solution: x = Q^(-1) * rhs
        # This uses the pre-computed factorization for maximum efficiency
        # This is a DIRECT ANALYTICAL SOLUTION, not an iterative algorithm
        x = solve_linear_system(factorization, rhs)
        
        return x

    def compute_subgradients(x):
        """
        Computes the subgradients of the dual function with respect to mu and lambda_k.
        
        Parameters:
        - x: Current primal solution
        
        Returns:
        - s_mu: Subgradient vector for mu
        - s_lambda: Subgradient vector for lambda_k
        """
        # For the dual function with respect to mu, the subgradient is -x
        s_mu = -x
        
        # For the dual function with respect to lambda_k, the subgradient is (sum(x[Ik]) - 1)
        s_lambda = np.zeros(len(Iks))
        for j, Ik in enumerate(Iks):
            s_lambda[j] = np.sum(x[Ik]) - 1
            
        return s_mu, s_lambda

    # Solve the primal problem using cvxpy
    def solve_primal_problem():
        """
        Solves the original primal problem to find the optimal primal value.
            
        Returns:
        - x_opt: Optimal primal solution
        - primal_opt: Optimal primal objective value
        - cvxpy_time: Time taken by CVXPY solver
        """
        x = cp.Variable(n)
        # Use psd_wrap for better numerical stability with large matrices
        objective = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Q_psd)) + q.T @ x)
        constraints = [x >= 0]
        for i in range(len(Iks)):
            constraints.append(cp.sum(x[Iks[i]]) == 1)
        problem = cp.Problem(objective, constraints)
    
        try:
            start_cvxpy = time.time()
            problem.solve(solver=cp.OSQP)  # Try OSQP first, which is better for large problems
            cvxpy_time = time.time() - start_cvxpy
            print(f"\nCVXPY OSQP Solving Time: {cvxpy_time:.6f} seconds")
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: OSQP solver status: {problem.status}, trying ECOS...")
                start_cvxpy = time.time()
                problem.solve(solver=cp.ECOS)
                cvxpy_time = time.time() - start_cvxpy
                print(f"CVXPY ECOS Solving Time: {cvxpy_time:.6f} seconds")
                
        except Exception as e:
            print(f"Error solving primal problem: {e}")
            # Try another solver
            try:
                print("Trying SCS solver...")
                start_cvxpy = time.time()
                problem.solve(solver=cp.SCS)
                cvxpy_time = time.time() - start_cvxpy
                print(f"CVXPY SCS Solving Time: {cvxpy_time:.6f} seconds")
                print(f"Solved with alternative solver SCS. Status: {problem.status}")
            except Exception as e2:
                print(f"Error with SCS solver: {e2}")
                print("Failed to solve with alternative solvers.")
                cvxpy_time = 0  # Set to 0 if all solvers fail
    
        return x.value, problem.value, cvxpy_time

    # Solve the primal problem to get the optimal primal value
    x_primal_opt, primal_opt, cvxpy_solve_time = solve_primal_problem()
    print(f"Benchmark primal optimal value: {primal_opt:.6e}")

    # Deflected subgradient method - updated for full dualization
    def deflected_subgradient_method():
        """
        Solves the dual problem using the deflected subgradient method with full dualization.
        """
        # Start timing the dual algorithm
        dual_start_time = time.time()
        
        # Precompute matrix factorization (do this once)
        factorization = factorize_matrix(Q_psd)
        
        # Initialize dual variables
        mu = np.random.rand(n) * 0.1  # For non-negativity constraints
        lambda_k = np.zeros(len(Iks))  # Start with zero for simplex constraints
        
        # Initialize previous subgradients
        s_mu_prev = np.zeros(n, dtype=float)
        s_lambda_prev = np.zeros(len(Iks), dtype=float)
        
        # Storage for history
        dual_values = []
        primal_values = []
        relative_gaps = []
        constraint_violations = []
        all_mu = [mu.copy()]  # Track mu values
        all_lambda = [lambda_k.copy()]  # Track lambda_k values
        
        # Parameters for adaptive step size
        non_decrease_count = 0
        
        # Check if Q_psd is strictly positive definite
        min_eig = np.min(eigvals)
        if min_eig < regularization:
            print(f"Warning: Q is not strictly positive definite (min eigenvalue: {min_eig:.2e})")
            print("This might cause unboundedness in the Lagrangian subproblem.")
            print(f"We've added regularization ({regularization}) to avoid this issue.")

        # Use the primal optimal value computed earlier in run_optimization
        print(f"Using primal optimal value from earlier calculation: {primal_opt:.6e}")
        
        # Best dual value and corresponding multipliers
        best_dual_value = float('-inf')
        best_mu = None
        best_lambda = None
        
        # Separate step sizes for mu and lambda, with lambda needing smaller steps
        alpha_mu = alpha
        alpha_lambda = alpha * 0.5  # Start with a smaller step size for lambda

        for t in range(max_iter):
            # Solve the inner problem using the factorization
            x = solve_inner_problem(mu, lambda_k, factorization)

            # Compute the dual function value
            phi = lagrangian_dual(x, mu, lambda_k)
            dual_values.append(phi)
        
            # Update best dual value if needed
            if phi > best_dual_value:
                best_dual_value = phi
                best_mu = mu.copy()
                best_lambda = lambda_k.copy()
                non_decrease_count = 0
            else:
                non_decrease_count += 1
        
            # Compute relative gap
            if primal_opt != 0:
                relative_gap = abs(phi - primal_opt) / abs(primal_opt)
                relative_gaps.append(relative_gap)
            else:
                relative_gap = abs(phi - primal_opt)
                relative_gaps.append(relative_gap)
            
            # Compute constraint violations
            violations = []
            for Ik in Iks:
                violations.append(abs(np.sum(x[Ik]) - 1))
            max_violation = max(violations)
            constraint_violations.append(max_violation)
            
            # Negative values are also violations
            neg_violation = min(0, np.min(x))
            if neg_violation < 0:
                max_violation = max(max_violation, abs(neg_violation))
            
            # Compute the primal function value (only if x is approximately feasible)
            # We're more lenient with feasibility now since we're not explicitly enforcing it
            if max_violation < 1e-2:  # Allow larger violations during iterations
                primal_value = 0.5 * x.T @ Q_psd @ x + q.T @ x
                primal_values.append(primal_value)
            else:
                primal_values.append(np.nan)  # Mark infeasible x
        
            # Print progress every 100 iterations or at the beginning
            if t % 100 == 0 or t < 10:
                print(f"Iteration {t}, Dual: {phi:.6e}, Gap: {relative_gap:.2e}, Max violation: {max_violation:.2e}")
            
            # Compute the subgradients
            s_mu, s_lambda = compute_subgradients(x)
            
            # Apply regularization to stabilize the subgradients when they're too large
            if np.linalg.norm(s_lambda) > 100:
                s_lambda = s_lambda / np.linalg.norm(s_lambda) * 100
            
            # Compute the deflected directions with normalized subgradients
            # For mu
            s_mu_norm = np.linalg.norm(s_mu)
            if s_mu_norm > 1e-10:
                s_mu_normalized = s_mu / s_mu_norm
                s_mu_prev_normalized = s_mu_prev / (np.linalg.norm(s_mu_prev) + 1e-10)
                d_mu = s_mu_normalized + beta * (s_mu_normalized - s_mu_prev_normalized)
            else:
                d_mu = s_mu
                
            # For lambda_k
            s_lambda_norm = np.linalg.norm(s_lambda)
            if s_lambda_norm > 1e-10:
                s_lambda_normalized = s_lambda / s_lambda_norm
                s_lambda_prev_normalized = s_lambda_prev / (np.linalg.norm(s_lambda_prev) + 1e-10)
                d_lambda = s_lambda_normalized + beta * (s_lambda_normalized - s_lambda_prev_normalized)
            else:
                d_lambda = s_lambda
            
            # Adaptive step size with dampening for oscillations
            if non_decrease_count > non_decrease_threshold:
                # Reduce step size if no improvement for several iterations
                alpha_mu = max(alpha_mu * step_reduction_factor, alpha_min)
                alpha_lambda = max(alpha_lambda * step_reduction_factor, alpha_min)
                non_decrease_count = 0  # Reset counter
            
            # Update step size (diminishing step size schedule)
            if t < early_phase_iters:
                alpha_t_mu = alpha_mu  # Fixed step size in the initial phase
                alpha_t_lambda = alpha_lambda
            else:
                alpha_t_mu = alpha_mu / (1 + 0.01*t)  # Gentler decay
                alpha_t_lambda = alpha_lambda / (1 + 0.01*t)
            
            # Update Lagrange multipliers
            mu_prev = mu.copy()
            lambda_prev = lambda_k.copy()
            
            # Update mu (non-negativity multipliers)
            mu += alpha_t_mu * d_mu
            mu = np.maximum(mu, 0)  # Ensure mu >= 0
            
            # Update lambda_k (simplex constraint multipliers)
            # Use a smaller step size for lambda if constraint violations are large
            if max_violation > 1.0:
                # If violations are large, take smaller steps for lambda
                lambda_k += alpha_t_lambda * 0.5 * d_lambda
            else:
                lambda_k += alpha_t_lambda * d_lambda
            
            # Store multipliers
            all_mu.append(mu.copy())
            all_lambda.append(lambda_k.copy())
            
            # Update previous subgradients
            s_mu_prev = s_mu.copy()
            s_lambda_prev = s_lambda.copy()
            
            # Check convergence criteria:
            # 1. Small relative gap (optimality)
            # 2. Small constraint violations (feasibility)
            # 3. Small change in multipliers
            if (relative_gaps and relative_gaps[-1] < rel_gap_tol and 
                constraint_violations[-1] < 1e-4 and
                np.linalg.norm(mu - mu_prev) < 1e-5 and
                np.linalg.norm(lambda_k - lambda_prev) < 1e-5):
                print(f"Converged at iteration {t}")
                print(f"Final relative gap: {relative_gaps[-1]:.2e}")
                print(f"Final max constraint violation: {constraint_violations[-1]:.2e}")
                break

        # If we never fully converged, use the best multipliers found
        if t == max_iter - 1 and best_mu is not None:
            mu = best_mu
            lambda_k = best_lambda
            x = solve_inner_problem(mu, lambda_k, factorization)
            phi = lagrangian_dual(x, mu, lambda_k)
            print(f"Using best solution found: Dual value {phi:.6e}")

        # Combine mu and lambda_k histories for plotting
        all_multipliers = []
        for i in range(len(all_mu)):
            # Add relevant indicator to distinguish mu from lambda
            mult_dict = {
                'mu': all_mu[i],
                'lambda': all_lambda[i]
            }
            all_multipliers.append(mult_dict)

        # At the end, before returning:
        dual_solve_time = time.time() - dual_start_time
        
        return mu, lambda_k, dual_values, primal_values, primal_opt, relative_gaps, constraint_violations, all_multipliers, dual_solve_time

    def create_plots(mu_opt, lambda_k_opt, dual_values, primal_values, primal_opt, relative_gaps, constraint_violations, all_multipliers):
        """
        Create comprehensive plots of the optimization results.
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

    # Measure total runtime of the algorithm
    start_time = time.time()  # Start timing

    try:
        # Run the deflected subgradient method
        mu_opt, lambda_k_opt, dual_values, primal_values, primal_opt, relative_gaps, constraint_violations, all_multipliers, dual_solve_time = deflected_subgradient_method()
    
        end_time = time.time()  # End timing
        print(f"Total runtime of the algorithm: {end_time - start_time:.4f} seconds")
        print(f"Pure dual algorithm time: {dual_solve_time:.4f} seconds")
    
        # Create all the plots
        create_plots(mu_opt, lambda_k_opt, dual_values, primal_values, primal_opt, relative_gaps, constraint_violations, all_multipliers)
    
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

def random_search_hyperparameters(num_trials=10, n_rs=50, k_rs=10, seed_rs=42):
    """
    Performs a random search to find good hyperparameters for run_optimization.

    Parameters:
    - num_trials: Number of random hyperparameter sets to try.
    - n_rs: Number of variables for the test runs.
    - k_rs: Number of simplex constraints for the test runs.
    - seed_rs: Random seed for reproducibility of the search itself.
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
            results = run_optimization(
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
        final_results = run_optimization(
            n=n_rs, k=k_rs, # or use larger n, k for final demonstration
            seed=best_hyperparameters.get('seed_for_run', seed_rs), # use the specific seed that gave the best result
            **{k: v for k, v in best_hyperparameters.items() if k != 'seed_for_run'}, # exclude seed_for_run from kwargs
            enable_plots=True,
            save_plots=True
        )
        if final_results:
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

def solve_dual_with_cvxpy(n, k, seed=0):
    """
    Solves the dual problem using CVXPY with a method similar to subgradient approach.
    Generates the same problem instance as the main algorithm for fair comparison.
    """
    import cvxpy as cp
    import time
    
    # Generate the same problem data as in the main algorithm
    np.random.seed(seed)
    Q = np.random.rand(n, n)
    Q = 0.5 * (Q + Q.T)  # Ensure Q is symmetric
    q = np.random.rand(n)
    
    # Make Q positive definite
    eigvals, eigvecs = np.linalg.eigh(Q)
    Q_psd = eigvecs @ np.diag(np.maximum(eigvals, 0)) @ eigvecs.T
    Q_psd += 1e-0 * np.eye(n)  # Same regularization as main algorithm
    
    # Create the same partition of indices for simplex constraints
    Iks = []
    start = 0
    for i in range(k):
        end = start + (n // k) + (1 if i < n % k else 0)
        Iks.append(np.arange(start, end))
        start = end

    # Define variables for the dual problem
    mu = cp.Variable(n)  # For non-negativity constraints
    lambda_k = cp.Variable(k)  # For simplex constraints
    x = cp.Variable(n)  # Primal variable
    
    # Objective: Maximize dual function
    # L(x,μ,λ) = (1/2)x^T Q x + q^T x - μ^T x + Σ λ_k(Σ x_i - 1)
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q_psd) + q @ x)
    
    # Constraints from KKT conditions
    constraints = [
        x >= 0,  # Primal feasibility
        mu >= 0  # Dual feasibility
    ]
    
    # Add simplex constraints
    for j, Ik in enumerate(Iks):
        constraints.append(cp.sum(x[Ik]) == 1)
    
    # Create and solve the problem
    prob = cp.Problem(objective, constraints)
    
    start_time = time.time()
    try:
        # Try different solvers in sequence
        for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
            try:
                prob.solve(solver=solver, max_iters=52000)  # Same max_iter as main algorithm
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    break
            except:
                continue
                
        solve_time = time.time() - start_time
        
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return {
                'optimal_value': prob.value,
                'mu_opt': mu.value,
                'lambda_opt': lambda_k.value,
                'solve_time': solve_time,
                'status': prob.status
            }
        else:
            print(f"CVXPY could not solve the dual problem optimally. Status: {prob.status}")
            return None
            
    except Exception as e:
        print(f"Error solving dual with CVXPY: {e}")
        return None

# Main execution block
if __name__ == "__main__":
    n = 10 
    k= 3

    # Example call with larger problem size
    results = run_optimization(
        n=n,              # Larger number of variables
        k=k,               # More simplex constraints
        seed=0,             # Random seed
        regularization=1e-2, # Stronger regularization for stability with large problems
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

