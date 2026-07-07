"""
Dual problem solver module.
Handles solving the dual optimization problem using deflected subgradient method.
"""

import numpy as np
import time
from matrix_operations import factorize_matrix, solve_linear_system


def lagrangian_dual(x, mu, lambda_k, Q_psd, q, Iks):
    """
    Computes the dual function value for a given x, mu, and lambda_k.

    Parameters:
    - x: Primal variables
    - mu: Dual variables for non-negativity constraints (Lagrange multipliers)
    - lambda_k: Dual variables for simplex constraints
    - Q_psd: Positive semidefinite matrix
    - q: Linear cost vector
    - Iks: Partition of indices for simplex constraints

    Returns:
    - dual_value: Value of the Lagrangian dual function
    """
    # Full dualization Lagrangian: f(x) - μᵀx + Σ λ_k * (Σ x_i - 1)
    value = 0.5 * x.T @ Q_psd @ x + q.T @ x - mu.T @ x
    for j, Ik in enumerate(Iks):
        value += lambda_k[j] * (np.sum(x[Ik]) - 1)
    return value


def solve_inner_problem(mu, lambda_k, factorization, q, Iks):
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
    - q: Linear cost vector
    - Iks: Partition of indices for simplex constraints
    
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


def compute_subgradients(x, Iks):
    """
    Computes the subgradients of the dual function with respect to mu and lambda_k.
    
    Parameters:
    - x: Current primal solution
    - Iks: Partition of indices for simplex constraints
    
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


def deflected_subgradient_method(n, Q_psd, q, Iks, eigvals, primal_opt, 
                                 alpha, beta, max_iter, rel_gap_tol,
                                 alpha_min, step_reduction_factor, 
                                 non_decrease_threshold, early_phase_iters,
                                 regularization):
    """
    Solves the dual problem using the deflected subgradient method with full dualization.
    
    Parameters:
    - n: Number of variables
    - Q_psd: Positive semidefinite matrix
    - q: Linear cost vector
    - Iks: Partition of indices for simplex constraints
    - eigvals: Eigenvalues of Q
    - primal_opt: Optimal primal value
    - alpha: Initial step size
    - beta: Deflection parameter
    - max_iter: Maximum number of iterations
    - rel_gap_tol: Relative gap tolerance for convergence
    - alpha_min: Minimum step size
    - step_reduction_factor: Factor to reduce step size
    - non_decrease_threshold: Number of iterations without improvement before reducing step size
    - early_phase_iters: Number of iterations to use fixed step size
    - regularization: Regularization parameter
    
    Returns:
    - mu: Optimal dual variables for non-negativity constraints
    - lambda_k: Optimal dual variables for simplex constraints
    - dual_values: History of dual function values
    - primal_values: History of primal function values
    - primal_opt: Optimal primal value
    - relative_gaps: History of relative gaps
    - constraint_violations: History of constraint violations
    - all_multipliers: History of all multipliers
    - dual_solve_time: Time taken by dual algorithm
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
        x = solve_inner_problem(mu, lambda_k, factorization, q, Iks)

        # Compute the dual function value
        phi = lagrangian_dual(x, mu, lambda_k, Q_psd, q, Iks)
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
        s_mu, s_lambda = compute_subgradients(x, Iks)
        
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
        x = solve_inner_problem(mu, lambda_k, factorization, q, Iks)
        phi = lagrangian_dual(x, mu, lambda_k, Q_psd, q, Iks)
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
