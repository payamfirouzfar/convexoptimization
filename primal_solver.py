"""
Primal problem solver module.
Handles solving the primal optimization problem using CVXPY.
"""

import numpy as np
import cvxpy as cp
import time


def solve_primal_problem(n, Q_psd, q, Iks):
    """
    Solves the original primal problem to find the optimal primal value.
        
    Parameters:
    - n: Number of variables
    - Q_psd: Positive semidefinite matrix
    - q: Linear cost vector
    - Iks: Partition of indices for simplex constraints
    
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


def solve_dual_with_cvxpy(n, k, seed=0):
    """
    Solves the dual problem using CVXPY with a method similar to subgradient approach.
    Generates the same problem instance as the main algorithm for fair comparison.
    
    Parameters:
    - n: Number of variables
    - k: Number of simplex constraints
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary with optimal_value, mu_opt, lambda_opt, solve_time, status
    """
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
