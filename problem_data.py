"""
Problem data generation module.
Handles generation of optimization problem matrices and constraints.
"""

import numpy as np


def generate_problem_data(n, k, seed, regularization=1e-6):
    """
    Generate problem data for the optimization problem.
    
    Parameters:
    - n: Number of variables
    - k: Number of simplex constraints
    - seed: Random seed for reproducibility
    - regularization: Value added to diagonal of Q to ensure positive definiteness
    
    Returns:
    - Q_psd: Positive semidefinite matrix
    - q: Linear cost vector
    - A: Constraint matrix
    - b: Constraint vector
    - Iks: Partition of indices for simplex constraints
    - eigvals: Eigenvalues of Q
    - eigvecs: Eigenvectors of Q
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
    
    return Q_psd, q, A, b, Iks, eigvals, eigvecs
