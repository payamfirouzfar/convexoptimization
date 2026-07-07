# convexoptimization

📖 Overview
Welcome! This repository introduces a high-performance solver specifically engineered for convex quadratic optimization problems subject to simplex constraints. These mathematical formulations are foundational to numerous advanced disciplines: optimizing portfolio selection in finance, driving resource allocation in operations research, calibrating mixture models in machine learning, and resolving network flows in game theory.

Generic black-box solvers can technically process these models, but they routinely fail to capitalize on the problem's underlying geometry. Our custom approach explicitly exploits this unique architecture to deliver an algorithm that is computationally superior.

🧮 Mathematical Formulation
Our computational engine solves the following standard generic formulation. We minimize the quadratic cost function:

f(x)= 
2
1
​
 x 
T
 Qx+q 
T
 x
This is subject to block-simplex constraints and non-negativity:

i∈I 
k
​
 
∑
​
 x 
i
​
 =1,k=1,…,K
x 
i
​
 ≥0,i=1,…,n
In this model, x represents the non-negative decision variable vector. The matrix Q is a positive semidefinite matrix (which may be singular), and q acts as the linear cost vector. The subsets {I 
1
​
 ,…,I 
K
​
 } constitute a complete partition of the variables {1,…,n}, enforcing the requirement that each defined group forms a valid convex combination (a simplex).

🚀 Algorithmic Insight
Because Q is positive semidefinite, the entire problem is convex and guarantees strong duality under mild regularity conditions. The primary breakthrough of this repository lies in bypassing standard, generic interior-point routines. Instead, our algorithm leverages the special structure of the block-simplex constraints.

By transitioning into the Lagrangian dual space, our method isolates an inner optimization problem that is perfectly analytically solvable. This mathematical exploitation yields a specialized solver that drastically reduces computational overhead and memory bottlenecks compared to black-box alternatives. We invite you to explore the source code, review our benchmarks, and leverage this solver for your own allocation problems!
