# Outer optimization loop: runs the topology optimization iterations.
#
# optimize(mesh, rho, objective_fn, constraint_fn, mma, ft, params) drives:
#   1. Apply density filter to rho
#   2. Call objective_fn(rho) under a Warp tape -> get J and dJ/drho
#   3. Call constraint_fn(rho) -> get g and dg/drho
#   4. Apply sensitivity filter to gradients
#   5. Pass J, dJ, g, dg to MMA -> get updated rho
#   6. Repeat until convergence or max iterations
#
# objective_fn is a callable defined in the problem file (e.g. compliance.py).
# It wraps the Warp tape setup so this loop stays problem-agnostic.
#
# Convergence is measured as the L-inf norm of the change in rho between
# iterations (same as JAX version: change = ||rho_new - rho_old||_inf).
