# Linear solver: solves the FEM system K_free * u_free = f_free, where
# K_free and f_free are K and f restricted to the free (unconstrained) DOFs.
#
# This is the most architecturally significant part of the Warp port.
# JAX used jax.scipy.linalg.solve (dense Cholesky). Options here:
#
#   dense   — direct dense solve via cuSolver through Warp/CuPy.
#             Simple, matches JAX baseline, but O(ndof^3) — won't scale.
#
#   sparse  — convert K to CSR format, call cuSPARSE via CuPy.
#             K is actually very sparse (most entries zero), so this is
#             the right long-term approach.
#
#   iterative — Conjugate Gradient implemented as Warp kernels.
#             Fully differentiable through Warp's tape, but CG convergence
#             needs a preconditioner to be practical.
#
# The solver interface is kept as a swappable backend so we can benchmark
# all three without changing the problem files.
#
# Gradients flow back through the solve via the adjoint:
#   given dJ/du, we need dJ/dK — which requires solving K * lambda = dJ/du.

import warp as wp
import numpy as np
import scipy.linalg 

def solveKuf(K, f, freeDofs, ndof):
    K_np = K.numpy()
    K_free = K_np[freeDofs, :][:, freeDofs]
    f_free = f[freeDofs]
    u_free = scipy.linalg.solve(K_free, f_free)
    u_np = np.zeros(ndof)
    u_np[freeDofs] = u_free
    return wp.array(u_np, dtype=wp.float64, device="cuda")

