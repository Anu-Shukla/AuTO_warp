# Microstructure / unit cell design via homogenization.
#
# Goal: find rho for a repeating unit cell that maximizes a homogenized
# modulus (bulk modulus, shear modulus, or targets a Poisson's ratio).
#
# This problem is more complex than the others:
#   - Uses periodic boundary conditions (not fixed-wall BCs)
#   - Requires 3 linear solves simultaneously (one per unit strain state)
#   - Computes a homogenized elasticity tensor from all 3 displacement fields
#   - Objective is an entry (or combination of entries) of that tensor
#
# Forward pass (run under Warp tape for autodiff):
#   E  = material.simp(rho)             # rho -> stiffness per element
#   K  = assembly.assemble(K0, E)       # global stiffness matrix
#   Kr, F = apply_periodic_bc(K)        # condense periodic DOFs
#   U  = solver.solve(Kr, F, free)      # 3 simultaneous solves (ndof x 3)
#   CH = homogenize(U, K0, rho)         # homogenized elasticity tensor (3x3)
#   J  = -CH[i, j]                      # maximize chosen modulus entry
#
# Note: this is the most computationally intensive problem and should be
# implemented last, after the other three are validated.
