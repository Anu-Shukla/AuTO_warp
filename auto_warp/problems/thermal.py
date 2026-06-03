# Thermal compliance minimization.
#
# Goal: find rho that minimizes J = dot(u, K @ u) subject to mean(rho) <= vf.
# J measures total heat dissipation — lower means better heat conduction.
#
# Forward pass (run under Warp tape for autodiff):
#   k = material.simp(rho, ...)     # rho -> conductivity per element
#   K = assembly.assemble(K0, k)    # element conductivities -> global matrix
#   u = solver.solve(K, f, free)    # Ku = f -> temperature field vector
#   J = dot(u, K @ u)               # scalar objective
#
# Uses the thermal K0 (4x4) from mesh.py and 1 DOF per node (temperature).
# Boundary conditions: fixed temperature nodes, uniform heat flux body load.
# Constraint: global volume fraction mean(rho)/vf - 1 <= 0.
# Optional: max length scale constraint on void regions.
