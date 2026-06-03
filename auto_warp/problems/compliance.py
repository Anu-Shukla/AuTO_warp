# Structural compliance minimization.
#
# Goal: find rho that minimizes J = dot(f, u) subject to mean(rho) <= vf.
# J measures how much the structure deforms under load — lower is stiffer.
#
# Forward pass (run under Warp tape for autodiff):
#   E = material.simp(rho)          # rho -> stiffness per element
#   K = assembly.assemble(K0, E)    # element stiffnesses -> global matrix
#   u = solver.solve(K, f, free)    # Ku = f -> displacement vector
#   J = dot(f, u)                   # scalar objective
#
# Uses the structural K0 (8x8) from mesh.py and 2 DOFs per node.
# Boundary conditions: one edge fully fixed, point load applied at one node.
# Constraint: global volume fraction mean(rho)/vf - 1 <= 0.
