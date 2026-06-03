# Compliant mechanism design.
#
# Goal: find rho that maximizes displacement at an output node when force
# is applied at an input node (e.g. a force-transmitting flexure or gripper).
# Objective: J = u[node_out]  (or mutual strain energy ratio MSE/SE).
#
# Forward pass (run under Warp tape for autodiff):
#   E = material.simp(rho)          # rho -> stiffness per element
#   K = assembly.assemble(K0, E)    # element stiffnesses -> global matrix
#   K = add_springs(K, node_in, node_out, k_spring)  # small springs at I/O
#   u = solver.solve(K, f_in, free) # primary solve: input load
#   J = u[node_out]                 # output displacement (to maximize -> negate)
#
# For the MSE/SE objective variant, a second solve is needed:
#   v = solver.solve(K, f_out, free) # adjoint-like solve with dummy output load
#   J = -dot(v, K @ u) / dot(u, K @ u)
#
# Uses structural K0 (8x8) and 2 DOFs per node. Same as compliance but
# different objective and spring additions to K.
