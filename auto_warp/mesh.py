# Mesh utilities: builds the DOF connectivity tables used by assembly.
#
# Given a grid of nelx x nely elements, computes:
#   edofMat  — (numElems x dofs_per_elem) array mapping each element to its
#              global DOF indices
#   idx      — (iK, jK) tuple of flattened row/col indices for scatter assembly
#
# Supports both structural meshes (2 DOFs per node, 8 DOFs per element)
# and thermal meshes (1 DOF per node, 4 DOFs per element).
#
# Also computes K0 — the reference element stiffness matrix (structural 8x8
# or thermal 4x4) that gets scaled by material properties in assembly.py.
#
# No Warp here — pure numpy, runs on CPU once at setup time.
