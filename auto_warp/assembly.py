# Stiffness matrix assembly: scatters scaled element matrices into the
# global stiffness matrix K.
#
# assembleK(K0, E, idx, ndof) — for each element e, adds K0 * E[e] into
#   the global K at the positions given by idx. This is a scatter operation:
#   multiple elements write to overlapping DOF positions, so atomic adds
#   are required.
#
# In JAX this was: K = K.at[idx].add(K_elem)
# In Warp this becomes a kernel using wp.atomic_add, one thread per
# element, writing its 8x8 (or 4x4) block into the global matrix.
#
# K is dense here (same as JAX baseline). Sparse representation is a
# future optimization once correctness is established.
#
# Implemented as a Warp kernel — runs on GPU and participates in the
# autodiff tape so gradients flow back through assembly to E (and then rho).
