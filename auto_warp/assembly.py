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

import warp as wp

@wp.kernel
def assembleK_kernel(KE_flat: wp.array[wp.float32],
                     E: wp.array[wp.float32],
                     iK: wp.array[wp.int32],
                     jK: wp.array[wp.int32],
                     K: wp.array2d[wp.float32],
                     n_dof_per_elem: int):
    e = wp.tid()
    for n in range(n_dof_per_elem):
        val = KE_flat[n] * E[e]
        wp.atomic_add(K, iK[e * n_dof_per_elem + n], jK[e * n_dof_per_elem + n], val)


def assembleK(KE_flat, E, iK, jK, ndof):
    K = wp.zeros((ndof, ndof), dtype=wp.float32, device=E.device, requires_grad=True)
    KE_flat_wp = wp.array(KE_flat, dtype=wp.float32, device=E.device)
    iK_wp = wp.array(iK, dtype=wp.int32, device=E.device)
    jK_wp = wp.array(jK, dtype=wp.int32, device=E.device)

    wp.launch(kernel=assembleK_kernel, dim=E.shape[0], inputs = [KE_flat_wp, E, iK_wp, jK_wp, K, KE_flat.shape[0]], device="cuda")

    return K


