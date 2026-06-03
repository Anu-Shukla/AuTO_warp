# Mesh utilities: builds the DOF (degrees of freedom) connectivity tables used by assembly.
#
# Given a grid of nelx x nely elements, computes:
#   edofMat  — (numElems x dofs_per_elem) array mapping each element to its
#              global DOF indices
#   idx      — (iK, jK) tuple of flattened row/col indices for scatter assembly
#
# Supports both structural meshes (2 DOFs per node, 8 DOFs per element)
# and thermal meshes (1 DOF per node, 4 DOFs per element).
#
# Also computes KE — the reference element stiffness matrix (structural 8x8
# or thermal 4x4) that gets scaled by material properties in assembly.py.

import numpy as np


def get_structural_mesh(nelx, nely, matProp):
    # returns edofMat: array of size (numElemsX8) with
    # the global dof of each elem
    # idx: A tuple informing the position for assembly of computed entries
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1 = (nely+1)*elx+ely
            n2 = (nely+1)*(elx+1)+ely
            edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2,
                                        2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
    iK = tuple(np.kron(edofMat, np.ones((8, 1))).flatten().astype(int))
    jK = tuple(np.kron(edofMat, np.ones((1, 8))).flatten().astype(int))
    idx = (iK, jK)

    # with the material defined, we can now calculate the base
    # constitutive matrix
    # the base constitutive matrix assumes unit
    # area element with E = 1. and nu prescribed.
    # the material is also assumed to be isotropic.
    # returns a matrix of size (8X8)
    E = 1.
    nu = matProp['nu']
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])

    return edofMat, idx, KE


def get_thermal_mesh(nelx, nely):
    nelx, nely = nelx, nely  # too lazy to write
    ndof = (nelx+1)*(nely+1)
    nodenrs = np.reshape(np.arange(0, ndof), (1+nelx, 1+nely)).T
    edofVec = np.reshape(nodenrs[0:-1, 0:-1]+1, nelx*nely, order='F')

    edofMat = np.tile(edofVec, (4, 1)).T + \
              np.tile(np.array([0, nely+1, nely, -1]), (nelx*nely, 1))
    edofMat = edofMat.astype(int)

    iK = np.kron(edofMat, np.ones((4, 1))).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((1, 4))).flatten().astype(int)
    idx = (iK, jK)

    KE = np.array([
         2./3., -1./6., -1./3., -1./6.,
        -1./6.,  2./3., -1./6., -1./3.,
        -1./3., -1./6.,  2./3., -1./6.,
        -1./6., -1./3., -1./6.,  2./3.,
    ]).reshape(4, 4)

    return edofMat, idx, KE
