# Density filter: smooths the design variable vector rho to prevent
# checkerboard patterns and enforce a minimum length scale.
#
# computeFilter(mesh, rmin) — precomputes the filter weight matrix H
#   and its row sums Hs. Run once at setup time. Pure numpy, CPU only.
#
# applySensitivityFilter(ft, x, dc, dv) — chains the filter into the
#   gradient: modifies dc and dv so MMA sees filtered sensitivities.
#
# computeLocalElements(mesh, dist) — precomputes element neighborhood matrix
#   used by the max length scale constraint in thermal.py.

import numpy as np


def computeFilter(mesh, rmin):
    nelx, nely = mesh['nelx'], mesh['nely']
    H = np.zeros((nelx*nely, nelx*nely))

    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = (i1)*nely+j1
            imin = max(i1-(np.ceil(rmin)-1), 0.)
            imax = min(i1+(np.ceil(rmin)), nelx)
            for i2 in range(int(imin), int(imax)):
                jmin = max(j1-(np.ceil(rmin)-1), 0.)
                jmax = min(j1+(np.ceil(rmin)), nely)
                for j2 in range(int(jmin), int(jmax)):
                    e2 = i2*nely+j2
                    H[e1, e2] = max(0., rmin-np.sqrt((i1-i2)**2+(j1-j2)**2))

    Hs = np.sum(H, 1)
    return H, Hs


def applySensitivityFilter(ft, x, dc, dv):
    if (ft['type'] == 1):
        dc = np.matmul(ft['H'],
                       np.multiply(x, dc)/ft['Hs']/np.maximum(1e-3, x))
    elif (ft['type'] == 2):
        dc = np.matmul(ft['H'], (dc/ft['Hs']))
        dv = np.matmul(ft['H'], (dv/ft['Hs']))
    return dc, dv


def computeLocalElements(mesh, dist, avgLocality=False):
    nelx, nely = mesh['nelx'], mesh['nely']
    dx, dy = mesh['elemSize'][0], mesh['elemSize'][1]
    localElems = np.zeros((nelx*nely, nelx*nely))
    for elem in range(nelx*nely):
        ex = dx*elem//nely
        ey = dy*elem%nely
        for neigh in range(nelx*nely):
            nx = dx*neigh//nely
            ny = dy*neigh%nely
            r = (nx-ex)**2 + (ny-ey)**2
            if(r <= dist):
                localElems[elem, neigh] = 1
        if(avgLocality):
            localElems[elem, :] /= np.sum(localElems[elem, :])
    return localElems
