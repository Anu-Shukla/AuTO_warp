# Outer optimization loop: runs the topology optimization iterations.
#
# optimize(mesh, rho, objective_fn, constraint_fn, mma, ft, params) drives:
#   1. Apply density filter to rho
#   2. Call objective_fn(rho) under a Warp tape -> get J and dJ/drho
#   3. Call constraint_fn(rho) -> get g and dg/drho
#   4. Apply sensitivity filter to gradients
#   5. Pass J, dJ, g, dg to MMA -> get updated rho
#   6. Repeat until convergence or max iterations
#
# objective_fn is a callable defined in the problem file (e.g. compliance.py).
# It wraps the Warp tape setup so this loop stays problem-agnostic.
#
# Convergence is measured as the L-inf norm of the change in rho between
# iterations (same as JAX version: change = ||rho_new - rho_old||_inf).
import numpy as np
import time
import matplotlib.pyplot as plt
from auto_warp.filter import applySensitivityFilter
from auto_warp.mma import MMA


def optimize(mesh, optimizationParams, ft,
             objectiveHandle, consHandle, numConstraints):
    rho = np.ones((mesh['nelx']*mesh['nely']))
    loop = 0
    change = 1.
    m = numConstraints
    n = mesh['numElems']
    mma = MMA()
    mma.setNumConstraints(numConstraints)
    mma.setNumDesignVariables(n)
    mma.setMinandMaxBoundsForDesignVariables(np.zeros((n,1)), np.ones((n,1)))

    xval = rho[np.newaxis].T
    xold1, xold2 = xval.copy(), xval.copy()
    mma.registerMMAIter(xval, xold1, xold2)
    mma.setLowerAndUpperAsymptotes(np.ones((n,1)), np.ones((n,1)))
    mma.setScalingParams(1.0, np.zeros((m,1)), 10000*np.ones((m,1)), np.zeros((m,1)))
    mma.setMoveLimit(0.2)

    mmaTime = 0

    t0 = time.perf_counter()

    while( (change > optimizationParams['relTol'])
           and (loop < optimizationParams['maxIters'])
           or (loop < optimizationParams['minIters'])):
        loop = loop + 1

        J, dJ = objectiveHandle(rho)

        vc, dvc = consHandle(rho, loop)

        dJ, dvc = applySensitivityFilter(ft, rho, dJ, dvc)

        J, dJ = J, dJ[np.newaxis].T
        tmr = time.perf_counter()
        mma.setObjectiveWithGradient(J, dJ)
        mma.setConstraintWithGradient(vc, dvc)

        xval = rho.copy()[np.newaxis].T
        mma.mmasub(xval)

        xmma, _, _ = mma.getOptimalValues()
        xold2 = xold1.copy()
        xold1 = xval.copy()
        rho = xmma.copy().flatten()
        mma.registerMMAIter(rho, xval.copy(), xold2.copy())

        mmaTime += time.perf_counter() - tmr

        status = 'Iter {:d}; J {:.2F}; vf {:.2F}'.format(loop, J, np.mean(rho))
        print(status)
        if(loop%10 == 0):
            plt.imshow(-np.flipud(rho.reshape((mesh['nelx'], mesh['nely'])).T), cmap='gray')
            plt.title(status)
            plt.show()

    totTime = time.perf_counter() - t0

    print('total time(s): ', totTime)
    print('mma time(s): ', mmaTime)
    print('FE time(s): ', totTime - mmaTime)
    return rho
