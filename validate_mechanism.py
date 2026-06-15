import numpy as np
from auto_warp.problems.mechanism import CompliantMechanismMinimizer
from auto_warp.filter import computeFilter

nelx, nely = 40, 20
mesh = {
    'nelx': nelx,
    'nely': nely,
    'elemSize': np.array([1., 1.]),
    'ndof': 2*(nelx+1)*(nely+1),
    'numElems': nelx*nely
}

material = {'Emax': 1., 'Emin': 1e-3, 'nu': 0.3, 'penal': 3.}

ndof = mesh['ndof']
force = np.zeros(ndof)
forceOut = np.zeros(ndof)
dofs = np.arange(ndof)
nodeIn = 2 * nely
nodeOut = 2*(nelx+1)*(nely+1) - 2
fixed = dofs[np.r_[0:4:1, 2*(nely+1)-1:2*(nelx+1)*(nely+1):2*(nely+1)]]
force[nodeIn] = 1
forceOut[nodeOut] = -1
free = np.setdiff1d(dofs, fixed)

bc = {
    'nodeIn': nodeIn,
    'nodeOut': nodeOut,
    'force': force,
    'forceOut': forceOut,
    'fixed': fixed,
    'free': free,
    'methodType': 'uOut', # uOut' # 'MSE_SE' # 'wMSE'
}

globalvolCons = {'vf': 0.35}

H, Hs = computeFilter(mesh, rmin=1.5)
ft = {'type': 1, 'H': H, 'Hs': Hs}
optimizationParams = {'maxIters': 200, 'minIters': 100, 'relTol': 0.02}

opt = CompliantMechanismMinimizer(mesh, bc, material, globalvolCons)
opt.TO(optimizationParams, ft)
