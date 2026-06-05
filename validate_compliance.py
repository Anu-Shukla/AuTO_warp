import numpy as np
from auto_warp.problems.compliance import ComplianceMinimizer
from auto_warp.filter import computeFilter

nelx, nely = 60, 30
mesh = {
    'nelx': nelx,
    'nely': nely,
    'elemSize': np.array([1., 1.]),
    'ndof': 2*(nelx+1)*(nely+1),
    'numElems': nelx*nely
}

material = {'Emax': 1., 'Emin': 1e-3, 'nu': 0.3, 'penal': 3.}

force = np.zeros(mesh['ndof'])
dofs = np.arange(mesh['ndof'])
fixed = dofs[0:2*(nely+1)]
free = np.setdiff1d(dofs, fixed)
force[2*(nelx+1)*(nely+1)-2*nely+1] = -1
bc = {'force': force, 'fixed': fixed, 'free': free}

globalvolCons = {'vf': 0.5}

H, Hs = computeFilter(mesh, rmin=1.5)
ft = {'type': 1, 'H': H, 'Hs': Hs}
optimizationParams = {'maxIters': 100, 'minIters': 100, 'relTol': 0.05}

opt = ComplianceMinimizer(mesh, bc, material, globalvolCons)
opt.TO(optimizationParams, ft)
