import numpy as np
from auto_warp.problems.thermal import ThermalComplianceMinimizer
from auto_warp.filter import computeFilter

nelx, nely = 40, 30
mesh = {
    'nelx': nelx,
    'nely': nely,
    'elemSize': np.array([1., 1.]),
    'ndof': (nelx+1)*(nely+1),
    'numElems': nelx*nely
}

material = {'k0': 1., 'penal': 3.}

force = 0.01 * np.ones(mesh['ndof'])
fixed = int(nely / 2 + 1 - nely / 20)
free = np.setdiff1d(np.arange(mesh['ndof']), fixed)
bc = {'heat': force, 'fixedTempNodes': fixed, 'freeTempNodes': free}

globalvolCons = {'vf': 0.5}

H, Hs = computeFilter(mesh, rmin=1.5)
ft = {'type': 1, 'H': H, 'Hs': Hs}
optimizationParams = {'maxIters': 250, 'minIters': 100, 'relTol': 0.02}

opt = ThermalComplianceMinimizer(mesh, bc, material, globalvolCons)
opt.TO(optimizationParams, ft)
