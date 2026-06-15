# Thermal compliance minimization.
#
# Goal: find rho that minimizes J = dot(heat, u) subject to mean(rho) <= vf.
# J measures total heat dissipation — lower means better heat conduction.
#
# Forward pass (run under Warp tape for autodiff):
#   k = simp(rho, 0, k0, penal)      # rho -> conductivity per element
#   K = assembleK(KE_flat, k, ...)   # element conductivities -> global matrix
#   u = solver(K, heat, free, ndof)  # Ku = heat -> temperature field
#   J = dot(heat, u)                 # scalar objective (= u^T K u since Ku=heat)
#
# Uses the thermal K0 (4x4) from mesh.py and 1 DOF per node (temperature).
# Boundary conditions: single fixed temperature node, uniform body heat load.
# Constraint: global volume fraction mean(rho)/vf - 1 <= 0.

import warp as wp
import numpy as np
from auto_warp.mesh import get_thermal_mesh
from auto_warp.material import simp
from auto_warp.assembly import assembleK
from auto_warp.solver import solveKuf
from auto_warp.optimizer import optimize


class ThermalComplianceMinimizer:
    def __init__(self, mesh, bc, material, globalvolCons):
        self.mesh = mesh
        self.bc = bc
        self.material = material
        self.globalVolumeConstraint = globalvolCons

        _, idx, KE = get_thermal_mesh(mesh['nelx'], mesh['nely'])
        self.iK = np.array(idx[0])
        self.jK = np.array(idx[1])
        self.KE_flat = KE.flatten()

        self.objectiveHandle = self.computeThermalCompliance
        self.consHandle = self.computeConstraints
        self.numConstraints = 1

    def computeThermalCompliance(self, rho):
        rho_wp = wp.array(rho, dtype=wp.float32, device="cuda", requires_grad=True)

        tape = wp.Tape()
        with tape:
            k = simp(rho_wp, 0., self.material['k0'], self.material['penal'])
            K = assembleK(self.KE_flat, k, self.iK, self.jK, self.mesh['ndof'])

        u = solveKuf(K, self.bc['heat'], self.bc['freeTempNodes'], self.mesh['ndof'])
        u_np = u.numpy()

        J = np.dot(self.bc['heat'], u_np)
        K.grad = wp.array(-np.outer(u_np, u_np).astype(np.float32), dtype=wp.float32, device="cuda")

        tape.backward()

        return J, rho_wp.grad.numpy()

    def computeConstraints(self, rho, epoch):
        vf = self.globalVolumeConstraint['vf']
        n = len(rho)
        c = np.array([[np.mean(rho) / vf - 1.]])
        gradc = np.ones((1, n)) / (n * vf)
        return c, gradc

    def TO(self, optimizationParams, ft):
        optimize(self.mesh, optimizationParams, ft,
                 self.objectiveHandle, self.consHandle, self.numConstraints)
