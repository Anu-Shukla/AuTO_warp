# Structural compliance minimization.
#
# Goal: find rho that minimizes J = dot(f, u) subject to mean(rho) <= vf.
# J measures how much the structure deforms under load — lower is stiffer.
#
# Forward pass (run under Warp tape for autodiff):
#   E = material.simp(rho)          # rho -> stiffness per element
#   K = assembly.assemble(K0, E)    # element stiffnesses -> global matrix
#   u = solver.solve(K, f, free)    # Ku = f -> displacement vector
#   J = dot(f, u)                   # scalar objective
#
# Uses the structural K0 (8x8) from mesh.py and 2 DOFs per node.
# Boundary conditions: one edge fully fixed, point load applied at one node.
# Constraint: global volume fraction mean(rho)/vf - 1 <= 0.

import warp as wp
import numpy as np
from auto_warp.mesh import get_structural_mesh
from auto_warp.material import simp
from auto_warp.assembly import assembleK
from auto_warp.solver import solveKuf
from auto_warp.optimizer import optimize


class ComplianceMinimizer:
    def __init__(self, mesh, bc, material, globalvolCons):
        self.mesh = mesh
        self.bc = bc
        self.material = material
        self.globalVolumeConstraint = globalvolCons

        _, idx, KE = get_structural_mesh(mesh['nelx'], mesh['nely'], material)
        self.iK = np.array(idx[0])
        self.jK = np.array(idx[1])
        self.KE_flat = KE.flatten()

        self.objectiveHandle = self.computeCompliance
        self.consHandle = self.computeConstraints
        self.numConstraints = 1

    def computeCompliance(self, rho):
        rho_wp = wp.array(rho, dtype=wp.float32, device="cuda", requires_grad=True)

        tape = wp.Tape()
        with tape:
            E = simp(rho_wp, self.material['Emin'], self.material['Emax'], self.material['penal'])
            K = assembleK(self.KE_flat, E, self.iK, self.jK, self.mesh['ndof'])

        u = solveKuf(K, self.bc['force'], self.bc['free'], self.mesh['ndof'])
        u_np = u.numpy()

        J = np.dot(self.bc['force'], u_np)
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
