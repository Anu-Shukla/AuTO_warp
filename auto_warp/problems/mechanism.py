# Compliant mechanism design.
#
# Goal: find rho that maximizes displacement at an output node when force
# is applied at an input node (e.g. a force-transmitting flexure or gripper).
#
# Forward pass (run under Warp tape for autodiff):
#   E = simp(rho)                        # rho -> stiffness per element
#   K = assembleK(KE_flat, E, ...)       # element stiffnesses -> global matrix
#   K[nodeIn, nodeIn]  += k_spring       # small springs at I/O nodes (outside tape)
#   K[nodeOut, nodeOut] += k_spring
#   u = solver(K, force, free, ndof)     # primary solve: input load
#   v = solver(K, forceOut, free, ndof)  # dummy solve: unit load at output node
#
# Three objective variants (bc['methodType']):
#   'uOut'   : J = u[nodeOut]                       -- maximize output displacement
#   'MSE_SE' : J = -MSE/SE                          -- mutual strain energy ratio
#   'wMSE'   : J = -w*MSE + (1-w)*SE               -- weighted combination
#
# where MSE = v^T K u, SE = u^T K u (= dot(force, u) since Ku = force).
#
# Manual adjoints (dJ/dK, injected as K.grad before tape.backward()):
#   'uOut'   : K.grad =  outer(v, u)
#   'MSE_SE' : K.grad =  outer(u, v)/SE - (MSE/SE^2) * outer(u, u)
#   'wMSE'   : K.grad =  w * outer(u, v) - (1-w) * outer(u, u)
#
# Note: forceOut[nodeOut] = -1 (same convention as JAX notebook), so
# v = K^{-1} forceOut = -K^{-1} e_nodeOut, hence the sign in 'uOut' adjoint.
#
# Springs are added outside the tape: they are constants (independent of rho),
# so dJ/dK is the same whether springs are inside or outside the tape.

import warp as wp
import numpy as np
from auto_warp.mesh import get_structural_mesh
from auto_warp.material import simp
from auto_warp.assembly import assembleK
from auto_warp.solver import solveKuf
from auto_warp.optimizer import optimize


class CompliantMechanismMinimizer:
    def __init__(self, mesh, bc, material, globalvolCons):
        self.mesh = mesh
        self.bc = bc
        self.material = material
        self.globalVolumeConstraint = globalvolCons

        _, idx, KE = get_structural_mesh(mesh['nelx'], mesh['nely'], material)
        self.iK = np.array(idx[0])
        self.jK = np.array(idx[1])
        self.KE_flat = KE.flatten()

        self.objectiveHandle = self.computeMechanism
        self.consHandle = self.computeConstraints
        self.numConstraints = 1

    def computeMechanism(self, rho):
        rho_wp = wp.array(rho, dtype=wp.float32, device="cuda", requires_grad=True)

        tape = wp.Tape()
        with tape:
            E = simp(rho_wp, self.material['Emin'], self.material['Emax'], self.material['penal'])
            K = assembleK(self.KE_flat, E, self.iK, self.jK, self.mesh['ndof'])

        # add springs outside tape (constants, no gradient contribution to rho)
        K_np = K.numpy()
        K_np[self.bc['nodeIn'], self.bc['nodeIn']] += 0.1
        K_np[self.bc['nodeOut'], self.bc['nodeOut']] += 0.1
        K_spring = wp.array(K_np, dtype=wp.float32, device="cuda")

        u = solveKuf(K_spring, self.bc['force'], self.bc['free'], self.mesh['ndof'])
        v = solveKuf(K_spring, self.bc['forceOut'], self.bc['free'], self.mesh['ndof'])
        u_np = u.numpy()
        v_np = v.numpy()

        methodType = self.bc['methodType']
        if methodType == 'uOut':
            J = float(u_np[self.bc['nodeOut']])
            K_grad = np.outer(v_np, u_np).astype(np.float32)

        elif methodType == 'MSE_SE':
            MSE = np.dot(v_np, self.bc['force'])
            SE = np.dot(u_np, self.bc['force'])
            J = -MSE / SE
            K_grad = (np.outer(u_np, v_np) / SE - (MSE / SE**2) * np.outer(u_np, u_np)).astype(np.float32)

        elif methodType == 'wMSE':
            w = 0.9
            MSE = np.dot(v_np, self.bc['force'])
            SE = np.dot(u_np, self.bc['force'])
            J = -w * MSE + (1 - w) * SE
            K_grad = (w * np.outer(u_np, v_np) - (1 - w) * np.outer(u_np, u_np)).astype(np.float32)

        K.grad = wp.array(K_grad, dtype=wp.float32, device="cuda")
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
