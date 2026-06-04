# Material models: maps the density vector rho -> stiffness vector E.
#
# simp(rho, Emin, Emax, penal) — Solid Isotropic Material with Penalization:
#   E = Emin + (Emax - Emin) * rho^penal
#   Penalization (penal=3) pushes rho toward 0 or 1 during optimization.
#
# ramp(rho, Emax, S) — RAMP model, an alternative to SIMP:
#   E = Emax * rho / (1 + S*(1 - rho))
#
# Both are elementwise operations over the full density vector.
# Implemented as Warp kernels so they run on GPU and participate in
# the Warp autodiff tape (gradients flow back to rho automatically).

import warp as wp

@wp.kernel
def simp_kernel(rho: wp.array[wp.float32],
                E: wp.array[wp.float32],
                Emin: float,
                Emax: float,
                penal: float):
    i = wp.tid()
    
    value = Emin + (Emax - Emin) * wp.pow(rho[i] + 0.01, penal)
    E[i] = value
 
def simp(rho, Emin, Emax, penal):
    E = wp.zeros(rho.shape[0], dtype=wp.float32, device=rho.device, requires_grad=True)
    wp.launch(kernel=simp_kernel, dim=rho.shape[0], inputs=[rho, E, Emin, Emax, penal], device="cuda")

    return E

@wp.kernel
def ramp_kernel(rho: wp.array[wp.float32],
                E: wp.array[wp.float32],
                Emax: float,
                S: float):
    i = wp.tid()

    value = Emax * rho[i] / (1.0 + S * (1.0 - rho[i]))
    E[i] = value

    
def ramp(rho, Emax, S): 
    E = wp.zeros(rho.shape[0], dtype=wp.float32, device=rho.device, requires_grad=True)
    wp.launch(kernel = ramp_kernel, dim=rho.shape[0], inputs=[rho, E, Emax, S], device="cuda")

    return E
