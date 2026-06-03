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
