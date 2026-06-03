# Density filter: smooths the design variable vector rho to prevent
# checkerboard patterns and enforce a minimum length scale.
#
# computeFilter(nelx, nely, rmin) — precomputes the filter weight matrix H
#   and its row sums Hs. Run once at setup time. Pure numpy, CPU only.
#
# applyFilter(H, Hs, rho) — applies the filter: rho_filtered = H @ rho / Hs.
#   Called each iteration before passing rho to the physics pipeline.
#
# applySensitivityFilter(H, Hs, x, dc, dv) — chains the filter into the
#   gradient: modifies dc and dv so MMA sees filtered sensitivities.
#
# No Warp here — pure numpy, runs on CPU.
