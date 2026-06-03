# Method of Moving Asymptotes (MMA) optimizer.
# Ported directly from the JAX version (Svanberg 1987).
#
# MMA is a gradient-based optimizer for constrained nonlinear problems.
# At each iteration it receives:
#   - the objective value J and its gradient dJ/drho
#   - the constraint value g and its gradient dg/drho
# and returns an updated rho that satisfies the move limits and constraints.
#
# This file is pure numpy / CPU — no Warp, no GPU.
# The optimizer itself does not need to be differentiated; only the
# forward physics pipeline (material -> assembly -> solver -> objective)
# needs to run on GPU under the Warp tape.
