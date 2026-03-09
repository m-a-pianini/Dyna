import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import sys
from os.path import dirname
sys.path.insert(1, dirname(dirname(__file__)))

from src.flows import samelson_flow, trajectory_plot, stream_plot
from src.lyapunov import mLCE_map

# Choose the dynamic system to be simulated
# Along with its parameters
pars = {
    "A0": 1.064,
    "C": 0.25,
    "L": 2,
    "h": 0.007,
    "wf": 0.058,
}

rhs = lambda t, z: samelson_flow(t, z, pars)

# Flow visualization
X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
U, V = samelson_flow(t=0, z=(X, Y), params=pars)

stream_plot(X, Y, U, V, density=5)

# Choose the parameters of the simulation
# Initial conditions, Total T, dt, total size, dx, ...

z0 = [-1.9546, -2.1656]
Tot_T = 180
dt = 0.1

# Trajectory solution & visualization
first = solve_ivp(
    rhs,
    t_span=(0, Tot_T),
    y0=z0,
    method="Radau",
    rtol=1e-9,
    atol=1e-12
).y

trajectory_plot(first[0], first[1])


dt = Tot_T/(len(first[0]) + 1)

# PROBLEM: the solve_ivp gives me array of different size
# SOLUTION: just take the second element of the array as output (it always exist as the algorithm does at least one step)
# It works, but it's inefficient obviously
# OTHER SOLUTION: write function that uses the whole trajectory (which is the same that would eventually be calculated), it halves computations

dynamic = lambda z: np.array([i[1] for i in solve_ivp(
    rhs,
    t_span=(0, dt),
    y0=z,
    method="Radau",
    rtol=1e-9,
    atol=1e-12
).y])

# Calculate mLCE
iters = int(len(first[0]))

lyap = mLCE_map(dynamic, z0, iters)
print(f'Estimated mLCE (map) for initial condition {z0}: {lyap:.10f}')
