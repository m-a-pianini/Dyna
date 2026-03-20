import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
import diffrax as dfx

import sys
from os.path import dirname
sys.path.insert(1, dirname(dirname(__file__)))

from src.flows import samelson_flow, trajectory_plot, stream_plot
from src.lyapunov import lyapunov_spectrum, kaplan_yorke, poincare_sos

jax.config.update("jax_enable_x64", True)


# Choose the dynamic system to be simulated
# Along with its parameters
pars = {
    "A0": 1.064,
    "C": 0.25,
    "L": 2,
    "h": 0.07,
    "wf": 0.058,
}

rhs = lambda t, z, args: samelson_flow(t, z, args)

# Flow visualization
X, Y = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 400), np.linspace(-5, 5, 400))
U, V = samelson_flow(t=0, z=(X, Y), params=pars)

stream_plot(X, Y, U, V, density=5)

# Choose the parameters of the simulation
# Initial conditions, Total T, dt, total size, dx, ...

# Trajectory in the chaos
z0 = jnp.array([-1.9546, -2.1656])
Tot_T = 3800
timesteps = np.linspace(0, Tot_T, 5000)
dt = 0.01

# Trajectory solution & visualization
solver = dfx.Dopri5()
term = dfx.ODETerm(rhs)
saveat = dfx.SaveAt(ts=timesteps)

first = dfx.diffeqsolve(
    term,
    solver,
    t0=0,
    t1=Tot_T,
    dt0=dt,
    y0=z0,
    saveat=saveat,
    args=pars,
    max_steps=1200000
).ys.transpose()

# Plot of the trajectory
trajectory_plot(first[0], first[1])

# Wrapped plot
wrapped = (np.array(first[0]) + np.pi)% 2* np.pi - np.pi
jump_idx = np.where(np.abs(np.diff(wrapped)) > np.pi)[0]

xw_plot = wrapped.copy()
xw_plot[jump_idx + 1] = np.nan

x_segments = np.split(xw_plot, jump_idx + 1)
y_segments = np.split(np.array(first[1]), jump_idx + 1)

for ys, xs in zip(y_segments, x_segments):
    plt.plot(xs, ys)
plt.show()

# Poincaré surface of section
# I would need a func that:
# Input = ndarray, indexes list, criterion (a crossing or modulo op)
times, p_idxs = poincare_sos(timesteps, section_val=0, tol=1e-2)


# Calculate lyapunov spectrum
steps = 100
N_int = 1e4
lyap = lyapunov_spectrum(flow=rhs, solver=dfx.Dopri5(), z0=z0, params=pars,
                                   dt=dt, interval=steps*dt, n_intervals=N_int)
print(f'Estimated mLCE (map) for initial condition {z0}: {lyap}')

# Whomp whomp :(
hdim = kaplan_yorke(lyap)
print(f'Kaplan-Yorke extimate: {hdim}')


# EXPERIMENT PROPER:
# Prep: 
# GRAPH 1: lyapunov exponents in function of number of iteration
# GRAPH 2: K-Y Extimate dimension (in function of lyapunov exponents iterations)
# GRAPH 3: BOX COUNTING PLOT
