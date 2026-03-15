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
from src.lyapunov import flow_lyapunov_spectrum

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
X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
U, V = samelson_flow(t=0, z=(X, Y), params=pars)

stream_plot(X, Y, U, V, density=5)

# Choose the parameters of the simulation
# Initial conditions, Total T, dt, total size, dx, ...

z0 = jnp.array([-1.9546, -2.1656])
Tot_T = 3800
dt = 0.1

# Trajectory solution & visualization
solver = dfx.Dopri5()
    
term = dfx.ODETerm(rhs)

# 5: savepoints (Important: this are the instants at which the solution is recorded)
saveat = dfx.SaveAt(ts=jnp.linspace(0, Tot_T, 5000))

first = dfx.diffeqsolve(
    term,
    solver,
    t0=0,
    t1=Tot_T,
    dt0=dt,
    y0=z0,
    saveat=saveat,
    args=pars,
    max_steps=120000
).ys.transpose()

trajectory_plot(first[0], first[1])

# Calculate lyapunov spectrum
steps = 1
N_int = 3000
lyap = flow_lyapunov_spectrum(flow=rhs, solver=dfx.Dopri5(), z0=z0, params=pars,
                                   dt=dt, interval=steps*dt, n_intervals=N_int)
print(f'Estimated mLCE (map) for initial condition {z0}: {lyap}')

# TODO: fractal dimension
