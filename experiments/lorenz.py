import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax as dfx

import sys
from os.path import dirname
sys.path.insert(1, dirname(dirname(__file__)))

from src.flows import lorenz_system, trajectory_plot, stream_plot
from src.lyapunov import *

jax.config.update("jax_enable_x64", True)
FIG_PATH = dirname(dirname(__file__)) + "/figs/"

# Choose the dynamic system to be simulated
# Along with its parameters
pars = {
    "sigma": 16.0,
    "beta": 4.0,
    "rho": 45.92,
}


rhs = lambda t, z, args: lorenz_system(t, z, args)

# Choose the parameters of the simulation
# Initial conditions, Total T, dt, total size, dx, ...

# Trajectory in the chaos
z0 = jnp.array([5, 5, 25])

# Integration
Tot_T = 100
dt = 0.0003

term = dfx.ODETerm(rhs)

solver = dfx.Kvaerno5()
stepsc = dfx.PIDController(rtol=1e-8, atol=1e-12)

timesteps = jnp.linspace(0, Tot_T, 15000)

steps = 10
burns = 0.1

# Analysis
boxes = np.logspace(-2, 2, 20)

# Calculate lyapunov spectrum
traject, cums = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=pars, jacobian=False,
                                   dt=dt, interval=steps*dt, n_intervals=int(Tot_T/(steps*dt)), stepsize=stepsc, burn_in=int(Tot_T/(steps*dt)*burns))

xs, ys, zs = traject.transpose()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs, ys, zs)
plt.show()

# Lyapunov extimate over iterations
print(f'Estimated mLCE (map) for initial condition {z0}: {cums[-1]}')
plt.plot(cums)
plt.show()

hdim = kaplan_yorke_dim(cums[-1])
print(f'Kaplan-Yorke extimate: {hdim}')

# Box counting plotting
ax, D = boxcount_plot(trajectory=traject[1000: ,:], 
                     box_sizes=boxes)
plt.show()
