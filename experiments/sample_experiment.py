import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax as dfx

import sys
from os.path import dirname
sys.path.insert(1, dirname(dirname(__file__)))

from dyna.flows import samelson_flow, trajectory_plot, stream_plot
from dyna.lyapunov import *

jax.config.update("jax_enable_x64", True)
FIG_PATH = dirname(dirname(__file__)) + "/figs/"

# Choose the dynamic system to be simulated
# Along with its parameters
pars = {
    "A0": 0.5,
    "C": 0.25,
    "L": 2.0,
    "h": 0,
    "wf": 0,
}

rhs = lambda t, z, args: samelson_flow(t, z, args)

# Phase portrait (assuming 2d system)

X, Y = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 400), np.linspace(-5, 5, 400))
U, V = samelson_flow(t=0, z=(X, Y), params=pars)
stream_plot(X, Y, U, V, density=5)

# Parameters for integration
# Initial condition
z0 = jnp.array([-np.pi/2, 0])

dt = 0.001
n_inters = 2e4
steps = 500
burns = 0.2

term = dfx.ODETerm(rhs)

solver = dfx.Kvaerno5()
stepsc = dfx.PIDController(rtol=1e-8, atol=1e-12)

timesteps = jnp.linspace(0, steps*dt, 60)

# Lyapunov specter calculation
traject, cums, times = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=pars, save_at=timesteps,
                                   dt=dt, interval=steps*dt, n_intervals=n_inters, stepsize=stepsc, burn_in=int(n_inters*burns))

# Plotting (assuming 2d)
first = traject.transpose()
trajectory_plot(first[0], first[1], save=FIG_PATH + "Sam_Path_" + str(pars["h"]) + "_" + str(pars["wf"]) + ".png")

# Poincaré surface of section (time-wise)
period = 1
poinc_t, idx = poincare_sos(times, 0, (timesteps[1] - timesteps[0] - timesteps[1]*1e-3)/2, period, 0)
poinc_x = first[0, idx]
poinc_y = first[1, idx]

ax, first_w = plot_wrapped(poinc_x, poinc_y, kind="scatter", s=3)

