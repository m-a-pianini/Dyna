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
    "A0": 1.064,
    "C": 0.25,
    "L": 2,
    "h": 0.07,
    "wf": 0.058,
}

rhs = lambda t, z, args: samelson_flow(t, z, args)


# Integration
Tot_T = 600
dt = 0.001

term = dfx.ODETerm(rhs)

solver = dfx.Kvaerno5()
stepsc = dfx.PIDController(rtol=1e-8, atol=1e-8)

burns = 0.2

z0 = jnp.array([np.pi/2, 0])
pars_list = pars.copy()
pars_list.update({"h": 0})

step_range = jnp.linspace(1, 1576, 26)
steps = 1
n_iters = 2e5

traject, cums = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=pars_list,
                                   dt=dt, interval=steps*dt, n_intervals=n_iters, stepsize=stepsc, burn_in=int(n_iters*burns))

first = traject.transpose()
trajectory_plot(first[0], first[1])

plt.plot(cums)
plt.show()

exit()

maxlyap = []
for step in step_range:
    print(step)
    lyap = jnp.sort(fast_flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=pars_list, dt=dt, 
                                                interval=step*dt, n_intervals=int(Tot_T/(dt*step)), burn_in=int(Tot_T/(dt*step)*burns),
                                           stepsize=stepsc, jacobian=False))[-1]
    maxlyap.append(lyap)

plt.scatter(step_range, np.array(maxlyap))
plt.grid(True)
plt.xlabel("# Iteration"); plt.ylabel("Leading lyap-exp value"); plt.title("Unpertrubed map integration interval variation")
plt.savefig(FIG_PATH + "Interval_variation5.png", dpi=500)
plt.show()
