import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax as dfx

import sys
from os.path import dirname
sys.path.insert(1, dirname(dirname(__file__)))

from src.flows import samelson_flow, trajectory_plot, stream_plot
from src.lyapunov import *

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
Tot_T = 6000
dt = 0.01

term = dfx.ODETerm(rhs)

solver = dfx.Kvaerno5()
stepsc = dfx.PIDController(rtol=1e-8, atol=1e-12)

steps = 1
N_int = 2e5
burns = 0.3

flurry = jnp.concat([jnp.array([jnp.linspace(-jnp.pi, jnp.pi, 30), jnp.zeros(30)]).T,
                     jnp.array([jnp.linspace(-jnp.pi/2, jnp.pi/2, 10), jnp.full(10, 2.75)]).T,
                     jnp.array([jnp.linspace(-jnp.pi*3/2, -jnp.pi/2, 10), jnp.full(10, -2.75)]).T,
                                ])
t0_batch = jnp.zeros(len(flurry))

fig, ax = plt.subplots()

for i, init in enumerate(flurry):
    print(i, init)
    traject, cums = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=init, params=pars,
                                    dt=dt, interval=steps*dt, n_intervals=N_int, stepsize=stepsc, burn_in=int(N_int*burns))
    plot_wrapped(*traject.T, ax=ax, linewidth=0.75)

plt.grid(True)
plt.savefig(FIG_PATH + "Flurry_sam.png", dpi=1500)
plt.show()


exit()

# Old version
# Build mappable function
flurry_compute = make_batch_lyapunov_solver(flow=rhs, solver=solver, dt=dt, stepsize=stepsc, n_intervals=N_int, burn_in=int(N_int*burns))
batched_lyap = jax.jit(
    jax.vmap(flurry_compute, in_axes=(0, 0, None, None))
)
# Execute
flurry_trajects, flurry_cum_lyaps = batched_lyap(flurry, t0_batch, pars, steps*dt)

# Plot and results
plt.ylim(top=0.1, bottom=-0.1)
for cum in flurry_cum_lyaps:
    plt.plot(cum)
plt.grid(True)
plt.savefig(FIG_PATH + "Flurry_lyaps.png")
plt.show()
lyap_ext = flurry_cum_lyaps.mean(axis=0)[-1, :]
lyap_std = flurry_cum_lyaps.std(axis=0)[-1, :]
print(f"Lyapunov exponents extimate (averaged over random trajectories): {lyap_ext} +- {lyap_std}")

# Trajectories
flurry_trajects = np.transpose(flurry_trajects, axes=(0, -1, -2))
#with plt.style.context('Solarize_Light2'):
fig, ax = plt.subplots()
for traj in flurry_trajects:
    plot_wrapped(*traj, ax=ax, linewidth=0.75)
plt.grid(True)
plt.savefig(FIG_PATH + "Flurry_trajects.png", dpi=1500)
plt.show()
