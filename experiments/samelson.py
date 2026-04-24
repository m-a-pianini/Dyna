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
FIG_PATH = dirname(dirname(__file__)) + "/figs_backflow/"

# Choose the dynamic system to be simulated
# Along with its parameters
pars = {
    "A0": 0.5,
    "C": 0.25,
    "L": 2.0,
    "h": 0.05,
    "wf": 0.2133,
}

unp_pars = pars.copy()
unp_pars.update({"h": 0})

rhs = lambda t, z, args: samelson_flow(t, z, args)

# Flow visualization
X, Y = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 400), np.linspace(-5, 5, 400))
U, V = samelson_flow(t=0, z=(X, Y), params=unp_pars)

stream_plot(X, Y, U, V, density=5)

U, V = samelson_flow(t=0, z=(X, Y), params=pars)

stream_plot(X, Y, U, V, density=5)

# Choose the parameters of the simulation
# Initial conditions, Total T, dt, total size, dx, ...

# Trajectory in the chaos
#z0 = jnp.array([-1.9546, -2.1656])
z0 = jnp.array([-np.pi/2, 0])

# Integration
Tot_T = 6000
dt = 0.001
n_inters = 2e4
steps = 500
burns = 0.2

term = dfx.ODETerm(rhs)

solver = dfx.Kvaerno5()
stepsc = dfx.PIDController(rtol=1e-8, atol=1e-12)

timesteps = jnp.linspace(0, steps*dt, 60)
saveat = dfx.SaveAt(t1=True, ts=timesteps)


# Analysis
boxes = np.logspace(-3, 1, 20)

# Calculate lyapunov spectrum
traject, cums, times = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=pars, save_at=timesteps,
                                   dt=dt, interval=steps*dt, n_intervals=n_inters, stepsize=stepsc, burn_in=int(n_inters*burns))

# Plot of the trajectory (perturbed)
first = traject.transpose()
trajectory_plot(first[0], first[1], save=FIG_PATH + "Sam_Path_" + str(pars["h"]) + "_" + str(pars["wf"]) + ".png")

print(first.shape)
tim = times
trajectory_plot(times, first[0])

# Wrapped plot
ax, first_w = plot_wrapped(first[0], first[1], linewidth=0.65)

BOTH = True
if BOTH:
    traject2, cums2, times2 = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=jnp.array([-np.pi/2, 1]), params=pars, save_at=timesteps,
                                    dt=dt, interval=steps*dt, n_intervals=n_inters, stepsize=stepsc, burn_in=int(n_inters*burns))
    second = traject2.transpose()
    plot_wrapped(second[0], second[1], ax=ax, linewidth=0.65)
    plt.savefig(FIG_PATH + "Sam_Wrapped_path_" + str(pars["h"]) + "_" + str(pars["wf"]) + "_2.png", dpi=1500)
    plt.show()

    period = 2*np.pi/pars["wf"]
    poinc_t, idx = poincare_sos(tim, 0, timesteps[1] - timesteps[0] - 1e-5, period, 0)
    poinc_x = first[0, idx]
    poinc_y = first[1, idx]

    ax, first_w = plot_wrapped(poinc_x, poinc_y, kind="scatter", s=3)

    poinc_t2, idx2 = poincare_sos(times2, 0, timesteps[1] - timesteps[0] - 1e-5, period, 0)
    plot_wrapped(second[0, idx2], second[1, idx2], ax=ax, kind="scatter", s=3)
    plt.savefig(FIG_PATH + "Sam_SOS_" + str(pars["h"]) + "_" + str(pars["wf"]) + "_2.png", dpi=1500)
    plt.show()

    exit()
else:
    plt.savefig(FIG_PATH + "Sam_Wrapped_path_" + str(pars["h"]) + "_" + str(pars["wf"]) + ".png", dpi=1500)
    plt.show()

# Lyapunov extimate over iterations
print(f'Estimated mLCE (map) for initial condition {z0}: {cums[-1]}')
plt.plot(cums)
plt.show()

hdim = kaplan_yorke_dim(cums[-1])
print(f'Kaplan-Yorke extimate: {hdim}')

# Box counting plotting
ax, D = boxcount_plot(trajectory=np.array([(first[0] + np.pi) % (2* np.pi) - np.pi, first[1]]).transpose()[5000: ,:], 
                     box_sizes=boxes)
plt.savefig(FIG_PATH + "Path_boxcount_" + str(pars["h"]) + "_" + str(pars["wf"]) + ".png", dpi=1500)
plt.show()


flurry = jnp.concat([jnp.array([jnp.zeros(5), jnp.linspace(-4, 4, 5)]).T,
                     jnp.array([jnp.linspace(-2*jnp.pi, 0, 5), jnp.full(5, -jnp.pi/2)]).T,
                     jnp.array([jnp.linspace(-jnp.pi, jnp.pi, 5), jnp.full(5, jnp.pi/2)]).T,
                                ])

flurry_compute = make_batch_lyapunov_solver(flow=rhs, solver=solver, dt=dt, stepsize=stepsc, n_intervals=n_inters, burn_in=int(n_inters*burns))
batched_lyap = jax.jit(
    jax.vmap(flurry_compute, in_axes=(0, None, None, None))
)
flurry_traj, flurry_lyaps, _ = batched_lyap(flurry, 0, pars, steps)

exit()

print("="*35 + " UNPERTURBED MAP " + "="*35)

# Unperturbed map for reference
"""unpert = dfx.diffeqsolve(
    term,
    solver,
    t0=0,
    t1=Tot_T,
    dt0=dt,
    y0=z0,
    saveat=saveat,
    args=unp_pars,
    max_steps=1200000,
    stepsize_controller=stepsc,
).ys.transpose()"""

unpert, unpert_cums, _ = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=unp_pars,
                                   dt=dt, interval=steps*dt, n_intervals=Tot_T/(steps*dt), stepsize=stepsc, burn_in=int(Tot_T/(steps*dt)*burns))

# Plot of the trajectory
unpert = unpert.transpose()
trajectory_plot(unpert[0], unpert[1], save=FIG_PATH + "Sam_unpert_Path.png")

# Wrapped plot
ax, unpert_w = plot_wrapped(unpert[0], unpert[1])
plt.savefig(FIG_PATH + "Sam_unpert_Wrapped_path.png", dpi=1500)
plt.show()

# Lyapunov extimate over iterations
print(f'Estimated mLCE (map) for initial condition {z0}: {unpert_cums[-1]}')
plt.plot(unpert_cums)
plt.show()

unpert_hdim = kaplan_yorke_dim(unpert_cums[-1])
print(f'Kaplan-Yorke extimate: {unpert_hdim}')

# Box counting plotting
ax, D = boxcount_plot(trajectory=np.array([(unpert[0] + np.pi) % (2* np.pi) - np.pi, unpert[1]]).transpose()[5000: ,:], 
                     box_sizes=boxes)
plt.savefig(FIG_PATH + "Unpert_Path_boxcount.png", dpi=1500)
plt.show()
