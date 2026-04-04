import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax as dfx
from time import time

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

from src.flows import duffing, trajectory_plot, stream_plot
from src.lyapunov import flow_lyapunov_spectrum, make_batch_lyapunov_solver, kaplan_yorke_dim, boxcount_dimension, poincare_sos

jax.config.update("jax_enable_x64", True)


# Choose the dynamic system to be simulated
# Along with its parameters
pars = {
    "a": 0.1,
    "b": 13.5,
    "w": 1,
}

pars = {
    "c": 1,
    "d": -1,
    "a": 0.2,
    "b": 0.3,
    "w": 1,
}

rhs = duffing

# Flow visualization
X, Y = np.meshgrid(np.linspace(-10, 10, 400), np.linspace(-10, 10, 400))
U, V = duffing(t=0, z=(X, Y), params=pars)

stream_plot(X, Y, U, V, density=5)

# Choose the parameters of the simulation
# Initial conditions, Total T, dt, total size, dx, ...

# Trajectory in the chaos
z0 = jnp.array([1, 1])
Tot_T = 400
timesteps = np.linspace(0, Tot_T, 50000)
dt = 0.001

# Integration
solver = dfx.Dopri5()
stepsc = dfx.ConstantStepSize()
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
    max_steps=1200000,
    stepsize_controller=stepsc,
).ys.transpose()

# Plot of the trajectory
trajectory_plot(first[0], first[1])

# Poincaré surface of section
# I would need a func that:
# Input = ndarray, indexes list, criterion (a crossing or modulo op)
#times, p_idxs = poincare_sos(timesteps, section_val=0, tol=1e-2)


# Calculate lyapunov spectrum
steps = 1
N_iters = 2e5
burns = 0.3
traject, cums = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=pars,
                                   dt=dt, interval=steps*dt, n_intervals=N_iters, stepsize=stepsc, jacobian=False, burn_in=int(N_iters*burns))

print(f'Estimated mLCE (map) for initial condition {z0}: {cums[-1]}')
print(f'Kaplan-Yorke extimate: {kaplan_yorke_dim(cums[-1])}')

plt.plot(cums)
#plt.plot(dims)
plt.grid(True)
plt.show()


flurry = jnp.array([jnp.array([1.2, 1]), jnp.array([1, 1.2]), jnp.array([1, 1]),
                    jnp.array([2, 1]), jnp.array([1, 2]), jnp.array([3, 3]),
                    jnp.array([-2, 1]), jnp.array([10, 2]), jnp.array([-7, 3]),
                    ])
t0_batch = jnp.zeros(len(flurry))

compute = make_batch_lyapunov_solver(flow=rhs, solver=solver, dt=dt, stepsize=stepsc, n_intervals=N_iters, burn_in=int(N_iters*burns), jacobian=False)
batched_lyap = jax.jit(
    jax.vmap(compute, in_axes=(0, 0, None, None))
)

start = time()
trajects,cum_lyaps = batched_lyap(flurry, t0_batch, pars, steps*dt)
end = time()

for cum in cum_lyaps:
    plt.plot(cum)
plt.grid(True)
plt.show()
print(f"Elapsed time: {end - start}")

cum_dims = kaplan_yorke_dim(cum_lyaps)

for cum in cum_dims:
    plt.plot(cum)
plt.grid(True)
plt.show()

D, sizes, counts, i0, i1 = boxcount_dimension(first.transpose()[5000: ,:])
print(f"Box counting extimate: {D}")

log_s = np.log10(1 / sizes)
log_c = np.log10(counts)

s_fit = np.array([log_s[i0], log_s[i1]])
c_fit = np.polyval(np.polyfit(log_s[i0:i1+1], log_c[i0:i1+1], 1), s_fit)

fig, ax = plt.subplots()
ax.loglog(1 / sizes, counts, 'o', label='all scales')
ax.loglog(1 / sizes[i0:i1+1], counts[i0:i1+1], 'o', color='red', label='linear region')

# Convert back from log10 space to data space for the fit line
ax.loglog(10**s_fit, 10**c_fit, '--', label=f'fit D={D:.3f}')

ax.set_xlabel('1/r (inverse box size)')
ax.set_ylabel('N(r) (box count)')
ax.legend(); plt.grid(True)
plt.show()
