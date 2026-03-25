import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax as dfx

import sys
from os.path import dirname
sys.path.insert(1, dirname(dirname(__file__)))

from src.flows import samelson_flow, trajectory_plot, stream_plot
from src.lyapunov import flow_lyapunov_spectrum, make_batch_lyapunov_solver, kaplan_yorke_dim, boxcount_dimension, poincare_sos

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
timesteps = np.linspace(0, Tot_T, 9000)
dt = 0.01

# Integration
solver = dfx.Kvaerno5()
stepsc = dfx.PIDController(rtol=1e-8, atol=1e-8)
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
#times, p_idxs = poincare_sos(timesteps, section_val=0, tol=1e-2)


# Calculate lyapunov spectrum
steps = 1
N_int = 2e5
burns = 0.3
cums = flow_lyapunov_spectrum(flow=rhs, solver=solver, z0=z0, params=pars,
                                   dt=dt, interval=steps*dt, n_intervals=N_int, stepsize=stepsc, burn_in=int(N_int*burns))
print(f'Estimated mLCE (map) for initial condition {z0}: {cums[-1]}')
plt.plot(cums)
plt.show()
# Whomp whomp :(
hdim = kaplan_yorke_dim(cums[-1])
print(f'Kaplan-Yorke extimate: {hdim}')


# EXPERIMENT PROPER:
# Prep: 
# GRAPH 2: K-Y Extimate dimension (in function of lyapunov exponents iterations)
# GRAPH 3: BOX COUNTING PLOT

flurry = jnp.array([jnp.array([-1.2, -1]), jnp.array([-2, 2]), jnp.array([2, 2]),
                    jnp.array([3, 1]), jnp.array([1, 4]), jnp.array([-3, 3]),
                    jnp.array([-2, -6]), jnp.array([10, -2]), jnp.array([-7, 3]),
                    ])
t0_batch = jnp.zeros(len(flurry))

compute = make_batch_lyapunov_solver(flow=rhs, solver=solver, dt=dt, stepsize=stepsc, n_intervals=N_int, burn_in=int(N_int*burns), jacobian=False)
batched_lyap = jax.jit(
    jax.vmap(compute, in_axes=(0, 0, None, None))
)

cum_lyaps = batched_lyap(flurry, t0_batch, pars, steps*dt)

for cum in cum_lyaps:
    plt.plot(cum)
plt.grid(True)
plt.show()

cum_dims = kaplan_yorke_dim(cum_lyaps)

for cum in cum_dims:
    plt.plot(cum)
plt.grid(True)
plt.show()

D, sizes, counts, i0, i1 = boxcount_dimension(np.array((np.array(first) + np.pi)% 2* np.pi - np.pi).transpose()[5000: ,:])
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
