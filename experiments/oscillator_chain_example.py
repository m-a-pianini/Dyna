import jax
import jax.numpy as jnp
import diffrax as dfx
import matplotlib.pyplot as plt
from time import time

from dyna.dynsys import VarSpec, DynamicalSystem, connect, make_clock
from dyna.lyapunov import *


def make_oscillator(name, drive_amp=0.0, drive_omega=1.0, k=1.0, c=0.2, m=1.0):
    def fn(x, u=jnp.zeros(1), params = {}, t=0):
        pos, vel = x[0], x[1]
        force = params.get("drive_amp", drive_amp) * jnp.sin(params.get("drive_omega", drive_omega) * t)   # coupling + optional forcing
        dpos = vel
        dvel = (-params.get("k", k) * (pos - u[0]) - params.get("c", c) * vel + force) / params.get("m", m)
        return jnp.stack([dpos, dvel])

    return DynamicalSystem(
        name, [VarSpec("pos"), VarSpec("vel")], fn,
        input_vars=[VarSpec("coupling_in")], outputs=["pos"],
        params={"k": k, "c": c, "m": m, "drive_amp": drive_amp, "drive_omega": drive_omega},
        domain="continuous",
    )

#Number of oscillators
N = 5

 # osc0, osc2, osc4 get an oscillating drive
driven = {0, 2, 4}

oscillators = [make_oscillator(f"osc{i}", drive_amp=1.0 if i in driven else 0.0) for i in range(N)]

edges = [(f"osc{i}", "pos", f"osc{(i+1)}", "coupling_in") for i in range(N-1)]  # each feeds the next with loop
chain = connect(oscillators, edges, name="chain")


# NOTE: lists in python behave in this way
# If we use the same list to build two different composite systems, they will share the EXACT same list object
# This leads to bug (see below)
chain2 = connect(oscillators, edges, name="chain2")
print(chain2.subsystems == chain.subsystems)

# Check everything is as expected
print(chain)                                  # state_size=10
print([v.name for v in chain.input_vars])     # empty because loop connection

# Compose the composite system
# Notice the difference: outputs attribute is a list of strings, while input_vars is a list of Vars
print((chain2.name, chain2.outputs[0], chain.name, chain.input_vars[0].name))
full_links = [(chain2.name, chain2.outputs[0], chain.name, chain.input_vars[0].name),
              (chain.name, chain.outputs[0], chain2.name, chain2.input_vars[0].name)]
full_chain = connect([chain, chain2], full_links, name="full_chain")
print(full_chain)
print(full_chain.flatten_params())

# Now let's change a single subsystem's param:
pars = full_chain.flatten_params()
# BUG: this updates both chain and chain2 osc4, but only if the chain and chain2 objects are
# created with the same list.
# It stays identical down to EVERY single object, even when storing the attribute value in an external variable
# It is not (arguably) an interface problem, it's a Python problem
# Could be fixed by making deep copies of all the init variables of the single objects
pars["chain2.osc4"]["drive_omega"] = 3
print(pars)

# We now update the params of the object
pars = full_chain.unflatten_params(pars)
full_chain.default_params = pars

z0 = jnp.zeros((full_chain.state_size,))
dz = full_chain(z0, jnp.zeros((full_chain.input_size,)), full_chain.default_params, t=0)
print(dz)

osc_jittedfn = jax.jit(lambda t, x, pars=None: full_chain(x=x, t=t, p=pars))

print(osc_jittedfn(0.5, z0))

# Lets try the autonomize
# NOTE: the lyapunov spectrum estimation can slow down quite a lot 
# This is probably due to the bad scaling in the number of dimensions of the system
clock = make_clock("clock")
canon_chain = full_chain.autonomize(name= "canon_chain", clock=clock)

print(canon_chain)

jitted_canon = jax.jit(lambda t, x, pars=None: canon_chain(x=x, t=t, p=pars))

# ==========================================
# Now let's calculate the lyapunov spectrum:


Tot_T = 200
Tot_iters = 1e5
steps = 50
n_inters = int(Tot_iters/steps) # Unelegantly, this has to be done outside the main function
burns = int(n_inters*0.2)

timesteps = jnp.linspace(0, Tot_T/n_inters, steps-1)

solver = dfx.Kvaerno5()
stepsc = dfx.PIDController(rtol=1e-10, atol=1e-12, pcoeff=0.4, icoeff=0.3)

saveat = dfx.SaveAt(t1=True, ts=timesteps)

# Calculate lyapunov spectrum
now = time()
traject, cums, times = flow_spectrum(flow=jitted_canon, solver=solver, z0=jnp.zeros((canon_chain.state_size,)), params=pars, save_at=timesteps,
                                t1=Tot_T, qr_every=steps, n_intervals=n_inters, stepsize=stepsc, burn_in=burns)
later = time()

print(f"Elapsed time: {later - now:.6f}")
# Plot of the trajectory
first = traject.transpose()
plt.plot(first[0], first[1])
plt.show()

plt.plot(cums)
plt.show()
