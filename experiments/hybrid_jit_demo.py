"""
experiments/hybrid_jit_demo.py

Test/example for the jit-compatible hybrid integrators
(`integrate_hybrid_jit` / `spiking_integrator_jit` in
dyna/integrators/integrators.py), on the same spiking-neuron +
discretely-updated-synapse composite used in hybrid_demo.py.

Run directly (`python experiments/hybrid_jit_demo.py`): each section prints
what it checks and asserts the result, so a clean run with no
AssertionError is a passing "test", while the printed values double as a
usage example.

Covers:
  1. spiking_integrator_jit vs. the eager spiking_integrator agree exactly
     (same spike times, same final state).
  2. Wrapping the whole thing in `jax.jit` (with the documented static args).
  3. `max_events` as a hard, static bound: too small -> correctly truncates,
     and `n_events == max_events` flags it.
  4. `jax.vmap` over a batch of initial conditions / drive currents -- the
     actual payoff of having a jit-compatible version: simulate many
     neurons at once instead of a Python loop over `spiking_integrator`.
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
import optimistix as optx
import matplotlib.pyplot as plt

from dyna.dynsys import VarSpec, DynamicalSystem, connect
from dyna.integrators.integrators import spiking_integrator, spiking_integrator_jit, spiking_integrator_jit_trajectory


# ---- the same LIF-style neuron + discrete synapse weight as hybrid_demo.py ----
def driven_neuron_fn(x, u, p, t):
    v = x[0]
    return jnp.atleast_1d(p["I"] - v / p["tau"])   # charges toward I*tau

neuron = DynamicalSystem(
    "neuron", [VarSpec("v")], driven_neuron_fn, outputs=["v"],
    params={"tau": 1.0, "I": 2.0}, domain="continuous",
)

def synapse_fn(x, u, p, t):
    return jnp.atleast_1d(x[0] + p["dw"])   # bumped by a fixed increment per spike

synapse = DynamicalSystem(
    "synapse", [VarSpec("w")], synapse_fn, outputs=["w"],
    params={"dw": 0.1}, domain="discrete",
)

net = connect([neuron, synapse], edges=[], name="net")   # synapse only sees the spike via the jump
print(net)

y0 = jnp.array([0.0, 0.0])   # [v, w]
T0, T1, DT0 = 0.0, 5.001, 0.01
VOLTAGE_IDX = [0]
THRESHOLD, V_RESET = 1.0, 0.0


# ---------------------------------------------------------------------
# 1) spiking_integrator_jit matches the eager spiking_integrator exactly
# ---------------------------------------------------------------------
sol_eager = spiking_integrator(net, VOLTAGE_IDX, THRESHOLD, V_RESET,
                                y0, net.default_params, T0, T1, DT0, update_synapses=True)
n_eager = len(sol_eager.event_times)
print(f"\n[1] eager: {n_eager} spikes, event_times[:5]={sol_eager.event_times[:5]}")
print(sol_eager.event_times, sol_eager.ys)
plt.plot(sol_eager.ys.transpose()[0])
plt.show()

MAX_EVENTS = 2000
t_final, y_final, event_times, event_states, n_events = spiking_integrator_jit(
    net, VOLTAGE_IDX, THRESHOLD, V_RESET,
    y0, net.default_params, T0, T1, DT0, max_events=MAX_EVENTS, diffeqsolve_kwargs={"saveat": dfx.SaveAt(t1=True)}
)
print(f"[1] jit-style: {int(n_events)} spikes, event_times[:5]={np.asarray(event_times)[:5]}")
print(event_times[:n_events], event_states[:n_events])

ys, ts, n_events, event_times, event_states, sol_idx = spiking_integrator_jit_trajectory(
    net, VOLTAGE_IDX, THRESHOLD, V_RESET,
    y0, net.default_params, T0, T1, DT0, n_intervals=MAX_EVENTS
)
print(f"[1] jit-style-trajectory: {int(n_events)} spikes, event_times[:5]={np.asarray(event_times)[:5]}")
print(event_times, event_states, ys, ts)
plt.plot(ys.transpose()[0])
plt.show()

assert int(n_events) == n_eager
assert np.allclose(np.asarray(event_times)[:n_eager], sol_eager.event_times, atol=1e-6)
assert np.all(np.isnan(np.asarray(event_times)[n_eager:]))     # NaN padding past the real count
assert np.allclose(np.asarray(y_final), sol_eager.ys[-1], atol=1e-6)
print("[1] PASS: jit-style output matches eager reference exactly")


# ---------------------------------------------------------------------
# 2) actually jit-compiled (max_events/solver marked static, as documented)
# ---------------------------------------------------------------------
jitted = jax.jit(
    functools.partial(spiking_integrator_jit,
                       t0=T0, t1=T1, dt0=DT0, max_events=MAX_EVENTS),
    static_argnames=("max_events",),
)
t_final_j, y_final_j, event_times_j, event_states_j, n_events_j = jitted(
    net, VOLTAGE_IDX, THRESHOLD, V_RESET, y0, net.default_params,
)
assert int(n_events_j) == n_eager
assert np.allclose(np.asarray(y_final_j), sol_eager.ys[-1], atol=1e-6)
print("[2] PASS: identical result under an actual jax.jit wrapper")


# ---------------------------------------------------------------------
# 3) max_events as a hard bound: too small -> truncates, n_events flags it
# ---------------------------------------------------------------------
SMALL_BOUND = 3
_, _, event_times_small, _, n_events_small = spiking_integrator_jit(
    net, VOLTAGE_IDX, THRESHOLD, V_RESET,
    y0, net.default_params, T0, T1, DT0, max_events=SMALL_BOUND,
)
assert int(n_events_small) == SMALL_BOUND
assert np.allclose(np.asarray(event_times_small), sol_eager.event_times[:SMALL_BOUND], atol=1e-6)
print(f"[3] PASS: max_events={SMALL_BOUND} correctly truncates "
      f"(n_events == max_events == {int(n_events_small)}, matches the first "
      f"{SMALL_BOUND} true spike times)")


# ---------------------------------------------------------------------
# 4) vmap over a batch of initial conditions / drive currents
# ---------------------------------------------------------------------
batch_size = 6
I_values = jnp.linspace(1.2, 3.0, batch_size)   # different drive current per neuron
y0_batch = jnp.tile(y0, (batch_size, 1))

def run_one(I_value, y0_i):
    params_i = dict(net.default_params)
    params_i["neuron"] = dict(params_i["neuron"])
    params_i["neuron"]["I"] = I_value
    return spiking_integrator_jit(
        net, VOLTAGE_IDX, THRESHOLD, V_RESET,
        y0_i, params_i, T0, T1, DT0, max_events=MAX_EVENTS,
    )

batched = jax.vmap(run_one, in_axes=(0, 0))(I_values, y0_batch)
t_final_b, y_final_b, event_times_b, event_states_b, n_events_b = batched

print("\n[4] batch results (drive current -> spike count, final synapse weight):")
for i in range(batch_size):
    print(f"    I={float(I_values[i]):.2f}  n_spikes={int(n_events_b[i])}  "
          f"w_final={float(y_final_b[i][1]):.3f}")

# higher drive current -> fires more often in the same window -> more spikes
assert np.all(np.diff(np.asarray(n_events_b)) >= 0)
# synapse weight is always exactly 0.1 * (number of spikes for that neuron)
assert np.allclose(np.asarray(y_final_b)[:, 1], 0.1 * np.asarray(n_events_b), atol=1e-6)
print("[4] PASS: vmap batches cleanly over initial current, spike counts "
      "increase monotonically with drive, synapse weights match spike counts")

print("\nALL CHECKS PASSED")
