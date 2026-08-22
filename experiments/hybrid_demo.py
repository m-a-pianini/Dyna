"""
experiments/hybrid_demo.py

Demonstrates integrating a HYBRID CompositeSystem (mixing continuous and
discrete subsystems) by operator splitting:
  - `.flow(x, u, p, t)`  is a valid ODE vector field for ANY composite --
    zero on the discrete part -- hand it straight to diffrax.
  - `.step(x, u, p, t)`  is a valid discrete update for ANY composite --
    identity on the continuous part -- apply it once per discrete tick.

At each macro time-step of length `dt`:
  1. integrate the continuous part over [t, t+dt] with diffrax, holding the
     discrete part fixed (its `.flow` contribution is exactly zero there);
  2. apply `.step` once, holding the continuous part fixed (its `.step`
     contribution is exactly identity), to update the discrete part.

This requires no per-system special-casing: `.flow`/`.step` are generic on
DynamicalSystem and CompositeSystem, and recurse correctly no matter how
deep/mixed the composition tree is (see dyna/dynsys.py).
"""
import numpy as np
import jax.numpy as jnp
import diffrax as dfx

from dyna.dynsys import VarSpec, DynamicalSystem, connect


# ---- continuous: a damped oscillator, forced by an external input ----
def osc_fn(x, u, p, t):
    pos, vel = x[0], x[1]
    return jnp.stack([vel, (-p["k"] * pos - p["c"] * vel + u[0]) / p["m"]])

osc = DynamicalSystem(
    "osc", [VarSpec("pos"), VarSpec("vel")], osc_fn,
    input_vars=[VarSpec("force")], outputs=["pos"],
    params={"k": 2.0, "c": 0.3, "m": 1.0}, domain="continuous",
)

# ---- discrete: a threshold counter driven by the oscillator's position ----
def counter_fn(x, u, p, t):
    n = x[0]
    crossing = u[0]
    return jnp.atleast_1d(n + jnp.where(crossing > p["thresh"], 1.0, 0.0))

counter = DynamicalSystem(
    "counter", [VarSpec("n")], counter_fn,
    input_vars=[VarSpec("pos_in")], outputs=["n"],
    params={"thresh": 0.5}, domain="discrete",
)

hybrid = connect([osc, counter], edges=[("osc", "pos", "counter", "pos_in")], name="hybrid")
print(hybrid)                          # domain == "hybrid"
print("continuous_mask:", hybrid.continuous_mask)   # [True, True, False]


def hybrid_integrate(composite, x0, params, t0, dt, n_macro_steps, solver=dfx.Tsit5()):
    """Operator-split integration: diffrax for the flow, a plain tick for the step."""
    u = jnp.zeros((composite.input_size,))
    ts, xs = [t0], [x0]
    x, t = x0, t0
    for _ in range(n_macro_steps):
        term = dfx.ODETerm(lambda tt, xx, args: composite.flow(xx, u, args, tt))
        sol = dfx.diffeqsolve(
            term, solver, t0=t, t1=t + dt, dt0=dt / 10, y0=x,
            args=params, saveat=dfx.SaveAt(t1=True),
        )
        x = sol.ys[-1]
        x = composite.step(x, u, params, t + dt)   # discrete tick, continuous part passes through
        t = t + dt
        ts.append(t)
        xs.append(x)
    return np.asarray(ts), np.asarray(xs)


if __name__ == "__main__":
    x0 = jnp.array([1.0, 0.0, 0.0])   # [pos, vel, n]
    ts, xs = hybrid_integrate(hybrid, x0, hybrid.default_params, t0=0.0, dt=0.5, n_macro_steps=20)
    print("t        pos       vel       n")
    for t, (pos, vel, n) in zip(ts, xs):
        print(f"{t:6.2f}  {pos:8.4f}  {vel:8.4f}  {n:4.0f}")
