import jax
import jax.numpy as jnp
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

from dyna.dynsys import VarSpec, DynamicalSystem, connect


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


N, driven = 5, {0, 2, 4}   # osc0, osc2, osc4 get an oscillating drive; osc1, osc3 don't
oscillators = [make_oscillator(f"osc{i}", drive_amp=1.0 if i in driven else 0.0) for i in range(N)]

edges = [(f"osc{i}", "pos", f"osc{(i+1)%N}", "coupling_in") for i in range(N)]  # each feeds the next with loop
chain = connect(oscillators, edges, name="chain")

print(chain)                                  # state_size=10, only osc0's coupling_in stays free
print([v.name for v in chain.input_vars])     # ['osc0.coupling_in']

x0 = jnp.zeros((chain.state_size,))
dx = chain(x0, jnp.zeros((chain.input_size,)), chain.default_params, t=0)
print(dx)

osc_jittedfn = jax.jit(chain)
dih = {"osc0": {"drive_amp": 0,}} # This get modified, while the other subs keep using their default parameters
print(osc_jittedfn(x=x0, p=dih, t=0.5))
