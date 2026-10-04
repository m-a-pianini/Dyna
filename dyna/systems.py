from dyna.dynsys import VarSpec, DynamicalSystem, connect

def izhikevich_neuron(name="", suffix="", a=None, b=None, pars=None):
    name = name or "izhikevich_neuron"
    name += suffix

    def func(x, u, p: dict, t):
        V, U  = x[0], x[1]
        dV = 0.04*(V**2) + 5*V - U + 140 + jnp.sum(u, axis=0)
        dU = p["a"]*(p["b"]*V - U)
        return jnp.stack([dV, dU])

    pars = pars or {"a": a, "b": b}
    return DynamicalSystem(name=name, state_vars=[VarSpec("V"), VarSpec("U")],
                           fn=func, 
                           input_vars=[VarSpec("I_app"), VarSpec("I_syn")], outputs=["V"],
                           params=pars, domain="continuous")

def izhikevich_threshold():
    pass

def izhikevich_synapse(name="", suffix="", ):
    def fn(x, u, p, t):
        return jnp.atleast_1d(x[0] + p["dw"])
    pass

def izhikevich_spike():
    pass

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import diffrax as dfx
    import matplotlib.pyplot as plt

    from dyna.integrators import spiking_integrator, spiking_integrator_jit, spiking_integrator_jit_trajectory

    par = pars={"a": 2, "b": 3}
    iz_neu = izhikevich_neuron(a=1, b=1, pars=par)
    iz2 = izhikevich_neuron(a=1, b=1, pars=par)
    print(iz2 == iz_neu)
    print(iz_neu(jnp.array([1, 2,]), jnp.array([2, 1])))

    