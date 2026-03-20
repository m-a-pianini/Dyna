import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax as dfx
from diffrax import SaveAt


# A flow should have as inputs the starting coordinates (in some space) and parameters
# As output the right hand side of the flow equations
# E.g. hamiltonian flow

def hamiltonian_flow(H):

    gradH = jax.grad(H)

    def flow(t, z, params):

        d = z.shape[0] // 2
        q = z[:d]
        p = z[d:]

        gq, gp = jnp.split(gradH(jnp.concatenate([q, p]), params), 2)

        dq = gp
        dp = -gq

        return jnp.concatenate([dq, dp])

    return flow

def ichikievich(t, z, params):

    a = params.get('a', 1.0)
    b = params.get('b', 0.25)
    c = params.get('c', 2)
    d = params.get('d', 2)
    I = 0

    z = jnp.array(z, dtype=jnp.float64)

    v, u = z

    ushape = u.shape
    vshape = v.shape
    v = v.unravel()
    u = u.unravel()
    if v >= 30:
        v = c
        u = u+d
    dv = 0.04*(c**2) + 5*v + 140 - u + I
    du = a*(b*v - u)

    return jnp.array([du, dv])

def samelson_flow(t, z, params):
    """
    Calculate flow function for Samelson's model of the Bjerknes jet.

    Parameters
    ----------
    state : array-like
        State variables [u, v, T] where u is zonal velocity, 
        v is meridional velocity, T is temperature
    t : float
        Time (may be unused depending on model)
    params : dict
        Model parameters including:
        - beta: Coriolis parameter gradient
        - gamma: Thermal damping coefficient
        - Q: Heat forcing
        - mu: Friction coefficient
    
    Returns
    ----------
    derivatives : array-like
        Time derivatives [du/dt, dv/dt, dT/dt]
    """
    boundaries = params.get("boundaries", "pbc")
    A0 = params.get('A0', 1.0)
    C = params.get('C', 0.25)
    L = params.get('L', 2)

    h = params.get('h', 0)
    wf = params.get('wf', 1)

    z = jnp.array(z, dtype=jnp.float64)
    x, y = z

    A = A0 + h*jnp.cos(wf*t)

    B = (y - A*jnp.cos(x))/ (L * jnp.sqrt(1+(A*jnp.sin(x))**2))
    phi = -jnp.tanh(B) + C*y

    phi_x = ((A*jnp.sin(x)* L*jnp.sqrt(1+(A*jnp.sin(x))**2) - (((y - A*jnp.cos(x))*L*((A**2)*jnp.sin(2*x))) / (2* jnp.sqrt(1+(A*jnp.sin(x))**2))) ) / ((1+(A*jnp.sin(x))**2)*(L**2)))  *  ((jnp.tanh(B))**2 - 1)
    phi_y = C - (1 - (jnp.tanh(B))**2)*(1/(L * jnp.sqrt(1+(A*jnp.sin(x))**2)))
    
    return jnp.array([-phi_y, phi_x])

# Visualization utils
def trajectory_plot(x, y):
    plt.figure(figsize=(8,4))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple plot")
    plt.grid(True)
    plt.show()

def stream_plot(X, Y, U , V, density = 2):
    plt.figure(figsize=(8,4))
    plt.streamplot(x=X, y=Y, u=U, v=V, density=density)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle trajectory in the Bickley jet")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # Test plot to see if the map is correct
    # it is :,)
    samelsons_pars = {
        "A0": 1.064,
        "C": 0.2,
        "L": 1.8,
        "h": 0.01,
        "wf": 0.058,
    }

    ich_pars = {
        "a": 1,
        "b": 1,
        "c" : 1,
        "d": 8,
    }

    X, Y = np.meshgrid(np.linspace(-5, 5, 850), np.linspace(-5, 5, 850))
    U, V = samelson_flow(t=0, z=(X, Y), params=samelsons_pars)
    stream_plot(X, Y, U, V, density=3)

    # Test for the solution finding and display
    # It works

    # Steps to work with diffrax:
    # 0: no type back and forth bewteen numpy/others, all in jax

    # 1: Function (args argument will be passed via diffeqsolve "args")
    rhs = lambda t, z, args: samelson_flow(t, z, args)

    # 2: Initial condition
    z0 = jnp.array([-2, -1])

    # 3: solver
    solver = dfx.Dopri5()
    
    # 4: vector field term
    term = dfx.ODETerm(rhs)

    # 5: savepoints (Important: this are the instants at which the solution is recorded)
    saveat = SaveAt(ts=jnp.linspace(0, 100, 1000))

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=100,
        dt0=0.01,
        y0=z0,
        saveat=saveat,
        args=samelsons_pars,
        max_steps=120000
    )
    print(sol.ys)

    x = sol.ys.transpose()[0]
    y = sol.ys.transpose()[1]
    print(sol.ys.shape, x[0], y[0])

    trajectory_plot(x, y)
    