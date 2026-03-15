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

samelsons_pars = {
    "A0": 1.064,
    "C": 0.2,
    "L": 1.8,
    "h": 0.01,
    "wf": 0.058,
}


if __name__ == "__main__":

    # Test plot to see if the map is correct
    # it is :,)
    X, Y = np.meshgrid(np.linspace(-5, 5, 850), np.linspace(-5, 5, 850))
    U, V = samelson_flow(t=0, z=(X, Y), params=samelsons_pars)
    stream_plot(X, Y, U, V)

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
    