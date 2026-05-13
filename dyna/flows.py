import numpy as np
import sympy
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

def lorenz_system(t, z, params):

    sigma = params.get('sigma', 16.0)
    rho = params.get('rho', 45.92)
    beta = params.get('beta', 4.0)

    X, Y, Z = z

    dx = sigma*(Y-X)
    dy = X*(rho - Z) - Y
    dz = X*Y - beta*Z

    return jnp.array([dx, dy, dz])

def samelson_phi(t, z, params):
    pass
# TODO: this has to be a symbolic formula to be differentiated and called with subs in the function below
_samelson_phi_poly = sympy.Add()
# TODO: lambdify

def samelson_flow(t, z, params={}):
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
    L = params.get('L', 2.0)

    h = params.get('h', 0.0)
    wf = params.get('wf', 1.0)

    z = jnp.array(z, dtype=jnp.float64)
    x, y = z

    A = A0 + h*jnp.cos(wf*t)
    B = (y - A*jnp.cos(x))/ (L * jnp.sqrt(1+(A*jnp.sin(x))**2))
    phi = -jnp.tanh(B) + C*y

    #phi_x = ((A*jnp.sin(x)* L * jnp.sqrt(1+(A*jnp.sin(x))**2) - (((y - A*jnp.cos(x))*L*((A**2)*jnp.sin(2*x))) / (2* jnp.sqrt(1+(A*jnp.sin(x))**2))) ) / ((1+(A*jnp.sin(x))**2)*(L**2)))  *  ((jnp.tanh(B))**2 - 1)
    phi_y = C - (1 - (jnp.tanh(B))**2) / (L * jnp.sqrt(1+(A*jnp.sin(x))**2))

    phi_x = (jnp.tanh(B)**2 - 1)/L * (A*jnp.sin(x)/(jnp.sqrt(1+(A*jnp.sin(x))**2))
                                        + jnp.sin(2*x) * (A*jnp.cos(x) * (A**2)
                                            - y*(A**2)) / (2*(1+(A*jnp.sin(x))**2)**(3/2)))
    
    return jnp.array([-phi_y, phi_x])

def duffing(t, z, params):
    a = params.get("a", 1)
    b = params.get("b", 1)
    c = params.get("c", 1)
    d = params.get("d", 0)

    w = params.get("w", 2*jnp.pi)

    z = jnp.array(z, dtype=jnp.float64)
    x, y = z
    dx = y
    dy = -a*y -d*x - c*x*x*x + b*jnp.cos(w*t)
    return jnp.array([dx, dy])

# Visualization utils
def trajectory_plot(x, y, save=None):
    plt.figure(figsize=(8,4))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple plot")
    plt.grid(True)
    if save is not None:
        plt.savefig(save)
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
        "A0": 0.5,
        "C": 0.25,
        "L": 2,
        "h": 0*0.3,
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
    z0 = jnp.array([-jnp.pi/2, 0])

    # 3: solver
    solver = dfx.Kvaerno5()
    
    # optional (but not for all solvers): stepsize controller
    stepsc = dfx.PIDController(rtol=1e-8, atol=1e-8)

    # 4: vector field term
    term = dfx.ODETerm(rhs)

    # 5: savepoints (Important: this are the instants at which the solution is recorded)
    saveat = SaveAt(ts=jnp.linspace(0, 300, 3000))

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=300,
        dt0=0.001,
        y0=z0,
        saveat=saveat,
        args=samelsons_pars,
        max_steps=12000000,
        stepsize_controller=stepsc
    )
    print(sol.ys)

    x = sol.ys.transpose()[0]
    y = sol.ys.transpose()[1]
    print(sol.ys.shape, x[0], y[0])

    plt.figure()
    plt.scatter(x, y)
    plt.show()
    #trajectory_plot()
    