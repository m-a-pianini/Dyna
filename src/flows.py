import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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

    z = np.array(z, dtype=np.float128)
    x, y = z

    A = A0 + h*np.cos(wf*t)

    B = (y - A*np.cos(x))/ (L * np.sqrt(1+(A*np.sin(x))**2))
    phi = -np.tanh(B) + C*y

    phi_x = ((A*np.sin(x)* L*np.sqrt(1+(A*np.sin(x))**2) - (((y - A*np.cos(x))*L*((A**2)*np.sin(2*x))) / (2* np.sqrt(1+(A*np.sin(x))**2))) ) / ((1+(A*np.sin(x))**2)*(L**2)))  *  ((np.tanh(B))**2 - 1)
    phi_y = C - (1 - (np.tanh(B))**2)*(1/(L * np.sqrt(1+(A*np.sin(x))**2)))
    
    return [-phi_y, phi_x]

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
    X, Y = np.meshgrid(np.linspace(-5, 5, 150), np.linspace(-5, 5, 150))
    U, V = samelson_flow(t=0, z=(X, Y), params=samelsons_pars)
    stream_plot(X, Y, U, V)

    # Test for the solution finding and display
    # It works
    hs = np.linspace(0.001, 0.1, 10)
    wfs = np.linspace(0.056, 0.062, 10)

    rhs = lambda t, z: samelson_flow(t, z, samelsons_pars)

    z0 = [-2, -1]

    sol = solve_ivp(
        rhs,
        t_span=(0, 100),
        y0=z0,
        method="Radau",
        rtol=1e-9,
        atol=1e-12
    )
    
    x = sol.y[0]
    y = sol.y[1]
    print(len(sol.y), len(sol.y[0]), x[0], y[0])

    trajectory_plot(x, y)
    