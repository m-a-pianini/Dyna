from typing import Callable, Tuple, Iterable
import numpy as np
from maps import iterate_map, standard_map


def poincare_sos(traj: np.ndarray, section_index: int = 0, tol: float = 1e-6) -> np.ndarray:
    """Extract points near a Poincaré section defined by a coordinate index (zero crossing not implemented).

    This simple helper collects states where coordinate at section_index is near zero (within tol).
    """
    return traj[np.abs(traj[:, section_index]) <= tol]

# TODO: trajectory 2d scatter plot


def mLCE_map(map_func: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, N: int, delta0: float = 1e-8) -> float:
    """Estimate maximal Lyapunov exponent for a discrete map using Benettin's algorithm (Benettin et al. 1980).
    Returns the estimated exponent (1 / iteration units).

    The Standard method solves the problem of 

    """
    x0 = np.asarray(x0)
    # create a small orthogonal perturbation
    dim = x0.size
    # random unit vector
    v = np.random.randn(dim)
    v /= np.linalg.norm(v)
    x = x0.copy()
    y = x0 + delta0 * v
    s = 0.0
    for i in range(N):
        x = np.asarray(map_func(x))
        y = np.asarray(map_func(y))
        diff = y - x
        dist = np.linalg.norm(diff)
        if dist == 0:
            return -np.inf
        s += np.log(dist / delta0)
        # renormalize perturbation
        diff = (delta0 / dist) * diff
        y = x + diff
    return s / N

# TODO: fix this: add internal ivp solver, 
def mLCE_flow(f: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t: np.ndarray, delta0: float = 1e-9) -> float:
    """Estimate maximal Lyapunov exponent for a flow by finite-difference shadowing (approx).
    Integrates two nearby trajectories with RK4 and applies Benettin renormalization at each time step.

    """
    t = np.asarray(t)
    dt = t[1] - t[0]
    y = np.asarray(y0)
    dim = y0.size
    v = np.random.randn(dim)
    v /= np.linalg.norm(v)
    # Step 0: choose a starting point and create a close vector
    z = y + delta0 * v
    s = 0.0
    for i in range(len(t) - 1):
        def step_state(state):
            # RK4
            k1 = f(t[i], state)
            k2 = f(t[i] + dt / 2, state + dt * k1 / 2)
            k3 = f(t[i] + dt / 2, state + dt * k2 / 2)
            k4 = f(t[i] + dt, state + dt * k3)
            return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
        # Step 1: evolve one step each
        y = step_state(y)
        z = step_state(z)
        diff = z - y
        dist = np.linalg.norm(diff)
        if dist == 0:
            return -np.inf
        # Step 2: add the log difference of the evolution process divided by the initial difference = the expansion in one iteration
        s += np.log(dist / delta0)
        # Step 3: set the second point as the first (evolved) + the difference vector of norm delta0
        diff = (delta0 / dist) * diff
        z = y + diff
    # The result is a sum of the logs of all these expansions
    return s / (t[-1] - t[0])


if __name__ == '__main__':
    # small demo: standard map (Chirikov standard map)
    import argparse
    import matplotlib.pyplot as plt
    import itertools as it
    from scipy.integrate import solve_ivp

    # Parser nonsense
    parser = argparse.ArgumentParser(description='Integrator demo: maps and Hamiltonians')
    parser.add_argument('--demo', choices=['standard_map', 'pendulum'], default='standard_map')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--k', type=float, default=0.971635)
    args = parser.parse_args()

    if args.demo == 'standard_map':
        k = args.k + 0.001
        dynamic = lambda x: standard_map(x, k)
    
        init_theta = np.concat([np.linspace(1.2, 1.7, 10), np.linspace(0, 6, 5)])
        init_p = np.concat([np.linspace(2.5, 3, 10), np.linspace(0, 6, 5)])
        init_values = np.array(list(it.product(init_theta, init_p)))
        #init_values = np.array([np.array([np.pi]*100), np.linspace(0, 2*np.pi, 100)]).transpose()

        lyaps = []
        plt.figure(figsize=(6, 5))

        for i, init in enumerate(init_values):
            #init = np.array([0, init_values[i]])
            traj = iterate_map(dynamic, init, args.iters)
            sim = iterate_map(dynamic, np.array([2*np.pi, 2*np.pi]) - init, args.iters)
            plt.scatter(traj[::1, 0], traj[::1, 1], s=0.5)
            plt.scatter(sim[::1, 0], sim[::1, 1], s=0.5)
            lyaps.append(mLCE_map(dynamic, init, 2000))
            print(f'Estimated mLCE (map) for initial condition {init}: {lyaps[i]:.10f}')

        plt.xlabel('theta')
        plt.ylabel('p')
        plt.title(f'Standard map k={k} ({args.iters} iter)')
        plt.tight_layout()
        plt.show()


    elif args.demo == 'pendulum':
        # Simple pendulum with H = p^2/2 - cos(theta)
        def pendulum_flow(t: np.ndarray, z: np.ndarray, m=1, g=9.81, L=1) -> np.ndarray:
            theta, p = z

            return [p, -(m*g/L)*np.sin(theta)]

        q0 = 1.0
        p0 = 0.0
        t_bounds = (0, 100)
        sol = solve_ivp(pendulum_flow, t_bounds, y0 = [q0, p0],
                        method="RK45",
                        rtol=1e-9,
                        atol=1e-12
                        )
        qs, ps = sol.y
        plt.figure()
        plt.plot(qs[:] % (2 * np.pi) - np.pi, ps[:], linewidth=0.5)
        plt.xlabel('theta')
        plt.ylabel('p')
        plt.title('Pendulum phase portrait')
        plt.show()

        def f_flow(t_, y):
            q = y[0:1]
            p = y[1:2]
            dq = p
            dp = -np.sin(q)
            return np.concatenate([dq, dp])

        y0 = np.concatenate([q0, p0])
        t_short = np.linspace(0, 50, 2001)
        lyap_flow = mLCE_flow(f_flow, y0, t_short)
        print('Estimated maximal Lyapunov exponent (flow, approx):', lyap_flow)
