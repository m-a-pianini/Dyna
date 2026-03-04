"""Integrator utilities for maps and Hamiltonian systems.

Features:
- RK4 integrator for general ODEs
- Symplectic Leapfrog (velocity Verlet) for separable Hamiltonians H(q,p)=T(p)+V(q)
- Map iteration utilities
- Poincaré / phase plotting helpers
- Maximal Lyapunov exponent estimation (Benettin) for maps and flows (finite-difference tangent)

Usage: import functions below or run as script for small demos.
"""

from typing import Callable, Tuple, Iterable
import numpy as np


def integrate_rk4(f: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Classic RK4 integrator for y' = f(t,y).

    Args:
        f: function f(t,y) -> ydot
        y0: initial state vector
        t: 1D array of times (must be increasing)

    Returns:
        ys: array shape (len(t), len(y0))
    """
    t = np.asarray(t)
    y0 = np.asarray(y0)
    ys = np.zeros((len(t), y0.size))
    ys[0] = y0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        ti = t[i]
        yi = ys[i]
        k1 = f(ti, yi)
        k2 = f(ti + dt / 2, yi + dt * k1 / 2)
        k3 = f(ti + dt / 2, yi + dt * k2 / 2)
        k4 = f(ti + dt, yi + dt * k3)
        ys[i + 1] = yi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return ys


def leapfrog_step(q: np.ndarray, p: np.ndarray, dt: float, dVdq: Callable[[np.ndarray], np.ndarray], dTdp: Callable[[np.ndarray], np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Single step of velocity-Verlet / leapfrog for separable Hamiltonian H(q,p)=T(p)+V(q).

    Args:
        q, p: coordinates and momenta (arrays)
        dt: timestep
        dVdq: gradient of potential V wrt q
        dTdp: gradient of kinetic T wrt p; if None assume T(p)=p^2/2 -> dTdp=p

    Returns:
        (q_next, p_next)
    """
    if dTdp is None:
        dTdp = lambda p: p
    p_half = p - 0.5 * dt * dVdq(q)
    q_next = q + dt * dTdp(p_half)
    p_next = p_half - 0.5 * dt * dVdq(q_next)
    return q_next, p_next


def integrate_leapfrog(q0: np.ndarray, p0: np.ndarray, t: np.ndarray, dVdq: Callable[[np.ndarray], np.ndarray], dTdp: Callable[[np.ndarray], np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate separable Hamiltonian with leapfrog over times t (1D array).

    Returns arrays qs, ps shaped (len(t), dim)
    """
    qs = np.zeros((len(t), q0.size))
    ps = np.zeros((len(t), p0.size))
    qs[0] = q0
    ps[0] = p0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        qn, pn = leapfrog_step(qs[i], ps[i], dt, dVdq, dTdp)
        qs[i + 1] = qn
        ps[i + 1] = pn
    return qs, ps


def iterate_map(map_func: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, N: int) -> np.ndarray:
    """Iterate a discrete map x_{n+1} = F(x_n) N times.

    Returns array of shape (N+1, dim) including x0.
    """
    x0 = np.asarray(x0)
    traj = np.zeros((N + 1, x0.size))
    traj[0] = x0
    x = x0.copy()
    for i in range(1, N + 1):
        x = np.asarray(map_func(x))
        traj[i] = x
    return traj


def poincare_points(traj: np.ndarray, section_index: int = 0, tol: float = 1e-6) -> np.ndarray:
    """Extract points near a Poincaré section defined by a coordinate index (zero crossing not implemented).

    This simple helper collects states where coordinate at section_index is near zero (within tol).
    """
    return traj[np.abs(traj[:, section_index]) <= tol]


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

    parser = argparse.ArgumentParser(description='Integrator demo: maps and Hamiltonians')
    parser.add_argument('--demo', choices=['standard_map', 'pendulum'], default='standard_map')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--k', type=float, default=0.971635)
    args = parser.parse_args()

    if args.demo == 'standard_map':
        k = args.k + 0.001

        def standard_map(x: np.ndarray) -> np.ndarray:
            # x = [theta, p]
            theta, p = x
            p_new = (p + k * np.sin(theta)) % (2 * np.pi)
            theta_new = (theta + p_new) % (2 * np.pi)
            return np.array([theta_new, p_new])

        init_theta = np.concat([np.linspace(1.2, 1.7, 10), np.linspace(0, 6, 5)])
        init_p = np.concat([np.linspace(2.5, 3, 10), np.linspace(0, 6, 5)])
        init_values = np.array(list(it.product(init_theta, init_p)))
        #init_values = np.array([np.array([np.pi]*100), np.linspace(0, 2*np.pi, 100)]).transpose()

        lyaps = []
        plt.figure(figsize=(6, 5))

        for i, init in enumerate(init_values):
            #init = np.array([0, init_values[i]])
            traj = iterate_map(standard_map, init, args.iters)
            sim = iterate_map(standard_map, np.array([2*np.pi, 2*np.pi]) - init, args.iters)
            plt.scatter(traj[::1, 0], traj[::1, 1], s=0.5)
            plt.scatter(sim[::1, 0], sim[::1, 1], s=0.5)
            lyaps.append(mLCE_map(standard_map, init, 2000))
            print(f'Estimated mLCE (map) for initial condition {init}: {lyaps[i]:.10f}')

        plt.xlabel('theta')
        plt.ylabel('p')
        plt.title(f'Standard map k={k} ({args.iters} iter)')
        plt.tight_layout()
        plt.show()


    elif args.demo == 'pendulum':
        # Simple pendulum with H = p^2/2 - cos(q)
        def dVdq(q: np.ndarray) -> np.ndarray:
            return np.sin(q)

        q0 = np.array([1.0])
        p0 = np.array([0.0])
        t = np.linspace(0, 100, 5001)
        qs, ps = integrate_leapfrog(q0, p0, t, dVdq)
        plt.figure()
        plt.plot(qs[:, 0] % (2 * np.pi), ps[:, 0], linewidth=0.5)
        plt.xlabel('q (mod 2pi)')
        plt.ylabel('p')
        plt.title('Pendulum phase portrait (leapfrog)')
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
