from typing import Callable, Tuple, Iterable
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx


#u are cute <3
# Visualisation utils
def poincare_sos(traj: np.ndarray, section_index: int = 0, tol: float = 1e-6) -> np.ndarray:
    """Extract points near a Poincaré section defined by a coordinate index (zero crossing not implemented).

    This simple helper collects states where coordinate at section_index is near zero (within tol).
    """
    return traj[np.abs(traj[:, section_index]) <= tol]

# Algorithms for calculating lyapunov exponent(s)
def mLCE_map(map_func: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, N: int, delta0: float = 1e-8) -> np.float64:
    """Estimate maximal Lyapunov exponent for a discrete map using Benettin's algorithm (Benettin et al. 1980).
    We fix their step "s" to 1.
    Returns the estimated exponent (1 / iteration units).

    The Standard method solves the problem of 

    """
    # Step 0: choose a starting point and create a close vector
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
        # Step 1: evolve one step each vector
        x = np.asarray(map_func(x), dtype=np.float64)
        y = np.asarray(map_func(y), dtype=np.float64)
        # Step 2: add the log difference of the evolution process divided by the initial difference = the expansion in one iteration
        # This gives the expansion of the distance of the vectors
        diff = y - x
        dist = np.linalg.norm(diff)
        if dist == 0:
            return -np.inf
        s += np.log(dist / delta0)
        # renormalize perturbation
        diff = (delta0 / dist) * diff
        y = x + diff
    return s / N

def flow_lyapunov_spectrum(
    flow,
    solver,
    z0,
    params=None,
    dt=0.01,
    interval=0.1,
    n_intervals=1000,
):

    n = z0.shape[0]

    jacobian = jax.jacfwd(lambda z, t: flow(t, z, params))

    def augmented_rhs(t, state, args):
        
        z = state[:n]
        Q = state[n:].reshape((n, n))

        f = flow(t, z, params)
        J = jacobian(z, t)

        dQ = J @ Q

        return jnp.concatenate([f, dQ.reshape(-1)])

    term = dfx.ODETerm(augmented_rhs)

    def integrate(state, t0):

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t0 + interval,
            dt0=dt,
            y0=state,
            saveat=dfx.SaveAt(t1=True),
        )

        return sol.ys[-1]

    def step(carry, _):

        state, t, lyap = carry

        state = integrate(state, t)

        z = state[:n]
        Q = state[n:].reshape((n, n))

        Q, R = jnp.linalg.qr(Q)

        lyap = lyap + jnp.log(jnp.abs(jnp.diag(R)))

        state = jnp.concatenate([z, Q.reshape(-1)])

        return (state, t + interval, lyap), None

    Q0 = jnp.eye(n)

    state0 = jnp.concatenate([z0, Q0.reshape(-1)])

    carry0 = (state0, 0.0, jnp.zeros(n))

    carry, _ = jax.lax.scan(step, carry0, None, length=n_intervals)

    state, t, lyap = carry

    total_time = interval * n_intervals

    return lyap / total_time

lyapunov_spectrum = jax.jit(flow_lyapunov_spectrum, static_argnames=("flow", "solver", "n_intervals"))
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
    from maps import iterate_map, standard_map
    from time import time

    # Parser nonsense
    parser = argparse.ArgumentParser(description='Integrator demo: maps and Hamiltonians')
    parser.add_argument('--demo', choices=['standard_map', 'pendulum'], default='pendulum')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--k', type=float, default=0.971635)
    args = parser.parse_args()

    if args.demo == 'standard_map':
        k = args.k*0
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
        def pendulum_flow(t: jnp.ndarray, z: jnp.ndarray, m=1, g=9.81, L=1) -> jnp.ndarray:
            theta, p = z

            return jnp.array([p, -(m*g/L)*jnp.sin(theta)])

        z0 = jnp.array([1.0, 0.0])

        t_bounds = [0, 30]
        delta_t = 0.0001

        solver = dfx.Dopri5()
        term = dfx.ODETerm(lambda t, z, args: pendulum_flow(t, z))

        saveat = dfx.SaveAt(ts=jnp.linspace(t_bounds[0], t_bounds[1], 10000))

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t_bounds[0],
            t1=t_bounds[1],
            dt0=delta_t,
            y0=z0,
            saveat=saveat,
            args=None,
            max_steps=1200000
    )

        qs, ps = sol.ys.transpose()
        plt.figure()
        plt.plot(qs[:] % (2 * jnp.pi) - jnp.pi, ps[:], linewidth=0.5)
        plt.xlabel('theta')
        plt.ylabel('p')
        plt.title('Pendulum phase portrait')
        plt.show()

        steps = 10
        N_int = 3000
        start = time()
        lyap_f = flow_lyapunov_spectrum(flow=lambda t, z, args: pendulum_flow(t, z), solver=dfx.Dopri5(), z0=z0,
                                   dt=delta_t, interval=steps*delta_t, n_intervals=N_int)
        end = time()
        print(f"Elapsed time: {(end - start):.6f}")
        print('Estimated maximal Lyapunov exponent (flow, approx):', lyap_f)
"""        print('Second method:')
        start = time()
        lyap_f = flow_lyapunov_spectrum(flow=lambda t, z, args: pendulum_flow(t, z), solver=dfx.Dopri5(), z0=z0,
                                   dt=delta_t, interval=steps*delta_t, n_intervals=N_int)
        end = time()

        print(f"Elapsed time: {(end - start):.6f}")
        print('Estimated maximal Lyapunov exponent (flow, approx):', lyap_f)"""
