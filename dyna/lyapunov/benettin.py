from typing import Callable, Tuple, Iterable
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)


#u are cute <3
# Benettin algorithm implementations for calculating lyapunov exponent(s)

# ========= Maximum Lyapunov exponent =========
# =============================================


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

# TODO: implement these
@partial(jax.jit, static_argnames=("flow", "solver", "n_intervals", "burn_in", "save_at", "stepsize"))
def flow_mLCE(
    flow: Callable,
    solver,
    z0,
    t0=0.0,
    params=None,
    dt=0.01,
    interval=1,
    n_intervals=1000,
    burn_in=100,
    save_at=dfx.SaveAt(t1=True),
    stepsize=dfx.ConstantStepSize()
):
    pass

def map_mLCE(
    map: Callable,
    z0,
    t0=0.0,
    params=None,
    interval=1,
    n_intervals=1000,
    burn_in=100,
    jacobian=True
):
    pass


# =============== Full spectrum ===============
# =============================================


# TODO: implement this
def map_spectrum(
    map: Callable,
    z0,
    t0=0.0,
    params=None,
    interval=1,
    n_intervals=1000,
    burn_in=100,
    jacobian=True
):
    pass

# TODO: Look for eliminating intermediate arrays/reducing dimensionality
# This is the best I can do
@partial(jax.jit, static_argnames=("flow", "solver", "n_intervals", "burn_in", "stepsize", "jacobian"))
def flow_spectrum(
    flow: Callable,
    solver,
    z0,
    t0=0.0,
    t1=1.0,
    params=None,
    qr_every=1,
    n_intervals=1000,
    burn_in=100,
    save_at=dfx.SaveAt(t1=True),
    stepsize=dfx.ConstantStepSize(),
    jacobian=False
):
    "Returns the lyapunov exponents extimate by iteration via Benettin algorithm with QR orthogonalization"

    z_dim = z0.shape[0]
    dt = (t1 - t0)/qr_every/n_intervals
    interval = qr_every*dt

    if jacobian:
        jacob = jax.jacfwd(lambda z, t: flow(t, z, params))

    def augmented_rhs(t, state, args):
        
        z = state[:z_dim]
        Q = state[z_dim:].reshape((z_dim, z_dim))

        f = flow(t, z, params)

        if jacobian:
            J = jacob(z, t)
            dQ = J @ Q
        else:
            def jvp_column(v):
                _, Jv = jax.jvp(lambda z: flow(t, z, params), (z,), (v,))
                return Jv

            dQ = jax.vmap(jvp_column)(Q.T).T

        return jnp.concatenate([f, dQ.reshape(-1)])

    term = dfx.ODETerm(augmented_rhs)

    def integrate(state, t0, saver):

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t0 + interval,
            dt0=dt,
            y0=state,
            saveat=saver,
            stepsize_controller=stepsize,
        )

        return sol.ys, sol.ts
    
    @partial(jax.jit, donate_argnums=(0,))
    def step(carry, k):

        _state0, t, lyap = carry        

        if isinstance(save_at, jnp.ndarray):
            saver = dfx.SaveAt(t1=True, ts=save_at + t)
        else:
            saver = save_at

        sol, ts = integrate(_state0, t, saver)

        z = sol[-1, :z_dim]
        Q = sol[-1, z_dim:].reshape((z_dim, z_dim))

        Q, R = jnp.linalg.qr(Q)

        lyap = lyap + jnp.log(jnp.abs(jnp.diag(R)))

        state = jnp.concatenate([z, Q.reshape(-1)])

        current_time = (k + 1) * interval
        lam_est = lyap / current_time

        seq = jnp.array([sol[:, :z_dim], jnp.full_like(sol[:, :z_dim], lam_est),  jnp.repeat(ts[..., jnp.newaxis], z_dim, axis=-1)])

        return (state, t + interval, lyap), seq

    # Burn in setup
    Q0 = jnp.eye(z_dim)
    state0 = jnp.concatenate([z0, Q0.reshape(-1)])
    carry0 = (state0, t0, jnp.zeros(z_dim))
    k0 = jnp.arange(burn_in)
    carry, ser0 = jax.lax.scan(step, carry0, k0, length=burn_in)
    state_follow, t_follow, lyap = carry

    # Follow up
    remaining = n_intervals - burn_in
    ks = jnp.arange(remaining)
    carry_follow = (state_follow, t_follow, jnp.zeros(z_dim))
    carry, ser = jax.lax.scan(step, carry_follow, ks, length=remaining)

    state, t, lyap = carry

    total_time = interval * remaining
    
    # Zeroth axis: iteration
    # First axis: traj vs lyapunov vs time
    # Second axis: iteration time step
    # Third axis: dimension
    traj = jnp.concat([ser0[:, 0, ...], ser[:, 0, ...]])
    traj = jnp.concat(traj)

    lyap_ext = ser[:, 1, 0, ...]

    times = jnp.concat([ser0[:, 2, ...,0], ser[:, 2, ..., 0]])
    times = jnp.concat(times)
    return traj, lyap_ext, times

# Many-trajectories/parameters variation (vmappable) lyapunov exponent calculation
def batch_flow_spectrum(flow, solver, dt, n_intervals, stepsize, burn_in, jacobian=True, save_at=dfx.SaveAt(t1=True)):
    """
    # Example usage:
    compute = make_batch_lyapunov_solver(flow=rhs, solver=solver, dt=dt, stepsize=stepsc, n_intervals=N_iters, burn_in=50, jacobian=False)
    batched_lyap = jax.jit(
        jax.vmap(compute, in_axes=(0, 0, None, None))
    )

    cum_lyaps = batched_lyap(jnp.array([[0., 0], [1, 1]]), t0_batch, pars, steps*dt)

    # Alternative for memory filling
    results = []
    z0_all = jnp.array([[0., 0], [1, 1]])
    for i in range(0, len(z0_all), batch_size):
        z_chunk = z0_all[i:i+batch_size]
        lam = batched_lyap(z_chunk, t0_batch, pars, steps*dt)
        results.append(lam)

    cum_lyaps = jnp.concatenate(results, axis=0)"""
    @partial(jax.jit, static_argnames=())
    def compute(z0, t0, params, interval):

        return flow_spectrum(
            flow=flow,
            solver=solver,
            z0=z0,
            t0=t0,
            params=params,
            dt=dt,
            interval=interval,
            n_intervals=n_intervals,
            burn_in=burn_in,
            save_at=save_at,
            stepsize=stepsize,
            jacobian=jacobian,
        )

    return compute


# Deprecatable
# =============================================

@partial(jax.jit, static_argnames=("flow", "solver", "n_intervals", "burn_in", "save_at", "stepsize", "jacobian"))
def fast_flow_spectrum(
    flow: Callable,
    solver,
    z0,
    t0=0.0,
    params=None,
    dt=0.01,
    interval=1,
    n_intervals=1000,
    burn_in=100,
    save_at=dfx.SaveAt(t1=True),
    stepsize=dfx.ConstantStepSize(),
    jacobian=True
):
    "Returns the lyapunov exponents extimate by iteration via Benettin algorithm with QR orthogonalization"

    z_dim = z0.shape[0]

    if jacobian:
        jacob = jax.jacfwd(lambda z, t: flow(t, z, params))

    def augmented_rhs(t, state, args):
        
        z = state[:z_dim]
        Q = state[z_dim:].reshape((z_dim, z_dim))

        f = flow(t, z, params)

        if jacobian:
            J = jacob(z, t)
            dQ = J @ Q
        else:
            def jvp_column(v):
                _, Jv = jax.jvp(lambda z: flow(t, z, params), (z,), (v,))
                return Jv

            dQ = jax.vmap(jvp_column)(Q.T).T

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
            saveat=save_at,
            stepsize_controller=stepsize,
        )

        return sol.ys

    def step(carry, k):

        state, t, lyap = carry

        state = integrate(state, t)[-1]

        z = state[:z_dim]
        Q = state[z_dim:].reshape((z_dim, z_dim))

        Q, R = jnp.linalg.qr(Q)

        lyap = lyap + jnp.log(jnp.abs(jnp.diag(R)))

        state = jnp.concatenate([z, Q.reshape(-1)])

        return (state, t + interval, lyap), None

    # Burn in setup
    Q0 = jnp.eye(z_dim)
    state0 = jnp.concatenate([z0, Q0.reshape(-1)])
    carry0 = (state0, t0, jnp.zeros(z_dim))
    k0 = jnp.arange(burn_in)
    carry, _ = jax.lax.scan(step, carry0, k0, length=burn_in)
    state_follow, t_follow, lyap = carry

    # Follow up
    remaining = n_intervals - burn_in
    ks = jnp.arange(remaining)
    carry_follow = (state_follow, t_follow, jnp.zeros(z_dim))
    carry, _ = jax.lax.scan(step, carry_follow, ks, length=remaining)

    state, t, lyap = carry

    total_time = interval * remaining
    return lyap/total_time

def batch_fast_flow_spectrum(flow, solver, dt, n_intervals, stepsize, burn_in, jacobian=True):
    """
    # Example usage:
    compute = make_batch_lyapunov_solver(flow=rhs, solver=solver, dt=dt, stepsize=stepsc, n_intervals=N_iters, burn_in=50, jacobian=False)
    batched_lyap = jax.jit(
        jax.vmap(compute, in_axes=(0, 0, None, None))
    )

    cum_lyaps = batched_lyap(jnp.array([[0., 0], [1, 1]]), t0_batch, pars, steps*dt)

    # Alternative for memory filling
    results = []
    z0_all = jnp.array([[0., 0], [1, 1]])
    for i in range(0, len(z0_all), batch_size):
        z_chunk = z0_all[i:i+batch_size]
        lam = batched_lyap(z_chunk, t0_batch, pars, steps*dt)
        results.append(lam)

    cum_lyaps = jnp.concatenate(results, axis=0)"""
    @partial(jax.jit, static_argnames=())
    def compute(z0, t0, params, interval):

        return fast_flow_spectrum(
            flow=flow,
            solver=solver,
            z0=z0,
            t0=t0,
            params=params,
            dt=dt,
            interval=interval,
            n_intervals=n_intervals,
            burn_in=burn_in,
            stepsize=stepsize,
            jacobian=jacobian,
        )

    return compute


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dyna.flows import samelson_flow
    from time import time
    from datetime import datetime

    # =========== INTEGRATION & OUTPUT TEST ===========
    pars = {
        "A0": 0.5,
        "C": 0.25,
        "L": 2.0,
        "h": 0.05,
        "wf": 0.2133,
    }

    unp_pars = pars.copy()
    unp_pars.update({"h": 0})

    rhs = lambda t, z, args: samelson_flow(t, z, args)
    z0 = jnp.array([-np.pi/2, 0])

    # Integration
    Tot_T = 3000
    Tot_iters = 1e6
    steps = 100
    n_inters = int(Tot_iters/steps) # Unelegantly, this has to be done outside the main function

    burns = 0.2
    term = dfx.ODETerm(rhs)

    timesteps = jnp.linspace(0, Tot_T/n_inters, steps-1)

    solver = dfx.Kvaerno5()
    stepsc = dfx.PIDController(rtol=1e-10, atol=1e-12, pcoeff=0.4, icoeff=0.3)

    saveat = dfx.SaveAt(t1=True, ts=timesteps)

    # Analysis
    boxes = np.logspace(-3, 1, 20)

    # Calculate lyapunov spectrum
    now = time()
    traject, cums, times = flow_spectrum(flow=rhs, solver=solver, z0=z0, params=pars, save_at=timesteps,
                                    t1=Tot_T, qr_every=steps, n_intervals=n_inters, stepsize=stepsc, burn_in=int(n_inters*burns))
    later = time()

    print(f"Elapsed time: {later - now:.6f}")
    # Plot of the trajectory (perturbed)
    first = traject.transpose()
    plt.plot(first[0], first[1])
    plt.show()

    print(first.shape)
    print(times[0], times[-1])
    print(cums.shape)
    plt.plot(times, first[0])
    plt.show()
    plt.plot(cums)
    plt.show()

    # ================= COPILATION TEST ===================

    REPORT_PATH = "reports/"

    compiled = flow_spectrum.lower(flow=rhs, solver=solver, z0=z0, params=pars, save_at=timesteps,
                                    t1=Tot_T, qr_every=steps, n_intervals=n_inters, stepsize=stepsc, burn_in=int(n_inters*burns))
    
    with open(REPORT_PATH + "compiled_flow_lyap_spect" + str(datetime.now()) + ".txt", "w") as f:
        f.write(compiled.as_text())
        f.write("\n\nCOST ANALYSYS\n\n")
        f.write(str(compiled.cost_analysis()))
        # Unsupported by current version
        #f.write("\n\nMEMORY ANALYSYS\n\n")
        #f.write(compiled.memory_analysis())
