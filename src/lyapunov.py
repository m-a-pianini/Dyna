from typing import Callable, Tuple, Iterable
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
jax.config.update("jax_enable_x64", True)


#u are cute <3
# Visualisation utils

def poincare_sos(data: np.ndarray | None = None, section_val: float = 0, tol: float = 1e-6,
                     wrap_period: float | None = None, center: float = 0) -> np.ndarray:
    """Extract points near a Poincaré section defined by a coordinate index (zero crossing not implemented).

    This simple helper collects states where coordinate at section_index is near zero (within tol).

    data should have shape (n,)
    """
    if wrap_period is not None:
        wrapped = (data - center + wrap_period / 2) % wrap_period + center - wrap_period / 2
        idxes = np.where(np.abs(wrapped - section_val) <= tol)[0]
    else:
        idxes = np.where(np.abs(data - section_val) <= tol)[0]
    # optional function/series of points

    return data[idxes], idxes

def plot_wrapped(
    x: np.ndarray,
    y: np.ndarray,
    wrap_axis: str = "x",
    period: float = 2 * np.pi,
    center: float = 0.0,
    ax: plt.Axes | None = None,
    kind: str | None = None,
    **plot_kwargs,
) -> plt.Axes:
    """
    Plot a curve with one axis wrapped to a given period.

    Parameters:
        x:           1D array of x values
        y:           1D array of y values
        wrap_axis:   Which axis to wrap, 'x' or 'y'
        period:      Wrapping period (default: 2π)
        center:      Center of the wrapped range; the output lies in
                     [center - period/2, center + period/2) (default: 0)
        ax:          Matplotlib axes to plot into; creates a new figure if None
        **plot_kwargs: Passed directly to ax.plot()

    Returns:
        The axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    data = np.asarray(x if wrap_axis == "x" else y, dtype=float)
    other = np.asarray(y if wrap_axis == "x" else x, dtype=float)

    # Wrap to [center - period/2, center + period/2)
    wrapped = (data - center + period / 2) % period + center - period / 2

    # Detect discontinuities and insert NaN to break the line
    jump_idx = np.where(np.abs(np.diff(wrapped)) > period / 2)[0]
    plot_data = wrapped.copy()
    plot_data[jump_idx + 1] = np.nan
    
    if wrap_axis != "x":
        plot_data, other = other, plot_data
    
    match kind:
        case "scatter":
            ax.scatter(plot_data, other, **plot_kwargs)
        case _:
            ax.plot(plot_data, other, **plot_kwargs)

    return ax, np.array([plot_data, other])

def boxcount_plot(
    trajectory: np.ndarray,
    box_sizes: np.ndarray | None = None,
    n_sizes: int = 20,
    min_points: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, int, int]:
    
    D, sizes, counts, i0, i1 = boxcount_dimension(trajectory, box_sizes, n_sizes, min_points)
    print(f"Box counting extimate: {D}")

    log_s = np.log10(1 / sizes)
    log_c = np.log10(counts)

    s_fit = np.array([log_s[i0], log_s[i1]])
    c_fit = np.polyval(np.polyfit(log_s[i0:i1+1], log_c[i0:i1+1], 1), s_fit)

    _, ax = plt.subplots()
    ax.loglog(1 / sizes, counts, 'o', label='all scales')
    ax.loglog(1 / sizes[i0:i1+1], counts[i0:i1+1], 'o', color='red', label='linear region')

    # Convert back from log10 space to data space for the fit line
    ax.loglog(10**s_fit, 10**c_fit, '--', label=f'fit D={D:.3f}')

    ax.set_xlabel('1/r (inverse box size)')
    ax.set_ylabel('N(r) (box count)')
    ax.legend(); plt.grid(True)
    return ax, D

# Algorithms for calculating lyapunov exponent(s)

# Maximum Lyapunov exponent
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

# TODO: implement this
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

# Full spectrum

def map_lyapunov_spectrum(
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

@partial(jax.jit, static_argnames=("flow", "solver", "n_intervals", "burn_in", "stepsize", "jacobian"))
def flow_lyapunov_spectrum(
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

@partial(jax.jit, static_argnames=("flow", "solver", "n_intervals", "burn_in", "save_at", "stepsize", "jacobian"))
def fast_flow_lyapunov_spectrum(
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

# Efficient many-trajectories (vmappable) lyapunov exponent calculation
def make_batch_lyapunov_solver(flow, solver, dt, n_intervals, stepsize, burn_in, jacobian=True, save_at=dfx.SaveAt(t1=True)):
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

        return flow_lyapunov_spectrum(
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

def make_batch_fast_lyapunov(flow, solver, dt, n_intervals, stepsize, burn_in, jacobian=True):
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

        return fast_flow_lyapunov_spectrum(
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

# TODO: equilibrium point finder for flows and maps
# TODO: jacobian analyzer flows/maps: trace, det, eigenvalues varying the parameters
# TODO: bifurcation diagram of a flow/map: 


# Fractal dimension extimation 

def kaplan_yorke_dim(lyaps: jnp.ndarray):
    """
    Compute Kaplan–Yorke (Lyapunov) dimension.

    Works for:
        lyaps.shape = (n,) or (..., n)

    Assumes exponents are real and system is dissipative.
    """

    # sort descending along last axis
    lam = jnp.sort(lyaps, axis=-1)[..., ::-1]

    # cumulative sums
    cumsum = jnp.cumsum(lam, axis=-1)

    # find largest j such that sum >= 0
    mask = cumsum >= 0

    # number of non-negative partial sums 
    j = jnp.sum(mask, axis=-1)  # shape (...)

    # index of the first negative cumulant
    j_clipped = jnp.clip(j - 1, 0, lam.shape[-1] - 1)

    # sum up to j
    sum_j = jnp.clip(jnp.take_along_axis(cumsum, j_clipped[..., None], axis=-1)[..., 0], 0)

    # next exponent λ_{j+1}
    lam_next = jnp.take_along_axis(
        lam,
        jnp.clip(j_clipped + 1, 0, lam.shape[-1] - 1)[..., None],
        axis=-1
    )[..., 0]
    # print(j, j_clipped[..., None], cumsum, sum_j, lam_next) # Debug line

    # Kaplan–Yorke formula
    dim = j + (sum_j / jnp.abs(lam_next))

    return dim

def count_boxes(trajectory: np.ndarray, box_size: float) -> int:
    """
    Count the number of boxes of given size needed to cover the trajectory.

    Parameters:
        trajectory: Array of shape (n_points, d) representing the d-dimensional trajectory
        box_size: Size of each box

    Returns:
        Number of boxes needed to cover the trajectory
    """
    box_indices = np.floor(trajectory / box_size).astype(int)
    unique_boxes = np.unique(box_indices, axis=0)
    return len(unique_boxes)

def find_linear_region(x: np.ndarray, y: np.ndarray, 
                       min_points: int = 5,
                       r2_threshold: float = 0.999) -> tuple[int, int]:
    """
    Find the longest contiguous window in (x, y) that is well-described
    by a linear fit, using a two-pointer expansion strategy.

    For each candidate start point, the window is expanded rightward as long
    as the R² of the linear fit stays above r2_threshold. The longest such
    window across all start points is returned.

    Parameters:
        x:             1D array of x values (assumed sorted)
        y:             1D array of y values
        min_points:    Minimum number of points a valid window must contain
        r2_threshold:  Minimum R² to consider a window linear (default: 0.999)

    Returns:
        Tuple (start, end) of indices (inclusive) of the best linear region
    """
    n = len(x)
    best_start, best_end = 0, min_points - 1
    best_length = 0

    for start in range(n - min_points + 1):
        # Expand the window rightward while it remains linear
        last_good_end = None

        for end in range(start + min_points - 1, n):
            x_win = x[start:end + 1]
            y_win = y[start:end + 1]

            coeffs = np.polyfit(x_win, y_win, 1)
            y_pred = np.polyval(coeffs, x_win)

            ss_res = np.sum((y_win - y_pred) ** 2)
            ss_tot = np.sum((y_win - y_win.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

            if r2 >= r2_threshold:
                last_good_end = end
            else:
                break  # once R² drops, expanding further only makes it worse

        if last_good_end is not None:
            length = last_good_end - start + 1
            if length > best_length:
                best_length = length
                best_start, best_end = start, last_good_end

    return best_start, best_end

def boxcount_dimension(
    trajectory: np.ndarray,
    box_sizes: np.ndarray | None = None,
    n_sizes: int = 20,
    min_points: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, int, int]:
    """
    Estimate the fractal dimension of a d-dimensional trajectory using the
    box-counting algorithm, fitting only in the linear region of the log-log plot.

    Parameters:
        trajectory: Array of shape (n_points, d)
        box_sizes:  Box sizes to use; auto-generated if None
        n_sizes:    Number of sizes when auto-generating (ignored if box_sizes given)
        min_points: Minimum points required for a linear-region candidate

    Returns:
        Tuple of:
            fractal_dimension  - estimated D
            box_sizes_used     - all box sizes (including invalid ones filtered out)
            box_counts         - corresponding box counts
            linear_start       - index into the valid arrays where the fit begins
            linear_end         - index into the valid arrays where the fit ends
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[:, np.newaxis]

    mins = trajectory.min(axis=0)
    maxs = trajectory.max(axis=0)
    extent = np.max(maxs - mins)

    if box_sizes is None:
        n_points = len(trajectory)
        min_size = extent / n_points
        max_size = extent / 2
        box_sizes = np.logspace(np.log10(min_size), np.log10(max_size), n_sizes)

    trajectory_shifted = trajectory - mins

    counts = np.array([count_boxes(trajectory_shifted, s) for s in box_sizes])

    # Work in log space
    valid = counts > 0
    valid_sizes  = box_sizes[valid]
    valid_counts = counts[valid]

    log_inv_sizes = np.log(1.0 / valid_sizes)
    log_counts    = np.log(valid_counts)

    # Detect and fit only the linear region
    lin_start, lin_end = find_linear_region(log_inv_sizes, log_counts, min_points)
    coeffs = np.polyfit(
        log_inv_sizes[lin_start:lin_end + 1],
        log_counts[lin_start:lin_end + 1],
        1,
    )
    fractal_dimension = coeffs[0]

    return fractal_dimension, valid_sizes, valid_counts, lin_start, lin_end

# TODO: fix this
@partial(jax.jit, static_argnames=("n_scales",))
def correlation_dimension(
    trajectory,
    n_scales=20,
    eps_min=1e-3,
    eps_max=1e-1,
):

    N, d = trajectory.shape

    # --- anisotropic rescaling (important) ---
    mins = jnp.min(trajectory, axis=0)
    maxs = jnp.max(trajectory, axis=0)
    X = (trajectory - mins) / (maxs - mins + 1e-12)

    # --- pairwise distances (static shape!) ---
    diff = X[:, None, :] - X[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)

    # remove diagonal
    mask = ~jnp.eye(N, dtype=bool)

    # --- epsilon scales ---
    epsilons = jnp.logspace(
        jnp.log10(eps_max),
        jnp.log10(eps_min),
        n_scales
    )

    def corr_sum(eps):
        return jnp.sum((dist < eps) & mask) / (N * (N - 1))

    C = jax.vmap(corr_sum)(epsilons)

    # avoid log(0)
    max_C = jnp.maximum(C, 1e-12)

    log_eps = jnp.log(epsilons)
    log_C = jnp.log(max_C)

    # --- slope (dimension) ---
    A = jnp.stack([log_eps, jnp.ones_like(log_eps)], axis=1)
    coeffs = jnp.linalg.lstsq(A, log_C, rcond=None)[0]

    dim = coeffs[0]

    return dim, jnp.array([epsilons, C])


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
        delta_t = 1e-4

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

        steps = 0.03
        N_int = 500000
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
