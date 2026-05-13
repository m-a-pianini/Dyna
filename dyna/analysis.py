from typing import Callable
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute, root
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# Visualisation & manipulation utils

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


# ================== Analysis =================
# =============================================

# Find stationary points of a function
def find_stationary(func: Callable, seq, perc=5, tolerance=0, ):
    """
    Evaluates the function on the seq points and searches for roots from the top perc% candidates
    that scores nearest to zero\n 
    Percentile is from counted from the top, so the points that score closer to zero\n

    """
    # First, calculate the squared norm of the function on the seq to find root candidates
    last_dim = seq.shape[-1]
    batch_f = jax.jit(jax.vmap(func, in_axes=0))
    squared_norm = jnp.linalg.norm(batch_f(seq), axis=-1)
    # Calculate top p percentile (lowest squared norm)
    threshold = jnp.percentile(a=squared_norm, q=perc, axis=-1,)
    print(threshold)
    idx = jnp.argwhere(squared_norm <= threshold)

    top = jnp.take_along_axis(seq, idx, axis=0) # top scorers
    top_val = squared_norm[idx] # top values

    # Then, search for a root near those values
    top.reshape(-1, seq.shape[-1])
    sols = []
    vals = []
    for x0 in np.array(top):
        result = root(func, x0)
        sols.append(result.x)
        vals.append(result.fun)

    return top, top_val, jnp.array(sols), jnp.array(vals)


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


if __name__ == "__main__":
    from flows import samelson_flow
    flurry = jnp.concat([jnp.array([jnp.linspace(-jnp.pi, jnp.pi, 20), jnp.zeros(20)]).T,
                        jnp.array([jnp.linspace(-jnp.pi/2, jnp.pi/2, 20), jnp.full(20, 2.75)]).T,
                        jnp.array([jnp.linspace(-jnp.pi*3/2, -jnp.pi/2, 20), jnp.full(20, -2.75)]).T,
                                    ])
    print(flurry.shape)
    print(find_stationary(lambda x: samelson_flow(0, x), flurry))
