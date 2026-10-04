"""
Periodic orbits of hybrid (spiking) systems by multi-segment orthogonal
collocation, in JAX.

System
------
    dx/dt = f(x)                         (smooth flow between events)
    event type i:  g_i(x) = 0, crossing upward   ->   x <- R_i(x)

e.g. networks of Izhikevich neurons: g_i = v_i - 30, R_i = reset of neuron i
(+ any pulse-coupling jumps it causes in the other neurons).

Formulation
-----------
A periodic orbit with a *prescribed event sequence* (i_0, ..., i_{K-1}) is cut
into K smooth segments.  Segment s starts just after the reset of event
i_{s-1} and ends at event i_s.  Each segment is rescaled to tau in [0,1],

        dx/dtau = T_s f(x),

and discretised by orthogonal collocation at Gauss points (N intervals of
degree m, values stored at equidistant nodes), exactly as in the smooth
single-segment method.  Per segment:

    collocation   N*m*n equations
    continuity    (N-1)*n equations (between sub-intervals)
    event         g_{i_s}(x_s(1)) = 0                         1 equation
    junction      x_{s+1}(0) = R_{i_s}(x_s(1))                n equations
                  (cyclic: segment K's reset feeds segment 0)

Unknowns: nodal values (K*N*(m+1)*n) and durations T_s (K).  The system is
square.  No phase condition is needed: the event/reset fixes the time origin.
Orbit period = sum_s T_s.

Stability
---------
Floquet multipliers = eigenvalues of  M = S_{K-1} Phi_{K-1} ... S_0 Phi_0,
Phi_s = flow monodromy of segment s (variational equation), and the saltation
matrix at each event

    S = DR + (f(x+) - DR f(x-)) grad g^T / (grad g^T f(x-)),   x+ = R(x-).

One multiplier is always 1 (time shift).

Caveats
-------
* The event sequence is an input (it is a discrete feature of the orbit);
  `find_cycle` extracts it from a simulation.  A change of the sequence along a
  parameter path (spike adding, grazing, ...) needs a restart with a new one.
* Simultaneous events (synchronous spikes) are not handled.
"""
import numpy as np
import jax
import jax.numpy as jnp
from numpy.polynomial import polynomial as P
from numpy.polynomial.legendre import leggauss

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# Reference-interval collocation data
# --------------------------------------------------------------------------
def collocation_matrices(m):
    """Return nodes s_k=k/m, Gauss points, interpolation matrix W[i,k]=l_k(zeta_i),
    derivative matrix D[i,k]=l_k'(zeta_i)."""
    nodes = np.arange(m + 1) / m
    gauss = 0.5 * (leggauss(m)[0] + 1.0)
    W = np.zeros((m, m + 1))
    D = np.zeros((m, m + 1))
    for k in range(m + 1):
        others = np.delete(nodes, k)
        coef = P.polyfromroots(others) / np.prod(nodes[k] - others)
        W[:, k] = P.polyval(gauss, coef)
        D[:, k] = P.polyval(gauss, P.polyder(coef))
    return nodes, gauss, W, D


# --------------------------------------------------------------------------
# Hybrid system container + simulation (for initial guesses)
# --------------------------------------------------------------------------
# TODO: make this coherent with event based integrations
class HybridSystem:
    def __init__(self, f, events, resets):
        """
        f      : x -> dx/dt
        events : list of scalar functions g_i(x); event when g_i crosses 0 upward
        resets : list of maps R_i(x) -> x
        """
        assert len(events) == len(resets)
        self.f, self.events, self.resets = f, list(events), list(resets)
        self.n_events = len(events)
        self._next_event = jax.jit(self._make_next_event())

    def g_all(self, x):
        return jnp.stack([g(x) for g in self.events])

    def reset(self, i, x):
        return jax.lax.switch(i, self.resets, x)

    def _make_next_event(self):
        f, g_all = self.f, self.g_all

        def rk4(x, dt):
            k1 = f(x); k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        def next_event(x, dt, max_steps):
            c0 = (x, 0.0, jnp.array(False), 0, x)

            def body3(c):
                x, t, hit, k, _ = c
                xn = rk4(x, dt)
                hitn = jnp.any(g_all(xn) >= 0.0)
                return xn, t + dt, hitn, k + 1, x

            xn, tn, hit, k, xo = jax.lax.while_loop(
                lambda c: jnp.logical_and(~c[2], c[3] < max_steps), body3, c0)
            go, gn = g_all(xo), g_all(xn)
            crossed = gn >= 0.0
            i = jnp.argmax(crossed)
            frac = -go[i] / (gn[i] - go[i])
            x_ev = xo + frac * (xn - xo)
            t_ev = tn - dt + frac * dt
            return i, t_ev, x_ev, self.reset(i, x_ev), hit

        return next_event

    def simulate_events(self, x0, n_events, dt=0.01, max_time=1e4):
        """Return list of (event index, time since previous event,
        pre-event state, post-reset state)."""
        out, x = [], jnp.asarray(x0, float)
        for _ in range(n_events):
            i, te, xe, xp, hit = self._next_event(x, dt, int(max_time / dt))
            if not bool(hit):
                break
            out.append((int(i), float(te), np.asarray(xe), np.asarray(xp)))
            x = xp
        return out

    def find_cycle(self, x0, n_transient=200, max_K=12, n_events=None,
                   tol=1e-3, dt=0.01, verbose=True):
        """Simulate, discard transient events, find the smallest K such that
        the post-reset state repeats after K events.  Returns
        (sequence, durations, post-reset start state)."""
        n_events = n_events or (n_transient + 4 * max_K + 2)
        ev = self.simulate_events(x0, n_events, dt=dt)
        ev = ev[n_transient:]
        if len(ev) < 2:
            raise RuntimeError("system did not spike enough (no event cycle)")
        for K in range(1, max_K + 1):
            if len(ev) <= K:
                break
            ok = all(np.linalg.norm(ev[k][3] - ev[k + K][3]) < tol
                     * (1 + np.linalg.norm(ev[k][3])) and ev[k][0] == ev[k + K][0]
                     for k in range(len(ev) - K))
            if ok:
                # segment 0 starts after event ev[0]; ends with event ev[1]
                seq = [ev[1 + s][0] for s in range(K)]
                durs = [ev[1 + s][1] for s in range(K)]
                x_start = ev[0][3]
                if verbose:
                    print(f"  cycle found: K = {K}, sequence = {seq}, "
                          f"T = {sum(durs):.6f}")
                return seq, durs, x_start
        raise RuntimeError("no periodic event pattern found (try longer "
                           "transient, larger max_K, or looser tol)")


# --------------------------------------------------------------------------
# Multi-segment collocation solver
# --------------------------------------------------------------------------
class HybridCollocation:
    def __init__(self, system, n, sequence, N=30, m=4, mesh=None, grading=3.0):
        """mesh: optional explicit mesh on [0,1]; default is graded toward tau=1
        (tau_j = 1-(1-j/N)^grading) to resolve the spike upstroke, where the
        quadratic Izhikevich flow nearly blows up."""
        self.sys, self.n = system, n
        self.seq = list(sequence)
        self.K, self.N, self.m = len(self.seq), N, m
        self.mesh = (1 - (1 - np.linspace(0, 1, N + 1)) ** grading
                     if mesh is None else np.asarray(mesh))
        nodes, gauss, W, D = collocation_matrices(m)
        self.nodes = nodes
        self.h = jnp.asarray(np.diff(self.mesh))
        self.W, self.D = jnp.asarray(W), jnp.asarray(D)
        self.tau = self.mesh[:-1, None] + np.diff(self.mesh)[:, None] * nodes[None]
        self._fvm = jax.vmap(jax.vmap(jax.vmap(system.f)))
        self._newton_pieces = jax.jit(
            lambda z: (self.residual(z), jax.jacfwd(self.residual)(z)))

    # --- packing ---------------------------------------------------------
    def pack(self, U, T):
        return jnp.concatenate([jnp.ravel(U), jnp.ravel(jnp.asarray(T, float))])

    def unpack(self, z):
        K, N, m, n = self.K, self.N, self.m, self.n
        return z[:K * N * (m + 1) * n].reshape(K, N, m + 1, n), z[K * N * (m + 1) * n:]

    # --- residual --------------------------------------------------------
    def residual(self, z):
        U, T = self.unpack(z)
        Uz = jnp.einsum("ik,sjkn->sjin", self.W, U)
        dU = jnp.einsum("ik,sjkn->sjin", self.D, U) / self.h[None, :, None, None]
        coll = (dU - T[:, None, None, None] * self._fvm(Uz)).ravel()
        cont = (U[:, :-1, -1, :] - U[:, 1:, 0, :]).ravel()
        ends, starts = U[:, -1, -1, :], U[:, 0, 0, :]
        ev, junc = [], []
        for s, i in enumerate(self.seq):
            ev.append(self.sys.events[i](ends[s]))
            junc.append(starts[(s + 1) % self.K] - self.sys.resets[i](ends[s]))
        return jnp.concatenate([coll, cont, jnp.stack(ev), jnp.concatenate(junc)])

    # --- initial guess from a trajectory ---------------------------------
    def guess(self, x_start, durations, substeps=200):
        """Integrate from the post-reset start state through the K segments
        (applying the resets) and sample at the collocation nodes."""
        f, sys, n = self.sys.f, self.sys, self.n
        grid = np.unique(np.round(self.tau.ravel(), 13))
        dtau = jnp.asarray(np.diff(np.concatenate([[0.0], grid])))
        lookup = {round(t, 11): k for k, t in enumerate(grid)}
        idx = np.vectorize(lambda t: lookup[round(t, 11)])(self.tau)

        @jax.jit
        def run(x0, T):
            def seg(x, d):
                def st(x, _):
                    dt = d * T / substeps
                    k1 = f(x); k2 = f(x + .5 * dt * k1)
                    k3 = f(x + .5 * dt * k2); k4 = f(x + dt * k3)
                    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None
                x, _ = jax.lax.scan(st, x, None, length=substeps)
                return x, x
            _, traj = jax.lax.scan(seg, x0, dtau)
            return traj

        Us, x = [], jnp.asarray(x_start, float)
        for s, i in enumerate(self.seq):
            traj = run(x, durations[s])
            U = traj[idx]
            U = U.at[0, 0].set(x)
            Us.append(U)
            x = sys.resets[i](traj[-1])
        return self.pack(jnp.stack(Us), jnp.asarray(durations))

    # --- Newton ----------------------------------------------------------
    def solve(self, z0, tol=1e-8, maxit=40, verbose=True):
        z = jnp.asarray(z0)
        for it in range(maxit):
            r, J = self._newton_pieces(z)
            nr = float(jnp.linalg.norm(r, jnp.inf))
            if verbose:
                print(f"  Newton {it:2d}: |res|_inf = {nr:.3e}   "
                      f"T = {float(jnp.sum(z[-self.K:])):.10f}")
            if nr < tol:
                return z
            dz = jnp.linalg.solve(J, -r)
            lam = 1.0
            for _ in range(10):                     # backtracking
                zn = z + lam * dz
                if float(jnp.linalg.norm(self.residual(zn), jnp.inf)) < nr or lam < 1e-3:
                    break
                lam *= 0.5
            z = zn
        print("Warning: Newton did not converge")
        return z

    # --- results ---------------------------------------------------------
    def solution(self, z):
        U, T = self.unpack(z)
        return np.asarray(U), np.asarray(T)

    def period(self, z):
        return float(jnp.sum(self.unpack(z)[1]))

    def evaluate(self, z, n_per_interval=20):
        """Evaluate the piecewise polynomials -> (time, states) with time
        running through the whole period (segments concatenated)."""
        U, T = self.solution(z)
        m = self.m
        s = np.linspace(0, 1, n_per_interval, endpoint=False)
        L = np.ones((len(s), m + 1))
        for k in range(m + 1):
            for q in range(m + 1):
                if q != k:
                    L[:, k] *= (s - self.nodes[q]) / (self.nodes[k] - self.nodes[q])
        ts, xs, t0 = [], [], 0.0
        h = np.asarray(self.h)
        for sg in range(self.K):
            for j in range(self.N):
                ts.append(t0 + T[sg] * (self.mesh[j] + h[j] * s))
                xs.append(L @ U[sg, j])
            ts.append([t0 + T[sg]]); xs.append(U[sg, -1, -1][None])
            t0 += T[sg]
        return np.concatenate(ts), np.vstack(xs)

    def floquet_multipliers(self, z, steps=4000):
        """Eigenvalues of M = S_{K-1}Phi_{K-1} ... S_0 Phi_0."""
        U, T = self.unpack(z)
        f, n = self.sys.f, self.n
        dt = 1.0 / steps

        def flow_monodromy(x0, Ts):
            def rhs(y):
                x, Phi = y[:n], y[n:].reshape(n, n)
                return jnp.concatenate(
                    [Ts * f(x), (Ts * jax.jacfwd(f)(x) @ Phi).ravel()])

            def step(y, _):
                k1 = rhs(y); k2 = rhs(y + .5 * dt * k1)
                k3 = rhs(y + .5 * dt * k2); k4 = rhs(y + dt * k3)
                return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

            y, _ = jax.lax.scan(step, jnp.concatenate([x0, jnp.eye(n).ravel()]),
                                None, length=steps)
            return y[n:].reshape(n, n)

        M = jnp.eye(n)
        for s, i in enumerate(self.seq):
            Phi = flow_monodromy(U[s, 0, 0], T[s])
            xm = U[s, -1, -1]
            R, g = self.sys.resets[i], self.sys.events[i]
            xp = R(xm)
            DR, gg = jax.jacfwd(R)(xm), jax.grad(g)(xm)
            fm, fp = f(xm), f(xp)
            S = DR + jnp.outer(fp - DR @ fm, gg) / (gg @ fm)
            M = S @ Phi @ M
        return np.linalg.eigvals(np.asarray(M))


# --------------------------------------------------------------------------
# Demo: Izhikevich neurons
# --------------------------------------------------------------------------
def izhikevich_network(a, b, c, d, I, W, v_peak=30.0):
    """N Izhikevich neurons, state x = [v_0..v_{N-1}, u_0..u_{N-1}].
    Pulse coupling: a spike of neuron j adds W[i, j] to v_i of every other i."""
    a, b, c, d, I, W = (jnp.asarray(q, float) for q in (a, b, c, d, I, W))
    Nn = len(a)

    def f(x):
        v, u = x[:Nn], x[Nn:]
        return jnp.concatenate([0.04 * v**2 + 5 * v + 140 - u + I,
                                a * (b * v - u)])

    def make(j):
        def g(x):
            return x[j] - v_peak

        def R(x):
            v, u = x[:Nn], x[Nn:]
            v = v + W[:, j] * (jnp.arange(Nn) != j)          # synaptic jumps
            v = v.at[j].set(c[j])
            u = u.at[j].set(u[j] + d[j])
            return jnp.concatenate([v, u])
        return g, R

    gs, Rs = zip(*[make(j) for j in range(Nn)])
    return HybridSystem(f, gs, Rs)


def run_case(title, sys_, n, x0, N=30, m=4, **kw):
    print("\n" + "=" * 70 + "\n" + title)
    seq, durs, x_start = sys_.find_cycle(x0, **kw)
    lc = HybridCollocation(sys_, n, seq, N=N, m=m)
    z = lc.solve(lc.guess(x_start, durs))
    U, T = lc.solution(z)
    print(f"  segment durations: {np.round(T, 8)}   period = {T.sum():.10f}")
    print(f"  simulated durations: {np.round(durs, 8)}")
    mult = lc.floquet_multipliers(z)
    print("  Floquet multipliers:", np.round(mult, 6))
    rest = np.delete(mult, np.argmin(np.abs(mult - 1.0)))     # drop trivial mu=1
    print(f"  max |nontrivial multiplier| = {np.abs(rest).max():.6f}  ->",
          "stable" if np.abs(rest).max() < 1 else "unstable")
    return lc, z


if __name__ == "__main__":
    # 1) single regular-spiking neuron (K = 1)
    sys1 = izhikevich_network([0.02], [0.2], [-65.0], [8.0], [10.0], [[0.0]])
    run_case("Single regular-spiking Izhikevich neuron", sys1, 2,
             [-65.0, -13.0], n_transient=30)

    # 2) two identical neurons, inhibitory pulse coupling -> alternating spikes
    sys2 = izhikevich_network([0.02, 0.02], [0.2, 0.2], [-65.0, -65.0],
                              [8.0, 8.0], [10.0, 10.0],
                              [[0.0, -4.0], [-4.0, 0.0]])
    lc2, z2 = run_case("Two inhibitory-coupled neurons", sys2, 4,
                       [-65.0, -50.0, -13.0, -13.0], n_transient=1500,
                       tol=1e-4, dt=0.02)

    # 3) non-identical neurons with inhibitory coupling: 2:1 locking, neuron 0
    #    fires twice per spike of neuron 1 (event sequence of length 3)
    sys3 = izhikevich_network([0.02, 0.02], [0.2, 0.2], [-65.0, -65.0],
                              [8.0, 8.0], [14.0, 7.0],
                              [[0.0, -3.0], [-3.0, 0.0]])
    lc3, z3 = run_case("2:1 locked pair of non-identical neurons",
                       sys3, 4, [-65.0, -55.0, -13.0, -13.0],
                       n_transient=600, max_K=8, tol=1e-4, dt=0.02)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t, x = lc2.evaluate(z2)
        plt.plot(t, x[:, 0], label="v0"); plt.plot(t, x[:, 1], label="v1")
        plt.xlabel("t (ms)"); plt.legend(); plt.title("Hybrid limit cycle (collocation)")
        plt.savefig("hybrid_cycle.png", dpi=120)
        print("\nsaved hybrid_cycle.png")
    except ImportError:
        pass
