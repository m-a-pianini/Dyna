"""
Limit cycle computation by orthogonal collocation (Gauss points) using JAX.

Solves   du/dt = f(u),  u(t+T) = u(t)   by rescaling time, tau = t/T in [0,1]:

    du/dtau = T f(u),   u(0) = u(1),   integral phase condition.

Discretisation (Kuznetsov, "Elements of Applied Bifurcation Theory", 10.2):
  * mesh 0 = tau_0 < ... < tau_N = 1
  * on each subinterval, u^(j) is a degree-m vector polynomial represented by
    its values u_{j,k} at the equidistant points tau_{j,k} = tau_j + (k/m) h_j
  * collocation (10.22) at the m Gauss points of each subinterval
  * continuity  u_{j-1,m} = u_{j,0},  periodicity (10.24)  u_{0,0} = u_{N-1,m}
  * discrete phase condition (10.25)
        sum_{j,i} h_j w_i <u_{j,i}, vdot_{j,i}> = 0
    with Lagrange quadrature weights w_i and vdot the derivative of a
    reference (previous / initial) periodic solution.

Unknowns: u_{j,i} (N*(m+1)*n numbers) and T.
Equations: N*m*n (collocation) + (N-1)*n (continuity) + n (periodicity) + 1
(phase) = N*(m+1)*n + 1.  Solved by Newton's method; the Jacobian is obtained
with JAX forward-mode autodiff (dense solve; the block structure could be
exploited for large problems).
"""
import numpy as np
import jax
import jax.numpy as jnp
from numpy.polynomial import polynomial as P
from numpy.polynomial.legendre import leggauss

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# Reference-interval quantities (independent of f and of the mesh)
# --------------------------------------------------------------------------
def collocation_matrices(m):
    """
    On the reference interval [0,1] with equidistant nodes s_k = k/m:
      gauss : (m,)     Gauss points zeta_i (roots of Legendre P_m mapped to [0,1])
      D     : (m,m+1)  D[i,k] = l_k'(zeta_i)   (derivative of Lagrange basis)
      w     : (m+1,)   w[k]   = int_0^1 l_k(s) ds  (quadrature weights)
    """
    nodes = np.arange(m + 1) / m
    x, _ = leggauss(m)
    gauss = 0.5 * (x + 1.0)     # Interval [0, 1]
    D = np.zeros((m, m + 1))
    w = np.zeros(m + 1)
    for k in range(m + 1):
        others = np.delete(nodes, k)
        coef = P.polyfromroots(others)      # From a set of roots generate the coefficients of the monic poly
        coef = coef / np.prod(nodes[k] - others)      # l_k(s) in monomial basis
        D[:, k] = P.polyval(gauss, P.polyder(coef))
        w[k] = P.polyval(1.0, P.polyint(coef)) - P.polyval(0.0, P.polyint(coef))
    return gauss, D, w, nodes


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
class LimitCycleCollocation:
    def __init__(self, f, n, N=20, m=4, mesh=None):
        """
        f    : callable f(u) -> (n,) JAX-traceable vector field (parameters
               can be closed over, e.g. with functools.partial / lambda)
        n    : state dimension
        N    : number of mesh intervals
        m    : degree of polynomial on each interval (= number of Gauss points)
        mesh : optional increasing array of N+1 points with mesh[0]=0, mesh[-1]=1
        """
        self.f, self.n, self.N, self.m = f, n, N, m
        self.mesh = jnp.linspace(0, 1, N + 1) if mesh is None else jnp.asarray(mesh)
        assert len(self.mesh) == N + 1
        gauss, D, w, nodes = collocation_matrices(m)
        h = jnp.diff(self.mesh)
        self.h = jnp.asarray(h)
        self.D = jnp.asarray(D)
        self.w = jnp.asarray(w)
        # all points tau_{j,k}, shape (N, m+1)
        self.tau = self.mesh[:-1, None] + h[:, None] * nodes[None, :]
        self.zeta = self.mesh[:-1, None] + h[:, None] * gauss[None, :]
        self._fv = jax.vmap(jax.vmap(f))                # (N,m+1,n)->(N,m+1,n)
        self._fz = jax.vmap(jax.vmap(f))
        self._newton_pieces = jax.jit(self._residual_and_jac)

    # ----- packing -------------------------------------------------------
    def pack(self, U, T):
        return jnp.concatenate([jnp.ravel(U), jnp.atleast_1d(T)])

    def unpack(self, z):
        N, m, n = self.N, self.m, self.n
        return z[:-1].reshape(N, m + 1, n), z[-1]

    # ----- residual ------------------------------------------------------
    def residual(self, z, vdot):
        # System of equations 10.22, 24, 25 and continuity
        # To be fed to the newton solver
        U, T = self.unpack(z)
        # u^(j)(zeta_{j,i}) and derivative wrt tau:  (1/h_j) sum_k D[i,k] u_{j,k}
        Uz = jnp.einsum("ik,jkn->jin", self.W_interp, U)      # values at Gauss pts
        dU = jnp.einsum("ik,jkn->jin", self.D, U) / self.h[:, None, None]
        F = jax.vmap(jax.vmap(self.f))(Uz)
        coll = (dU - T * F).ravel()                           # (22)
        cont = (U[:-1, -1, :] - U[1:, 0, :]).ravel()          # continuity
        per = U[-1, -1, :] - U[0, 0, :]                       # (24)
        # (25): sum_j h_j sum_i w_i <u_ji, vdot_ji>
        phase = jnp.einsum("j,i,jin,jin->", self.h, self.w, U, vdot)
        return jnp.concatenate([coll, cont, per, jnp.atleast_1d(phase)])

    @property
    def W_interp(self):
        # interpolation matrix: values of l_k at Gauss points
        if not hasattr(self, "_W"):
            m = self.m
            nodes = np.arange(m + 1) / m
            gauss = 0.5 * (leggauss(m)[0] + 1.0)
            W = np.ones((m, m + 1))
            for k in range(m + 1):
                for q in range(m + 1):
                    if q != k:
                        W[:, k] *= (gauss - nodes[q]) / (nodes[k] - nodes[q])
            self._W = jnp.asarray(W)
        return self._W

    def _residual_and_jac(self, z, vdot):
        r = self.residual(z, vdot)
        J = jax.jacfwd(self.residual)(z, vdot)
        return r, J

    # ----- initial guess helpers ----------------------------------------
    def guess_from_function(self, u_of_tau, T):
        """u_of_tau: callable tau-array(N,m+1) -> states (N,m+1,n)."""
        U = jnp.asarray(u_of_tau(self.tau))
        return self.pack(U, T)

    # TODO: should use integrator step from integrators
    def guess_from_integration(self, u0, T, steps_per_interval=200):
        """Integrate du/dt=f(u) from u0 over time T (RK4) and sample at mesh."""
        f = self.f
        flat_tau = np.unique(np.round(self.tau.ravel(), 14))
        dt_total = jnp.asarray(np.diff(np.concatenate([[0.0], flat_tau])) * T)

        def rk4(u, dt):
            def step(u, _):
                dts = dt / steps_per_interval
                k1 = f(u); k2 = f(u + 0.5 * dts * k1)
                k3 = f(u + 0.5 * dts * k2); k4 = f(u + dts * k3)
                return u + dts / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None
            u, _ = jax.lax.scan(step, u, None, length=steps_per_interval)
            return u, u

        _, traj = jax.lax.scan(rk4, jnp.asarray(u0, dtype=float), dt_total)
        lookup = {round(t, 12): i for i, t in enumerate(flat_tau)}
        idx = np.vectorize(lambda t: lookup[round(t, 12)])(self.tau)
        U = traj[idx]
        # force exact periodic closure of the guess
        U = U.at[-1, -1].set(U[0, 0])
        return self.pack(U, T)

    # ----- Newton --------------------------------------------------------
    # TODO: should be function from analysis/other module
    def solve(self, z0, tol=1e-11, maxit=50, verbose=True, update_phase_ref=False):
        z = jnp.asarray(z0)
        U, T = self.unpack(z)
        vdot = jax.vmap(jax.vmap(self.f))(U)       # reference derivative
        for it in range(maxit):
            r, J = self._newton_pieces(z, vdot)
            nr = float(jnp.linalg.norm(r, jnp.inf))
            if verbose:
                print(f"  Newton {it:2d}: |res|_inf = {nr:.3e}   T = {float(z[-1]):.10f}")
            if nr < tol:
                break
            # Solving method
            dz = jnp.linalg.solve(J, -r)
            z = z + dz
            if update_phase_ref:
                U, T = self.unpack(z)
                vdot = jax.vmap(jax.vmap(self.f))(U)
        else:
            print("Warning: Newton did not converge")
        return z

    # ----- post-processing ----------------------------------------------
    def solution(self, z):
        U, T = self.unpack(z)
        return np.asarray(U), float(T)

    def evaluate(self, z, n_per_interval=20):
        """Evaluate the piecewise polynomial on a fine grid -> (tau, u)."""
        U, T = self.solution(z)
        m = self.m
        nodes = np.arange(m + 1) / m
        s = np.linspace(0, 1, n_per_interval, endpoint=False)
        L = np.ones((len(s), m + 1))
        for k in range(m + 1):
            for q in range(m + 1):
                if q != k:
                    L[:, k] *= (s - nodes[q]) / (nodes[k] - nodes[q])
        taus, us = [], []
        for j in range(self.N):
            taus.append(self.mesh[j] + self.h[j] * s)
            us.append(L @ U[j])
        taus.append([1.0]); us.append(U[-1, -1][None, :])
        return np.concatenate(taus), np.vstack(us)

    # TODO: should be function from lyapubov module
    def floquet_multipliers(self, z):
        """Monodromy matrix via variational equation (RK4), for stability."""
        U, T = self.unpack(z)
        f = self.f
        u0 = U[0, 0]
        n = self.n

        def rhs(x):
            u, Phi = x[:n], x[n:].reshape(n, n)
            return jnp.concatenate([T * f(u), (T * jax.jacfwd(f)(u) @ Phi).ravel()])

        x = jnp.concatenate([u0, jnp.eye(n).ravel()])
        K = 2000
        dt = 1.0 / K

        def step(x, _):
            k1 = rhs(x); k2 = rhs(x + 0.5 * dt * k1)
            k3 = rhs(x + 0.5 * dt * k2); k4 = rhs(x + dt * k3)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

        x, _ = jax.lax.scan(step, x, None, length=K)
        return np.linalg.eigvals(np.asarray(x[n:].reshape(n, n)))


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from dyna.flows import hodgkin_huxley, van_der_pol, hopf
    mu = 1.0

    lc = LimitCycleCollocation(lambda u: hodgkin_huxley(u, I_ext=15), n=4, N=50, m=10)

    # crude guess: circle of radius 2, period 2*pi, refined by integration
    """    z0 = lc.guess_from_function(
        lambda tau: jnp.stack([2 * jnp.cos(2 * jnp.pi * tau),
                               -2 * jnp.sin(2 * jnp.pi * tau)], axis=-1),
        T=6.5)
    print("Van der Pol, mu =", mu)"""

    z0_hh = lc.guess_from_integration(u0=jnp.stack([-74, 0.03, 0.125, 0.68]), T=0.012, steps_per_interval=200)
    print(z0_hh.shape, z0_hh)

    z = lc.solve(z0_hh, tol=1e-8)
    U, T = lc.solution(z)
    print(f"\nPeriod T = {T:.10f}   (literature: 6.6632868593 for mu=1)")
    print("Floquet multipliers:", lc.floquet_multipliers(z),
          " (one multiplier = 1, the other inside unit circle -> stable)")

    # Hopf normal form: r' = r(1 - r^2), theta' = 1  -> circle r=1, T = 2 pi
    def hopf(u):
        x, y = u
        r2 = x * x + y * y
        return jnp.array([x * (1 - r2) - y, y * (1 - r2) + x])

    lc2 = LimitCycleCollocation(hopf, n=2, N=10, m=3)
    z0 = lc2.guess_from_function(
        lambda tau: jnp.stack([1.5 * jnp.cos(2 * jnp.pi * tau),
                               1.5 * jnp.sin(2 * jnp.pi * tau)], axis=-1), T=5.5)
    print("\nHopf normal form")
    z = lc2.solve(z0)
    U, T = lc2.solution(z)
    print(f"T = {T:.12f}  (exact 2*pi = {2*np.pi:.12f})")
    print("max | |u|-1 | =", np.abs(np.linalg.norm(U, axis=-1) - 1).max())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = lc.solve(z0=z0_hh, tol=1e-8, verbose=False)
        _, u = lc.evaluate(z)
        plt.plot(u[:, 0], u[:, 1]); plt.xlabel("x"); plt.ylabel("y")
        plt.title("Limit cycle"); plt.savefig("limit_cycle.png", dpi=120)
        print("saved limit_cycle.png")
    except ImportError:
        pass
