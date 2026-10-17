"""The 3D 1-form wave equation in the stellarator-like tube (talk, Section 8), on the Struphy Derham complex with polar
splines.

Maxwell's equations ``dE/dt = curl B``, ``dB/dt = -curl E`` with ``E x n = 0`` on the boundary, semidiscretized with the
polar 1- and 2-forms of a 3D Struphy ``Derham`` complex (masses ``WeightedMassOperators.M1, M2``, the polar curl,
``PolarMassMatrixPreconditioner``), integrated by the leapfrog of ``MaxwellWeakAmpere(algo="leapfrog")`` with
``dt = 0.9 * 2 / sqrt(lambda_max)``. The domain is ``HelicalTube("stellarator")``, the mesh ``n x 2n x 8n`` cells of
degree 4 (as the talk). The initial field ``B_0 = curl A_0`` is the magnetic pulse of
``examples/Maxwell/gaussian_helical_tube`` (``A_0 = (0, 0, a)`` in logical components, a Gaussian next to the axis);
``B_0`` is the discrete curl of the projection of ``A_0``, so ``div B = 0`` exactly.

The error measure is the talk's: the maximum relative L-infinity error during the final 10% of the run against a
finer mesh of the highest regularity, sampled at logical points of the cross-sections ``eta3 = 0`` and ``1/2``.
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy.demos.plotting import CATEGORICAL, DIVERGING, clean_axis

WIDTH_FACTOR, AXIAL_WIDTH = 0.04, 0.1


def helical_tube():
    from struphy.geometry import domains

    return domains.HelicalTube("stellarator", num_elements=(8, 24, 64), degree=(4, 4, 4))


def initial_potential(domain):
    """The logical component ``a_3`` of ``A_0`` of the example (a Gaussian in the cross-section and along the axis)."""
    g = domain.geometry
    width = WIDTH_FACTOR * g["a"]

    def section(e1, e2):
        radius = e1 * (1 + g["eps"] * e1 * np.cos(2 * np.pi * g["k"] * e2))
        return radius * g["a"] * np.cos(2 * np.pi * e2), radius * g["b"] * np.sin(2 * np.pi * e2)

    center = section(0.05, 0.0)
    speed = 2 * np.pi * np.sqrt((g["R0"] + g["delta"]) ** 2 + (g["periods"] * g["delta_z"]) ** 2)

    def a3(e1, e2, e3):
        x2, y2 = section(e1, e2)
        arclength = speed * (np.mod(e3 + 0.5, 1.0) - 0.5)
        return np.exp(
            -((x2 - center[0]) ** 2 + (y2 - center[1]) ** 2) / (2 * width**2) - arclength**2 / (2 * AXIAL_WIDTH**2)
        )

    return a3


class Wave3D:
    """Polar 1-form wave (Maxwell) in the helical tube with regularity ``r`` on ``n x 2n x 8n`` cells."""

    def __init__(self, regularity, n, degree=4, domain=None, tol=1e-12):
        from feectools.linalg.solvers import inverse

        from struphy.feec.mass import WeightedMassOperators
        from struphy.feec.preconditioner import PolarMassMatrixPreconditioner
        from struphy.feec.psydac_derham import Derham
        from struphy.io.options import DerhamOptions
        from struphy.topology.grids import TensorProductGrid

        self.domain = helical_tube() if domain is None else domain
        self.regularity, self.n = regularity, n
        options = DerhamOptions(
            degree=(degree,) * 3,
            bcs=(("free", "dirichlet"), None, None),
            polar_splines=True,
            polar_regularity=regularity,
            nquads_proj=(degree + 2,) * 3,
        )
        grid = TensorProductGrid(num_elements=(n, 2 * n, 8 * n), mpi_dims_mask=(False, True, True))
        import warnings

        with warnings.catch_warnings():
            # the cross-sections of the tube rotate: the regularity r >= 2 holds near eta3 = 0 (see derham_blocks)
            warnings.simplefilter("ignore", UserWarning)
            self.derham = derham = Derham(grid, options, domain=self.domain)
        mass = WeightedMassOperators(derham, self.domain)
        self.M1, self.M2, self.curl = mass.M1, mass.M2, derham.curl
        self.M1_inv = inverse(self.M1, "pcg", pc=PolarMassMatrixPreconditioner(self.M1), tol=tol, maxiter=1000)
        self.weak_curl = self.M1_inv @ self.curl.T @ self.M2
        self.lambda_max = self._largest_eigenvalue()
        self.dt = 0.9 * 2 / np.sqrt(self.lambda_max)
        a3 = initial_potential(self.domain)
        A0 = derham.P1([lambda e1, e2, e3: 0 * e1, lambda e1, e2, e3: 0 * e1, a3])
        self.b0 = derham.boundary_ops["2"].dot(self.curl.dot(A0))

    def _largest_eigenvalue(self, iterations=25):
        """Power iteration on ``M1^{-1} C^T M2 C`` (the square of the largest discrete frequency)."""
        x = self.M1.domain.zeros()
        rng = np.random.default_rng(0)
        for block in getattr(x, "pol", []):
            block[:] = rng.standard_normal(block.shape)
        tp = x.tp if hasattr(x, "tp") else x
        for sub in getattr(tp, "blocks", [tp]):
            sub._data[:] = rng.standard_normal(sub._data.shape)
        x.update_ghost_regions()
        x = self.derham.boundary_ops["1"].dot(x)
        value = 0.0
        for _ in range(iterations):
            y = self.weak_curl.dot(self.curl.dot(x))
            value = float(np.sqrt(y.inner(self.M1.dot(y)) / x.inner(self.M1.dot(x))))
            x = y * (1.0 / np.sqrt(y.inner(y)))
        return value

    def evolve(self, final_time, output_times, sample):
        """Leapfrog ``b <- b - dt/2 C e``, ``e <- e + dt M1^{-1} C^T M2 b``, ``b <- b - dt/2 C e``; returns the sampled
        2-form proxies (``len(output_times)`` arrays) and the relative energy error."""
        steps = int(np.ceil(final_time / self.dt))
        dt = final_time / steps
        e = self.M1.domain.zeros()
        b = self.b0.copy()
        field = self.derham.create_spline_function("b", "Hdiv")
        frames, times, t = [], [], 0.0
        energy0 = 0.5 * b.inner(self.M2.dot(b))
        energies = []

        def record():
            field.vector = b
            frames.append(sample(field))
            times.append(t)

        record()
        for _ in range(steps):
            b -= 0.5 * dt * self.curl.dot(e)
            e += dt * self.weak_curl.dot(b)
            b -= 0.5 * dt * self.curl.dot(e)
            t += dt
            energies.append(0.5 * (b.inner(self.M2.dot(b)) + e.inner(self.M1.dot(e))))
            record()
        frames = np.array(frames)
        values = np.array([np.interp(output_times, times, frames[:, i]) for i in range(frames.shape[1])]).T
        energy_error = np.max(np.abs(np.array(energies) - energy0)) / energy0 if energies else 0.0
        return values, energy_error


def sample_points(n=24):
    """Logical points in the cross-sections ``eta3 = 0`` and ``1/2`` (markers)."""
    e1 = (np.arange(n) + 0.5) / n
    e2 = np.arange(2 * n) / (2 * n)
    E1, E2 = np.meshgrid(e1, e2, indexing="ij")
    return np.concatenate([np.column_stack([E1.ravel(), E2.ravel(), np.full(E1.size, z)]) for z in (0.0, 0.5)], axis=0)


def convergence(regularities=(1, 2, 3, 4), meshes=(4, 6, 8), reference_mesh=12, final_time=0.4, degree=4, verbose=True):
    """Maximum relative L-infinity errors during the final 10% against the finest mesh of the highest regularity."""
    markers = sample_points()
    output_times = np.linspace(0, final_time, 41)

    def sample(field):
        values = field(markers)
        return np.concatenate([np.asarray(v) for v in values])

    def solve(r, n):
        wave = Wave3D(r, n, degree)
        values, energy_error = wave.evolve(final_time, output_times, sample)
        if verbose:
            print(f"  3D wave r = {r}, n = {n}: dt = {wave.dt:.3g}, relative energy error {energy_error:.1e}")
        return values

    reference = solve(max(regularities), reference_mesh)
    scale = np.max(np.abs(reference), axis=1)
    traces, final = {}, {}
    for r in regularities:
        traces[r], final[r] = [], []
        for n in meshes:
            error = np.max(np.abs(solve(r, n) - reference), axis=1) / scale
            traces[r].append(error)
            final[r].append(np.max(error[output_times >= 0.9 * final_time]))
    return dict(form=1, times=output_times, cells=np.array(meshes), traces=traces, final=final, degree=degree)


def figure_cross_section(wave, coefficients=None, eta3=0.0, title=None):
    """The eta3-component of B (physical density of the 2-form) in a cross-section of the tube."""
    field = wave.derham.create_spline_function("b", "Hdiv")
    field.vector = wave.b0 if coefficients is None else coefficients
    e1, e2 = np.linspace(1e-3, 1, 81), np.linspace(0, 1, 161)
    b3 = np.asarray(field(e1, e2, np.array([eta3]))[2])[:, :, 0]
    det = np.asarray(wave.domain.jacobian_det(e1, e2, np.array([eta3])))[:, :, 0]
    X, Y, Z = (np.asarray(c)[:, :, 0] for c in wave.domain(e1, e2, np.array([eta3])))
    R = np.hypot(X, Y)
    values = b3 / det
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    limit = np.max(np.abs(values)) or 1.0
    ax.pcolormesh(R, Z, values, cmap=DIVERGING, vmin=-limit, vmax=limit, shading="gouraud")
    ax.set_aspect("equal")
    clean_axis(ax)
    ax.set_title(title or rf"$B_\zeta$ at $\zeta = {eta3}$, $\mathfrak{{r}} = {wave.regularity}$")
    return fig


def figure_convergence(study):
    hs = 1 / study["cells"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (r, e) in enumerate(study["final"].items()):
        ax.loglog(hs, e, "o-", color=CATEGORICAL[i % len(CATEGORICAL)], label=rf"$\mathfrak{{r}} = {r}$")
    rate = study["degree"]
    e = np.array(study["final"][max(study["final"])])
    ax.loglog(hs, 0.5 * e[-1] * (hs / hs[-1]) ** rate, "k:", label=f"$O(h^{rate})$")
    ax.set_xlabel("$h = 1/n$")
    ax.set_ylabel(r"$\Vert B_h - B_\mathrm{ref}\Vert_{L^\infty} / \Vert B_\mathrm{ref}\Vert_{L^\infty}$")
    ax.set_title("3D 1-form wave: maximum error during the final 10%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
