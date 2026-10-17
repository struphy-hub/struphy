"""Guiding-centre orbits near the magnetic axis and their energy conservation (talk, Section 9).

The vacuum field is ``HelicalVacuumField`` on the cylinder ``HollowCylinder(a1=0, a2=0.3, Lz=2 pi)`` (as
``examples/GuidingCenter/vacuum_near_axis``), projected into the polar spline spaces of a Struphy Derham complex of
regularity ``r``: ``|B|_h = P0 |B|``, its discrete gradient, ``b_h = P1 b`` and ``curl b_h``. The guiding centres follow

    dX/dt = (v_par B* + eps mu b x grad |B|_h) / B*_par,    dv_par/dt = -mu B* . grad |B|_h / B*_par,

``B* = |B|_h b + eps v_par curl b``, ``B*_par = b . B*``, integrated with the classical RK4. The exact flow conserves
``E = v_par^2 / 2 + mu |B|_h``; the relative energy error ``max_t |E(t) - E(0)| / (mu |B|)`` after ``20`` transits
measures the time-integration error, which converges like ``dt^4`` only when the projected field is smooth enough at
the axis. ``angular="trig"`` gives the compatible spaces of the talk (trigonometric functions in ``eta2``), the
default B-splines the B-spline panel. (Struphy's particle pushers, used by the example, need B-splines in every
direction; this integrator evaluates the fields with ``SplineFunction``.)
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy.demos.plotting import CATEGORICAL

A, LZ = 0.3, 2 * np.pi
MU, EPSILON = 1e-3, 0.01
MARKERS = tuple(
    (r / A, angle, 0.0, sign)
    for r, angle in ((0.01, 0.0), (0.02, 0.125), (0.03, 0.25), (0.04, 0.375))
    for sign in (1.0, -1.0)
)


class ProjectedVacuumField:
    """The helical vacuum field projected into the polar splines of regularity ``r`` (B-splines or trigonometric
    functions in ``eta2``), evaluated in Cartesian coordinates."""

    def __init__(self, regularity, angular="bspline", num_elements=(8, 12, 16), degree=(4, 4, 3)):
        from struphy.feec.psydac_derham import Derham
        from struphy.fields_background import equils
        from struphy.geometry import domains
        from struphy.io.options import DerhamOptions
        from struphy.topology.grids import TensorProductGrid

        n2 = num_elements[1] + (1 - num_elements[1] % 2 if angular == "trig" else 0)
        self.domain = domains.HollowCylinder(a1=0.0, a2=A, Lz=LZ)
        options = DerhamOptions(
            degree=degree,
            bcs=(("free", "free"), None, None),
            polar_splines=True,
            polar_regularity=regularity,
            angular_basis=angular,
            nquads_proj=tuple(p + 3 for p in degree),
        )
        grid = TensorProductGrid(
            num_elements=(num_elements[0], n2, num_elements[2]), mpi_dims_mask=(True, False, False)
        )
        self.derham = d = Derham(grid, options, domain=self.domain)
        self.equil = equils.HelicalVacuumField(B0=1.0, delta=0.5, m=2, k=3.0)
        self.equil.domain = self.domain
        self.absB = d.create_spline_function("absB", "H1")
        self.absB.vector = d.P0(self.equil.absB0)
        self.gradB = d.create_spline_function("gradB", "Hcurl")
        self.gradB.vector = d.grad.dot(self.absB.vector)
        self.b = d.create_spline_function("b", "Hcurl")
        self.b.vector = d.P1([lambda e1, e2, e3, i=i: self.equil.unit_b1(e1, e2, e3)[i] for i in range(3)])
        self.curl_b = d.create_spline_function("curl_b", "Hdiv")
        self.curl_b.vector = d.curl.dot(self.b.vector)

    @staticmethod
    def logical(x):
        """Logical coordinates of Cartesian points of the cylinder (markers ``(N, 3)``)."""
        r = np.hypot(x[:, 0], x[:, 1])
        return np.column_stack(
            [np.minimum(r / A, 1.0), (np.arctan2(x[:, 1], x[:, 0]) / (2 * np.pi)) % 1.0, (x[:, 2] / LZ) % 1.0]
        )

    def __call__(self, x):
        """``|B|_h``, ``grad |B|_h``, ``b_h`` and ``curl b_h`` (Cartesian) at Cartesian points ``x`` (N, 3)."""
        eta = self.logical(x)
        # the polar fields are conforming: their physical values at the axis are the limits eta1 -> 0
        eta[:, 0] = np.maximum(eta[:, 0], 1e-9)
        DF = np.asarray(self.domain.jacobian(eta)).reshape(3, 3, -1).transpose(2, 0, 1)
        det = np.linalg.det(DF)
        inv_T = np.linalg.inv(DF).transpose(0, 2, 1)

        def one_form(field):
            return np.einsum("nij,nj->ni", inv_T, np.column_stack([np.asarray(c) for c in field(eta)]))

        def two_form(field):
            return np.einsum("nij,nj->ni", DF, np.column_stack([np.asarray(c) for c in field(eta)])) / det[:, None]

        return np.asarray(self.absB(eta)).ravel(), one_form(self.gradB), one_form(self.b), two_form(self.curl_b)

    def cartesian_markers(self):
        eta = np.array([m[:3] for m in MARKERS])
        x = np.asarray(self.domain(eta)).reshape(3, -1).T
        return x, np.array([m[3] for m in MARKERS])


def rhs(field, x, v):
    B, gradB, b, curl_b = field(x)
    b = b / np.linalg.norm(b, axis=1)[:, None]
    B_star = B[:, None] * b + EPSILON * v[:, None] * curl_b
    B_par = np.einsum("ni,ni->n", b, B_star)
    dx = (v[:, None] * B_star + EPSILON * MU * np.cross(b, gradB)) / B_par[:, None]
    dv = -MU * np.einsum("ni,ni->n", B_star, gradB) / B_par
    return dx, dv, B


def integrate(field, dt, periods=20.0, record_every=10):
    """RK4 for all markers; returns the recorded positions and the relative energy errors."""
    x, v = field.cartesian_markers()
    steps = int(np.ceil(periods * LZ / dt))
    dt = periods * LZ / steps
    _, _, B0 = rhs(field, x, v)
    energy0 = 0.5 * v**2 + MU * B0
    errors = np.zeros(len(v))
    path = [x.copy()]
    for step in range(steps):
        k1x, k1v, B = rhs(field, x, v)
        errors = np.maximum(errors, np.abs(0.5 * v**2 + MU * B - energy0) / (MU * B0))
        k2x, k2v, _ = rhs(field, x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v, _ = rhs(field, x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v, _ = rhs(field, x + dt * k3x, v + dt * k3v)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if (step + 1) % record_every == 0:
            path.append(x.copy())
    _, _, B = rhs(field, x, v)
    errors = np.maximum(errors, np.abs(0.5 * v**2 + MU * B - energy0) / (MU * B0))
    return np.array(path), errors


def energy_study(
    regularities=(0, 1, 2, 3, 4),
    dts=(0.04, 0.02, 0.01, 0.005, 0.0025),
    angular="bspline",
    periods=20.0,
    verbose=True,
    **kwargs,
):
    """Maximum relative energy error over the eight orbits for every regularity and time step."""
    errors = {}
    for r in regularities:
        field = ProjectedVacuumField(r, angular, **kwargs)
        errors[r] = []
        for dt in dts:
            _, e = integrate(field, dt, periods)
            errors[r].append(float(np.max(e)))
            if verbose:
                print(f"  {angular} r = {r}, dt = {dt}: max relative energy error {errors[r][-1]:.2e}")
    return dict(dts=np.array(dts), errors=errors, angular=angular, periods=periods)


def figure_energy(study):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    dts = study["dts"]
    for i, (r, e) in enumerate(study["errors"].items()):
        ax.loglog(dts, e, "o-", color=CATEGORICAL[i % len(CATEGORICAL)], label=rf"$\mathfrak{{r}} = {r}$")
    e = np.array(study["errors"][max(study["errors"])])
    ax.loglog(dts, e[-1] * (dts / dts[-1]) ** 4, "k:", label=r"$O(\Delta t^4)$")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\max_t |\mathcal{E}(t) - \mathcal{E}(0)| / (\mu |B|)$")
    kind = "compatible (trigonometric)" if study["angular"] == "trig" else "B-spline"
    ax.set_title(f"energy after {study['periods']:g} transits, {kind} spaces")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def figure_orbits(path, extent=None):
    """Orbits projected onto the cross-section (``extent``: half width of the close-up)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for i in range(path.shape[1]):
        ax.plot(path[:, i, 0], path[:, i, 1], ".", ms=1, color=CATEGORICAL[i % len(CATEGORICAL)])
    if extent is None:
        ax.add_patch(plt.Circle((0, 0), A, color="k", fill=False))
        extent = A
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("guiding-centre orbits projected onto a cross-section" if extent == A else "close-up near the axis")
    fig.tight_layout()
    return fig
