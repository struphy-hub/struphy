"""A stellarator-like magnetic field in the 3D polar complex of a twisted tube (port of PolarFEEC.jl
``examples/07_torus_3d.jl``, on the Struphy :class:`~struphy.feec.psydac_derham.Derham` with
``polar_splines=True``).

The tube :class:`~struphy.geometry.domains.HelicalTube` ``("islands")`` has a helical magnetic axis with four field
periods and an elliptical cross-section ``a x b = 0.28 x 0.14`` that rotates by half a turn per period. In logical
coordinates ``(r, theta, zeta)`` the field is the closed 2-form

    B = Phi'(r) dr ^ dtheta + dzeta ^ dpsi,    psi = psi_p(r) + sum eps_mn r^m cos 2 pi (m theta - n zeta),

with toroidal flux ``Phi' = 2 pi a b B0 r`` and poloidal flux ``psi_p' = iota Phi'``, ``iota(r) = 0.6 + 0.3 r^2``.
The modes ``(3, 2)`` and ``(5, 4)`` open island chains on the resonant surfaces ``iota = 2/3`` and ``iota = 4/5``.
Since ``dB = 0`` and the polar commuting projection commutes with ``d``, the projected field has ``div B = 0`` in polar
coefficients up to rounding, on the magnetic axis included. Field lines are traced in the projected field and their
winding is compared with ``iota``.

The field lines are integrated in logical coordinates with ``zeta`` as the time (``dr/dzeta = b_1 / b_3``,
``dtheta/dzeta = b_2 / b_3`` for the logical 2-form components) by RK4; PolarFEEC.jl traces them in physical space.
"""

import matplotlib.pyplot as plt
import numpy as np

from struphy.polar.homotopy.demos.plotting import CATEGORICAL

B0 = 1.0
MODES = ((3, 2, 8e-5), (5, 4, 5e-5))


def iota(r):
    return 0.6 + 0.3 * r**2


def build(num_elements=(8, 24, 32), degree=(3, 3, 3), regularity=2, domain_resolution=(8, 32, 128)):
    """The domain and the Derham complex with high-order polar splines."""
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import HelicalTube
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    domain = HelicalTube("islands", num_elements=domain_resolution, degree=(4, 4, 4))
    options = DerhamOptions(
        degree=degree,
        bcs=(("free", "free"), None, None),
        polar_splines=True,
        polar_regularity=regularity,
        nquads_proj=tuple(p + 3 for p in degree),
    )
    return domain, Derham(TensorProductGrid(num_elements=num_elements), options, domain=domain)


def magnetic_field(domain):
    """Logical 2-form components ``(b_theta_zeta, b_zeta_r, b_r_theta)`` of the field."""
    a, b = domain.geometry["a"], domain.geometry["b"]

    def phi_prime(r):
        return 2 * np.pi * a * b * B0 * r

    def components(r, t, z):
        d_theta_psi = np.zeros_like(r * t * z, dtype=float)
        d_r_psi = iota(r) * phi_prime(r) + 0 * t * z
        for m, n, eps in MODES:
            phase = 2 * np.pi * (m * t - n * z)
            d_theta_psi = d_theta_psi - 2 * np.pi * m * eps * r**m * np.sin(phase)
            d_r_psi = d_r_psi + m * eps * r ** (m - 1) * np.cos(phase)
        return -d_theta_psi, d_r_psi, phi_prime(r) + 0 * t * z

    return [lambda r, t, z, i=i: components(r, t, z)[i] for i in range(3)]


def project(derham, domain):
    """The projected field (polar coefficients) and the relative divergence in polar coefficients."""
    b = derham.P2(magnetic_field(domain))
    div = derham.div.dot(b)
    return b, float(np.sqrt(div.inner(div) / b.inner(b)))


def spline_field(derham, b):
    field = derham.create_spline_function("B", "Hdiv")
    field.vector = b
    return field


def trace(field, seeds, turns, steps_per_turn=100):
    """RK4 in zeta for all seeds ``(r, theta, zeta0)``; returns the logical points after every full turn and the
    winding numbers ``Delta theta / Delta zeta``."""
    state = np.array(seeds, dtype=float)
    h = 1.0 / steps_per_turn
    theta_unwrapped = state[:, 1].copy()

    def rhs(s):
        markers = np.column_stack([np.clip(s[:, 0], 1e-8, 1.0), s[:, 1] % 1.0, s[:, 2] % 1.0])
        b1, b2, b3 = field(markers)
        return np.column_stack([b1 / b3, b2 / b3])

    sections = [state[:, :2].copy()]
    for step in range(turns * steps_per_turn):
        k1 = rhs(state)
        s2 = state.copy()
        s2[:, :2] += 0.5 * h * k1
        s2[:, 2] += 0.5 * h
        k2 = rhs(s2)
        s3 = state.copy()
        s3[:, :2] += 0.5 * h * k2
        s3[:, 2] += 0.5 * h
        k3 = rhs(s3)
        s4 = state.copy()
        s4[:, :2] += h * k3
        s4[:, 2] += h
        k4 = rhs(s4)
        increment = h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        state[:, :2] += increment
        theta_unwrapped += increment[:, 1]
        state[:, 2] += h
        # field lines that cross the axis continue on the opposite ray
        crossed = state[:, 0] < 0
        state[crossed, 0] *= -1
        state[crossed, 1] += 0.5
        if (step + 1) % steps_per_turn == 0:
            sections.append(state[:, :2].copy())
    sections = np.array(sections)
    windings = (theta_unwrapped - np.array(seeds)[:, 1]) / turns
    return sections, windings


def resonant_radius(m, n):
    return np.sqrt((n / m - 0.6) / 0.3)


def seeds():
    """Seeds on the ray theta = 0 of the cut zeta = 0 and inside both island chains (as PolarFEEC.jl)."""
    out = [(r, 0.0, 0.0) for r in np.arange(0.08, 0.97, 0.08)]
    for m, n, _ in MODES:
        out += [(resonant_radius(m, n) + d, 0.0, 1 - 1 / (2 * n)) for d in (0.008, 0.018)]
    return out


def study(turns=30, steps_per_turn=100, verbose=True, **kwargs):
    domain, derham = build(**kwargs)
    b, divergence = project(derham, domain)
    field = spline_field(derham, b)
    seed_list = seeds()
    sections, windings = trace(field, seed_list, turns, steps_per_turn)
    if verbose:
        print(f"||div b|| / ||b|| in polar coefficients = {divergence:.1e}")
        for (r, _, z), w in zip(seed_list, windings):
            if z == 0.0 and round(r, 2) in (0.24, 0.40, 0.56):
                print(f"r = {r:.2f}: iota of the traced field line {w:.5f}, exact {iota(r):.5f}")
    return dict(
        domain=domain, derham=derham, b=b, divergence=divergence, seeds=seed_list, sections=sections, windings=windings
    )


def figure_poincare(result):
    """Poincare section at zeta = 0 (in the logical plane r cos 2 pi theta, r sin 2 pi theta) and iota(r)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    sections = result["sections"]
    for i, (r0, _, z0) in enumerate(result["seeds"]):
        r, t = sections[:, i, 0], sections[:, i, 1]
        # sections are taken at zeta = z0 + k; shift the island seeds to zeta = 0 by their phase
        color = CATEGORICAL[i % len(CATEGORICAL)] if z0 == 0.0 else "k"
        axes[0].plot(r * np.cos(2 * np.pi * t), r * np.sin(2 * np.pi * t), ".", ms=1.5, color=color)
    axes[0].set_aspect("equal")
    axes[0].set_title(r"Poincare section (logical $r\cos 2\pi\theta$, $r\sin 2\pi\theta$)")
    rs = np.linspace(0, 1, 101)
    axes[1].plot(rs, iota(rs), "k", label=r"exact $\iota(r) = 0.6 + 0.3 r^2$")
    radii = [s[0] for s in result["seeds"] if s[2] == 0.0]
    windings = [w for s, w in zip(result["seeds"], result["windings"]) if s[2] == 0.0]
    axes[1].plot(radii, windings, "o", color=CATEGORICAL[0], label="traced field lines")
    for m, n, _ in MODES:
        axes[1].axhline(n / m, color="gray", ls=":")
    axes[1].set_xlabel("$r$")
    axes[1].set_ylabel(r"$\iota$")
    axes[1].legend()
    fig.tight_layout()
    return fig
