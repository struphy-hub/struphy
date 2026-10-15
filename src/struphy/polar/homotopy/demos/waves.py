"""0- and 1-form wave equations on the perturbed disk (talk, Sections 1, 3 and 5).

The semidiscretization in polar coefficients is (form degree ``k = 0, 1``, ``D_k`` the tensor-product gradient or curl)::

    M_p v'' + K_p v = 0,    M_p = E_k^T M_k E_k,    K_p = (D_k E_k)^T M_{k+1} (D_k E_k),

with Dirichlet conditions (``P = 0``, resp. ``A x n = 0``), integrated by the explicit staggered leapfrog with
``pi = M_p v'`` (one stiffness product and one mass solve per step)::

    pi^{n+1/2} = pi^{n-1/2} - dt K_p v^n,      v^{n+1} = v^n + dt M_p^{-1} pi^{n+1/2},

and each regularity uses its own CFL step ``dt = 0.9 * 2 / sqrt(lambda_max)``. This is the method of the wave script of
PolarFEEC.jl (``examples/polar_1form_wave.jl``, moved to the seminar repository). The observable is ``u`` for 0-forms
and ``curl A`` for 1-forms.

Initial data. The talk uses the Gaussian ``exp(-((x - 0.05)^2 + y^2) / (2 * 0.04^2))`` with zero velocity, for ``P``
and for ``curl A``; the Julia script uses a compactly supported bump. For 1-forms the initial ``A`` is the homotopy
lift ``A = h_2 (I - B) b`` of the projected (zero-mean) 2-form ``b``, so ``curl A = b - B b``.

Not stated in the talk and chosen here: the reference solution of the convergence study (the finest mesh with the
same degree), the sampling of the L-infinity norm (a physical grid) and the final time ``T = 1``.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

from struphy.polar.homotopy import (
    ClampedBSplineSpace,
    DiskMapping,
    PeriodicBSplineSpace,
    PolarComplex2D,
    TrigonometricSpace,
    zernike_basis,
)
from struphy.polar.homotopy.demos.plotting import CATEGORICAL, DIVERGING, clean_axis, draw_mesh
from struphy.polar.homotopy.linalg import as_dense

AMPLITUDE, PETALS = 0.2, 3
CENTER, WIDTH = (0.05, 0.0), 0.04


def perturbed_disk():
    return DiskMapping(radius=lambda r, t: r * (1 + AMPLITUDE * r * np.cos(2 * np.pi * PETALS * t)))


def gaussian(x, y, center=CENTER, width=WIDTH):
    return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * width**2))


def bump(x, y, center=(0.38, 0.0), radius=0.24):
    """Laplacian of the compact bump ``psi = exp(-1/(1 - s))`` of the Julia script (its 0-form initial condition)."""
    dx, dy = x - center[0], y - center[1]
    s = (dx**2 + dy**2) / radius**2
    out = np.zeros_like(s)
    inside = s < 1
    d = 1 - s[inside]
    out[inside] = -4 * (1 - s[inside] - s[inside] ** 2) * np.exp(-1 / d - 4 * np.log(d)) / radius**2
    return out


def angular_space(kind, degree, cells):
    """``"bspline"``: periodic B-splines with ``cells`` cells; ``"trig"``: ``cells`` (odd) real trigonometric functions."""
    if kind == "bspline":
        return PeriodicBSplineSpace(degree, cells)
    if cells % 2 == 0:
        raise ValueError("the number of trigonometric functions must be odd")
    return TrigonometricSpace((cells - 1) // 2)


class WaveProblem:
    """Polar wave semidiscretization of degree ``form`` and regularity ``r``.

    ``mapping`` defines the pushforward; ``space_mapping`` (default: the same) the logical spaces of the polar
    complex (the "non-matching" study of the talk uses the unit disk for the spaces and the perturbed disk for the
    pushforward).
    """

    def __init__(
        self,
        form,
        regularity,
        degree=5,
        radial_cells=8,
        angular="bspline",
        angular_cells=16,
        enrichment="cochain_chain",
        mapping=None,
        space_mapping=None,
        initial=gaussian,
    ):
        self.form, self.regularity = form, regularity
        mapping = perturbed_disk() if mapping is None else mapping
        radial = ClampedBSplineSpace(degree, radial_cells)
        ang = angular_space(angular, degree, angular_cells)
        # the polynomial projections use space_mapping; the physical spaces (masses, observables) use mapping
        construction = mapping if space_mapping is None else space_mapping
        complex_ = PolarComplex2D(
            radial,
            ang,
            construction,
            regularity,
            enrichment=enrichment,
            boundary="dirichlet",
            basis=zernike_basis(regularity, radius=degree / radial_cells),
            check_assumptions=False,
        )
        if space_mapping is not None:
            for space in complex_.spaces:
                space.mapping = mapping
            complex_._cache.clear()
        self.complex = c = complex_
        E = c.extraction(form)
        D = c.spaces[form].derivative_matrix
        self.E, self.DE = E, (D @ E).tocsr()
        M, Mc = c.full_mass_matrix(form), c.full_mass_matrix(form + 1)
        self.mass = ((E.T @ M @ E + (E.T @ M @ E).T) / 2).tocsc()
        self.stiffness = (self.DE.T @ Mc @ self.DE).tocsr()
        self.mass_lu = spla.splu(self.mass)
        lam = spla.eigsh(self.stiffness, k=1, M=self.mass, which="LM", return_eigenvectors=False)[0]
        self.lambda_max = float(lam)
        self.dt = 0.9 * 2 / np.sqrt(self.lambda_max)
        self.initial = self._initial_coefficients(initial)

    def _initial_coefficients(self, initial):
        c = self.complex
        if self.form == 0:
            return c.mass_solver(0).solve(c.extraction(0).T @ c.spaces[0].inner_product(initial, 2))
        b = c.mass_solver(2).solve(c.extraction(2).T @ c.spaces[2].inner_product(initial, 2))
        return as_dense(c.homotopy(2)) @ b if c.boundary == "dirichlet" else c.homotopy(2) @ b

    def observable(self, v):
        """Full coefficients of ``u`` (0-forms) or ``curl A`` (1-forms)."""
        return self.E @ v if self.form == 0 else self.DE @ v

    @property
    def observable_space(self):
        return self.complex.spaces[0 if self.form == 0 else 2]

    def evolve(self, final_time, record_every=1):
        """Staggered leapfrog; returns times, observable snapshots and the relative staggered-energy error."""
        dt = self.dt
        steps = int(np.ceil(final_time / dt))
        v = self.initial.copy()
        force = self.stiffness @ v
        momentum = 0.5 * dt * force  # pi^{-1/2} for zero initial velocity
        times, snapshots = [0.0], [self.observable(v)]
        energies = []
        for step in range(1, steps + 1):
            momentum -= dt * force
            velocity = self.mass_lu.solve(momentum)
            v += dt * velocity
            force = self.stiffness @ v
            energies.append(0.5 * momentum @ velocity + 0.5 * (v @ force - dt * velocity @ force))
            if step % record_every == 0 or step == steps:
                times.append(step * dt)
                snapshots.append(self.observable(v))
        energies = np.array(energies)
        energy_error = np.max(np.abs(energies - energies[0])) / abs(energies[0]) if energies.size else 0.0
        return np.array(times), np.array(snapshots), energy_error


def interpolate(times, snapshots, t):
    i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    w = (t - times[i]) / (times[i + 1] - times[i])
    return (1 - w) * snapshots[i] + w * snapshots[i + 1]


def sample_points(n=121, radius=1.0):
    """A physical sampling grid inside the domain (logical points, uniform in a box around the disk)."""
    mapping = perturbed_disk()
    x = np.linspace(-1.2 * radius, 1.2 * radius, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    logical_x, logical_y = mapping.inverse(X.ravel(), Y.ravel())
    inside = logical_x < 1 - 1e-12
    return logical_x[inside], logical_y[inside]


def sampled(problem, coefficients, points):
    return problem.observable_space.evaluate_points(coefficients, *points)


def run_family(form, regularities=(0, 1, 2, 3, 4, 5), final_time=1.0, output_times=None, verbose=True, **kwargs):
    """Solutions for every regularity, interpolated to common output times."""
    output_times = np.linspace(0, final_time, 101) if output_times is None else output_times
    family = {}
    for r in regularities:
        problem = WaveProblem(form, r, **kwargs)
        times, snapshots, energy_error = problem.evolve(final_time)
        family[r] = dict(
            problem=problem,
            frames=np.array([interpolate(times, snapshots, t) for t in output_times]),
            energy_error=energy_error,
        )
        if verbose:
            print(
                f"  {form}-form r = {r}: {problem.mass.shape[0]} polar modes, dt = {problem.dt:.4g} "
                f"({len(times) - 1} steps), max relative staggered-energy error {energy_error:.2e}"
            )
    return dict(form=form, output_times=output_times, family=family)


def figure_snapshots(result, frames=(0, 50, 100), radial_cells=8):
    """The observable at the first, middle and last output time, one row per regularity."""
    family = result["family"]
    rs, ts = np.linspace(1e-3, 1, 101), np.linspace(0, 1, 241)
    fig, axes = plt.subplots(len(family), len(frames), figsize=(3 * len(frames), 3 * len(family)), squeeze=False)
    reference = family[max(family)]
    for i, (r, entry) in enumerate(family.items()):
        problem = entry["problem"]
        mapping = problem.observable_space.mapping
        for j, frame in enumerate(frames):
            values = problem.observable_space.evaluate(entry["frames"][frame], rs, ts)
            limit = np.max(np.abs(reference["problem"].observable_space.evaluate(reference["frames"][frame], rs, ts)))
            X, Y = mapping(*np.meshgrid(rs, ts, indexing="ij"))
            axes[i, j].pcolormesh(X, Y, values, cmap=DIVERGING, vmin=-limit, vmax=limit, shading="gouraud")
            draw_mesh(
                axes[i, j],
                mapping,
                np.linspace(0, 1, problem.complex.radial.cells + 1),
                np.linspace(0, 1, problem.complex.angular.n + 1),
                alpha=0.2,
            )
            axes[i, j].set_aspect("equal")
            clean_axis(axes[i, j])
            axes[i, j].set_title(rf"$\mathfrak{{r}} = {r}$, $t = {result['output_times'][frame]:.2f}$", fontsize=10)
    name = "u" if result["form"] == 0 else r"\mathrm{curl}\,A"
    fig.suptitle(f"${name}$ of the {result['form']}-form wave equation")
    fig.tight_layout()
    return fig


def convergence(
    form,
    regularities=(0, 1, 2, 3, 4, 5),
    radial_cells=(8, 16, 32),
    reference_cells=64,
    final_time=1.0,
    angular="bspline",
    angular_factor=2,
    degree=5,
    points=None,
    verbose=True,
    **kwargs,
):
    """Relative L-infinity errors against a fine-mesh reference: time traces and the maximum over the final 10%."""
    output_times = np.linspace(0, final_time, 51)
    points = sample_points(81) if points is None else points

    def solve(r, n):
        cells = angular_factor * n if angular == "bspline" else 2 * (angular_factor * n // 2) + 1
        problem = WaveProblem(form, r, degree=degree, radial_cells=n, angular=angular, angular_cells=cells, **kwargs)
        times, snapshots, _ = problem.evolve(final_time)
        frames = np.array([sampled(problem, interpolate(times, snapshots, t), points) for t in output_times])
        return frames

    reference = solve(max(regularities), reference_cells)
    scale = np.max(np.abs(reference), axis=1)
    traces, final = {}, {}
    for r in regularities:
        traces[r], final[r] = [], []
        for n in radial_cells:
            error = np.max(np.abs(solve(r, n) - reference), axis=1) / scale
            traces[r].append(error)
            final[r].append(np.max(error[output_times >= 0.9 * final_time]))
            if verbose:
                print(f"  {form}-form r = {r}, N^x = {n}: max error over the final 10% {final[r][-1]:.3e}")
    return dict(form=form, times=output_times, cells=np.array(radial_cells), traces=traces, final=final)


def figure_convergence(study, degree=5):
    hs = 1 / study["cells"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (r, e) in enumerate(study["final"].items()):
        ax.loglog(hs, e, "o-", color=CATEGORICAL[i % len(CATEGORICAL)], label=rf"$\mathfrak{{r}} = {r}$")
    rate = degree + 1 if study["form"] == 0 else degree
    e = np.array(study["final"][max(study["final"])])
    ax.loglog(hs, 0.5 * e[-1] * (hs / hs[-1]) ** rate, "k:", label=f"$O(h^{rate})$")
    name = "P" if study["form"] == 0 else "B"
    ax.set_xlabel("$h$")
    ax.set_ylabel(
        rf"$\Vert {name}_h - {name}_\mathrm{{ref}}\Vert_{{L^\infty}} / \Vert {name}_\mathrm{{ref}}\Vert_{{L^\infty}}$"
    )
    ax.set_title("maximum error during the final 10%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def figure_time_traces(study, index=0):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (r, traces) in enumerate(study["traces"].items()):
        ax.semilogy(
            study["times"], traces[index], color=CATEGORICAL[i % len(CATEGORICAL)], label=rf"$\mathfrak{{r}} = {r}$"
        )
    name = "P" if study["form"] == 0 else "B"
    ax.set_xlabel("$t$")
    ax.set_ylabel(
        rf"$\Vert {name}_h - {name}_\mathrm{{ref}}\Vert_{{L^\infty}} / \Vert {name}_\mathrm{{ref}}\Vert_{{L^\infty}}$"
    )
    ax.set_title(f"$N^x = {study['cells'][index]}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def animate(result, path, fps=25, regularity=None):
    """Write an animation (gif or mp4) of the observable for one regularity (as the videos of the talk)."""
    from matplotlib.animation import FuncAnimation

    r = max(result["family"]) if regularity is None else regularity
    entry = result["family"][r]
    problem = entry["problem"]
    mapping = problem.observable_space.mapping
    rs, ts = np.linspace(1e-3, 1, 81), np.linspace(0, 1, 181)
    X, Y = mapping(*np.meshgrid(rs, ts, indexing="ij"))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    values = problem.observable_space.evaluate(entry["frames"][0], rs, ts)
    limit = np.max(np.abs(values))
    mesh = ax.pcolormesh(X, Y, values, cmap=DIVERGING, vmin=-limit, vmax=limit, shading="gouraud")
    draw_mesh(
        ax,
        mapping,
        np.linspace(0, 1, problem.complex.radial.cells + 1),
        np.linspace(0, 1, problem.complex.angular.n + 1),
        alpha=0.2,
    )
    ax.set_aspect("equal")
    clean_axis(ax)

    def update(frame):
        values = problem.observable_space.evaluate(entry["frames"][frame], rs, ts)
        limit = np.max(np.abs(values)) or 1.0
        mesh.set_array(values.ravel())
        mesh.set_clim(-limit, limit)
        ax.set_title(rf"$\mathfrak{{r}} = {r}$, $t = {result['output_times'][frame]:.2f}$")
        return (mesh,)

    animation = FuncAnimation(fig, update, frames=len(entry["frames"]))
    animation.save(path, fps=fps)
    plt.close(fig)
    return path
