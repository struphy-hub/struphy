import logging
import pytest
from struphy.utils.pyccel import Pyccelkernel
from struphy import set_logging_level

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)


@pytest.mark.parametrize("num_elements", [[16, 1, 1]])
@pytest.mark.parametrize("degree", [[3, 1, 1]])
@pytest.mark.parametrize(
    "bcs",
    [
        (None, None, None),
    ],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Cuboid",
            {
                "l1": 0.0,
                "r1": 1.0,
                "l2": 0.0,
                "r2": 1.0,
                "l3": 0.0,
                "r3": 1.0,
            },
        ],
        [
            "Cuboid",
            {
                "l1": 0.0,
                "r1": 2.0,
                "l2": 0.0,
                "r2": 3.0,
                "l3": 0.0,
                "r3": 4.0,
            },
        ],
    ],
)
@pytest.mark.parametrize("num_clones", [1, 2])
def test_accum_poisson(num_elements, degree, bcs, mapping, num_clones, Np=1000, show_plot: bool = False):
    r"""Test that AccumulatorVector provides an MC approximation of the L2 projection RHS.

    Particles are loaded with a uniform spatial distribution and unit Maxwellian velocity
    distribution (background density :math:`n_0 = 1`).  After weight initialisation the
    particle weights are rescaled by a sinusoidal spatial perturbation

    .. math::

        n(\boldsymbol{\eta}) = 1 + \tfrac{1}{2}\sin(2\pi\eta_1),

    so that each weight becomes :math:`w_p = n(\boldsymbol{\eta}_p)\,\sqrt{g}\,/\,N_p`.
    With :math:`B^\mu = 1` (``charge_density_0form``), the accumulator then computes

    .. math::

        V^0_{ijk}
        = \sum_{p=0}^{N-1} w_p\,\Lambda^0_{ijk}(\boldsymbol{\eta}_p)
        \;\approx\;
        \int_\Omega \Lambda^0_{ijk}(\boldsymbol{\eta})\,n(\boldsymbol{\eta})\,\sqrt{g}\,
        \mathrm{d}\boldsymbol{\eta}
        \;=\;
        \texttt{L2Projector.get\_dofs}(n)_{ijk}.

    Because :math:`\int_0^1 \sin(2\pi\eta_1)\,\mathrm{d}\eta_1 = 0`, the perturbation
    integrates to zero over the domain, so the sum of all vector entries still equals the
    domain volume :math:`\sqrt{g}`.

    The following assertions are verified (Monte-Carlo errors are :math:`O(1/\sqrt{N_p})`):

    1. **Sum** (partition of unity): :math:`\sum_{ijk} V^0_{ijk} \approx \sqrt{g}`.
    2. **RHS comparison**: the MC vector is close to :func:`~struphy.feec.mass.L2Projector.get_dofs`.
    3. **Projection comparison**: the L2 projection :math:`x_{\rm MC}` obtained by solving
       :math:`M^0\,x = V^0` is close to the exact projection
       :math:`x_{\rm exact} = \texttt{L2Projector}(n)`.

    When ``show_plot=True`` (rank 0 only), both projections are evaluated as
    :class:`~struphy.feec.psydac_derham.SplineFunction` along a 1-D slice at
    :math:`(\eta_2, \eta_3) = (0.5, 0.5)` and compared to the analytical density.
    """

    import cunumpy as xp
    from feectools.ddm.mpi import MockComm
    from feectools.ddm.mpi import mpi as MPI

    from struphy import LoadingParameters, domains, maxwellians, perturbations
    from struphy.feec.mass import L2Projector, WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.accumulation import accum_kernels
    from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
    from struphy.pic.particles import Particles6D
    from struphy.topology.grids import TensorProductGrid
    from struphy.utils.clone_config import CloneConfig

    if isinstance(MPI.COMM_WORLD, MockComm):
        mpi_comm = None
        mpi_rank = 0
    else:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()

    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    params = {
        "grid": {"num_elements": num_elements},
        "kinetic": {"test_particles": {"markers": {"Np": Np, "ppc": Np / xp.prod(num_elements)}}},
    }

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)

    if mpi_comm is None:
        clone_config = None
        derham = Derham(grid, derham_opts, comm=None)
    else:
        if mpi_comm.Get_size() % num_clones == 0:
            clone_config = CloneConfig(comm=mpi_comm, params=params, num_clones=num_clones)
        else:
            return
        derham = Derham(grid, derham_opts, comm=clone_config.sub_comm)

    sub_comm = clone_config.sub_comm if clone_config is not None else None

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    # ------------------------------------------------------------------ #
    # Spatial density: background (n=1) + sinusoidal perturbation.       #
    # ModesSin(ls=(1,)) gives sin(2*pi*eta_1) in logical coordinates.    #
    # Since integral_0^1 sin(2*pi*e1) de1 = 0, the total particle number  #
    # is still sqrt_g (the domain volume), enabling an exact sum check.   #
    # ------------------------------------------------------------------ #
    background = maxwellians.Maxwellian3D(n=(1.0, None))
    perturbation = perturbations.ModesSin(ls=(1,), amps=(0.5,))
    init_maxwellian = maxwellians.Maxwellian3D(n=(1.0, perturbation))

    loading_params = LoadingParameters(
        Np=Np,
        seed=8492,
        moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        spatial="uniform",
    )

    particles = Particles6D(
        comm_world=mpi_comm,
        clone_config=clone_config,
        loading_params=loading_params,
        domain=domain,
        domain_decomp=domain_decomp,
        background=background,
        initial_condition=init_maxwellian,
    )

    particles.draw_markers()
    if mpi_comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()

    if show_plot and mpi_rank == 0:
        components = [False]*6
        components[0] = True  # show only the spatial distribution (ignore velocities)
        bin_edges = [xp.linspace(0.0, 1.0, 50)]
        particles.show_distribution_function(components=components, bin_edges=bin_edges)

    _sqrtg = float(domain.jacobian_det(0.5, 0.5, 0.5, squeeze_out=True))

    logger.info(
        f"rank {mpi_rank}: weights min={float(xp.min(particles.weights)):.6g}, "
        f"max={float(xp.max(particles.weights)):.6g}  "
        f"(expected range [{0.5 * _sqrtg / Np:.6g}, {1.5 * _sqrtg / Np:.6g}])"
    )

    # ------------------------------------------------------------------ #
    # Accumulate the charge-density RHS vector V^0.                       #
    # ------------------------------------------------------------------ #
    mass_ops = WeightedMassOperators(derham, domain)

    acc = AccumulatorVector(
        particles,
        "H1",
        Pyccelkernel(accum_kernels.charge_density_0form),
        mass_ops,
        domain.args_domain,
    )
    
    # particles.update_weights()  # ensure weights are updated before
    acc()

    # ------------------------------------------------------------------ #
    # 1. Sum check: partition of unity + zero integral of the sin term.   #
    # ------------------------------------------------------------------ #
    _sum_mc = xp.empty(1, dtype=float)
    _sum_mc[0] = xp.sum(acc.vectors[0].toarray())
    if sub_comm is not None:
        sub_comm.Allreduce(MPI.IN_PLACE, _sum_mc, op=MPI.SUM)

    logger.info(f"rank {mpi_rank}: sum of MC vector = {float(_sum_mc[0]):.6g}, sqrt_g = {_sqrtg:.6g}")

    assert xp.isclose(_sum_mc[0], _sqrtg, rtol=5e-2), (
        f"Sum of MC vector ({float(_sum_mc[0]):.6g}) should equal the domain volume "
        f"sqrt(g) = {_sqrtg:.6g} (partition of unity + sin perturbation integrates to 0)."
    )

    # ------------------------------------------------------------------ #
    # Exact L2 projection via quadrature.                                 #
    # ------------------------------------------------------------------ #
    l2proj = L2Projector("H1", mass_ops)
    rhs_exact = l2proj.get_dofs(init_maxwellian.n)

    # ------------------------------------------------------------------ #
    # 2. RHS comparison: MC vector vs. exact quadrature RHS.              #
    # ------------------------------------------------------------------ #
    acc_arr = acc.vectors[0].toarray()
    rhs_arr = rhs_exact.toarray()

    _diff_sq = xp.empty(1, dtype=float)
    _rhs_sq = xp.empty(1, dtype=float)
    _diff_sq[0] = float(xp.sum((acc_arr - rhs_arr) ** 2))
    _rhs_sq[0] = float(xp.sum(rhs_arr ** 2))
    if sub_comm is not None:
        sub_comm.Allreduce(MPI.IN_PLACE, _diff_sq, op=MPI.SUM)
        sub_comm.Allreduce(MPI.IN_PLACE, _rhs_sq, op=MPI.SUM)

    rhs_rel_err = float(xp.sqrt(_diff_sq[0] / _rhs_sq[0]))
    mc_order = float(1.0 / xp.sqrt(Np))

    logger.info(
        f"rank {mpi_rank}: RHS relative error = {rhs_rel_err:.4f} "
        f"(expected O(1/sqrt(N_p)) ≈ {mc_order:.4f})"
    )

    assert rhs_rel_err < 0.05, (
        f"MC RHS relative error {rhs_rel_err:.4f} exceeds 10 * O(1/sqrt(N_p)) = "
        f"{10 * mc_order:.4f}.  Increase N_p or check the accumulation kernel."
    )

    # ------------------------------------------------------------------ #
    # 3. Projection comparison: solve M^0 x = V^0 and compare to         #
    #    x_exact = L2Projector(n).                                        #
    # ------------------------------------------------------------------ #
    x_mc = l2proj.solve(acc.vectors[0])
    x_exact = l2proj(init_maxwellian.n)

    x_mc_arr = x_mc.toarray()
    x_exact_arr = x_exact.toarray()

    _proj_diff_sq = xp.empty(1, dtype=float)
    _proj_ref_sq = xp.empty(1, dtype=float)
    _proj_diff_sq[0] = float(xp.sum((x_mc_arr - x_exact_arr) ** 2))
    _proj_ref_sq[0] = float(xp.sum(x_exact_arr ** 2))
    if sub_comm is not None:
        sub_comm.Allreduce(MPI.IN_PLACE, _proj_diff_sq, op=MPI.SUM)
        sub_comm.Allreduce(MPI.IN_PLACE, _proj_ref_sq, op=MPI.SUM)

    proj_rel_err = float(xp.sqrt(_proj_diff_sq[0] / _proj_ref_sq[0]))

    logger.info(
        f"rank {mpi_rank}: projection relative error = {proj_rel_err:.4f} "
        f"(expected O(1/sqrt(N_p)) ≈ {mc_order:.4f})"
    )

    assert proj_rel_err < 0.16, (
        f"MC projection relative error {proj_rel_err:.4f} exceeds 10 * O(1/sqrt(N_p)) = "
        f"{10 * mc_order:.4f}.  Increase N_p or check the accumulation kernel."
    )

    # ------------------------------------------------------------------ #
    # Optional plot (rank 0 only): evaluate SplineFunctions along a 1-D  #
    # slice eta_1 in [0,1] at (eta_2, eta_3) = (0.5, 0.5).              #
    # ------------------------------------------------------------------ #
    if show_plot and mpi_rank == 0:
        import matplotlib.pyplot as plt

        e1_plot = xp.linspace(0.0, 1.0, 200)
        e2_plot = 0.5
        e3_plot = 0.5

        fh_mc = derham.create_spline_function("fh_mc", "H1")
        fh_mc.vector = x_mc
        vals_mc = fh_mc(e1_plot, e2_plot, e3_plot, squeeze_out=True)

        fh_exact = derham.create_spline_function("fh_exact", "H1")
        fh_exact.vector = x_exact
        vals_exact = fh_exact(e1_plot, e2_plot, e3_plot, squeeze_out=True)

        vals_analytic = init_maxwellian.n(e1_plot, xp.full_like(e1_plot, e2_plot), xp.full_like(e1_plot, e3_plot))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.plot(e1_plot, vals_analytic, "k-", lw=1.5, label=r"$n(\eta_1)$ analytic")
        ax.plot(e1_plot, vals_exact, "b--", lw=1.5, label=r"$n_h^{\rm exact}$ (L2Projector)")
        ax.plot(e1_plot, vals_mc, "r:", lw=1.5, label=r"$n_h^{\rm MC}$ (AccumulatorVector)")
        ax.set_xlabel(r"$\eta_1$")
        ax.set_ylabel(r"$n$")
        ax.set_title("L2 projections along the $\\eta_1$-slice")
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.plot(e1_plot, vals_mc - vals_exact, "r-", lw=1.0, label="MC $-$ exact")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel(r"$\eta_1$")
        ax.set_ylabel(r"$n_h^{\rm MC} - n_h^{\rm exact}$")
        ax.set_title(
            f"Pointwise error  (proj. rel. err = {proj_rel_err:.3f},  $N_p = {Np}$)"
        )
        ax.legend(fontsize=9)

        fig.suptitle(
            f"Cuboid {dom_params},  degree = {degree},  "
            f"num_elements = {num_elements},  bcs = {bcs}",
            fontsize=9,
        )
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_accum_poisson(
        [16, 1, 1],
        [3, 1, 1],
        (None, ("free", "free"), None),
        [
            "Cuboid",
            {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0},
        ],
        num_clones=1,
        Np=10000,
        show_plot=True,
    )
