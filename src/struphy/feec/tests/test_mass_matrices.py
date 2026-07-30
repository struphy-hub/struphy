import logging

import pytest
from matplotlib import pyplot as plt

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("matrix_free", [False])
@pytest.mark.parametrize("num_elements", [(32, 32, 32)])
@pytest.mark.parametrize("degree", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("bcs", [(("free", "dirichlet"), None, None)])
@pytest.mark.parametrize(
    "map_and_equil",
    [
        ("Cuboid", "HomogenSlab"),
        ("Colella", "HomogenSlab"),
        ("HollowCylinder", "ScrewPinch"),
        ("HollowTorus", "AdhocTorus"),
    ],
)
def test_mass(num_elements, degree, bcs, map_and_equil, matrix_free, show_plots=False):
    """Test weighted mass matrices by recovering projected functions from the DeRham complex.

    For each mass operator in ``{M0, M1, M2, M3, Mv, M1n, M2n, Mvn, M1ninv, M0ad, M0ad_withT}``,
    the test:

    1. Projects known trigonometric right-hand-side functions onto the
       corresponding finite-element space using :class:`~struphy.feec.mass.L2Projector`.
    2. Solves the linear system ``M * u = rhs`` with a CG solver.
    3. Evaluates the recovered field ``u`` on a uniform test grid and compares
       it point-wise to the exact function.

    The density-weighted operators (``M1n``, ``M2n``, ``Mvn``, ``M0ad``) are
    tested against ``exact / n0``, ``M0ad_withT`` is tested against ``exact * t0 / n0``, and the inverse-density operator
    (``M1ninv``) is tested against ``exact * n0``.
    """

    from types import MethodType

    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI
    from feectools.linalg.solvers import inverse

    from struphy import domains, equils
    from struphy.feec.mass import L2Projector, WeightedMassOperator, WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.fields_background.projected_equils import ProjectedMHDequilibrium
    from struphy.geometry.base import Domain
    from struphy.geometry.domains import HollowCylinder
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    mpi_comm.Barrier()

    logger.debug(f"Rank {mpi_rank} | Start test_mass with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, map_and_equil[0])
    if map_and_equil[0] == "HollowCylinder":
        R0 = 3.0
        domain: HollowCylinder = domain_class(a1=0.3, Lz=2 * xp.pi * R0)
    else:
        domain: Domain = domain_class()
    logger.debug(f"{domain = }")

    # equilibrium
    equil_class = getattr(equils, map_and_equil[1])
    if map_and_equil[1] == "HomogenSlab":
        equil: equils.HomogenSlab = equil_class(n0=2.0)
    elif map_and_equil[1] == "ScrewPinch":
        equil: equils.ScrewPinch = equil_class(na=0.5, n1=1.0, n2=1.0, R0=R0)
    elif map_and_equil[1] == "AdhocTorus":
        equil: equils.AdhocTorus = equil_class(na=0.4)
    equil.domain = domain
    logger.debug(f"{equil = }")

    if show_plots and False:
        equil.show()

    # derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=mpi_comm, domain=domain)

    logger.debug(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # projected equilibrium for mass matrices with spline weights
    projected_equil = ProjectedMHDequilibrium(equil, derham)

    # mass matrices object
    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil, matrix_free=matrix_free)

    # right-hand side, integrated against the basis functions
    def rhs_0(e1, e2, e3):
        return xp.sin(2 * xp.pi * e1) * xp.cos(4 * xp.pi * e2) * xp.cos(2 * xp.pi * e3)

    def rhs_1(e1, e2, e3):
        return xp.sin(2 * xp.pi * e1) * xp.cos(2 * xp.pi * e2) * xp.cos(2 * xp.pi * e3)

    def rhs_2(e1, e2, e3):
        return xp.zeros_like(e1)

    l2proj_0 = L2Projector("H1", mass_ops)
    l2proj_1 = L2Projector("Hcurl", mass_ops)
    l2proj_2 = L2Projector("Hdiv", mass_ops)
    l2proj_3 = L2Projector("L2", mass_ops)
    l2proj_v = L2Projector("H1vec", mass_ops)

    rhs = {}
    rhs["M0"] = l2proj_0.get_dofs(rhs_0, apply_bc=True)
    rhs["M0ad"] = rhs["M0"]
    rhs["M0ad_withT"] = rhs["M0"]
    rhs["M1"] = l2proj_1.get_dofs((rhs_0, rhs_1, rhs_2), apply_bc=True)
    rhs["M1n"] = rhs["M1"]
    rhs["M1ninv"] = rhs["M1"]
    rhs["M2"] = l2proj_2.get_dofs((rhs_0, rhs_1, rhs_2), apply_bc=True)
    rhs["M2n"] = rhs["M2"]
    rhs["M2B"] = rhs["M2"]
    rhs["M3"] = l2proj_3.get_dofs(rhs_0, apply_bc=True)
    rhs["Mv"] = l2proj_v.get_dofs((rhs_0, rhs_1, rhs_2), apply_bc=True)
    rhs["Mvn"] = rhs["Mv"]
    rhs["WMM"] = rhs["Mv"]
    rhs["WMMnew"] = rhs["Mv"]

    # test mass matrices
    e1 = xp.linspace(0, 1, 8)
    e2 = xp.linspace(0, 1, 16)
    e3 = xp.linspace(0, 1, 12)
    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

    if min(degree) == 1:
        err_bound = 2.0e-1
    elif min(degree) == 2:
        err_bound = 2.6e-2

    names = ["M0", "M1", "M2", "M3", "Mv", "M1n", "M2n", "Mvn", "M1ninv", "M0ad", "M0ad_withT", "WMM", "WMMnew"]
    for name in names:
        if name == "WMM":
            intermediate = mass_ops.WMM
            intermediate.update_weight(projected_equil.n3)
            M: WeightedMassOperator = mass_ops.WMM.massop
        elif name == "WMMnew":
            M: WeightedMassOperator = mass_ops.WMMnew
            logger.debug(f"{M.spline_functions = }")
            M.spline_functions["l2_field"].vector = projected_equil.n3
            M.assemble()
        else:
            M: WeightedMassOperator = getattr(mass_ops, name)
        space_id = M.domain_symbolic_name

        if space_id in ("H1", "L2"):
            exact = rhs_0(ee1, ee2, ee3)
        else:
            exact = xp.array([rhs_0(ee1, ee2, ee3), rhs_1(ee1, ee2, ee3), rhs_2(ee1, ee2, ee3)])

        solver = "cg"
        if name == "M0ad_withT":
            exact *= equil.t0(e1, e2, e3)
        if name in ["M1n", "M2n", "Mvn", "M0ad", "WMM", "WMMnew", "M0ad_withT"]:
            # solve n0 * u = f, where n0 is the equilibrium density
            exact /= equil.n0(e1, e2, e3)
        elif name == "M1ninv":
            # solve u1 / n0 = f1, where n0 is the equilibrium density
            exact *= equil.n0(e1, e2, e3)

        result = derham.create_spline_function("result", space_id)
        Minv = inverse(M, solver, tol=1e-8, maxiter=1000, verbose=False)
        result.vector = Minv.dot(rhs[name])

        result_values = xp.array(result(e1, e2, e3))
        logger.debug(f"{result_values.shape = }")

        if show_plots:
            if space_id in ("H1", "L2"):
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.pcolor(e1, e2, result_values[:, :, 0].T)
                plt.colorbar()
                plt.title(f"{name} with assembled matrix")
                plt.subplot(1, 2, 2)
                plt.pcolor(e1, e2, exact[:, :, 0].T)
                plt.colorbar()
                plt.title("exact")
                plt.show()
            else:
                plt.figure(figsize=(24, 10))
                plt.subplot(2, 3, 1)
                plt.pcolor(e1, e2, result_values[0, :, :, 0].T)
                plt.colorbar()
                plt.title(f"{name} with assembled matrix, component 1")
                plt.subplot(2, 3, 2)
                plt.pcolor(e1, e2, result_values[1, :, :, 0].T)
                plt.colorbar()
                plt.title(f"{name} with assembled matrix, component 2")
                plt.subplot(2, 3, 3)
                plt.pcolor(e1, e2, result_values[2, :, :, 0].T)
                plt.colorbar()
                plt.title(f"{name} with assembled matrix, component 3")
                plt.subplot(2, 3, 4)
                plt.pcolor(e1, e2, exact[0, :, :, 0].T)
                plt.colorbar()
                plt.title("exact, component 1")
                plt.subplot(2, 3, 5)
                plt.pcolor(e1, e2, exact[1, :, :, 0].T)
                plt.colorbar()
                plt.title("exact, component 2")
                plt.subplot(2, 3, 6)
                plt.pcolor(e1, e2, exact[2, :, :, 0].T)
                plt.colorbar()
                plt.title("exact, component 3")
                plt.show()

        err = xp.max(xp.abs(result_values - exact)) / xp.max(xp.abs(exact))
        print(f"{name} relative max-error: {err:.2e}")
        assert err < err_bound, f"{name} relative max-error {err:.2e} exceeds bound of {err_bound:.2e}"
        logger.info(f"Test passed for {name}")


@pytest.mark.parametrize("case", ["1-form", "2-form"])
@pytest.mark.parametrize("matrix_free", [False])
@pytest.mark.parametrize("eps", [1.0])
@pytest.mark.parametrize("num_elements", [(32, 32, 32)])
@pytest.mark.parametrize("degree", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("bcs", [(("free", "dirichlet"), None, None)])
@pytest.mark.parametrize(
    "map_and_equil",
    [
        ("Cuboid", "HomogenSlab"),
        ("Colella", "HomogenSlab"),
        ("HollowCylinder", "ScrewPinch"),
        ("HollowTorus", "AdhocTorus"),
    ],
)
def test_rotation(case, num_elements, degree, bcs, map_and_equil, eps, matrix_free, show_plots=False):
    """Test the rotation-stabilized mass operators on the Hdiv and Hcurl spaces.

    The test verifies that the perp-to-field component of the numerical
    solution matches the analytically derived exact solution for the following
    regularised rotation problems:

    1. B and u as 2-forms: eps * u2 + B2 x u2 = G*f2,

    2. B and u as 1-forms: eps * u1 + B1 x u1 = G^{-1}*sqrt(g)*f1,

    where eps is a regularisation parameter.

    The exact perpendicular solution is computed analytically from the
    right-hand-side trigonometric functions, the local rotation matrix built
    from the equilibrium magnetic 2-form components, and the domain metric
    tensor.  Only the component of the numerical result perpendicular to the
    background magnetic field is compared to the exact solution.
    """

    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI
    from feectools.linalg.solvers import inverse

    from struphy import domains, equils
    from struphy.feec.mass import L2Projector, WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import LocalRotationMatrix
    from struphy.geometry.base import Domain
    from struphy.geometry.domains import Cuboid, HollowCylinder
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    mpi_comm.Barrier()

    logger.debug(f"Rank {mpi_rank} | Start test_mass with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, map_and_equil[0])
    if map_and_equil[0] == "Cuboid":
        domain: Cuboid = domain_class(l1=0.0, r1=10.0, l2=0.0, r2=3.0, l3=0.0, r3=4.0)
    elif map_and_equil[0] == "HollowCylinder":
        R0 = 3.0
        domain: HollowCylinder = domain_class(a1=0.3, Lz=2 * xp.pi * R0)
    else:
        domain: Domain = domain_class()
    logger.debug(f"{domain = }")

    # equilibrium
    equil_class = getattr(equils, map_and_equil[1])
    if map_and_equil[1] == "HomogenSlab":
        equil: equils.HomogenSlab = equil_class(n0=2.0)
    elif map_and_equil[1] == "ScrewPinch":
        equil: equils.ScrewPinch = equil_class(na=0.5, n1=1.0, n2=1.0, R0=R0)
    elif map_and_equil[1] == "AdhocTorus":
        equil: equils.AdhocTorus = equil_class(na=0.4)
    equil.domain = domain
    logger.debug(f"{equil = }")

    if show_plots and False:
        equil.show()

    # derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=mpi_comm)

    logger.debug(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # mass matrices object
    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil, matrix_free=matrix_free)

    # right-hand side, integrated against the basis functions
    def rhs_0(e1, e2, e3):
        return xp.sin(2 * xp.pi * e1) * xp.cos(4 * xp.pi * e2) * xp.cos(2 * xp.pi * e3)

    def rhs_1(e1, e2, e3):
        return xp.sin(2 * xp.pi * e1) * xp.cos(2 * xp.pi * e2) * xp.cos(2 * xp.pi * e3)

    def rhs_2(e1, e2, e3):
        return xp.zeros_like(e1)

    if case == "1-form":
        l2proj = L2Projector("Hcurl", mass_ops)
    elif case == "2-form":
        l2proj = L2Projector("Hdiv", mass_ops)
    rhs = l2proj.get_dofs((rhs_0, rhs_1, rhs_2), apply_bc=True)

    # test mass matrices
    e1 = xp.linspace(0, 1, 8)
    e2 = xp.linspace(0, 1, 16)
    e3 = xp.linspace(0, 1, 12)
    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

    if min(degree) == 1:
        err_bound = 1.15e-1
    elif min(degree) == 2:
        err_bound = 1.4e-2

    # exact solution to the rotation problem u + B x u = rhs
    if case == "1-form":
        rot_B = LocalRotationMatrix(equil.b1_1, equil.b1_2, equil.b1_3)(ee1, ee2, ee3)
    elif case == "2-form":
        rot_B = LocalRotationMatrix(equil.b2_1, equil.b2_2, equil.b2_3)(ee1, ee2, ee3)
    logger.debug(f"{rot_B.shape = }")

    G = domain.metric(ee1, ee2, ee3, change_out_order=True)
    Ginv = domain.metric_inv(ee1, ee2, ee3, change_out_order=True)
    sqrt_g = domain.jacobian_det(ee1, ee2, ee3)
    logger.debug(f"{G.shape = }, {Ginv.shape = }")

    # numpy operates on the last two indices with @
    rhs_mat = xp.array([rhs_0(ee1, ee2, ee3), rhs_1(ee1, ee2, ee3), rhs_2(ee1, ee2, ee3)])
    tmp = xp.transpose(rhs_mat, axes=(1, 2, 3, 0))
    logger.debug(f"{tmp.shape = }")

    if case == "1-form":
        f = xp.matvec(Ginv, tmp)
        f *= sqrt_g[..., xp.newaxis]
        absBsq = equil.b1_1(ee1, ee2, ee3) ** 2 + equil.b1_2(ee1, ee2, ee3) ** 2 + equil.b1_3(ee1, ee2, ee3) ** 2
    elif case == "2-form":
        f = xp.matvec(G, tmp)
        absBsq = equil.b2_1(ee1, ee2, ee3) ** 2 + equil.b2_2(ee1, ee2, ee3) ** 2 + equil.b2_3(ee1, ee2, ee3) ** 2

    logger.debug(f"{xp.min(xp.abs(absBsq)) = }")

    f_rot_B = -xp.transpose(xp.matvec(rot_B, f), axes=(3, 0, 1, 2))
    tmp = -xp.matvec(rot_B, xp.matvec(rot_B, f))
    f_perp = xp.transpose(tmp, axes=(3, 0, 1, 2)) / absBsq

    exact = (f_rot_B + eps * f_perp) / (eps**2 + absBsq)
    logger.debug(f"{exact.shape = }")

    # numerical solution (weak form)
    solver = "gmres"

    if case == "1-form":
        stab = mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            weights=("Identity",),
            name="M1stab_for_rot",
            assemble=True,
        )

        rot_B1 = LocalRotationMatrix(
            equil.b1_1,
            equil.b1_2,
            equil.b1_3,
        )

        M = mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            weights=(rot_B1,),
            name="M1B1",
            assemble=True,
        )

    elif case == "2-form":
        stab = mass_ops.M2stab_for_rot
        M = mass_ops.M2B

    # stabilization and solver
    M += eps * stab
    Minv = inverse(M, solver, tol=1e-7, maxiter=1000, verbose=False)

    if case == "1-form":
        result = derham.create_spline_function("result", "Hcurl")
    elif case == "2-form":
        result = derham.create_spline_function("result", "Hdiv")

    result.vector = Minv.dot(rhs)

    result_values = xp.array(result(e1, e2, e3))
    logger.debug(f"{result_values.shape = }")

    tmp = xp.matvec(rot_B, xp.transpose(result_values, axes=(1, 2, 3, 0)))
    tmp2 = -xp.matvec(rot_B, tmp)
    result_values_perp = xp.transpose(tmp2, axes=(3, 0, 1, 2)) / absBsq
    logger.debug(f"{result_values_perp.shape = }")

    if show_plots:
        plt.figure(figsize=(24, 10))
        plt.subplot(2, 3, 1)
        plt.pcolor(e1, e2, result_values_perp[0, :, :, 0].T)
        plt.colorbar()
        plt.title("solution with assembled matrix, component 1")
        plt.subplot(2, 3, 2)
        plt.pcolor(e1, e2, result_values_perp[1, :, :, 0].T)
        plt.colorbar()
        plt.title("solution with assembled matrix, component 2")
        plt.subplot(2, 3, 3)
        plt.pcolor(e1, e2, result_values_perp[2, :, :, 0].T)
        plt.colorbar()
        plt.title("solution with assembled matrix, component 3")
        plt.subplot(2, 3, 4)
        plt.pcolor(e1, e2, exact[0, :, :, 0].T)
        plt.colorbar()
        plt.title("exact, component 1")
        plt.subplot(2, 3, 5)
        plt.pcolor(e1, e2, exact[1, :, :, 0].T)
        plt.colorbar()
        plt.title("exact, component 2")
        plt.subplot(2, 3, 6)
        plt.pcolor(e1, e2, exact[2, :, :, 0].T)
        plt.colorbar()
        plt.title("exact, component 3")
        plt.show()

    err = xp.max(xp.abs(result_values_perp - exact)) / xp.max(xp.abs(exact))
    print(f"relative max-error: {err:.2e}")
    assert err < err_bound, f"relative max-error {err:.2e} exceeds bound of {err_bound:.2e}"


@pytest.mark.parametrize("num_elements", [(8, 9, 11)])
@pytest.mark.parametrize("degree", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("bcs", [(("free", "dirichlet"), None, None)])
@pytest.mark.parametrize("matrix_free", [False])
@pytest.mark.parametrize(
    "map_and_equil",
    [
        ("Cuboid", "HomogenSlab"),
        ("Colella", "HomogenSlab"),
        ("HollowCylinder", "ScrewPinch"),
        ("HollowTorus", "AdhocTorus"),
    ],
)
def test_identity_mapping_equivalence(num_elements, degree, bcs, matrix_free, map_and_equil):
    """Test whether different choices of basis for the magnetic background yield the same rotation-stabilized mass operator."""

    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI
    from feectools.linalg.solvers import inverse

    from struphy import domains, equils
    from struphy.feec.mass import L2Projector, WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import LocalRotationMatrix
    from struphy.geometry.base import Domain
    from struphy.geometry.domains import Cuboid, HollowCylinder
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    mpi_comm.Barrier()

    logger.debug(f"Rank {mpi_rank} | Start test_mass with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, map_and_equil[0])
    domain: Domain = domain_class()
    logger.debug(f"{domain = }")

    # equilibrium
    equil_class = getattr(equils, map_and_equil[1])
    equil = equil_class()
    equil.domain = domain
    logger.debug(f"{equil = }")

    # derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=mpi_comm)

    logger.debug(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # mass matrices object
    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil, matrix_free=matrix_free)

    # different spaces for B:
    rot_B1 = LocalRotationMatrix(
        equil.b1_1,
        equil.b1_2,
        equil.b1_3,
    )

    rot_B2 = LocalRotationMatrix(
        equil.b2_1,
        equil.b2_2,
        equil.b2_3,
    )

    e = xp.array([0.5])
    ee1, ee2, ee3 = xp.meshgrid(e, e, e, indexing="ij")

    if isinstance(domain, domains.Cuboid):
        assert xp.all(rot_B1(ee1, ee2, ee3) == rot_B2(ee1, ee2, ee3)), (
            "Rotation matrices for B1 and B2 are not equal at the same point."
        )

    M1B1 = mass_ops.create_weighted_mass(
        "Hcurl",
        "Hcurl",
        weights=(rot_B1,),
        name="M1B1",
        assemble=True,
    )

    M1B2 = mass_ops.create_weighted_mass(
        "Hcurl",
        "Hcurl",
        weights=(
            "Ginv",
            rot_B2,
            "Ginv",
            "sqrt_g",
        ),
        name="M1B2",
        assemble=True,
    )

    print(f"{M1B1.toarray().shape = }")
    print(f"{M1B2.toarray().shape = }")

    assert xp.all(xp.isclose(M1B1.toarray(), M1B2.toarray())), "Mass matrices for B1 and B2 are not equal."


@pytest.mark.parametrize("num_elements", [[8, 12, 6]])
@pytest.mark.parametrize("degree", [[2, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (("free", "dirichlet"), None, None),
        (("free", "free"), None, ("free", "free")),
        (("free", "dirichlet"), None, ("free", "dirichlet")),
        (("free", "free"), None, ("dirichlet", "free")),
    ],
)
@pytest.mark.parametrize("mapping", [["IGAPolarCylinder", {"a": 1.0, "Lz": 3.0}]])
def test_mass_polar(num_elements, degree, bcs, mapping, show_plots=False):
    """Compare Struphy polar mass matrices to Struphy-legacy polar mass matrices."""

    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarVector
    from struphy.topology.grids import TensorProductGrid

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        logger.info("")

    mpi_comm.Barrier()

    logger.info(f"Rank {mpi_rank} | Start test_mass_polar with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(
        num_elements=num_elements[:2], degree=degree[:2], a=mapping[1]["a"], Lz=mapping[1]["Lz"]
    )

    if show_plots:
        import matplotlib.pyplot as plt

        domain.show(grid_info=num_elements)

    # load MHD equilibrium
    eq_mhd = ScrewPinch(
        a=mapping[1]["a"], R0=mapping[1]["Lz"], B0=1.0, q0=1.05, q1=1.8, n1=3.0, n2=4.0, na=0.0, beta=0.1,
    )

    if show_plots:
        eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs, polar_splines=True)
    derham = Derham(
        grid,
        derham_opts,
        comm=mpi_comm,
        domain=domain,
    )

    logger.info(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # compare to old STRUPHY
    bc_old = [[None, None], [None, None], [None, None]]
    for i in range(3):
        if bcs[i] is not None:
            for j in range(2):
                if bcs[i][j] == "dirichlet":
                    bc_old[i][j] = "d"
                else:
                    bc_old[i][j] = "f"

    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(derham.V0fem, seed=1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(derham.V1fem, seed=1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(derham.V2fem, seed=8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(derham.V3fem, seed=8196, flattened=True)

    # set polar vectors
    x0_pol_psy = PolarVector(derham.V0pol)
    x1_pol_psy = PolarVector(derham.V1pol)
    x2_pol_psy = PolarVector(derham.V2pol)
    x3_pol_psy = PolarVector(derham.V3pol)

    x0_pol_psy.tp = x0_psy
    x1_pol_psy.tp = x1_psy
    x2_pol_psy.tp = x2_psy
    x3_pol_psy.tp = x3_psy

    xp.random.seed(1607)
    x0_pol_psy.pol = [xp.random.rand(x0_pol_psy.pol[0].shape[0], x0_pol_psy.pol[0].shape[1])]
    x1_pol_psy.pol = [xp.random.rand(x1_pol_psy.pol[n].shape[0], x1_pol_psy.pol[n].shape[1]) for n in range(3)]
    x2_pol_psy.pol = [xp.random.rand(x2_pol_psy.pol[n].shape[0], x2_pol_psy.pol[n].shape[1]) for n in range(3)]
    x3_pol_psy.pol = [xp.random.rand(x3_pol_psy.pol[0].shape[0], x3_pol_psy.pol[0].shape[1])]

    # apply boundary conditions to old STRUPHY
    x0_pol_str = x0_pol_psy.toarray(True)
    x1_pol_str = x1_pol_psy.toarray(True)
    x2_pol_str = x2_pol_psy.toarray(True)
    x3_pol_str = x3_pol_psy.toarray(True)

    r0_pol_psy = mass_mats.M0.dot(x0_pol_psy, apply_bc=True)
    r1_pol_psy = mass_mats.M1.dot(x1_pol_psy, apply_bc=True)
    r2_pol_psy = mass_mats.M2.dot(x2_pol_psy, apply_bc=True)
    r3_pol_psy = mass_mats.M3.dot(x3_pol_psy, apply_bc=True)

    rn_pol_psy = mass_mats.M2n.dot(x2_pol_psy, apply_bc=True)
    rJ_pol_psy = mass_mats.M2J.dot(x2_pol_psy, apply_bc=True)

    # perfrom matrix-vector products (without boundary conditions)
    r0_pol_psy = mass_mats.M0.dot(x0_pol_psy, apply_bc=False)
    r1_pol_psy = mass_mats.M1.dot(x1_pol_psy, apply_bc=False)
    r2_pol_psy = mass_mats.M2.dot(x2_pol_psy, apply_bc=False)
    r3_pol_psy = mass_mats.M3.dot(x3_pol_psy, apply_bc=False)

    logger.info(f"Rank {mpi_rank} | All tests passed!")


@pytest.mark.parametrize("num_elements", [[8, 12, 6]])
@pytest.mark.parametrize("degree", [[2, 3, 2]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (("free", "dirichlet"), None, None),
        (("free", "free"), None, ("free", "free")),
        (("free", "dirichlet"), None, ("free", "dirichlet")),
        (("free", "free"), None, ("dirichlet", "free")),
    ],
)
@pytest.mark.parametrize("mapping", [["HollowCylinder", {"a1": 0.1, "a2": 1.0, "Lz": 18.84955592153876}]])
def test_mass_preconditioner(num_elements, degree, bcs, mapping, show_plots=False):
    """Compare mass matrix-vector products with Kronecker products of preconditioner,
    check PC * M = Id and test PCs in solve."""

    import time

    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI
    from feectools.linalg.solvers import inverse

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch, ShearedSlab
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        logger.info("")

    mpi_comm.Barrier()

    logger.info(f"Rank {mpi_rank} | Start test_mass_preconditioner with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    if show_plots:
        import matplotlib.pyplot as plt

        domain.show()

    # load MHD equilibrium
    if mapping[0] == "Cuboid":
        eq_mhd = ShearedSlab(
            a=(mapping[1]["r1"] - mapping[1]["l1"]), R0=(mapping[1]["r3"] - mapping[1]["l3"]) / (2 * xp.pi), B0=1.0, q0=1.05, q1=1.8, n1=3.0, n2=4.0, na=0.0, beta=0.1,
        )

    elif mapping[0] == "Colella":
        eq_mhd = ShearedSlab(
            a=mapping[1]["Lx"], R0=mapping[1]["Lz"] / (2 * xp.pi), B0=1.0, q0=1.05, q1=1.8, n1=3.0, n2=4.0, na=0.0, beta=0.1,
        )

        if show_plots:
            eq_mhd.plot_profiles()

    elif mapping[0] == "HollowCylinder":
        eq_mhd = ScrewPinch(
            a=mapping[1]["a2"], R0=3.0, B0=1.0, q0=1.05, q1=1.8, n1=3.0, n2=4.0, na=0.0, beta=0.1,
        )

        if show_plots:
            eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=mpi_comm)

    fem_spaces = [derham.V0fem, derham.V1fem, derham.V2fem, derham.V3fem, derham.Vvfem]

    logger.info(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # exact mass matrices
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # assemble preconditioners
    if mpi_rank == 0:
        logger.info("Start assembling preconditioners")

    M0pre = MassMatrixPreconditioner(mass_mats.M0)
    M1pre = MassMatrixPreconditioner(mass_mats.M1)
    M2pre = MassMatrixPreconditioner(mass_mats.M2)
    M3pre = MassMatrixPreconditioner(mass_mats.M3)
    Mvpre = MassMatrixPreconditioner(mass_mats.Mv)

    M1npre = MassMatrixPreconditioner(mass_mats.M1n)
    M2npre = MassMatrixPreconditioner(mass_mats.M2n)
    Mvnpre = MassMatrixPreconditioner(mass_mats.Mvn)

    M1Bninvpre = MassMatrixPreconditioner(mass_mats.M1Bninv)

    if mpi_rank == 0:
        logger.info("Done")

    # create random input arrays
    x0 = create_equal_random_arrays(fem_spaces[0], seed=1234, flattened=True)[1]
    x1 = create_equal_random_arrays(fem_spaces[1], seed=1568, flattened=True)[1]
    x2 = create_equal_random_arrays(fem_spaces[2], seed=8945, flattened=True)[1]
    x3 = create_equal_random_arrays(fem_spaces[3], seed=8196, flattened=True)[1]
    xv = create_equal_random_arrays(fem_spaces[4], seed=2038, flattened=True)[1]

    # compare mass matrix-vector products with Kronecker products of preconditioner
    do_this_test = False

    if (mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder") and do_this_test:
        if mpi_rank == 0:
            logger.info("Start matrix-vector products in stencil format for mapping Cuboid/HollowCylinder")

        r0 = mass_mats.M0.dot(x0)
        r1 = mass_mats.M1.dot(x1)
        r2 = mass_mats.M2.dot(x2)
        r3 = mass_mats.M3.dot(x3)
        rv = mass_mats.Mv.dot(xv)

        r1n = mass_mats.M1n.dot(x1)
        r2n = mass_mats.M2n.dot(x2)
        rvn = mass_mats.Mvn.dot(xv)

        r1Bninv = mass_mats.M1Bninv.dot(x1)

        if mpi_rank == 0:
            logger.info("Done")

        if mpi_rank == 0:
            logger.info("Start matrix-vector products in KroneckerStencil format for mapping Cuboid/HollowCylinder")

        r0_pre = M0pre.matrix.dot(x0)
        r1_pre = M1pre.matrix.dot(x1)
        r2_pre = M2pre.matrix.dot(x2)
        r3_pre = M3pre.matrix.dot(x3)
        rv_pre = Mvpre.matrix.dot(xv)

        r1n_pre = M1npre.matrix.dot(x1)
        r2n_pre = M2npre.matrix.dot(x2)
        rvn_pre = Mvnpre.matrix.dot(xv)

        r1Bninv_pre = M1Bninvpre.matrix.dot(x1)

        if mpi_rank == 0:
            logger.info("Done")

        # compare output arrays
        assert xp.allclose(r0.toarray(), r0_pre.toarray())
        assert xp.allclose(r1.toarray(), r1_pre.toarray())
        assert xp.allclose(r2.toarray(), r2_pre.toarray())
        assert xp.allclose(r3.toarray(), r3_pre.toarray())
        assert xp.allclose(rv.toarray(), rv_pre.toarray())

        assert xp.allclose(r1n.toarray(), r1n_pre.toarray())
        assert xp.allclose(r2n.toarray(), r2n_pre.toarray())
        assert xp.allclose(rvn.toarray(), rvn_pre.toarray())

        assert xp.allclose(r1Bninv.toarray(), r1Bninv_pre.toarray())

    # test if preconditioner satisfies PC * M = Identity
    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert xp.allclose(mass_mats.M0.dot(M0pre.solve(x0)).toarray(), derham.boundary_ops["0"].dot(x0).toarray())
        assert xp.allclose(mass_mats.M1.dot(M1pre.solve(x1)).toarray(), derham.boundary_ops["1"].dot(x1).toarray())
        assert xp.allclose(mass_mats.M2.dot(M2pre.solve(x2)).toarray(), derham.boundary_ops["2"].dot(x2).toarray())
        assert xp.allclose(mass_mats.M3.dot(M3pre.solve(x3)).toarray(), derham.boundary_ops["3"].dot(x3).toarray())
        assert xp.allclose(mass_mats.Mv.dot(Mvpre.solve(xv)).toarray(), derham.boundary_ops["v"].dot(xv).toarray())

    # test preconditioner in iterative solver
    M0inv = inverse(mass_mats.M0, "pcg", pc=M0pre, tol=1e-8, maxiter=1000)
    M1inv = inverse(mass_mats.M1, "pcg", pc=M1pre, tol=1e-8, maxiter=1000)
    M2inv = inverse(mass_mats.M2, "pcg", pc=M2pre, tol=1e-8, maxiter=1000)
    M3inv = inverse(mass_mats.M3, "pcg", pc=M3pre, tol=1e-8, maxiter=1000)
    Mvinv = inverse(mass_mats.Mv, "pcg", pc=Mvpre, tol=1e-8, maxiter=1000)

    M1ninv = inverse(mass_mats.M1n, "pcg", pc=M1npre, tol=1e-8, maxiter=1000)
    M2ninv = inverse(mass_mats.M2n, "pcg", pc=M2npre, tol=1e-8, maxiter=1000)
    Mvninv = inverse(mass_mats.Mvn, "pcg", pc=Mvnpre, tol=1e-8, maxiter=1000)

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M0 with preconditioner")
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0))
    else:
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M0inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M1 with preconditioner")
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1))
    else:
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M1inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M2 with preconditioner")
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2))
    else:
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M2inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M3 with preconditioner")
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3))
    else:
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M3inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert Mv with preconditioner")
        rv = Mvinv.dot(derham.boundary_ops["v"].dot(xv))
    else:
        rv = Mvinv.dot(derham.boundary_ops["v"].dot(xv))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert Mvinv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Apply M1n with preconditioner")
        r1n = M1ninv.dot(derham.boundary_ops["1"].dot(x1))
    else:
        r1n = M1ninv.dot(derham.boundary_ops["1"].dot(x1))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M1ninv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Apply M2n with preconditioner")
        r2n = M2ninv.dot(derham.boundary_ops["2"].dot(x2))
    else:
        r2n = M2ninv.dot(derham.boundary_ops["2"].dot(x2))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M2ninv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Apply Mvn with preconditioner")
        rvn = Mvninv.dot(derham.boundary_ops["v"].dot(xv))
    else:
        rvn = Mvninv.dot(derham.boundary_ops["v"].dot(xv))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert Mvninv._info["niter"] == 2

    time.sleep(2)
    logger.info(f"Rank {mpi_rank} | All tests passed!")


@pytest.mark.parametrize("num_elements", [[8, 9, 6]])
@pytest.mark.parametrize("degree", [[2, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (("free", "dirichlet"), None, None),
        (("free", "free"), None, ("free", "free")),
        (("free", "dirichlet"), None, ("free", "dirichlet")),
        (("free", "free"), None, ("dirichlet", "free")),
    ],
)
@pytest.mark.parametrize("mapping", [["IGAPolarCylinder", {"a": 1.0, "Lz": 3.0}]])
def test_mass_preconditioner_polar(num_elements, degree, bcs, mapping, show_plots=False):
    """Compare polar mass matrix-vector products with Kronecker products of preconditioner,
    check PC * M = Id and test PCs in solve."""

    import time

    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI
    from feectools.linalg.solvers import inverse

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarVector
    from struphy.topology.grids import TensorProductGrid

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        logger.info("")

    mpi_comm.Barrier()

    logger.info(f"Rank {mpi_rank} | Start test_mass_preconditioner_polar with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(
        num_elements=num_elements[:2], degree=degree[:2], a=mapping[1]["a"], Lz=mapping[1]["Lz"]
    )

    if show_plots:
        import matplotlib.pyplot as plt

        domain.show()

    # load MHD equilibrium
    eq_mhd = ScrewPinch(
        a=mapping[1]["a"], R0=mapping[1]["Lz"], B0=1.0, q0=1.05, q1=1.8, n1=3.0, n2=4.0, na=0.0, beta=0.1,
    )

    if show_plots:
        eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # derham object
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs, polar_splines=True)
    derham = Derham(
        grid,
        derham_opts,
        comm=mpi_comm,
        domain=domain,
    )

    logger.info(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # exact mass matrices
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # preconditioners
    if mpi_rank == 0:
        logger.info("Start assembling preconditioners")

    M0pre = MassMatrixPreconditioner(mass_mats.M0)
    M1pre = MassMatrixPreconditioner(mass_mats.M1)
    M2pre = MassMatrixPreconditioner(mass_mats.M2)
    M3pre = MassMatrixPreconditioner(mass_mats.M3)

    M1npre = MassMatrixPreconditioner(mass_mats.M1n)
    M2npre = MassMatrixPreconditioner(mass_mats.M2n)

    if mpi_rank == 0:
        logger.info("Done")

    # create random input arrays
    x0 = create_equal_random_arrays(derham.V0fem, seed=1234, flattened=True)[1]
    x1 = create_equal_random_arrays(derham.V1fem, seed=1568, flattened=True)[1]
    x2 = create_equal_random_arrays(derham.V2fem, seed=8945, flattened=True)[1]
    x3 = create_equal_random_arrays(derham.V3fem, seed=8196, flattened=True)[1]

    # set polar vectors
    x0_pol = PolarVector(derham.V0pol)
    x1_pol = PolarVector(derham.V1pol)
    x2_pol = PolarVector(derham.V2pol)
    x3_pol = PolarVector(derham.V3pol)

    x0_pol.tp = x0
    x1_pol.tp = x1
    x2_pol.tp = x2
    x3_pol.tp = x3

    xp.random.seed(1607)
    x0_pol.pol = [xp.random.rand(x0_pol.pol[0].shape[0], x0_pol.pol[0].shape[1])]
    x1_pol.pol = [xp.random.rand(x1_pol.pol[n].shape[0], x1_pol.pol[n].shape[1]) for n in range(3)]
    x2_pol.pol = [xp.random.rand(x2_pol.pol[n].shape[0], x2_pol.pol[n].shape[1]) for n in range(3)]
    x3_pol.pol = [xp.random.rand(x3_pol.pol[0].shape[0], x3_pol.pol[0].shape[1])]

    # test preconditioner in iterative solver and compare to case without preconditioner
    M0inv = inverse(mass_mats.M0, "pcg", pc=M0pre, tol=1e-8, maxiter=500)
    M1inv = inverse(mass_mats.M1, "pcg", pc=M1pre, tol=1e-8, maxiter=500)
    M2inv = inverse(mass_mats.M2, "pcg", pc=M2pre, tol=1e-8, maxiter=500)
    M3inv = inverse(mass_mats.M3, "pcg", pc=M3pre, tol=1e-8, maxiter=500)

    M1ninv = inverse(mass_mats.M1n, "pcg", pc=M1npre, tol=1e-8, maxiter=500)
    M2ninv = inverse(mass_mats.M2n, "pcg", pc=M2npre, tol=1e-8, maxiter=500)

    M0inv_nopc = inverse(mass_mats.M0, "pcg", pc=None, tol=1e-8, maxiter=500)
    M1inv_nopc = inverse(mass_mats.M1, "pcg", pc=None, tol=1e-8, maxiter=500)
    M2inv_nopc = inverse(mass_mats.M2, "pcg", pc=None, tol=1e-8, maxiter=500)
    M3inv_nopc = inverse(mass_mats.M3, "pcg", pc=None, tol=1e-8, maxiter=500)

    M1ninv_nopc = inverse(mass_mats.M1n, "pcg", pc=None, tol=1e-8, maxiter=500)
    M2ninv_nopc = inverse(mass_mats.M2n, "pcg", pc=None, tol=1e-8, maxiter=500)

    # =============== M0 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M0 with preconditioner")
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0_pol))
        logger.info(f"Number of iterations : {M0inv._info['niter']}")
    else:
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0_pol))

    assert M0inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M0 without preconditioner")
        r0 = M0inv_nopc.dot(derham.boundary_ops["0"].dot(x0_pol))
        logger.info(f"Number of iterations : {M0inv_nopc._info['niter']}")
    else:
        r0 = M0inv_nopc.dot(derham.boundary_ops["0"].dot(x0_pol))

    assert M0inv._info["niter"] < M0inv_nopc._info["niter"]
    # =======================================================

    # =============== M1 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M1 with preconditioner")
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1_pol))
        logger.info(f"Number of iterations : {M1inv._info['niter']}")
    else:
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M1 without preconditioner")
        r1 = M1inv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))
        logger.info(f"Number of iterations : {M1inv_nopc._info['niter']}")
    else:
        r1 = M1inv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1inv._info["niter"] < M1inv_nopc._info["niter"]
    # =======================================================

    # =============== M2 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M2 with preconditioner")
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2_pol))
        logger.info(f"Number of iterations : {M2inv._info['niter']}")
    else:
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M2 without preconditioner")
        r2 = M2inv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))
        logger.info(f"Number of iterations : {M2inv_nopc._info['niter']}")
    else:
        r2 = M2inv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2inv._info["niter"] < M2inv_nopc._info["niter"]
    # =======================================================

    # =============== M3 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M3 with preconditioner")
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3_pol))
        logger.info(f"Number of iterations : {M3inv._info['niter']}")
    else:
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3_pol))

    assert M3inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M3 without preconditioner")
        r3 = M3inv_nopc.dot(derham.boundary_ops["3"].dot(x3_pol))
        logger.info(f"Number of iterations : {M3inv_nopc._info['niter']}")
    else:
        r3 = M3inv_nopc.dot(derham.boundary_ops["3"].dot(x3_pol))

    assert M3inv._info["niter"] < M3inv_nopc._info["niter"]
    # =======================================================

    # =============== M1n ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M1n with preconditioner")
        r1 = M1ninv.dot(derham.boundary_ops["1"].dot(x1_pol))
        logger.info(f"Number of iterations : {M1ninv._info['niter']}")
    else:
        r1 = M1ninv.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1ninv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M1n without preconditioner")
        r1 = M1ninv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))
        logger.info(f"Number of iterations : {M1ninv_nopc._info['niter']}")
    else:
        r1 = M1ninv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1ninv._info["niter"] < M1ninv_nopc._info["niter"]
    # =======================================================

    # =============== M2n ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M2n with preconditioner")
        r2 = M2ninv.dot(derham.boundary_ops["2"].dot(x2_pol))
        logger.info(f"Number of iterations : {M2ninv._info['niter']}")
    else:
        r2 = M2ninv.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2ninv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        logger.info("Invert M2n without preconditioner")
        r2 = M2ninv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))
        logger.info(f"Number of iterations : {M2ninv_nopc._info['niter']}")
    else:
        r2 = M2ninv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2ninv._info["niter"] < M2ninv_nopc._info["niter"]
    # =======================================================

    time.sleep(2)
    logger.info(f"Rank {mpi_rank} | All tests passed!")


@pytest.mark.parametrize("num_elements", [[12, 13, 14]])
@pytest.mark.parametrize("mpi_mask", [(False, False, True), (True, False, True)])
@pytest.mark.parametrize("degree", [[2, 2, 3], [1, 1, 1], [1, 4, 2]])
@pytest.mark.parametrize(
    "bcs",
    [
        (None, None, None),
        (("free", "dirichlet"), None, None),
        (None, ("dirichlet", "dirichlet"), ("dirichlet", "dirichlet")),
    ],
)
def test_average_operator(num_elements, mpi_mask, degree, bcs, show_plots=False):
    """This function tests the AverageOperator implementation by comparing the result
    from the AverageOperator/StencilVector multiplication and a basic averaging with np.mean.
    The vector averaged with numpy is produced by evaluating a FEECVariable that contains the StencilVector."""
    import cunumpy as xp
    import matplotlib.pyplot as plt
    from feectools.ddm.mpi import mpi as MPI

    from struphy import DerhamOptions, domains, grids
    from struphy.feec.mass import AverageOperator
    from struphy.feec.psydac_derham import Derham
    from struphy.models.variables import FEECVariable

    derham_opt = DerhamOptions(degree, bcs)
    domain = domains.Cuboid()
    grid = grids.TensorProductGrid(num_elements, mpi_mask)
    comm = MPI.COMM_WORLD
    derham = Derham(grid, derham_opt, comm=(comm if comm.Get_size() > 1 else None), domain=domain)

    n_points = 100
    linspace = xp.linspace(0, 1, n_points)
    grid_eval = xp.meshgrid(linspace, linspace, linspace, indexing="ij")

    var = FEECVariable("H1")
    var.allocate(derham, domain)
    v = derham.V0.zeros()
    v._data = xp.random.random(v._data.shape)
    v.update_ghost_regions()
    var.spline.vector = v
    var_eval = var.spline(*grid_eval)

    var_out = FEECVariable("H1")
    var_out.allocate(derham, domain)

    for dir in range(3):
        av_op = AverageOperator(derham, "H1", dir)

        out = av_op.dot(v)
        sl_out = tuple([(slice(None) if i != dir else 0) for i in range(3)])

        var_out.spline.vector = out

        var_out_eval = var_out.spline(*grid_eval)
        var_averaged = var_eval.mean(axis=dir)

        max_diff = xp.max(xp.abs(var_out_eval[sl_out] - var_averaged))
        logger.info(f"{max_diff=} for direction={dir}")
        assert max_diff < 0.03

        if show_plots:
            e1, e2 = grid_eval[(dir + 1) % 3][sl_out], grid_eval[(dir + 2) % 3][sl_out]
            xlabel = 1 if dir == 0 else 0
            ylabel = 1 if dir == 2 else 2
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.pcolor(e1, e2, var_averaged)
            plt.colorbar()
            plt.title("simple average using np.mean")
            plt.xlabel("eta_" + str(xlabel))
            plt.ylabel("eta_" + str(ylabel))
            plt.subplot(1, 2, 2)
            plt.pcolor(e1, e2, var_out_eval[sl_out])
            plt.colorbar()
            plt.title("average using linear operator")
            plt.xlabel("eta_" + str(xlabel))
            plt.ylabel("eta_" + str(ylabel))
            plt.show()


if __name__ == "__main__":
    # test_mass(
    #    num_elements=(32, 32, 32),
    #    degree=(1, 1, 1),
    #    bcs=(("dirichlet", "dirichlet"), None, None),
    #    # bcs=(None, None, None),
    #    map_and_equil=("Cuboid", "HomogenSlab"),
    #    # map_and_equil=("Colella", "HomogenSlab"),
    #    # map_and_equil=("HollowCylinder", "ScrewPinch"),
    #    # map_and_equil=("HollowTorus", "AdhocTorus"),
    #    matrix_free=False,
    #    show_plots=True,
    # )
    test_rotation(
        case="1-form",
        num_elements=(32, 32, 32),
        degree=(1, 1, 1),
        bcs=(("dirichlet", "dirichlet"), None, None),
        # bcs=(None, None, None),
        # map_and_equil=("Cuboid", "HomogenSlab"),
        # map_and_equil=("Colella", "HomogenSlab"),
        # map_and_equil=("HollowCylinder", "ScrewPinch"),
        map_and_equil=("HollowTorus", "AdhocTorus"),
        eps=1.0,
        matrix_free=False,
        show_plots=True,
    )
    # test_average_operator(
    #     num_elements=(12, 13, 14),
    #     mpi_mask=(True, True, True),
    #     degree=(2, 3, 4),
    #     bcs=(("dirichlet", "dirichlet"), None, ("dirichlet", "dirichlet")),
    #     show_plots=True,
    # test_identity_mapping_equivalence(
    #     num_elements=(3, 1, 1),
    #     degree=(1, 1, 1),
    #     bcs=(None, None, None),
    #     # bcs=(("dirichlet", "dirichlet"), None, None),
    # )
