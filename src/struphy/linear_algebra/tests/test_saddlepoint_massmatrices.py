import pytest


@pytest.mark.parametrize(
    "method_for_solving,Nel,p,spl_kind,dirichlet_bc,mapping",
    [
        (
            "SaddlePointSolverUzawaNumpy",
            [15, 15, 1],
            [3, 3, 1],
            [True, False, True],
            [[False, False], [False, False], [False, False]],
            ["Cuboid", {"l1": 0.0, "r1": 2.0, "l2": 0.0, "r2": 3.0, "l3": 0.0, "r3": 6.0}],
        )
    ],
)
def test_saddlepointsolver(method_for_solving, Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Test saddle-point-solver with manufactured solutions."""

    import time

    import numpy as np
    import scipy as sc
    from mpi4py import MPI
    from psydac.linalg.basic import IdentityOperator
    from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace

    from struphy.feec.basis_projection_ops import BasisProjectionOperators, BasisProjectionOperatorLocal
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.fields_background.equils import FluxAlignedTokamak, HomogenSlab
    from struphy.geometry import domains
    from struphy.linear_algebra.saddle_point import (
        SaddlePointSolverGMRES,
        SaddlePointSolverGMRESwithPC,
        SaddlePointSolverUzawaNumpy,
    )

    from struphy.initial import perturbations
    from struphy.feec.projectors import L2Projector
    from struphy.feec.psydac_derham import TransformedPformComponent

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    mpi_comm.Barrier()

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, dirichlet_bc=dirichlet_bc, local_projectors=True)
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])
    fem_spaces = [derham.Vh_fem["0"], derham.Vh_fem["1"], derham.Vh_fem["2"], derham.Vh_fem["3"], derham.Vh_fem["v"]]
    derhamnumpy = Derham(Nel, p, spl_kind, domain=domain)

    # Mhd equilibirum (slab)
    mhd_equil_params = {"B0x": 0.0, "B0y": 0.0, "B0z": 1.0, "beta": 2.0, "n0": 1.0}
    eq_mhd = HomogenSlab(**mhd_equil_params)

    # mhd_equil_params = {'a': 1.45, 'R0': 6.5, 'q_kind': 1, 'p_kind': 0}

    # eq_mhd = AdhocTorus(**mhd_equil_params)
    eq_mhd.domain = domain

    # create random input array
    x1_rdm_block, x1_rdm = create_equal_random_arrays(fem_spaces[1], seed=1568, flattened=False)
    x2_rdm_block, x2_rdm = create_equal_random_arrays(fem_spaces[1], seed=68, flattened=False)
    y1_rdm_block, y1_rdm = create_equal_random_arrays(fem_spaces[3], seed=1268, flattened=False)

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    hodge_mats = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)
    # mass_old_fortesting = WeightedMassOperatorsOldForTesting(derham, domain, eq_mhd=eq_mhd)

    fun = []
    for m in range(3):
        fun += [[]]
        for n in range(3):
            fun[-1] += [
                lambda e1, e2, e3, m=m, n=n: hodge_mats.G(e1, e2, e3)[:, :, :, m, n] / hodge_mats.sqrt_g(e1, e2, e3),
            ]

    S21 = BasisProjectionOperatorLocal(derham._Ploc["1"], derham.Vh_fem["2"], fun, transposed=False)
    Hodge = hodge_mats.S21
    M2R = mass_mats.M2B
    M2 = mass_mats.M2
    C = derham.curl
    D = derham.div
    G = derham.grad  # _bcfree
    M3 = mass_mats.M3
    B0 = 1.0
    nue = 0.01 * 100
    nu = 1.0
    dt = 0.001
    eps = 1e-6
    eps2 = eps  # 1e-5#1. #Preconditioner Ae
    method_to_solve = "ScipySparse"  # 'ScipySparse', 'DirectNPInverse', 'InexactNPInverse', , 'SparseSolver'
    preconditioner = False
    spectralanalysis = False
    manufactured_solution = False

    # Create the solver
    rho = 0.0005  # Example descent parameter
    tol = 1e-7
    max_iter = 2000
    pc = None  # M2pre # Preconditioner
    # Conjugate gradient solver  'bicg', 'bicgstab',  'lsmr', 'gmres', 'cg', 'pcg', 'minres'
    solver_name = "gmres"
    verbose = False

    A11 = M2 / dt + nu * (D.T @ M3 @ D + 1.0 * S21.T @ C.T @ M2 @ C @ S21) - 1.0 * M2R
    A12 = None
    A21 = A12
    A22 = eps * IdentityOperator(A11.domain) + nue * (D.T @ M3 @ D + 1.0 * S21.T @ C.T @ M2 @ C @ S21) + 1.0 * M2R
    B1 = -M3 @ D
    B1T = B1.T
    B2 = M3 @ D
    B2T = B2.T
    x1 = derham.curl.dot(x1_rdm)
    x2 = derham.curl.dot(x2_rdm)
    F1 = (
        A11.dot(x1) + B1T.dot(y1_rdm)
    )  # -0.*nu*(D.T @ M3 @ D.dot(x1)+ 0.*S21.T @ C.T @ M2 @ C @ S21.dot(x1) ) +0.5*M2R.dot(x1) #implicit/ explicit for diffusion terms
    F2 = A22.dot(x2) + B2T.dot(
        y1_rdm
    )  # -0.*nue*(D.T @ M3 @ D.dot(x2)+ 0.*S21.T @ C.T @ M2 @ C @ S21.dot(x2)) -0.5*M2R.dot(x2)

    # Preconditioner
    _A11 = M2 / dt + nu * (D.T @ M3 @ D)  # + S21.T @ C.T @ M2 @ C @ S21
    _A22 = nue * (D.T @ M3 @ D) + M2  # +eps2*IdentityOperator(A22.domain)  #

    # Change to numpy
    if method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
        M2np = M2._mat.toarray()
        M3np = M3._mat.toarray()
        Dnp = derhamnumpy.div.toarray()
        Cnp = derhamnumpy.curl.toarray()
        # Dnp = D.toarray()
        # Cnp = C.toarray()
        S21np = S21.toarray
        Hodgenp = Hodge.toarray
        M2Bnp = M2R._mat.toarray()
        x1np = x1.toarray()
        x2np = x2.toarray()
    elif method_to_solve in ("SparseSolver", "ScipySparse"):
        M2np = M2._mat.tosparse()
        M3np = M3._mat.tosparse()
        Dnp = derhamnumpy.div.tosparse()
        Cnp = derhamnumpy.curl.tosparse()
        # Dnp = D.tosparse()
        # Cnp = C.tosparse()
        S21np = S21.tosparse
        Hodgenp = Hodge.tosparse
        M2Bnp = M2R._mat.tosparse()
        x1np = x1.toarray()
        x2np = x2.toarray()

    A11np = M2np / dt + nu * (Dnp.T @ M3np @ Dnp + 1.0 * S21np.T @ Cnp.T @ M2np @ Cnp @ S21np) - 1.0 * M2Bnp
    if method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
        A22np = (
            eps * np.identity(A11np.shape[0])
            + nue * (Dnp.T @ M3np @ Dnp + 1.0 * S21np.T @ Cnp.T @ M2np @ Cnp @ S21np)
            + 1.0 * M2Bnp
        )
        _A22np = np.identity(A22np.shape[0])  # nue*(Dnp.T @ M3np @ Dnp) +eps2*np.identity(A22np.shape[0])  #+M2np #
    elif method_to_solve in ("SparseSolver", "ScipySparse"):
        A22np = (
            eps * sc.sparse.identity(A11np.shape[0], format="csr")
            + nue * (Dnp.T @ M3np @ Dnp + 1.0 * S21np.T @ Cnp.T @ M2np @ Cnp @ S21np)
            + 1.0 * M2Bnp
        )
        # nue*(Dnp.T @ M3np @ Dnp) +eps2*sc.sparse.identity(A22np.shape[0], format='csr')  #+M2np #
        _A22np = sc.sparse.identity(A22np.shape[0], format="csr")
        _A22np = _A22np.tocsr()
    B1np = -M3np @ Dnp
    B2np = M3np @ Dnp
    ynp = y1_rdm.toarray()
    F1np = A11np.dot(x1np) + (B1np.T).dot(ynp)
    F2np = A22np.dot(x2np) + (B2np.T).dot(ynp)

    Anp = [A11np, A22np]
    Bnp = [B1np, B2np]
    Fnp = [F1np, F2np]
    _A11np = M2np / dt + nu * (Dnp.T @ M3np @ Dnp)  # + S21np.T @ Cnp.T @ M2np @ Cnp @ S21np
    Anppre = [_A11np, _A22np]

    if A12 is not None:
        assert A11.codomain == A12.codomain
    if A21 is not None:
        assert A22.codomain == A21.codomain
    assert B1.codomain == B2.codomain
    if A12 is not None:
        assert A11.domain == A12.domain == B1.domain
    if A21 is not None:
        assert A21.domain == A22.domain == B2.domain
    assert A22.domain == B2.domain
    assert A11.domain == B1.domain

    block_domainA = BlockVectorSpace(A11.domain, A22.domain)
    block_codomainA = block_domainA
    block_domainB = block_domainA
    block_codomainB = B2.codomain
    blocks = [[A11, A12], [A21, A22]]
    A = BlockLinearOperator(block_domainA, block_codomainA, blocks=blocks)
    B = BlockLinearOperator(block_domainB, block_codomainB, blocks=[[B1, B2]])
    F = BlockVector(block_domainA, blocks=[F1, F2])
    x = BlockVector(block_domainA, blocks=[x1, x2])

    # Manufactured solution
    if manufactured_solution == True:
        _forceterm_logical = lambda e1, e2, e3: 0 * e1

        # Manufactured solution
        _ux = getattr(perturbations, "ManufacturedSolutionVelocity_x")(B0)
        _uy = getattr(perturbations, "ManufacturedSolutionVelocity_y")(B0)
        _uex = getattr(perturbations, "ManufacturedSolutionVelocityElectrons_x")(B0)
        _uey = getattr(perturbations, "ManufacturedSolutionVelocityElectrons_x")(B0)
        _pot = getattr(perturbations, "ManufacturedSolutionPotential")(B0)

        # get callable(s) for specified init type
        velocity_class = [_ux, _uy, _forceterm_logical]
        velocityelectrons_class = [_uex, _uey, _forceterm_logical]
        potential_class = [_pot]

        # pullback callable
        velx = TransformedPformComponent(velocity_class, fun_basis="physical", out_form="2", comp=0, domain=domain)
        vely = TransformedPformComponent(velocity_class, fun_basis="physical", out_form="2", comp=1, domain=domain)
        vel_electronsx = TransformedPformComponent(
            velocityelectrons_class, fun_basis="physical", out_form="2", comp=0, domain=domain
        )
        vel_electronsy = TransformedPformComponent(
            velocityelectrons_class, fun_basis="physical", out_form="2", comp=1, domain=domain
        )
        potential = TransformedPformComponent(
            potential_class, fun_basis="physical", out_form="3", comp=0, domain=domain
        )
        l2_proj = L2Projector(space_id="Hdiv", mass_ops=mass_mats)
        x1mf = l2_proj([velx, vely, _forceterm_logical])
        x2mf = l2_proj([vel_electronsx, vel_electronsy, _forceterm_logical])
        l2_proj_L2 = L2Projector(space_id="L2", mass_ops=mass_mats)
        ymf = l2_proj_L2(potential)
        x1mfnp = x1mf.toarray()
        x2mfnp = x2mf.toarray()
        ymfnp = ymf.toarray()
        xmf = BlockVector(block_domainA, blocks=[x1mf, x2mf])

        # ###Restelli
        # _forceterm_logical = lambda e1, e2, e3: 0 * e1
        # _ux = getattr(perturbations, "AnalyticSolutionRestelliVelocity_x")()
        # _uy = getattr(perturbations, "AnalyticSolutionRestelliVelocity_y")()
        # _uz = getattr(perturbations, "AnalyticSolutionRestelliVelocity_z")()
        # _pot = getattr(perturbations, "AnalyticSolutionRestelliPotential")()

        # get callable(s) for specified init type
        # velocity_class = [_ux, _uy, _uz]
        # potential_class = [_pot]

        # pullback callable
        # velx = TransformedPformComponent(velocity_class, fun_basis="physical", out_form="2", comp=0, domain=domain)
        # vely = TransformedPformComponent(velocity_class, fun_basis="physical", out_form="2", comp=1, domain=domain)
        # velz = TransformedPformComponent(velocity_class, fun_basis="physical", out_form="2", comp=2, domain=domain)
        # potential = TransformedPformComponent(
        #     potential_class, fun_basis="physical", out_form="3", comp=0, domain=domain
        # )
        # l2_proj = L2Projector(space_id='Hdiv', mass_ops=mass_mats)
        # x1mf = l2_proj([velx, vely, velz])
        # x1mf = x1mf
        # x2mf = x1mf
        # l2_proj_L2 = L2Projector(space_id='L2', mass_ops=mass_mats)
        # ymf = l2_proj_L2(potential)
        # x1mfnp = x1mf.toarray()
        # x2mfnp = x2mf.toarray()
        # ymfnp = ymf.toarray()
        # xmf = BlockVector(block_domainA, blocks=[x1mf, x2mf])

        # Foreceterm

        _funx = getattr(perturbations, "ManufacturedSolutionForceterm_x")(B0, nu)
        _funy = getattr(perturbations, "ManufacturedSolutionForceterm_y")(B0, nu)
        _funelectronsx = getattr(perturbations, "ManufacturedSolutionForcetermElectrons_x")(B0, nue)
        _funelectronsy = getattr(perturbations, "ManufacturedSolutionForcetermElectrons_y")(B0, nue)

        # get callable(s) for specified init type
        forceterm_class = [_funx, _funy, _forceterm_logical]
        forcetermelectrons_class = [_funelectronsx, _funelectronsy, _forceterm_logical]

        # pullback callable
        funx = TransformedPformComponent(forceterm_class, fun_basis="physical", out_form="2", comp=0, domain=domain)
        funy = TransformedPformComponent(forceterm_class, fun_basis="physical", out_form="2", comp=1, domain=domain)
        fun_electronsx = TransformedPformComponent(
            forcetermelectrons_class, fun_basis="physical", out_form="2", comp=0, domain=domain
        )
        fun_electronsy = TransformedPformComponent(
            forcetermelectrons_class, fun_basis="physical", out_form="2", comp=1, domain=domain
        )
        F1mf = l2_proj([funx, funy, _forceterm_logical])
        F2mf = l2_proj([fun_electronsx, fun_electronsy, _forceterm_logical])
        F1mfnp = F1mf.toarray()
        F2mfnp = F2mf.toarray()

        # Fmfnp = [F1mfnp + 1 / dt * M2np.dot(x1np), F2mfnp]
        # Fmf = BlockVector(block_domainA, blocks=[F1mf + 1 / dt * M2.dot(x1mf), F2mf])
        F1mf = A11.dot(x1mf) + B1T.dot(ymf)  # - 1 / dt * M2.dot(x1mf)
        F2mf = A22.dot(x2mf) + B2T.dot(ymf)
        Fmf = BlockVector(block_domainA, blocks=[F1mf, F2mf])
        F1mfnp = A11np.dot(x1mfnp) + (B1np.T).dot(ymfnp)  # - 1 / dt * M2np.dot(x1mfnp)
        F2mfnp = A22np.dot(x2mfnp) + (B2np.T).dot(ymfnp)
        Fmfnp = [F1mfnp, F2mfnp]

        # TestA = F[0]-A11.dot(x1) - B1T.dot(y1_rdm)
        TestAmf = (
            Fmf[0]
            - (M2 / dt + nu * (D.T @ M3 @ D + 1.0 * S21.T @ C.T @ M2 @ C @ S21) - 1.0 * M2R).dot(x1mf)
            - (B[0, 0].T).dot(ymf)
        )
        TestAemf = (
            Fmf[1]
            - (nue * (D.T @ M3 @ D + 1.0 * S21.T @ C.T @ M2 @ C @ S21) + 1.0 * M2R).dot(x2mf)
            - (B[0, 1].T).dot(ymf)
        )
        # TestAemf = Fmf[1]-A22.dot(x2mf) - B2T.dot(ymf)
        RestAmf = np.linalg.norm(TestAmf.toarray())
        RestAemf = np.linalg.norm(TestAemf.toarray())
        TestAmfnp = F1mfnp - A11np.dot(x1mfnp) - B1np.T.dot(ymfnp)
        TestAemfnp = F2mfnp - A22np.dot(x2mfnp) - B2np.T.dot(ymfnp)
        RestAmfnp = np.linalg.norm(TestAmfnp)
        RestAemfnp = np.linalg.norm(TestAemfnp)
        TestDivmfnp = B1np.dot(x1mfnp) + B2np.dot(x2mfnp)
        RestDivmfnp = np.linalg.norm(TestDivmfnp)
        TestDivmf = B1.dot(x1mf) + B2.dot(x2mf)
        RestDivmf = np.linalg.norm(TestDivmf.toarray())
        print(f"{RestAmf =}")
        print(f"{RestAemf =}")
        print(f"{RestAmfnp =}")
        print(f"{RestAemfnp =}")
        print(f"{RestDivmfnp =}")
        print(f"{RestDivmf =}")

    else:
        # TestA = F[0]-A11.dot(x1) - B1T.dot(y1_rdm)
        TestA = (
            F[0]
            - (M2 / dt + nu * (D.T @ M3 @ D + 1.0 * S21.T @ C.T @ M2 @ C @ S21) - 1.0 * M2R).dot(x1)
            - (B[0, 0].T).dot(y1_rdm)
        )
        TestAe = (
            F[1]
            - (nue * (D.T @ M3 @ D + 1.0 * S21.T @ C.T @ M2 @ C @ S21) + 1.0 * M2R).dot(x2)
            - (B[0, 1].T).dot(y1_rdm)
        )
        # TestAemf = Fmf[1]-A22.dot(x2mf) - B2T.dot(ymf)
        RestA = np.linalg.norm(TestA.toarray())
        RestAe = np.linalg.norm(TestAe.toarray())
        TestAnp = (
            F1np
            - (M2np / dt + nu * (Dnp.T @ M3np @ Dnp + S21np.T @ Cnp.T @ M2np @ Cnp @ S21np) - M2Bnp).dot(x1np)
            - B1np.T.dot(ynp)
        )
        TestAenp = (
            F2np
            - (nue * (Dnp.T @ M3np @ Dnp + S21np.T @ Cnp.T @ M2np @ Cnp @ S21np) + M2Bnp).dot(x2np)
            - B2np.T.dot(ynp)
        )
        RestAnp = np.linalg.norm(TestAnp)
        RestAenp = np.linalg.norm(TestAenp)
        TestDivnp = -B1np.dot(x1np) + B2np.dot(x2np)
        RestDivnp = np.linalg.norm(TestDivnp)
        TestDiv = -B1.dot(x1) + B2.dot(x2)
        RestDiv = np.linalg.norm(TestDiv.toarray())
        print(f"{RestA =}")
        print(f"{RestAe =}")
        print(f"{RestAnp =}")
        print(f"{RestAenp =}")
        print(f"{RestDivnp =}")
        print(f"{RestDiv =}")

    # Manufactured solution
    # uanalyt0 = lambda x, y, z: -np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    # uanalyt1 = -np.cos(2*np.pi*x_grid)*np.cos(2*np.pi*y_grid)
    # uanalyt2 = 0.0*x_grid
    # u_eanalyt0 = -np.sin(4*np.pi*x_grid)*np.sin(4*np.pi*y_grid)
    # u_eanalyt1 = -np.cos(4*np.pi*x_grid)*np.cos(4*np.pi*y_grid)
    # u_eanalyt2 = 0.0*x_grid
    # uanalyt = [uanalyt0, uanalyt1, uanalyt2]
    # u_eanalyt = [u_eanalyt0, u_eanalyt1, u_eanalyt2]
    # potential_analytical = [np.cos(2*np.pi*x_grid)+np.sin(2*np.pi*y_grid)]

    # F1analyticx = -2.0*np.pi*np.sin(2*np.pi*x_grid)+np.cos(2*np.pi*x_grid)*np.cos(2*np.pi*y_grid)*b0-nu*8.0*np.pi**2*np.sin(2*np.pi*x_grid)*np.sin(2*np.pi*y_grid)
    # F1analyticy = 2.0*np.pi*np.cos(2*np.pi*y_grid)-np.sin(2*np.pi*x_grid)*np.sin(2*np.pi*y_grid)*b0-nu*8.0*np.pi**2*np.cos(2*np.pi*x_grid)*np.cos(2*np.pi*y_grid)
    # F1analytic = [F1analyticx, F1analyticy, 0*x_grid]

    # F2analyticx = 2.0*np.pi*np.sin(2*np.pi*x_grid)-np.cos(4*np.pi*x_grid)*np.cos(4*np.pi*y_grid)*b0-nu_e*32.0*np.pi**2*np.sin(4*np.pi*x_grid)*np.sin(4*np.pi*y_grid)
    # F2analyticy = -2.0*np.pi*np.cos(2*np.pi*y_grid)+np.sin(4*np.pi*x_grid)*np.sin(4*np.pi*y_grid)*b0-nu_e*32.0*np.pi**2*np.cos(4*np.pi*x_grid)*np.cos(4*np.pi*y_grid)
    # F2analytic = [F2analyticx, F2analyticy, 0*x_grid]

    # Compare numpy to psydac
    c1 = C.dot(x1_rdm)
    c2 = Cnp.dot(x1_rdm.toarray())
    compare_arrays(c1, c2, mpi_rank, atol=1e-5)
    xblock, xdiv_rdm = create_equal_random_arrays(fem_spaces[2], seed=1568, flattened=False)
    d1 = D.dot(xdiv_rdm)
    d2 = Dnp.dot(xdiv_rdm.toarray())
    compare_arrays(d1, d2, mpi_rank, atol=1e-5)
    TestA11composed = M2np / dt + Dnp.T @ M3np @ Dnp + S21np.T @ Cnp.T @ M2np @ Cnp @ S21np
    TestA11 = M2 / dt + nu * D.T @ M3 @ D + S21.T @ C.T @ M2 @ C @ S21
    # TestA11np = (M2 / dt + nu * D.T @ M3 @ D+S21.T @ C.T @ M2 @ C @ S21).toarray_struphy()
    # TestA11npdot = TestA11np.dot(x1.toarray())
    TestA11composeddot = TestA11composed.dot(x1.toarray())
    TestA11dot = TestA11.dot(x1)
    compare_arrays(TestA11dot, TestA11composeddot, mpi_rank, atol=1e-5)
    # compare_arrays(TestA11dot, TestA11npdot, mpi_rank, atol=1e-5)
    print(f"Comparison numpy to psydac succesfull.")

    M2pre = MassMatrixPreconditioner(mass_mats.M2)

    start_time = time.time()

    if method_for_solving == "SaddlePointSolverUzawaNumpy":
        if manufactured_solution == True:
            solver = SaddlePointSolverUzawaNumpy(
                Anp, Bnp, Fmfnp, Anppre, method_to_solve, preconditioner, spectralanalysis, tol=tol, max_iter=max_iter
            )
            x_u, x_ue, y_uzawa, info, residual_norms = solver(0.9 * x1mf, 0.9 * x2mf, 1.1 * ymf)
        else:
            ###wrong initialization to check if changed
            solver = SaddlePointSolverUzawaNumpy(
                Anppre,
                Bnp,
                [Anppre[0].dot(x1np), Anppre[0].dot(x1np)],
                Anppre,
                method_to_solve,
                preconditioner,
                spectralanalysis,
                tol=tol,
                max_iter=max_iter,
            )
            # solver = SaddlePointSolverUzawaNumpy(
            #     Anp,  Bnp, [Anppre[0].dot(x1np), Anppre[0].dot(x1np)], Anppre, method_to_solve, preconditioner, spectralanalysis,  tol=tol, max_iter=max_iter
            # )
            solver.A = Anp
            solver.B = Bnp
            solver.F = Fnp
            solver.Apre = Anppre
            x_u, x_ue, y_uzawa, info, residual_norms = solver(
                0.9 * x1, 0.9 * x2, 1.1 * y1_rdm
            )  # 0.9*x1, 0.9*x2, 1.1*y1_rdm
        x_uzawa = {}
        x_uzawa[0] = x_u
        x_uzawa[1] = x_ue
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverGMRES":
        if manufactured_solution == True:
            solver = SaddlePointSolverGMRES(
                A, B, Fmf, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
            )
            x_uzawa, y_uzawa, info = solver(0.9 * x1mf, 0.9 * x2mf, 1.1 * ymf)
        else:
            solver = SaddlePointSolverGMRES(
                A, B, F, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
            )
            x_uzawa, y_uzawa, info = solver(0.9 * x1, 0.9 * x2, 1.1 * y1_rdm)
    elif method_for_solving == "SaddlePointSolverGMRESwithPC":
        solver = SaddlePointSolverGMRESwithPC(
            A,
            B,
            _A11,
            _A22,
            F,
            M2pre,
            rho=rho,
            solver_name=solver_name,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            pc=pc,
            derham=derham,
        )
        x_uzawa, y_uzawa, info = solver(dt, 0.9 * x1, 0.9 * x2, 1.1 * y1_rdm)

    end_time = time.time()

    print(f"{method_for_solving}{info}")

    elapsed_time = end_time - start_time
    print(f"Method execution time: {elapsed_time:.6f} seconds")

    if isinstance(x_uzawa[0], np.ndarray):
        if manufactured_solution == True:
            Rx1 = x1mfnp - x_uzawa[0]
            Rx2 = x2mfnp - x_uzawa[1]
            Ry = ymfnp - y_uzawa
            residualx_normx1 = np.linalg.norm(Rx1)
            residualx_normx2 = np.linalg.norm(Rx2)
            residualy_norm = np.linalg.norm(Ry)
            TestRest1 = F1mfnp - A11np.dot(x_uzawa[0]) - B1np.T.dot(y_uzawa)
            TestRest1val = np.max(abs(TestRest1))
            Testoldy1 = F1mfnp - A11np.dot(x_uzawa[0]) - B1np.T.dot(ymfnp)
            Testoldy1val = np.max(abs(Testoldy1))
            TestRest2 = F2mfnp - A22np.dot(x_uzawa[1]) - B2np.T.dot(y_uzawa)
            TestRest2val = np.max(abs(TestRest2))
            Testoldy2 = F2mfnp - A22np.dot(x_uzawa[1]) - B2np.T.dot(ymfnp)
            Testoldy2val = np.max(abs(Testoldy2))
            Testoldy1y = F1mfnp - A11np.dot(x1mfnp) - B1np.T.dot(y_uzawa)
            Testoldy1valy = np.max(abs(Testoldy1y))
            TestRest2y = F2mfnp - A22np.dot(x2mfnp) - B2np.T.dot(y_uzawa)
            Testoldy2valy = np.max(abs(TestRest2y))
            print(f"{TestRest1val =}")
            print(f"{TestRest2val =}")
            print(f"{Testoldy1val =}")
            print(f"{Testoldy2val =}")
            print(f"{Testoldy1valy =}")
            print(f"{Testoldy2valy =}")
            print(f"Residual x1 norm: {residualx_normx1}")
            print(f"Residual x2 norm: {residualx_normx2}")
            print(f"Residual y norm: {residualy_norm}")

            compare_arrays(ymf, y_uzawa, mpi_rank, atol=1e-5)
            compare_arrays(x1mf, x_uzawa[0], mpi_rank, atol=1e-5)
            compare_arrays(x2mf, x_uzawa[1], mpi_rank, atol=1e-5)
            print(f"{info = }")
        else:
            # Output as np.ndarray
            Rx1 = x1np - x_uzawa[0]
            Rx2 = x2np - x_uzawa[1]
            Ry = ynp - y_uzawa
            residualx_normx1 = np.linalg.norm(Rx1)
            residualx_normx2 = np.linalg.norm(Rx2)
            residualy_norm = np.linalg.norm(Ry)
            TestRest1 = F1np - A11np.dot(x_uzawa[0]) - B1np.T.dot(y_uzawa)
            TestRest1val = np.max(abs(TestRest1))
            Testoldy1 = F1np - A11np.dot(x_uzawa[0]) - B1np.T.dot(ynp)
            Testoldy1val = np.max(abs(Testoldy1))
            TestRest2 = F2np - A22np.dot(x_uzawa[1]) - B2np.T.dot(y_uzawa)
            TestRest2val = np.max(abs(TestRest2))
            Testoldy2 = F2np - A22np.dot(x_uzawa[1]) - B2np.T.dot(ynp)
            Testoldy2val = np.max(abs(Testoldy2))
            print(f"{TestRest1val =}")
            print(f"{TestRest2val =}")
            print(f"{Testoldy1val =}")
            print(f"{Testoldy2val =}")
            print(f"Residual x1 norm: {residualx_normx1}")
            print(f"Residual x2 norm: {residualx_normx2}")
            print(f"Residual y norm: {residualy_norm}")

            compare_arrays(y1_rdm, y_uzawa, mpi_rank, atol=1e-5)
            compare_arrays(x1, x_uzawa[0], mpi_rank, atol=1e-5)
            compare_arrays(x2, x_uzawa[1], mpi_rank, atol=1e-5)
            print(f"{info = }")
    elif isinstance(x_uzawa[0], BlockVector):
        if manufactured_solution == True:
            # Output as Blockvector
            Rx1 = x1mf - x_uzawa[0]
            Rx2 = x2mf - x_uzawa[1]
            Ry = ymf - y_uzawa
            residualx_normx1 = np.linalg.norm(Rx1.toarray())
            residualx_normx2 = np.linalg.norm(Rx2.toarray())
            residualy_norm = np.linalg.norm(Ry.toarray())

            TestRest1 = F1mf - A11.dot(x_uzawa[0]) - B1T.dot(y_uzawa)
            TestRest1val = np.max(abs(TestRest1.toarray()))
            Testoldy1 = F1mf - A11.dot(x_uzawa[0]) - B1T.dot(ymf)
            Testoldy1val = np.max(abs(Testoldy1.toarray()))
            TestRest2 = F2mf - A22.dot(x_uzawa[1]) - B2T.dot(y_uzawa)
            TestRest2val = np.max(abs(TestRest2.toarray()))
            Testoldy2 = F2mf - A22.dot(x_uzawa[1]) - B2T.dot(ymf)
            Testoldy2val = np.max(abs(Testoldy2.toarray()))
            # print(f"{TestRest1val =}")
            # print(f"{TestRest2val =}")
            # print(f"{Testoldy1val =}")
            # print(f"{Testoldy2val =}")
            print(f"Residual x1 norm: {residualx_normx1}")
            print(f"Residual x2 norm: {residualx_normx2}")
            print(f"Residual y norm: {residualy_norm}")

            compare_arrays(ymf, y_uzawa.toarray(), mpi_rank, atol=1e-5)
            compare_arrays(x1mf, x_uzawa[0].toarray(), mpi_rank, atol=1e-5)
            compare_arrays(x2mf, x_uzawa[1].toarray(), mpi_rank, atol=1e-5)

        else:
            # Output as Blockvector
            Rx1 = x1 - x_uzawa[0]
            Rx2 = x2 - x_uzawa[1]
            Ry = y1_rdm - y_uzawa
            residualx_normx1 = np.linalg.norm(Rx1.toarray())
            residualx_normx2 = np.linalg.norm(Rx2.toarray())
            residualy_norm = np.linalg.norm(Ry.toarray())

            TestRest1 = F1 - A11.dot(x_uzawa[0]) - B1T.dot(y_uzawa)
            TestRest1val = np.max(abs(TestRest1.toarray()))
            Testoldy1 = F1 - A11.dot(x_uzawa[0]) - B1T.dot(y1_rdm)
            Testoldy1val = np.max(abs(Testoldy1.toarray()))
            TestRest2 = F2 - A22.dot(x_uzawa[1]) - B2T.dot(y_uzawa)
            TestRest2val = np.max(abs(TestRest2.toarray()))
            Testoldy2 = F2 - A22.dot(x_uzawa[1]) - B2T.dot(y1_rdm)
            Testoldy2val = np.max(abs(Testoldy2.toarray()))
            # print(f"{TestRest1val =}")
            # print(f"{TestRest2val =}")
            # print(f"{Testoldy1val =}")
            # print(f"{Testoldy2val =}")
            print(f"Residual x1 norm: {residualx_normx1}")
            print(f"Residual x2 norm: {residualx_normx2}")
            print(f"Residual y norm: {residualy_norm}")

            compare_arrays(y1_rdm, y_uzawa.toarray(), mpi_rank, atol=1e-5)
            compare_arrays(x1, x_uzawa[0].toarray(), mpi_rank, atol=1e-5)
            compare_arrays(x2, x_uzawa[1].toarray(), mpi_rank, atol=1e-5)


def _plot_residual_norms(residual_norms):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(residual_norms, label="Residual Norm")
    plt.yscale("log")  # Use logarithmic scale for better visualization
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Convergence of Residual Norm")
    plt.legend()
    plt.grid(True)
    plt.savefig("residual_norms_plot.png")


def _plot_velocity(data_reshaped):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("Agg")

    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(6, 5))
    plt.imshow(data_reshaped.T, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    plt.colorbar(label="u_x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Velocity Component u_x")
    plt.savefig("velocity.png")


if __name__ == "__main__":
    # test_saddlepointsolver('SaddlePointSolverUzawaNumpy',
    #                        [10, 10, 1],
    #                        [3, 3, 1],
    #                        [True, False, True],
    #                        [[False,  False], [False, False], [False, False]],
    #                        ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], True)

    # test_saddlepointsolver('SaddlePointSolverGMRES',
    #                        [15, 15, 1],
    #                        [3, 3, 1],
    #                        [True, False, True],
    #                        [[False,  False], [False, False], [False, False]],
    #                        ['Cuboid', {'l1': 0., 'r1': 2., 'l2': 0., 'r2': 3., 'l3': 0., 'r3': 6.}], True)
    test_saddlepointsolver(
        "SaddlePointSolverUzawaNumpy",
        [15, 15, 1],
        [3, 3, 1],
        [True, False, True],
        [[False, False], [False, False], [False, False]],
        ["Cuboid", {"l1": 0.0, "r1": 2.0, "l2": 0.0, "r2": 3.0, "l3": 0.0, "r3": 6.0}],
        True,
    )
