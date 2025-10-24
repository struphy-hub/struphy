import pytest


@pytest.mark.mpi_skip
@pytest.mark.parametrize("method_for_solving", ["SaddlePointSolverUzawaNumpy", "SaddlePointSolverGMRES"])
@pytest.mark.parametrize("Nel", [[12, 8, 1]])
@pytest.mark.parametrize("p", [[3, 3, 1]])
@pytest.mark.parametrize("spl_kind", [[False, True, True]])
@pytest.mark.parametrize("dirichlet_bc", [((False, False), (False, False), (False, False))])
@pytest.mark.parametrize("mapping", [["Cuboid", {"l1": 0.0, "r1": 2.0, "l2": 0.0, "r2": 3.0, "l3": 0.0, "r3": 6.0}]])
def test_saddlepointsolver(method_for_solving, Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Test saddle-point-solver with manufactured solutions."""

    import time

    import cunumpy as xp
    import scipy as sc
    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.basic import IdentityOperator
    from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace

    from struphy.examples.restelli2018 import callables
    from struphy.feec.basis_projection_ops import BasisProjectionOperatorLocal, BasisProjectionOperators
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.projectors import L2Projector
    from struphy.feec.psydac_derham import Derham, TransformedPformComponent
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.fields_background.equils import CircularTokamak, HomogenSlab
    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.linear_algebra.saddle_point import SaddlePointSolver

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    mpi_comm.Barrier()

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, dirichlet_bc=dirichlet_bc, local_projectors=False)
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
    x2_rdm_block, x2_rdm = create_equal_random_arrays(fem_spaces[1], seed=111, flattened=False)
    y1_rdm_block, y1_rdm = create_equal_random_arrays(fem_spaces[3], seed=8567, flattened=False)

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    hodge_mats = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    S21 = hodge_mats.S21
    M2R = mass_mats.M2B
    M2 = mass_mats.M2
    C = derham.curl
    D = derham.div
    M3 = mass_mats.M3
    B0 = 1.0
    nue = 0.01 * 100
    nu = 1.0
    dt = 0.001
    stab_sigma = 1e-4
    method_to_solve = "DirectNPInverse"  # 'ScipySparse', 'DirectNPInverse', 'InexactNPInverse', , 'SparseSolver'
    preconditioner = True
    spectralanalysis = False

    # Create the solver
    rho = 0.0005  # Example descent parameter
    tol = 1e-10
    max_iter = 4000
    pc = None  # M2pre # Preconditioner
    # Conjugate gradient solver  'bicg', 'bicgstab',  'lsmr', 'gmres', 'cg', 'pcg', 'minres'
    solver_name = "gmres"  # lsmr gmres
    verbose = False

    x1 = derham.curl.dot(x1_rdm)
    x2 = derham.curl.dot(x2_rdm)
    if method_for_solving in ("SaddlePointSolverGMRES", "SaddlePointSolverGMRESwithPC"):
        A11 = M2 / dt + nu * (D.T @ M3 @ D + S21.T @ C.T @ M2 @ C @ S21) - M2R
        A12 = None
        A21 = A12
        A22 = stab_sigma * IdentityOperator(A11.domain) + nue * (D.T @ M3 @ D + S21.T @ C.T @ M2 @ C @ S21) + M2R
        B1 = -M3 @ D
        B1T = B1.T
        B2 = M3 @ D
        B2T = B2.T
        F1 = A11.dot(x1) + B1T.dot(y1_rdm)
        F2 = A22.dot(x2) + B2T.dot(y1_rdm)
    elif method_for_solving in ("SaddlePointSolverUzawaNumpy"):
        # Change to numpy
        if method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
            M2np = M2._mat.toarray()
            M3np = M3._mat.toarray()
            Dnp = derhamnumpy.div.toarray()
            Cnp = derhamnumpy.curl.toarray()
            # Dnp = D.toarray()
            # Cnp = C.toarray()
            if derham.with_local_projectors == True:
                S21np = S21.toarray
            else:
                S21np = S21.toarray_struphy()
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
            if derham.with_local_projectors == True:
                S21np = S21.tosparse
            else:
                S21np = S21.toarray_struphy(is_sparse=True)
            M2Bnp = M2R._mat.tosparse()
            x1np = x1.toarray()
            x2np = x2.toarray()

        A11np = M2np / dt + nu * (Dnp.T @ M3np @ Dnp + S21np.T @ Cnp.T @ M2np @ Cnp @ S21np) - M2Bnp
        if method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
            A22np = (
                stab_sigma * xp.identity(A11np.shape[0])
                + nue * (Dnp.T @ M3np @ Dnp + S21np.T @ Cnp.T @ M2np @ Cnp @ S21np)
                + M2Bnp
            )
            # Preconditioner
            _A22np_pre = stab_sigma * xp.identity(A22np.shape[0])  # + nue*(Dnp.T @ M3np @ Dnp)
            _A11np_pre = M2np / dt  # + nu * (Dnp.T @ M3np @ Dnp)
        elif method_to_solve in ("SparseSolver", "ScipySparse"):
            A22np = (
                stab_sigma * sc.sparse.identity(A11np.shape[0], format="csr")
                + nue * (Dnp.T @ M3np @ Dnp + S21np.T @ Cnp.T @ M2np @ Cnp @ S21np)
                + M2Bnp
            )
            +nue * (Dnp.T @ M3np @ Dnp) + stab_sigma * sc.sparse.identity(A22np.shape[0], format="csr")  #
            # Preconditioner
            _A22np_pre = stab_sigma * sc.sparse.identity(A22np.shape[0], format="csr")  # + nue*(Dnp.T @ M3np @ Dnp)
            _A22np_pre = _A22np_pre.tocsr()
            _A11np_pre = M2np / dt  # + nu * (Dnp.T @ M3np @ Dnp)
            _A11np_pre = _A11np_pre.tocsr()
        B1np = -M3np @ Dnp
        B2np = M3np @ Dnp
        ynp = y1_rdm.toarray()
        F1np = A11np.dot(x1np) + (B1np.T).dot(ynp)
        F2np = A22np.dot(x2np) + (B2np.T).dot(ynp)

        Anp = [A11np, A22np]
        Bnp = [B1np, B2np]
        Fnp = [F1np, F2np]
        # Preconditioner not inverted
        Anppre = [_A11np_pre, _A22np_pre]

    if method_for_solving in ("SaddlePointSolverGMRES", "SaddlePointSolverGMRESwithPC"):
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
        blocksfalse = [[A22, A12], [A21, A11]]
        A = BlockLinearOperator(block_domainA, block_codomainA, blocks=blocks)
        Afalse = BlockLinearOperator(block_domainA, block_codomainA, blocks=blocksfalse)
        B = BlockLinearOperator(block_domainB, block_codomainB, blocks=[[B1, B2]])
        F = BlockVector(block_domainA, blocks=[F1, F2])
        Ffalse = BlockVector(block_domainA, blocks=[0.0 * F1, 0.0 * F2])

    # TestA = F[0]-A11.dot(x1) - B1T.dot(y1_rdm)
    if method_for_solving in ("SaddlePointSolverGMRES", "SaddlePointSolverGMRESwithPC"):
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
        TestDiv = -B1.dot(x1) + B2.dot(x2)
        RestDiv = xp.linalg.norm(TestDiv.toarray())
        RestA = xp.linalg.norm(TestA.toarray())
        RestAe = xp.linalg.norm(TestAe.toarray())
        print(f"{RestA =}")
        print(f"{RestAe =}")
        print(f"{RestDiv =}")
    elif method_for_solving in ("SaddlePointSolverUzawaNumpy"):
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
        RestAnp = xp.linalg.norm(TestAnp)
        RestAenp = xp.linalg.norm(TestAenp)
        TestDivnp = -B1np.dot(x1np) + B2np.dot(x2np)
        RestDivnp = xp.linalg.norm(TestDivnp)
        print(f"{RestAnp =}")
        print(f"{RestAenp =}")
        print(f"{RestDivnp =}")

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
        ###wrong initialization to check if changed
        solver = SaddlePointSolver(
            A=Anppre,
            B=Bnp,
            F=[Anppre[0].dot(x1np), Anppre[0].dot(x1np)],
            Apre=Anppre,
            method_to_solve=method_to_solve,
            preconditioner=preconditioner,
            spectralanalysis=spectralanalysis,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )
        solver.A = Anp
        solver.B = Bnp
        solver.F = Fnp
        solver.Apre = Anppre
        x_u, x_ue, y_uzawa, info, residual_norms, spectral_result = solver(0.9 * x1, 0.9 * x2, 1.1 * y1_rdm)
        x_uzawa = {}
        x_uzawa[0] = x_u
        x_uzawa[1] = x_ue
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverGMRES":
        # Wrong initialization to check if changed
        solver = SaddlePointSolver(
            A=Afalse,
            B=B,
            F=Ffalse,
            Apre=None,
            solver_name=solver_name,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            pc=pc,
        )
        solver.A = A
        solver.F = F
        x_uzawa, y_uzawa, info = solver(0.9 * x1, 0.9 * x2, 1.1 * y1_rdm)

    end_time = time.time()

    print(f"{method_for_solving}{info}")

    elapsed_time = end_time - start_time
    print(f"Method execution time: {elapsed_time:.6f} seconds")

    if isinstance(x_uzawa[0], xp.ndarray):
        # Output as xp.ndarray
        Rx1 = x1np - x_uzawa[0]
        Rx2 = x2np - x_uzawa[1]
        Ry = ynp - y_uzawa
        residualx_normx1 = xp.linalg.norm(Rx1)
        residualx_normx2 = xp.linalg.norm(Rx2)
        residualy_norm = xp.linalg.norm(Ry)
        TestRest1 = F1np - A11np.dot(x_uzawa[0]) - B1np.T.dot(y_uzawa)
        TestRest1val = xp.max(abs(TestRest1))
        Testoldy1 = F1np - A11np.dot(x_uzawa[0]) - B1np.T.dot(ynp)
        Testoldy1val = xp.max(abs(Testoldy1))
        TestRest2 = F2np - A22np.dot(x_uzawa[1]) - B2np.T.dot(y_uzawa)
        TestRest2val = xp.max(abs(TestRest2))
        Testoldy2 = F2np - A22np.dot(x_uzawa[1]) - B2np.T.dot(ynp)
        Testoldy2val = xp.max(abs(Testoldy2))
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
        print(f"{info =}")
    elif isinstance(x_uzawa[0], BlockVector):
        # Output as Blockvector
        Rx1 = x1 - x_uzawa[0]
        Rx2 = x2 - x_uzawa[1]
        Ry = y1_rdm - y_uzawa
        residualx_normx1 = xp.linalg.norm(Rx1.toarray())
        residualx_normx2 = xp.linalg.norm(Rx2.toarray())
        residualy_norm = xp.linalg.norm(Ry.toarray())

        TestRest1 = F1 - A11.dot(x_uzawa[0]) - B1T.dot(y_uzawa)
        TestRest1val = xp.max(abs(TestRest1.toarray()))
        Testoldy1 = F1 - A11.dot(x_uzawa[0]) - B1T.dot(y1_rdm)
        Testoldy1val = xp.max(abs(Testoldy1.toarray()))
        TestRest2 = F2 - A22.dot(x_uzawa[1]) - B2T.dot(y_uzawa)
        TestRest2val = xp.max(abs(TestRest2.toarray()))
        Testoldy2 = F2 - A22.dot(x_uzawa[1]) - B2T.dot(y1_rdm)
        Testoldy2val = xp.max(abs(Testoldy2.toarray()))
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
    import cunumpy as xp
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    x = xp.linspace(0, 1, 30)
    y = xp.linspace(0, 1, 30)
    X, Y = xp.meshgrid(x, y)

    plt.figure(figsize=(6, 5))
    plt.imshow(data_reshaped.T, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    plt.colorbar(label="u_x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Velocity Component u_x")
    plt.savefig("velocity.png")


if __name__ == "__main__":
    # test_saddlepointsolver(
    #     "SaddlePointSolverGMRES",
    #     [15, 15, 1],
    #     [3, 3, 1],
    #     [True, False, True],
    #     [[False, False], [False, False], [False, False]],
    #     ["Cuboid", {"l1": 0.0, "r1": 2.0, "l2": 0.0, "r2": 3.0, "l3": 0.0, "r3": 6.0}],
    #     True,
    # )
    test_saddlepointsolver(
        "SaddlePointSolverUzawaNumpy",
        [15, 15, 1],
        [3, 3, 1],
        [True, False, True],
        [[False, False], [False, False], [False, False]],
        ["Cuboid", {"l1": 0.0, "r1": 2.0, "l2": 0.0, "r2": 3.0, "l3": 0.0, "r3": 6.0}],
        True,
    )
