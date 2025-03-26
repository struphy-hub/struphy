def test_saddlepointsolver(method_for_solving, Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Test saddle-point-solver with manufactured solutions."""

    import time

    import numpy as np
    from mpi4py import MPI
    from psydac.linalg.basic import IdentityOperator
    from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace

    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.fields_background.equils import FluxAlignedTokamak, HomogenSlab
    from struphy.geometry import domains
    from struphy.linear_algebra.saddle_point import (
        SaddlePointSolver,
        SaddlePointSolverGMRES,
        SaddlePointSolverGMRESsolution,
        SaddlePointSolverInexactUzawa,
        SaddlePointSolverNoCG,
        SaddlePointSolverNoCGPaper,
        SaddlePointSolverTest,
        SaddlePointSolverGMRESwithPC,
    )

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    
    mpi_comm.Barrier()

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, dirichlet_bc=dirichlet_bc)

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    fem_spaces = [derham.Vh_fem["0"], derham.Vh_fem["1"], derham.Vh_fem["2"], derham.Vh_fem["3"], derham.Vh_fem["v"]]

    # Mhd equilibirum (slab)
    mhd_equil_params = {"B0x": 0.0, "B0y": 0.0, "B0z": 1.0, "beta": 2.0, "n0": 1.0}
    eq_mhd = HomogenSlab(**mhd_equil_params)

    # mhd_equil_params = {'a': 1.45, 'R0': 6.5, 'q_kind': 1, 'p_kind': 0}

    # eq_mhd = AdhocTorus(**mhd_equil_params)
    eq_mhd.domain = domain

    # create random input array
    x1_rdm_block, x1_rdm = create_equal_random_arrays(fem_spaces[1], seed=1568, flattened=False)
    x2_rdm_block, x2_rdm = create_equal_random_arrays(fem_spaces[1], seed=68, flattened=False)
    y1_rdm_block, y1_rdm = create_equal_random_arrays(fem_spaces[3], seed=1568, flattened=False)

    print(f"{np.max(abs(x1_rdm.toarray())) =}")
    print(f"{np.max(abs(x2_rdm.toarray())) =}")
    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    hodge_mats = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    # Change order of input in callable
    M2R = mass_mats.M2B
    M2 = mass_mats.M2
    Hodge = hodge_mats.S21
    C = derham.curl
    D = derham.div
    G = derham.grad
    M3 = mass_mats.M3
    nue = 0.01*100
    nu = 1.0
    dt = 0.01
    eps = 1e-7
    eps1 = 0. #Preconditioner A
    eps2 = 1e-7#1. #Preconditioner Ae
    
    # Create the solver
    rho = 0.0005  # Example descent parameter
    tol = 1e-9
    max_iter = 1500
    pc = None  # M2pre # Preconditioner
    # Conjugate gradient solver 'cg', 'pcg', 'bicg', 'bicgstab', 'minres', 'lsmr', 'gmres'
    solver_name = "gmres"
    verbose = False

    A11 = M2 / dt + nu * (D.transpose() @ M3 @ D + 1.0 * Hodge.transpose() @ C.transpose() @ M2 @ C @ Hodge) - 1.0 * M2R
    A12 = None
    A21 = A12
    A22 = (
        eps * IdentityOperator(A11.domain)
        + nue * (D.transpose() @ M3 @ D + 1.0 * Hodge.transpose() @ C.transpose() @ M2 @ C @ Hodge)
        + 1.0 * M2R
    )
    B1 = -M3 @ D
    B1T = B1.transpose()
    B2 = M3 @ D
    B2T = B2.transpose()
    x1 = derham.curl.dot(x1_rdm)
    x2 = derham.curl.dot(x2_rdm)
    F1 = (
        A11.dot(x1) + B1T.dot(y1_rdm)
    )  # -0.*nu*(D.T @ M3 @ D.dot(x1)+ 0.*Hodge.transpose() @ C.transpose() @ M2 @ C @ Hodge.dot(x1) ) +0.5*M2R.dot(x1) #implicit/ explicit for diffusion terms
    F2 = A22.dot(x2) + B2T.dot(
        y1_rdm
    )  # -0.*nue*(D.T @ M3 @ D.dot(x2)+ 0.*Hodge.transpose() @ C.transpose() @ M2 @ C @ Hodge.dot(x2)) -0.5*M2R.dot(x2)

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
    
    print(f'{type(F1.blocks[0])}')

    block_domainA = BlockVectorSpace(A11.domain, A22.domain)
    block_codomainA = block_domainA
    block_domainB = block_domainA
    block_codomainB = B2.codomain
    blocks = [[A11, A12], [A21, A22]]
    A = BlockLinearOperator(block_domainA, block_codomainA, blocks=blocks)
    B = BlockLinearOperator(block_domainB, block_codomainB, blocks=[[B1, B2]])
    F = BlockVector(block_domainA, blocks=[F1, F2])
    x = BlockVector(block_domainA, blocks=[x1, x2])
    
    # TestA11np = (M2 / dt + nu * D.transpose() @ M3 @ D).toarray()
    # M2np = M2.toarray_struphy()
    # Dnp = derham.div.toarray()
    # Cnp = C.toarray_struphy()
    # TestA11composed = M2.toarray_struphy()/dt + (D.T).toarray()@M3.toarray_struphy()@D.toarray()
    
    # TestA11composeddot = TestA11composed.dot(x1.toarray())
    # TestA11npdot = TestA11np.dot(x1.toarray())
    # compare_arrays(TestA11composeddot, TestA11npdot, mpi_rank, atol=1e-5)

    M2pre = MassMatrixPreconditioner(mass_mats.M2)

    #Preconditioner
    _A11 =  M2/dt + nu*(D.T @ M3 @ D) #+ Hodge.T @ C.T @ M2 @ C @ Hodge
    _A22 = nue*(D.T @ M3 @ D) +M2 # +eps2*IdentityOperator(A22.domain)  # 

    start_time = time.time()

    # SaddlePointSolver, SaddlePointSolverTest, SaddlePointSolverNoCG
    if method_for_solving == "SaddlePointSolverTest":
        solver = SaddlePointSolverTest(
            A, B, F, _A11, _A22, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
        )
        x_u, x_ue, y_uzawa, info = solver()
        x_uzawa={}
        x_uzawa[0]=x_u
        x_uzawa[1]=x_ue
    elif method_for_solving == "SaddlePointSolver":
        solver = SaddlePointSolver(
            A, B, F, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc, recycle=True
        )
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverNoCG":
        solver = SaddlePointSolverNoCG(
            A, B, F, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc, recycle=True
        )
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverInexactUzawa":
        solver = SaddlePointSolverInexactUzawa(
            A,  B, F, _A11, _A22, M2pre, rho=rho, derham = derham, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc, recycle=True
        )
        x_u, x_ue, y_uzawa, info, residual_norms = solver(0.9*x1, 0.9*x2, 1.1*y1_rdm) #0.9*x1, 0.9*x2, 1.1*y1_rdm
        x_uzawa={}
        x_uzawa[0]=x_u
        x_uzawa[1]=x_ue
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverGMRESsolution":
        solver = SaddlePointSolverGMRESsolution(
            A,
            B,
            F,
            rho=rho,
            solver_name=solver_name,
            massmatrix=M2,
            tol=tol,
            x=x,
            y=y1_rdm,
            max_iter=max_iter,
            verbose=verbose,
            pc=pc,
        )
        x_uzawa, y_uzawa, info = solver()
    elif method_for_solving == "SaddlePointSolverGMRES":
        solver = SaddlePointSolverGMRES(
            A, B, F, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
        )
        x_uzawa, y_uzawa, info = solver(dt)
    elif method_for_solving == "SaddlePointSolverGMRESwithPC":
        solver = SaddlePointSolverGMRESwithPC(
            A, B, _A11, _A22, F, M2pre, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc, derham = derham
        )
        x_uzawa, y_uzawa, info = solver(dt,0.9*x1, 0.9*x2, 1.1*y1_rdm)
    elif method_for_solving == "SaddlePointSolverNoCGPaper":
        solver = SaddlePointSolverNoCGPaper(
            A11,
            A22,
            B1,
            B2,
            F1,
            F2,
            Pinit=None,
            rho=rho,
            solver_name=solver_name,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            pc=pc,
        )
        x_u, x_ue, y_uzawa, info, residual_norms = solver()
        x_uzawa={}
        x_uzawa[0]=x_u
        x_uzawa[1]=x_ue

    end_time = time.time()

    print(f"{method_for_solving}{info}")

    elapsed_time = end_time - start_time
    print(f"Method execution time: {elapsed_time:.6f} seconds")
    
    if isinstance(x_uzawa[0], np.ndarray):
        # Output as np.ndarray
        Rx1 = x1.toarray() - x_uzawa[0]
        Rx2 = x2.toarray() - x_uzawa[1]
        Ry = y1_rdm.toarray() - y_uzawa
        residualx_normx1 = np.linalg.norm(Rx1)
        residualx_normx2 = np.linalg.norm(Rx2)
        residualy_norm = np.linalg.norm(Ry)
        TestRest1 = F1.toarray() - A11.toarray_struphy().dot(x_uzawa[0]) - B1T.toarray_struphy().dot(y_uzawa)
        TestRest1val = np.max(abs(TestRest1))
        Testoldy1 = F1.toarray() - A11.toarray_struphy().dot(x_uzawa[0]) - B1T.toarray_struphy().dot(y1_rdm.toarray())
        Testoldy1val = np.max(abs(Testoldy1))
        TestRest2 = F2.toarray() - A22.toarray_struphy().dot(x_uzawa[1]) - B2T.toarray_struphy().dot(y_uzawa)
        TestRest2val = np.max(abs(TestRest2))
        Testoldy2 = F2.toarray() - A22.toarray_struphy().dot(x_uzawa[1]) - B2T.toarray_struphy().dot(y1_rdm.toarray())
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
        print(f'{info = }')
    elif isinstance(x_uzawa[0], BlockVector):
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
        print(f"{TestRest1val =}")
        print(f"{TestRest2val =}")
        print(f"{Testoldy1val =}")
        print(f"{Testoldy2val =}")

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


if __name__ == "__main__":
    # test_saddlepointsolver('SaddlePointSolverGMRES',
    #                        [15, 15, 1],
    #                        [2, 2, 1],
    #                        [True, False, True],
    #                        [[False, False], [False, False], [False, False]],
    #                        ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], True)
    # test_saddlepointsolver('SaddlePointSolverGMRES',
    #                        [3, 4, 5],
    #                        [2, 2, 3],
    #                        [True, False, True],
    #                        [[False,  False], [False, False], [False, False]],
    #                        ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], True)
    # test_saddlepointsolver('SaddlePointSolverGMRES',
    #                        [5, 6, 7],
    #                        [2, 2, 3],
    #                        [True, False, True],
    #                        [[False,  False], [False, False], [False, False]],
    #                        ['Cuboid', {'l1': 0., 'r1': 2., 'l2': 0., 'r2': 3., 'l3': 0., 'r3': 6.}], True)
    # test_saddlepointsolver('SaddlePointSolverGMRES',
    #                        [5, 6, 7],
    #                        [2, 2, 3],
    #                        [True, False, True],
    #                        [[False,  False], [False, False], [False, False]],
    #                        ['Cuboid', {'l1': 0., 'r1': 2., 'l2': 0., 'r2': 3., 'l3': 0., 'r3': 6.}], True)
    # test_saddlepointsolver(
    #     "SaddlePointSolverNoCGPaper",
    #     [3, 4, 5],
    #     [2, 2, 3],
    #     [True, False, True],
    #     [[False, False], [False, False], [False, False]],
    #     [
    #         "Tokamak",
    #         {
    #             "Nel": [3, 4],
    #             "p": [2, 3],
    #             "psi_power": 0.75,
    #             "psi_shifts": [2.0, 2.0, 2.0],
    #             "xi_param": "equal_angle",
    #             "r0": 0.3,
    #         },
    #     ],
    #     True,
    # )
    # test_saddlepointsolver('SaddlePointSolverTest',
    #                        [5, 6, 1],
    #                        [2, 2, 1],
    #                        [True, False, True],
    #                        [[False,  False], [False, False], [False, False]],
    #                        ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], False)
    test_saddlepointsolver('SaddlePointSolverInexactUzawa',
                           [10, 10, 1],
                           [2, 2, 1],
                           [True, False, True],
                           [[False,  False], [False, False], [False, False]],
                           ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], True)
    # test_saddlepointsolver('SaddlePointSolverGMRESwithPC',
    #                        [5, 5, 1],
    #                        [1, 1, 1],
    #                        [True, False, True],
    #                        [[False,  False], [False, False], [False, False]],
    #                        ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], True)

