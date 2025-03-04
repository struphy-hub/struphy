def test_saddlepointsolver(method_for_solving, Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Test saddle-point-solver with manufactured solutions."""

    import time

    import numpy as np
    from mpi4py import MPI
    from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace, LinearOperator

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.linear_algebra.saddle_point import (
        SaddlePointSolver,
        SaddlePointSolverGMRES,
        SaddlePointSolverNoCG,
        SaddlePointSolverTest,
        SaddlePointSolverNoCGPaper,
    )

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm)

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    fem_spaces = [derham.Vh_fem["0"], derham.Vh_fem["1"], derham.Vh_fem["2"], derham.Vh_fem["3"], derham.Vh_fem["v"]]

    # create random input array
    x1_rdm_block, x1_rdm = create_equal_random_arrays(fem_spaces[1], seed=1568, flattened=True)
    x2_rdm_block, x2_rdm = create_equal_random_arrays(fem_spaces[1], seed=1568, flattened=True)
    y1_rdm_block, y1_rdm = create_equal_random_arrays(fem_spaces[3], seed=1568, flattened=True)

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain)
    A11 = mass_mats.M2 / 0.01
    A12 = None
    A21 = A12
    A22 = mass_mats.M2
    B1 = derham.div
    B1T = B1.transpose()
    B2 = derham.div
    B2T = B2.transpose()
    x1 = derham.curl.dot(x1_rdm)
    x2 = derham.curl.dot(x2_rdm)
    F1 = A11.dot(x1) + B1T.dot(y1_rdm)
    F2 = A22.dot(x2) + B2T.dot(y1_rdm)

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

    M2preblock = MassMatrixPreconditioner(mass_mats.M2)
    M2pre = BlockLinearOperator(block_domainA, block_codomainA, blocks=[[M2preblock, None], [None, M2preblock]])

    # Create the solver
    rho = 0.0005  # Example descent parameter
    tol = 1e-6
    max_iter = 2500
    pc = None  # M2pre # Preconditioner
    # Conjugate gradient solver 'cg', 'pcg', 'bicg', 'bicgstab', 'minres', 'lsmr', 'gmres'
    solver_name = "gmres"
    verbose = False

    start_time = time.time()

    # SaddlePointSolver, SaddlePointSolverTest, SaddlePointSolverNoCG
    if method_for_solving == "SaddlePointSolverTest":
        solver = SaddlePointSolverTest(
            A, B, F, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
        )
        x_uzawa, y_uzawa, info = solver()
    elif method_for_solving == "SaddlePointSolver":
        solver = SaddlePointSolver(
            A, B, F, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
        )
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverNoCG":
        solver = SaddlePointSolverNoCG(
            A, B, F, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
        )
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverNoCGPaper":
        solver = SaddlePointSolverNoCGPaper(
            A11, A22, B1, B2, F1, F2, Pinit=y1_rdm, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
        )
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == "SaddlePointSolverGMRES":
        solver = SaddlePointSolverGMRES(
            A, B, F, rho=rho, solver_name=solver_name, tol=tol, max_iter=max_iter, verbose=verbose, pc=pc
        )
        x_uzawa, y_uzawa, info = solver()

    end_time = time.time()

    print(f"{method_for_solving}{info}")

    elapsed_time = end_time - start_time
    print(f"Method execution time: {elapsed_time:.6f} seconds")
    Rx = x - x_uzawa
    Ry = y1_rdm - y_uzawa
    residualx_norm = np.linalg.norm(Rx.toarray())
    residualy_norm = np.linalg.norm(Ry.toarray())
    print(f"Residual x norm: {residualx_norm}")
    print(f"Residual y norm: {residualy_norm}")

    compare_arrays(x1, x_uzawa[0].toarray(), mpi_rank, atol=1e-4)
    compare_arrays(x2, x_uzawa[1].toarray(), mpi_rank, atol=1e-4)
    compare_arrays(y1_rdm, y_uzawa.toarray(), mpi_rank, atol=1e-4)


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
    test_saddlepointsolver(
        "SaddlePointSolverNoCGPaper",
        [5, 6, 7],
        [2, 2, 3],
        [True, False, True],
        [[False, True], [True, False], [False, False]],
        ["Colella", {"Lx": 1.0, "Ly": 6.0, "alpha": 0.1, "Lz": 10.0}],
        True,
    )
    # test_saddlepointsolver('SaddlePointSolver',
    #                        [5, 6, 7],
    #                        [2, 2, 3],
    #                        [True, False, True],
    #                        [[False,  True], [True, False], [False, False]],
    #                        ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], False)
