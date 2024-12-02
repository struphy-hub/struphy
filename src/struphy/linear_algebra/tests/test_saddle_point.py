
def test_saddlepointsolver(method_for_solving, Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    '''Test saddle-point-solver with manufactured solutions.'''

    from struphy.linear_algebra.saddle_point import SaddlePointSolver, SaddlePointSolverNoCG, SaddlePointSolverTest
    
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.feec.utilities import create_equal_random_arrays, compare_arrays
    from struphy.feec.mass import WeightedMassOperators
    import numpy as np
    import time
    from struphy.feec.preconditioner import MassMatrixPreconditioner

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm)

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    fem_spaces = [derham.Vh_fem['0'],
                  derham.Vh_fem['1'],
                  derham.Vh_fem['2'],
                  derham.Vh_fem['3'],
                  derham.Vh_fem['v']]

    # create random input array
    x1_rdm_block, x1_rdm = create_equal_random_arrays(
        fem_spaces[1], seed=1568, flattened=True)
    y1_rdm_block, y1_rdm = create_equal_random_arrays(
        fem_spaces[3], seed=1568, flattened=True)

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain)
    A = mass_mats.M2
    B = derham.div
    BT = B.transpose()
    x = derham.curl.dot(x1_rdm)
    F = A.dot(x) + BT.dot(y1_rdm)
    
    M2pre = MassMatrixPreconditioner(mass_mats.M2)

    # Create the solver
    rho = 0.001  # Example descent parameter
    tol = 1e-5
    max_iter = 1000
    pc = M2pre  # Preconditioner
    # Conjugate gradient solver 'cg', 'pcg', 'bicg', 'bicgstab', 'minres', 'lsmr', 'gmres'
    solver_name = 'pcg'
    verbose = False
    
    
    start_time = time.time()
    
    #SaddlePointSolver, SaddlePointSolverTest, SaddlePointSolverNoCG
    if method_for_solving == 'SaddlePointSolverTest':
        count = 0
        solver = SaddlePointSolverTest(A, B, F,
                                rho=rho,
                                solver_name=solver_name,
                                tol=tol,
                                max_iter=max_iter,
                                verbose=verbose,
                                pc=pc,
                                count=count)
        x_uzawa, y_uzawa, info = solver()
    elif method_for_solving == 'SaddlePointSolver':
        count = 0
        solver = SaddlePointSolver(A, B, F,
                                rho=rho,
                                solver_name=solver_name,
                                tol=tol,
                                max_iter=max_iter,
                                verbose=verbose,
                                pc=pc,
                                count=count)
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == 'SaddlePointSolverNoCG':
        count = 0
        solver = SaddlePointSolverNoCG(A, B, F,
                                rho=rho,
                                solver_name=solver_name,
                                tol=tol,
                                max_iter=max_iter,
                                verbose=verbose,
                                pc=pc,
                                count=count)
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)

    end_time = time.time()

    print(f"{method_for_solving}{info}")
    
    elapsed_time = end_time - start_time
    print(f"Method execution time: {elapsed_time:.6f} seconds")
    Rx=x-x_uzawa
    Ry=y1_rdm-y_uzawa
    residualx_norm = np.linalg.norm(Rx.toarray())
    residualy_norm = np.linalg.norm(Ry.toarray())
    print(f"Residual x norm: {residualx_norm}")
    print(f"Residual y norm: {residualy_norm}")

    compare_arrays(x, x_uzawa.toarray(), mpi_rank, atol=1e-3)
    compare_arrays(y1_rdm, y_uzawa.toarray(), mpi_rank, atol=1e-3)
    
def _plot_residual_norms(residual_norms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.plot(residual_norms, label="Residual Norm")
    plt.yscale('log')  # Use logarithmic scale for better visualization
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Convergence of Residual Norm")
    plt.legend()
    plt.grid(True)
    plt.savefig("residual_norms_plot.png")


if __name__ == '__main__':
    test_saddlepointsolver('SaddlePointSolverTest',
                           [5, 6, 7],
                           [2, 2, 3],
                           [True, False, True],
                           [[False,  True], [True, False], [False, False]],
                           ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], False)
    test_saddlepointsolver('SaddlePointSolver',
                           [5, 6, 7],
                           [2, 2, 3],
                           [True, False, True],
                           [[False,  True], [True, False], [False, False]],
                           ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], False)
