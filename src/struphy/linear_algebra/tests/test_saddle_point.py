
def test_saddlepointsolver(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    '''Test saddle-point-solver with manufactured solutions.'''

    from struphy.linear_algebra.saddle_point import SaddlePointSolver, SaddlePointSolverNoCG #, SaddlePointSolverTest
    from struphy.linear_algebra.saddle_point_uzawa import SaddlePointSolverTest

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.feec.utilities import create_equal_random_arrays, compare_arrays
    from struphy.feec.mass import WeightedMassOperators
    import numpy as np
    import time

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
    print(f"A shape: {A.shape}, A type: {type(A)}")
    B = derham.div
    print(f"B shape: {B.shape}, B type: {type(B)}")
    BT = B.transpose()
    print(f"BT shape: {BT.shape}, BT type: {type(B)}")
    x = derham.curl.dot(x1_rdm)
    F = A.dot(x) + BT.dot(y1_rdm)
    print(f"F shape: {F.shape}, F type: {type(F)}")

    # Create the Uzawa solver
    rho = 0.0001  # Example descent parameter
    tol = 1e-5
    max_iter = 1000
    pc = None  # No preconditioner
    # Conjugate gradient solver 'cg', 'pcg', 'bicg', 'bicgstab', 'minres', 'lsmr', 'gmres'
    solver_name = 'cg'
    verbose = True
    
    start_time = time.time()
    
    #SaddlePointSolver, SaddlePointSolverTest, SaddlePointSolverNoCG
    method_for_solving = 'SaddlePointSolverNoCG'
    if method_for_solving == 'SaddlePointSolverTest':
        solver = SaddlePointSolverTest(A, B, F,
                                rho=rho,
                                solver_name=solver_name,
                                tol=tol,
                                max_iter=max_iter,
                                verbose=verbose,
                                pc=pc)
        x_uzawa, y_uzawa, info = solver()
    elif method_for_solving == 'SaddlePointSolver':
        solver = SaddlePointSolver(A, B, F,
                                rho=rho,
                                solver_name=solver_name,
                                tol=tol,
                                max_iter=max_iter,
                                verbose=verbose,
                                pc=pc)
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)
    elif method_for_solving == 'SaddlePointSolverNoCG':
        solver = SaddlePointSolverNoCG(A, B, F,
                                rho=rho,
                                solver_name=solver_name,
                                tol=tol,
                                max_iter=max_iter,
                                verbose=verbose,
                                pc=pc)
        x_uzawa, y_uzawa, info, residual_norms = solver()
        if show_plots == True:
            _plot_residual_norms(residual_norms)

    end_time = time.time()

    print(f"x shape: {x.shape}, x type: {type(x)}")
    print(f"x_uzawa shape: {x_uzawa.shape}, x_uzawa type: {type(x_uzawa)}")
    print(f"y shape: {y1_rdm.shape}, y type: {type(y1_rdm)}")
    print(f"y_uzawa shape: {y_uzawa.shape}, y_uzawa type: {type(y_uzawa)}")
    print(f"Rank: {mpi_rank}") 
    print(info)
    
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
    test_saddlepointsolver([5, 6, 7],
                           [2, 2, 3],
                           [True, False, True],
                           [[False,  True], [True, False], [False, False]],
                           ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], True)
