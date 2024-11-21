
def test_saddlepointsolver(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    '''Test saddle-point-solver with manufactured solutions.'''

    from struphy.linear_algebra.saddle_point import SaddlePointSolver, SaddlePointSolverTest

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.feec.utilities import create_equal_random_arrays, compare_arrays
    from struphy.feec.mass import WeightedMassOperators
    import numpy as np
    from psydac.linalg.basic import InverseLinearOperator
    from psydac.linalg.block import BlockLinearOperator, BlockVectorSpace, BlockVector

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
    x2_rdm_block, x2_rdm = create_equal_random_arrays(
        fem_spaces[1], seed=1568, flattened=True)
    y1_rdm_block, y1_rdm = create_equal_random_arrays(
        fem_spaces[3], seed=1568, flattened=True)

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain)
    A11 = mass_mats.M2
    A12 = A11.copy()*0
    A22 = mass_mats.M2
    B1 = derham.div
    B1T = B1.transpose()
    B2 = derham.div
    B2T = B2.transpose()
    x1 = derham.curl.dot(x1_rdm)
    x2 = derham.curl.dot(x2_rdm)
    F1 = A11.dot(x1) + B1T.dot(y1_rdm)
    F2 = A22.dot(x2) + B2T.dot(y1_rdm)
    print(f"A11 shape: {A11.shape}, A type: {type(A11)}")
    print(f"B1 shape: {B1.shape}, B type: {type(B1)}")
    print(f"B1T shape: {B1T.shape}, BT type: {type(B1T)}")
    print(f"F1 shape: {F1.shape}, F type: {type(F1)}")
    
    print(f"A11 domain: {A11.domain}, codomain: {A11.codomain}")
    print(f"A12 domain: {A12.domain}, codomain: {A12.codomain}")
    print(f"BlockLinearOperator domain[0]: {derham.Vh['1'].spaces[0]}")
    print(f"BlockLinearOperator domain[1]: {derham.Vh['1'].spaces[1]}")
    
    block_domainA = BlockVectorSpace(A11.domain, A22.domain)
    block_codomainA = BlockVectorSpace(A11.codomain, A22.codomain)
    block_domainB = BlockVectorSpace(B1.domain)

    A = BlockLinearOperator(block_domainA, block_codomainA, blocks=[[A11, A12], [A12, A22]])
    B = BlockVector(block_domainB, blocks = [[B1],[B2]])
    F = BlockVector(block_domainB, blocks = [F1,F2])
    

    # Create the Uzawa solver
    rho = 0.01  # Example descent parameter
    tol = 1e-6
    max_iter = 1000
    pc = None  # No preconditioner
    # Conjugate gradient solver 'cg', 'pcg', 'bicg', 'bicgstab', 'minres', 'lsmr', 'gmres'
    solver_name = 'cg'
    verbose = True

    solver = SaddlePointSolverTest(A, B, F,
                               rho=rho,
                               solver_name=solver_name,
                               tol=tol,
                               max_iter=max_iter,
                               verbose=verbose,
                               pc=pc)

    x_uzawa, y_uzawa, info = solver()

    print(f"x shape: {x1.shape}, x type: {type(x1)}")
    print(f"x_uzawa shape: {x_uzawa.shape}, x_uzawa type: {type(x_uzawa)}")
    print(f"y shape: {y1_rdm.shape}, y type: {type(y1_rdm)}")
    print(f"y_uzawa shape: {y_uzawa.shape}, y_uzawa type: {type(y_uzawa)}")
    print(f"Rank: {mpi_rank}")
    
    Rx=x1-x_uzawa
    Ry=y1_rdm-y_uzawa
    residualx_norm = np.linalg.norm(Rx.toarray())
    residualy_norm = np.linalg.norm(Ry.toarray())
    print(f"Residual x norm: {residualx_norm}")
    print(f"Residual y norm: {residualy_norm}")
    

    compare_arrays(x1, x_uzawa.toarray(), mpi_rank, atol=1e-4)
    compare_arrays(y1_rdm, y_uzawa.toarray(), mpi_rank, atol=1e-4)


if __name__ == '__main__':
    test_saddlepointsolver([5, 6, 7],
                           [2, 2, 3],
                           [True, False, True],
                           [[False,  True], [True, False], [False, False]],
                           ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], False)
