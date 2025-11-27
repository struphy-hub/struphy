from struphy.linear_algebra.multigrid_solver import MultiGridPoissonSolver, build_operator
from struphy.geometry.domains import Cuboid
from struphy.feec.psydac_derham import Derham
import numpy as np
from mpi4py import MPI
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.utilities import create_equal_random_arrays

def test_multigrid_poisson_solver():
    """Test for the MultiGridPoissonSolver class solving a Poisson problem."""
    comm = MPI.COMM_WORLD
    Nel = [16,16,1]
    sp_key = '0'
    plist = [2,2,1]
    spl_kind = [True,True,True]
    N_levels = 3
    domain = Cuboid()
    derham = Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False)
    max_iter = [10, 25, 25]
    N_cycles = 5
    method = 'cg'
    
    # We build the operator A = grad^T * M1 * grad
    A = derham.grad.T @ WeightedMassOperators(derham, domain).M1 @ derham.grad

    # We build the multigrid solver
    multigrid = MultiGridPoissonSolver(derham, sp_key, N_levels, domain, max_iter, N_cycles, method)
    
    # We create a random solution u_star
    _, u_star = create_equal_random_arrays(derham.Vh_fem[sp_key], seed=8765)
    #We compute the rhs
    b = A.dot(u_star)
    b_arr = b.toarray()
    # We solve the system using the multigrid solver
    u = multigrid.solve(b, verbose=True) 
    
    # We check that the solution is correct
    b_ans_arr = A.dot(u).toarray()
    assert np.allclose(b_ans_arr, b_arr, atol=1e-6)

def test_build_operator():
    """Test for the build_operator function."""
    comm = MPI.COMM_WORLD
    Nel = [8,8,1]
    plist = [2,2,1]
    spl_kind = [True,True,True]
    domain = Cuboid()
    derham = Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False)
    
    expr = " grad.T @ M1 @ grad"
    
    namespace = {"grad.T": "derham.grad.T",
    "grad":   "derham.grad",
    "M1":     "WeightedMassOperators(derham, domain).M1"}
    
    
     # 1. Create a local context for the eval. It maps string names
    #    to the actual objects for the CURRENT level.
    eval_context = {
        "derham": derham,
        "domain": domain,
        "WeightedMassOperators": WeightedMassOperators 
    }

    # 2. Build the namespace by evaluating each string from the template
    #    within the context of the current level.
    updated_namespace = {
        key: eval(recipe_string, {}, eval_context)
        for key, recipe_string in namespace.items()
    }
    
    A = build_operator(expr, updated_namespace)
    
    A_ref = derham.grad.T @ WeightedMassOperators(derham, domain).M1 @ derham.grad
    A_arr = A.toarray()
    A_ref_arr = A_ref.toarray()

    assert np.allclose(A_arr, A_ref_arr, atol=1e-10)
    
    
if __name__ == "__main__":
    test_build_operator()
    test_multigrid_poisson_solver()