import pytest
from struphy.ode.utils import ButcherTableau

@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("spaces", [('0'), ('1'), ('2'), ('3'), ('v'),])
@pytest.mark.parametrize("algo", ButcherTableau.available_methods())
def test_one_variable(space, algo):
    '''Solve dy/dt = y for different feec variables y and with all available solvers
    from the ButcherTableau.'''

    from mpi4py import MPI
    import numpy as np
    from matplotlib import pyplot as plt

    from struphy.feec.psydac_derham import Derham
    from struphy.ode.solvers import ODEsolverFEEC

    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    Nel = [1, 8, 9]
    p = [1, 2, 3]
    spl_kind = [True]*3
    derham = Derham(Nel, p, spl_kind, comm=comm)
    
    var = derham.Vh[space].zeros()
    if isinstance(var, StencilVector):
        var[:] = 1.0
        var.update_ghost_regions()
    else:
        print('no block ...')
        exit()
        
    out = var.space.zeros()
    def f(t, y1, out=out):
        out *= 0.0
        out += y1
        return out
    
    vector_field = {var: f}
    print(f'{vector_field = }')
    print(f'{algo = }')
    
    solver = ODEsolverFEEC(vector_field, algo=algo)
    
    y_exact = lambda t: np.exp(t)
    
    hs = (.1, .05, .025, .0125)
    Tend = 2
    
    for h in hs:
        time = np.linspace(0, Tend, int(Tend/h) + 1)
        print(f'{h = }, {time.size = }')
        yvec = y_exact(time)
        ymin = {}
        ymax = {}
        for var in vector_field:
            var *= 0.0
            if isinstance(var, StencilVector):
                var[:] = 1.0
                var.update_ghost_regions()
            else:
                print('no block ...')
                exit()
            ymin[var] = np.ones_like(time)
            ymax[var] = np.ones_like(time)
        for n in range(time.size - 1):
            tn = h*n
            solver(tn, h)
            for var in vector_field:
                ymin[var][n + 1] = np.min(var.toarray())
                ymax[var][n + 1] = np.max(var.toarray())
            
        plt.figure()    
        plt.plot(time, yvec, label='exact')
        for j, var in enumerate(vector_field):
            plt.plot(time, ymin[var], label=f'{j} ymin')
            plt.plot(time, ymin[var], '--', label=f'{j} ymax')
        
        plt.legend()    
    plt.show()
  
if __name__ == '__main__':
    test_one_variable('0', 'rk4')
        
        
# @pytest.mark.mpi(min_size=1)
# @pytest.mark.parametrize("spaces", [('0'), ('1'), ('0', '2'), ('1', '3'), ('v', '1', '3'),])
# def test_two_variables(spaces):
#     '''Solve dy/dt = y for different feec variables y and with all available solvers
#     from the ButcherTableau.'''

#     from mpi4py import MPI
#     import numpy as np

#     from struphy.feec.psydac_derham import Derham
#     from struphy.feec.utilities import compare_arrays
#     from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

#     from psydac.linalg.stencil import StencilVector
#     from psydac.linalg.block import BlockVector

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
    
#     Nel = [1, 8, 9]
#     p = [1, 2, 3]
#     spl_kind = [True]*3
#     derham = Derham(Nel, p, spl_kind, comm=comm)
    
#     vars = []
#     fs = []
#     for space in spaces:
#         var = derham.Vh[space].zeros()
#         var += 1.0
#         vars += [var]
        
#         out = var.space.zeros()
#         if len(spaces) == 1:
#             def f(t, y1, out=out):
#                 out *= 0.0
#                 out += y1
#                 return out
#         elif len(spaces) == 2:
#             def f(t, y1, y2, out=out):
#                 out *= 0.0
#                 out += y1
#                 return out
        
#         fs += [f]

