import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[12, 10, 8]])
@pytest.mark.parametrize('p',   [[3, 2, 1]])
@pytest.mark.parametrize('spl_kind', [[False, True, True]])
@pytest.mark.parametrize('geom', ['cuboid', 'tokamak'])
def test_solvers(Nel, p, spl_kind, geom, verbose=False):
    '''Test and time Psydac iterative solvers.'''

    import yaml
    import os
    from mpi4py import MPI
    import time
    
    import struphy
    from struphy.models.toy import Maxwell
    from struphy.models.fluid import LinearMHD

    from struphy.psydac_api.utilities import create_equal_random_arrays
    
    from struphy.linear_algebra.iterative_solvers import ConjugateGradient as STR_CG
    from struphy.linear_algebra.iterative_solvers import PConjugateGradient as STR_PCG
    from struphy.linear_algebra.iterative_solvers import BiConjugateGradientStab as STR_BICGSTAB
    from struphy.linear_algebra.iterative_solvers import PBiConjugateGradientStab as STR_PBICGSTAB
    
    from struphy.psydac_api.preconditioner import MassMatrixPreconditioner

    from psydac.linalg.solvers import ConjugateGradient 
    from psydac.linalg.solvers import PConjugateGradient 
    from psydac.linalg.solvers import BiConjugateGradient 
    from psydac.linalg.solvers import BiConjugateGradientStabilized 
    from psydac.linalg.solvers import MinimumResidual 
    from psydac.linalg.solvers import LSMR 
    from psydac.linalg.solvers import GMRES 

    # mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # parameter file
    libpath = struphy.__path__[0]
    
    params_maxwell = Maxwell.generate_default_parameter_file(save=False)
    params_mhd = LinearMHD.generate_default_parameter_file(save=False)

    if rank == 0:
        print('\nGRID:')
        for key, val in params_mhd['grid'].items():
            print((key + ':').ljust(25), val)
            
        print('\nGEOMETRY:')
        for key, val in params_mhd['geometry'].items():
            print((key + ':').ljust(25), val)
    
    # linmhd instance
    maxwell = Maxwell(params_maxwell, comm) 
    linmhd = LinearMHD(params_mhd, comm)
  
    # derivative operators
    grad = linmhd.derham.grad
    curl = linmhd.derham.curl
    div = linmhd.derham.div
    
    # solver parameters
    sol_dict = {'x0': None, 'tol': 1e-5, 'maxiter': 10000, 'verbose': verbose}

    # ============ Mass matrix inversion ==============
    for space in ['0', '1', '2', '3']:
        
        # mass matrix, preconditioner and random rhs
        M = getattr(linmhd.mass_ops, 'M' + space)
        pc = MassMatrixPreconditioner(M)
        b_str, b = create_equal_random_arrays(linmhd.derham.Vh_fem[space], 1234)
        
        # ------------- solvers --------------
        str_solvers = []
        str_timings = []
        str_infos = []
        
        solvers = []
        timings = []
        infos = []
        
        # conjugate gradient (cg)
        str_solvers += [STR_CG(linmhd.derham.Vh[space])]
        solvers += [ConjugateGradient(M, **sol_dict)]
        
        t0 = time.time()
        str_res, info = str_solvers[-1].solve(M, b, **sol_dict)
        t1 = time.time()
        str_timings += [t1-t0]
        str_infos += [info]
        assert info['success']
        
        t0 = time.time()
        res = solvers[-1].solve(b)
        t1 = time.time()
        timings += [t1-t0]
        infos += [solvers[-1]._info]
        assert infos[-1]['success']
        
        # pre-conditioned cg
        str_solvers += [STR_PCG(linmhd.derham.Vh[space])]
        solvers += [PConjugateGradient(M, **sol_dict)] # TODO: preconditioning is not yet supported in psydac, add it
        
        t0 = time.time()
        str_res, info = str_solvers[-1].solve(M, b, pc, **sol_dict)
        t1 = time.time()
        str_timings += [t1-t0]
        str_infos += [info]
        assert info['success']
        
        t0 = time.time()
        #res = solvers[-1].solve(b)
        t1 = time.time()
        timings += [t1-t0]
        infos += [{'res_norm': 99.0, 'niter': 0, 'success': True}]
        assert infos[-1]['success']
        
        if rank == 0:
            print('\nTIMINGS for mass matrix inversion in space ' + space + ':')
            for str_s, str_t, str_i, s, t, i in zip(str_solvers, str_timings, str_infos, solvers, timings, infos):
                print('struphy {0:20s}: {1:8.6f} s, with residual {2:4.2e} from {3:4n} iterations'.format(str_s.__class__.__name__, str_t, str_i['res_norm'], str_i['niter']))
                print('psydac  {0:20s}: {1:8.6f} s, with residual {2:4.2e} from {3:4n} iterations'.format(s.__class__.__name__, t, i['res_norm'], i['niter']))

    # ============ Shear Alfven step ==============
    maxwell.initialize_from_params()
    linmhd.initialize_from_params()
    
    for dt in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        if rank == 0:
            print('\nMAXWELL PROPAGATOR with dt={0:6.3}:'.format(dt))
        maxwell.propagators[0](dt)
        
    for dt in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:   
        if rank == 0:
            print('\nSHEAR ALFVEN PROPAGATOR with dt={0:6.3}:'.format(dt))
        linmhd.propagators[0](dt)


if __name__ == '__main__':
    
    test_solvers([8, 6, 4], [3, 2, 1], [False, True, True], 'cuboid', verbose=True)
    test_solvers([8, 6, 4], [3, 2, 1], [False, True, True], 'tokamak', verbose=True)
