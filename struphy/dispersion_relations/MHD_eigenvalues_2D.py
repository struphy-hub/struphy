import time

import numpy        as np
import scipy.sparse as spa

from struphy.feec.spline_space import Spline_space_1d
from struphy.feec.spline_space import Tensor_spline_space

from struphy.feec.projectors.pro_global.mhd_operators_cc_lin_6d import MHDOperators


def solve_mhd_ev_problem_2d(num_params, eq_mhd, n_tor, basis_tor='i', dir_out=None):
    """
    Numerical solution of the ideal MHD eigenvalue problem for a given 2D axisymmetric equilibrium and a fixed toroial mode number.
    
    Parameters
    ----------
    num_params : dictionary
        numerical parameters : 
            * Nel      : list of ints, number of elements in [s, chi] direction 
            * p        : list of ints, spline degrees in [s, chi] direction
            * spl_kind : list of booleans, kind of splines in [s, chi] direction 
            * nq_el    : list of ints, number of quadrature points per element in [s, chi] direction 
            * nq_pr    : list of ints, number of quadrature points per projection interval in [s, chi] direction 
            * bc       : list of strings, boundary conditions in [s, chi] direction
            * polar_ck : int, C^k continuity at pole
        
    eq_mhd : MHD equilibrium object
        the MHD equilibrium for which the spectrum shall be computed
        
    n_tor : int
        toroidal mode number
        
    basis_tor : string 
        basis in toroidal direction : 
            * r : A(s, chi)*cos(n_tor*2*pi*phi) + B(s, chi)*sin(n_tor*2*pi*phi), 
            * i : A(s, chi)*exp(n_tor*2*pi*phi*i) 
        
    dir_out : string (optional)
        if given, directory to save the full spectrum as .npy
    """
    
    # extract numerical parameters
    Nel      = num_params['Nel']
    p        = num_params['p']
    spl_kind = num_params['spl_kind']
    nq_el    = num_params['nq_el']
    nq_pr    = num_params['nq_pr']
    bc       = num_params['bc']
    polar_ck = num_params['polar_ck']
    
    print('Numerical parameters : ', num_params)
   
    # set up 1d spline spaces and corresponding projectors 
    space_1d_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc[0])
    space_1d_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1], bc[1])
    
    space_1d_1.set_projectors(nq_pr[0])
    space_1d_2.set_projectors(nq_pr[1])
    
    # set up 2d tensor-product space    
    space_2d = Tensor_spline_space([space_1d_1, space_1d_2], polar_ck, eq_mhd.DOMAIN.cx[:, :, 0], eq_mhd.DOMAIN.cy[:, :, 0], n_tor, basis_tor)
    
    # set up 2d projectors
    space_2d.set_projectors('general')
    
    print('Initialization of FEM spaces done')
    
    # assemble mass matrix in V2 and V3 and apply boundary operators
    space_2d.assemble_Mk(eq_mhd.DOMAIN, 'V2')
    space_2d.assemble_Mk(eq_mhd.DOMAIN, 'V3')
    
    M2_0 = space_2d.B2.dot(space_2d.M2_mat.dot(space_2d.B2.T))
    M3_0 = space_2d.B3.dot(space_2d.M3_mat.dot(space_2d.B3.T))
    
    #print(M2_0.toarray())
    #print(M3_0.toarray())
    
    print('Assembly of mass matrices done')
    
    # create linear MHD operators
    mhd_ops = MHDOperators(space_2d, eq_mhd, 2)
    
    # assemble right-hand sides of degree of freedom projection matrices
    mhd_ops.assemble_dofs('EF')
    mhd_ops.assemble_dofs('MF')
    mhd_ops.assemble_dofs('PF')
    mhd_ops.assemble_dofs('PR')
    
    print('Assembly of projection matrices done')
    
    # assemble mass matrix weighted with 0-form density
    timea = time.time()
    mhd_ops.assemble_Mn()
    timeb = time.time()
    
    print('Assembly of weighted mass matrix done (density), time : ', timeb - timea)
    
    # assemble mass matrix weighted with J_eq x
    timea = time.time()
    mhd_ops.assemble_MJ()
    timeb = time.time()
    
    print('Assembly of weighted mass matrix done (current), time : ', timeb - timea)
    
    # final operators
    I1_11 = spa.kron(space_2d.projectors.I1_pol_0, space_2d.projectors.I_tor)
    I1_22 = spa.kron(space_2d.projectors.I0_pol_0, space_2d.projectors.H_tor)
    
    I2_11 = spa.kron(space_2d.projectors.I2_pol_0, space_2d.projectors.H_tor)
    I2_22 = spa.kron(space_2d.projectors.I3_pol_0, space_2d.projectors.I_tor)
    
    I3    = spa.kron(space_2d.projectors.I3_pol_0, space_2d.projectors.H_tor)
    
    I1 = spa.bmat([[I1_11, None], [None, I1_22]], format='csc')
    I2 = spa.bmat([[I2_11, None], [None, I2_22]], format='csc')
    I3 = I3.tocsc()
    
    EF = spa.linalg.inv(I1).dot(mhd_ops.dofs_EF).tocsr()
    PF = spa.linalg.inv(I2).dot(mhd_ops.dofs_PF).tocsr()
    PR = spa.linalg.inv(I3).dot(mhd_ops.dofs_PR).tocsr()
    
    L  = -space_2d.D0.dot(PF) - (5/3 - 1)*PR.dot(space_2d.D0)
    
    print('Application of inverse interpolation matrices on projection matrices done')
    
    # set up eigenvalue problem MAT*u = omega^2*u
    MAT = spa.linalg.inv(mhd_ops.Mn_mat.tocsc()).dot(EF.T.dot(space_2d.C0.conjugate().T.dot(M2_0.dot(space_2d.C0.dot(EF)))) + mhd_ops.MJ_mat.dot(space_2d.C0.dot(EF)) - space_2d.D0.conjugate().T.dot(M3_0.dot(L))).toarray()
    
    print('Assembly of final system matrix done --> start of eigenvalue calculation')
    
    omega2, U2_eig = np.linalg.eig(MAT)
    
    print('Eigenstates calculated')
    
    # save spectrum as .npy 
    if dir_out != None:
        
        np.save(dir_out, np.hstack((omega2.reshape(omega2.size, 1), U2_eig)))
        
        return 0.
    
    # or return eigenfrequencies, eigenvectors and system matrix
    else:
        return omega2, U2_eig, MAT