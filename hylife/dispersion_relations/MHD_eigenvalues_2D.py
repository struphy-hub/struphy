import numpy         as np
import scipy         as sc
import scipy.special as sp
import scipy.sparse  as spa

import hylife.utilitis_FEEC.Bspline                  as Bsp
import hylife.utilitis_FEEC.bsplines                 as bsp
import hylife.utilitis_FEEC.spline_space             as spl
import hylife.utilitis_FEEC.derivatives.derivatives  as der

import hylife.utilitis_FEEC.basics.mass_matrices_2d  as mass_2d
import hylife.utilitis_FEEC.basics.inner_products_2d as inner_2d

import hylife.utilitis_FEEC.projectors.projectors_global             as pro
import hylife.utilitis_FEEC.projectors.kernels_projectors_global_mhd as ker

import hylife.utilitis_FEEC.projectors.mhd_operators_2d as mhd

import hylife.geometry.domain_3d as dom
import hylife.geometry.polar_splines as pol

import hylife.geometry.mappings_3d as mapping


# numerical solution of the general ideal MHD eigenvalue problem in a cylinder using a 2d commuting diagram with B-splines
def solve_ev_problem_FEEC_2D(num_params, domain, equilibrium, gamma, n):
    """
    Parameters
    ----------
    num_params : list
        numerical parameters : num_params[0] : Nel, num_params[1] : p, num_params[2] : spl_kind, num_params[3] : nq_el, num_params[4] : nq_pr, num_params[5] : bc
    """
   
    # set up 1d spline spaces and create 2d tensor-product space
    space_1d = [spl.spline_space_1d(Nel, p, bc, nq) for Nel, p, bc, nq in zip(num_params[0], num_params[1], num_params[2], num_params[3])]
    space_2d =  spl.tensor_spline_space(space_1d)
    
    # create polar_splines extraction operators
    polar_splines = pol.polar_splines_2D(space_2d, domain.cx[:, :, 0], domain.cy[:, :, 0])
    
    # set boundary conditions and extraction operators
    space_2d.set_extraction_operators(num_params[5], polar_splines)
    
    # set discrete derivatives
    space_2d.set_derivatives_2D(n)

    # assemble mass matrix in V2 and V3
    space_2d.assemble_M2_2D(domain)
    space_2d.assemble_M3_2D(domain)
    
    # load equilibrium
    B1   = [equilibrium.b1_eq_1, equilibrium.b1_eq_2, equilibrium.b1_eq_3]
    B2   = [equilibrium.b2_eq_1, equilibrium.b2_eq_2, equilibrium.b2_eq_3]
    J2   = [equilibrium.j2_eq_1, equilibrium.j2_eq_2, equilibrium.j2_eq_3]
    Rho3 =  equilibrium.rho3_eq
    P3   =  equilibrium.p3_eq
    
    # 2D polar projectors
    projectors_2d = pro.projectors_global_2d(space_2d, num_params[4])
    
    # projection of equilibrium profiles
    b1    = projectors_2d.pi_1(B1  , True , 'tp')
    b2    = projectors_2d.pi_2(B2  , True , 'tp')
    b2_bc = projectors_2d.pi_2(B2  , False, 'tp')
    
    #j2    = projectors_2d.pi_2(J2  , True , 'tp')
      
    r3    = projectors_2d.pi_3(Rho3, True , 'tp')
    p3    = projectors_2d.pi_3(P3  , True , 'tp')
    
    # equilibrium current curl(b1)
    j2 = space_2d.C_all.dot(b1)
    
    #return j2
    
    MHD = mhd.operators_mhd(projectors_2d, 2)
    MHD.assemble_rhs_EF(domain, B2)
    MHD.assemble_rhs_F( domain, Rho3, 'm')
    MHD.assemble_rhs_F( domain, P3, 'p')
    MHD.assemble_rhs_PR(domain, P3)
    MHD.assemble_TF_V2( domain, J2)
    
    EF = spa.linalg.inv(projectors_2d.I1.tocsc()).dot(MHD.rhs_EF).tocsr()
    MF = spa.linalg.inv(projectors_2d.I2.tocsc()).dot(MHD.rhs_MF).tocsr()
    PF = spa.linalg.inv(projectors_2d.I2.tocsc()).dot(MHD.rhs_PF).tocsr()
    PR = spa.linalg.inv(projectors_2d.I3.tocsc()).dot(MHD.rhs_PR).tocsr()
    
    MF = (space_2d.M2.dot(MF) + MF.T.dot(space_2d.M2))/2
    
    L = -space_2d.D.dot(PF) - (gamma - 1)*PR.dot(space_2d.D)

    # ========================= solve eigenvalue problem ========================================
    MAT = np.linalg.inv(MF.toarray()).dot(EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(space_2d.C.dot(EF)))).toarray() + MHD.mat_TF.dot(space_2d.C.dot(EF)).toarray() - space_2d.D.conjugate().T.dot(space_2d.M3.dot(L)).toarray())
    
    print('operator assembly done')
    
    omega2, eig_vals = np.linalg.eig(MAT)
    
    print('eigenstates calculated')
    
    return omega2, eig_vals
    
    mode1 = 466
    print(np.real(omega2[mode1]))
    
    
    U2_1 = lambda eta1, eta2, eta3 : space_2d.evaluate_ND(eta1, eta2, np.real(eig_vals[:, mode1]), 'V2')*np.cos(2*np.pi*n*eta3) - space_2d.evaluate_ND(eta1, eta2, np.imag(eig_vals[:, mode1]), 'V2')*np.sin(2*np.pi*n*eta3)
    U2_2 = lambda eta1, eta2, eta3 : space_2d.evaluate_DN(eta1, eta2, np.real(eig_vals[:, mode1]), 'V2')*np.cos(2*np.pi*n*eta3) - space_2d.evaluate_DN(eta1, eta2, np.imag(eig_vals[:, mode1]), 'V2')*np.sin(2*np.pi*n*eta3)
    U2_3 = lambda eta1, eta2, eta3 : space_2d.evaluate_DD(eta1, eta2, np.real(eig_vals[:, mode1]), 'V2')*np.cos(2*np.pi*n*eta3) - space_2d.evaluate_DD(eta1, eta2, np.imag(eig_vals[:, mode1]), 'V2')*np.sin(2*np.pi*n*eta3)
    
    return U2_1, U2_2, U2_3
    
    # ======================== solve initial value problem ======================================
    dt   = 0.01
    Tend = 10.
    Nt   = int(Tend/dt)
    
    eU = np.zeros(Nt + 1, dtype=complex)
    
    u_coeff = np.zeros((Nt + 1, eig_vals.shape[0]), dtype=complex)
    b_coeff = np.zeros((Nt + 1, eig_vals.shape[0]), dtype=complex)
    p_coeff = np.zeros((Nt + 1, L.shape[0])  , dtype=complex)
    
    mode1 = 118
    #mode2 = 254
    
    omega_eig = np.sqrt(np.real(omega2[mode1]))
    print(omega_eig)
    
    u_coeff[0] = np.imag(eig_vals[:, mode1])
    #b_coeff[0] = -space_2d.C.dot(EF.dot(u_coeff[0]))/(-1j*omega_eig)
    #p_coeff[0] = L.dot(u_coeff[0])/(-1j*omega_eig)
    
    #u_coeff[0] = np.concatenate((eig_vals[:(2 + (space_2d.NbaseN[0] - 3)*space_2d.NbaseD[1]), mode1], np.zeros(2*(space_2d.NbaseD[0] - 1)*space_2d.NbaseN[1], dtype=complex)))
    
    #u_coeff[0] = np.random.rand(u_coeff[0].size)
    
    eU[0] = u_coeff[0].dot(MF.dot(np.conj(u_coeff[0])))
    
    S2    = MF + dt**2/4*EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(space_2d.C.dot(EF)))) + dt**2/4*MHD.mat_TF.dot(space_2d.C.dot(EF))
    S2_LU = spa.linalg.splu(S2.tocsc())
    
    S6    = MF - dt**2/4*space_2d.D.conjugate().T.dot(space_2d.M3.dot(L))
    S6_LU = spa.linalg.splu(S6.tocsc())
    
    print('start time integration')
    
    for i in range(Nt):
        
        if i%100 == 0:
            print(i)
        
        # update 1
        rhs = MF.dot(u_coeff[i]) - dt**2/4*EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(space_2d.C.dot(EF.dot(u_coeff[i]))))) - dt**2/4*MHD.mat_TF.dot(space_2d.C.dot(EF.dot(u_coeff[i]))) + dt*EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(b_coeff[i]))) + dt*MHD.mat_TF.dot(b_coeff[i])
        
        u_coeff[i + 1] = S2_LU.solve(rhs)
        
        b_coeff[i + 1] = b_coeff[i] - dt/2*space_2d.C.dot(EF.dot(u_coeff[i] + u_coeff[i + 1]))
        
        # update 2
        u_old = np.copy(u_coeff[i + 1])
        
        rhs = MF.dot(u_old) + dt**2/4*space_2d.D.conjugate().T.dot(space_2d.M3.dot(L.dot(u_old))) + dt*space_2d.D.conjugate().T.dot(space_2d.M3.dot(p_coeff[i]))
        
        u_coeff[i + 1] = S6_LU.solve(rhs)
        
        p_coeff[i + 1] = p_coeff[i] + dt/2*L.dot(u_coeff[i + 1] + u_old)
        
        eU[i + 1] = u_coeff[i + 1].dot(MF.dot(np.conj(u_coeff[i + 1])))
    
    return eU