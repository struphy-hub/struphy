import time

import numpy         as np
import scipy         as sc
import scipy.special as sp
import scipy.sparse  as spa

import hylife.utilitis_FEEC.spline_space as spl

import hylife.utilitis_FEEC.projectors.mhd_operators_2d_global as mhd


# numerical solution of the general ideal MHD eigenvalue problem in a cylinder using a 2d commuting diagram with B-splines
def solve_ev_problem_FEEC_2D(num_params, domain, equilibrium, n, project_profiles=False, return_kind=0, om2=0., dir_out='eigenstates.npy'):
    """
    Parameters
    ----------
    num_params : list
        numerical parameters : 
            num_params[0] : Nel     , number of elements in [radial, angular] direction 
            num_params[1] : p       , spline degrees in [radial, angular] direction
            num_params[2] : spl_kind, kind of splines in [radial, angular] direction 
            num_params[3] : nq_el   , number of quadrature points per element in [radial, angular] direction 
            num_params[4] : nq_pr   , number of quadrature points per projection interval in [radial, angular] direction 
            num_params[5] : bc      , boundary conditions in radial direction
        
    domain : domain object
        the computational domain given by a mapping x = F(eta), typically 'spline cylinder' or 'spline torus'
        
    equilibrium : equilibrium object
        the MHD equilibrium for which the spectrum shall be computed
        
    n : int
        toroidal mode number according to decomposition exp(i*n*phi + i*m*theta) (usually negative)
        
    project_profiles : boolean
        whether to project the equilibrium profiles on the respective splines spaces
        
    return_kind : int
        what to return : 
            0  : all eigenvalues and eigenfunctions, 
            1  : all eigenvalues and eigenfunctions with additional saving as .npy
            
            11 : real part of eigenfunction with squared frequency 'om2' as callable 
            12 : imag part of eigenfunction with squared frequency 'om2' as callable
            
            20 : eigenfunction with squared frequency 'om2' solved as an initial-value problem
            30 : projected equilibrium profiles
            
    om2 : float
        eigenfrequency of mode if return_kind = 11, 12 or 2
        
    dir_out : string
        directory to save eigenstates if return_kind = 1
    """
   
    # set up 1d spline spaces and create 2d tensor-product space
    space_1d_1 = spl.spline_space_1d(num_params[0][0], num_params[1][0], num_params[2][0], num_params[3][0], num_params[5])
    space_1d_2 = spl.spline_space_1d(num_params[0][1], num_params[1][1], num_params[2][1], num_params[3][1])
    space_2d   = spl.tensor_spline_space([space_1d_1, space_1d_2], n)
    
    # set polar splines, discrete derivatives and projectors
    space_2d.set_polar_splines(domain.cx[:, :, 0], domain.cy[:, :, 0])
    space_2d.set_projectors('general', num_params[4])
    
    print('Initialization done')
    
    # assemble mass matrix in V2 and V3
    space_2d.assemble_M2_2D(domain)
    space_2d.assemble_M3_2D(domain)
    
    print('Assembly of mass matrices done')
    
    # create additional splines space without boundary conditions and no dependence in third dimension
    space_1d_1_nobc = spl.spline_space_1d(num_params[0][0], num_params[1][0], num_params[2][0], num_params[3][0])
    space_1d_2_nobc = spl.spline_space_1d(num_params[0][1], num_params[1][1], num_params[2][1], num_params[3][1])
    space_2d_nobc   = spl.tensor_spline_space([space_1d_1_nobc, space_1d_2_nobc], 0)
    space_2d_nobc.set_polar_splines(domain.cx[:, :, 0], domain.cy[:, :, 0])
    
    # load equilibrium profiles
    B1 = [equilibrium.b1_eq_1, equilibrium.b1_eq_2, equilibrium.b1_eq_3]
    B2 = [equilibrium.b2_eq_1, equilibrium.b2_eq_2, equilibrium.b2_eq_3]
    J2 = [equilibrium.j2_eq_1, equilibrium.j2_eq_2, equilibrium.j2_eq_3]
    R3 =  equilibrium.r3_eq
    R0 =  equilibrium.r0_eq
    P3 =  equilibrium.p3_eq
    
    # projection of equilibrium profiles
    if project_profiles:
        B1 = space_2d.projectors.pi_1(B1, True , 'tensor_product')
        B2 = space_2d.projectors.pi_2(B2, True , 'tensor_product')
        
        R3 = space_2d.projectors.pi_3(R3, True , 'tensor_product')
        P3 = space_2d.projectors.pi_3(P3, True , 'tensor_product')
        
        J2 = np.real(space_2d_nobc.C.dot(B1))
        
        if return_kind == 30:
            return B1, B2, R3, P3, J2
        
    print('Loading of MHD equilibrium done')
    
    # create MHD operators
    MHD = mhd.operators_mhd(space_2d.projectors, 2)
    
    # assemble right-hand sides of projection matrices
    MHD.assemble_rhs_EF(domain, B2     )
    MHD.assemble_rhs_F( domain, P3, 'p')
    MHD.assemble_rhs_PR(domain, P3     )
    
    print('Assembly of projection matrices done')
    
    # assemble mass matrix weighted with 0-form density
    timea = time.time()
    MHD.assemble_MR(domain, R0)
    timeb = time.time()
    
    print('Assembly of weighted mass matrix done (density), time : ', timeb - timea)
    
    # assemble mass matrix weighted with J_eq x
    timea = time.time()
    MHD.assemble_JB_strong(domain, J2)
    timeb = time.time()
    
    print('Assembly of weighted mass matrix done (current), time : ', timeb - timea)
    
    # final operators
    EF = spa.linalg.inv(space_2d.projectors.I1.tocsc()).dot(MHD.rhs_EF).tocsr()
    PF = spa.linalg.inv(space_2d.projectors.I2.tocsc()).dot(MHD.rhs_PF).tocsr()
    PR = spa.linalg.inv(space_2d.projectors.I3.tocsc()).dot(MHD.rhs_PR).tocsr()
    
    print('Application of inverse interpolation matrices on projection matrices done')
    
    L  = -space_2d.D.dot(PF) - (equilibrium.gamma - 1)*PR.dot(space_2d.D)

    # ========================= solve eigenvalue problem ========================================
    MAT = spa.linalg.inv(MHD.MR.tocsc()).dot(EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(space_2d.C.dot(EF)))) + MHD.mat_JB.dot(space_2d.C.dot(EF)) - space_2d.D.conjugate().T.dot(space_2d.M3.dot(L))).toarray()
    
    print('Operator assembly finished --> start of eigenvalue calculation')
    
    omega2, eig_vals = np.linalg.eig(MAT)
    
    print('Eigenstates calculated')
    
    if return_kind == 0:
        return omega2, eig_vals
    
    if return_kind == 1:
        np.save(dir_out, np.hstack((omega2.reshape(omega2.size, 1), eig_vals)))
        return omega2, eig_vals
    
    if return_kind == 11:
        
        mode = np.where((np.real(omega2) < om2 + 1e-6) & (np.real(omega2) > om2 - 1e-6))[0][0]

        U2_1 = lambda eta1, eta2, eta3 : np.kron(space_2d.evaluate_ND(eta1, eta2, np.real(eig_vals[:, mode]), 'V2')[:, :, None], np.cos(2*np.pi*n*eta3)) - np.kron(space_2d.evaluate_ND(eta1, eta2, np.imag(eig_vals[:, mode]), 'V2')[:, :, None], np.sin(2*np.pi*n*eta3))
        U2_2 = lambda eta1, eta2, eta3 : np.kron(space_2d.evaluate_DN(eta1, eta2, np.real(eig_vals[:, mode]), 'V2')[:, :, None], np.cos(2*np.pi*n*eta3)) - np.kron(space_2d.evaluate_DN(eta1, eta2, np.imag(eig_vals[:, mode]), 'V2')[:, :, None], np.sin(2*np.pi*n*eta3))
        U2_3 = lambda eta1, eta2, eta3 : np.kron(space_2d.evaluate_DD(eta1, eta2, np.real(eig_vals[:, mode]), 'V2')[:, :, None], np.cos(2*np.pi*n*eta3)) - np.kron(space_2d.evaluate_DD(eta1, eta2, np.imag(eig_vals[:, mode]), 'V2')[:, :, None], np.sin(2*np.pi*n*eta3))

        return U2_1, U2_2, U2_3, np.real(omega2[mode])
    
    if return_kind == 12:
        
        mode = np.where((np.real(omega2) < om2 + 1e-6) & (np.real(omega2) > om2 - 1e-6))[0][0]

        U2_1 = lambda eta1, eta2, eta3 : np.kron(space_2d.evaluate_ND(eta1, eta2, np.imag(eig_vals[:, mode]), 'V2')[:, :, None], np.cos(2*np.pi*n*eta3)) + np.kron(space_2d.evaluate_ND(eta1, eta2, np.real(eig_vals[:, mode]), 'V2')[:, :, None], np.sin(2*np.pi*n*eta3))
        U2_2 = lambda eta1, eta2, eta3 : np.kron(space_2d.evaluate_DN(eta1, eta2, np.imag(eig_vals[:, mode]), 'V2')[:, :, None], np.cos(2*np.pi*n*eta3)) + np.kron(space_2d.evaluate_DN(eta1, eta2, np.real(eig_vals[:, mode]), 'V2')[:, :, None], np.sin(2*np.pi*n*eta3))
        U2_3 = lambda eta1, eta2, eta3 : np.kron(space_2d.evaluate_DD(eta1, eta2, np.imag(eig_vals[:, mode]), 'V2')[:, :, None], np.cos(2*np.pi*n*eta3)) + np.kron(space_2d.evaluate_DD(eta1, eta2, np.real(eig_vals[:, mode]), 'V2')[:, :, None], np.sin(2*np.pi*n*eta3))

        return U2_1, U2_2, U2_3, np.real(omega2[mode])
    
    # ======================== solve initial value problem ======================================
    dt   = 4.0
    Tend = 1000.
    Nt   = int(Tend/dt)
    
    mode = np.where((np.real(omega2) < om2 + 1e-6) & (np.real(omega2) > om2 - 1e-6))[0][0]
    
    eU = np.zeros(Nt + 1, dtype=complex)
    
    u_coeff = np.zeros((Nt + 1, space_2d.M2.shape[0]), dtype=complex)
    b_coeff = np.zeros((Nt + 1, space_2d.M2.shape[0]), dtype=complex)
    p_coeff = np.zeros((Nt + 1, space_2d.M3.shape[0]), dtype=complex)
    
    
    # initialization
    u_coeff[0] = eig_vals[:, mode]
    eU[0] = u_coeff[0].dot(A.dot(np.conj(u_coeff[0])))
    
    #u_coeff[0, :] = projectors_2d.pi_2([u2_1_ini, u2_2_ini, u2_3_ini], False, 'normal')
    
    #return u_coeff[0]
    
    #b_coeff[0] = -space_2d.C.dot(EF.dot(u_coeff[0]))/(-1j*omega_eig)
    #p_coeff[0] = L.dot(u_coeff[0])/(-1j*omega_eig)
    
    #u_coeff[0] = np.concatenate((eig_vals[:(2 + (space_2d.NbaseN[0] - 3)*space_2d.NbaseD[1]), mode1], np.zeros(2*(space_2d.NbaseD[0] - 1)*space_2d.NbaseN[1], dtype=complex)))
    
    
    # create liner operators
    loc_jeq = 'step_2'
    
    S2 = A + dt**2/4*EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(space_2d.C.dot(EF))))
    
    if loc_jeq == 'step_2':
        S2 += dt**2/4*MHD.mat_JB.dot(space_2d.C.dot(EF))
    
    S6 = A - dt**2/4*space_2d.D.conjugate().T.dot(space_2d.M3.dot(L))
    
    S2_LU = spa.linalg.splu(S2.tocsc())
    S6_LU = spa.linalg.splu(S6.tocsc())
    
    print('start time integration')
    
    for i in range(Nt):
        
        if i%100 == 0:
            print(i)
        
        # update 1
        rhs  = A.dot(u_coeff[i]) 
        rhs -= dt**2/4*EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(space_2d.C.dot(EF.dot(u_coeff[i])))))
        rhs += dt*EF.T.dot(space_2d.C.conjugate().T.dot(space_2d.M2.dot(b_coeff[i])))
        
        if loc_jeq == 'step_2': 
            rhs -= dt**2/4*MHD.mat_JB.dot(space_2d.C.dot(EF.dot(u_coeff[i])))
            rhs += dt*MHD.mat_JB.dot(b_coeff[i])
        
        u_coeff[i + 1] = S2_LU.solve(rhs)
        
        b_coeff[i + 1] = b_coeff[i] - dt/2*space_2d.C.dot(EF.dot(u_coeff[i] + u_coeff[i + 1]))
        
        # update 2
        u_old = np.copy(u_coeff[i + 1])
        
        rhs  = A.dot(u_old)
        rhs += dt**2/4*space_2d.D.conjugate().T.dot(space_2d.M3.dot(L.dot(u_old)))
        rhs += dt*space_2d.D.conjugate().T.dot(space_2d.M3.dot(p_coeff[i]))
        
        if loc_jeq == 'step_6': 
            rhs += dt*MHD.mat_JB.dot(b_coeff[i + 1])
        
        u_coeff[i + 1] = S6_LU.solve(rhs)
        
        p_coeff[i + 1] = p_coeff[i] + dt/2*L.dot(u_coeff[i + 1] + u_old)
        
        eU[i + 1] = u_coeff[i + 1].dot(A.dot(np.conj(u_coeff[i + 1])))
    
    return eU, u_coeff