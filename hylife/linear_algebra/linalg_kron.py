'''Matrix-vector product and solvers for matrices with a 3D Kronecker product structure.

    M_ijkmno = A_im * B_jn * C_ko
    
    where matrices A, B, C stem from 1D problems.

    COMMENT: the reshape of a matrix can be viewed as ravel+reshape.
        Let r = (r_ijk) be a 3D matrix of size M*N*O.
        ravel(r) = [r_111, r112, ... , r_MNO] (row major always --> last index runs fastest)
        reshape(ravel(r), (M, N*O)) = [[r_111, r112, ... , r_1NO], 
                                        [r_211, r212, ... , r_2NO], 
                                        ...,
                                        [r_M11, rM12, ... , r_MNO]]
'''


from scipy.sparse.linalg import splu
from scipy.linalg import solve_circulant 


def kron_matvec_3d(kmat, vec3d):
    """3D Kronecker matrix-vector product.
    
    res_ijk = (A_im * B_jn * C_ko) * vec3d_mno

    implemented as three matrix-matrix multiplications with intermediate reshape and transpose.
    step1(v1*v2,k0) <= ( kmat[0](k0,v0) * reshaped_vec3d(v0,v1*v2) )^T
    step2(v2*k0,k1) <= ( kmat[1](k1,v1) * reshaped_step1(v1,v2*k0) )^T
    step3(k0*k1*k2) <= ( kmat[2](k2,v2) * reshaped_step2(v2,k0*k1) )^T
    res <= reshaped_step3(k0,k1,k2)

    no overhead of numpy reshape command, as they do NOT copy the data.
    
    Parameters
    ----------
    kmat : 3 sparse matrices for each direction, 
              of size (k0,v0),(k1,v1),(k2,v2)

    vec3d : 3d array of size (v0,v1,v2)   


    Returns
    -------
    res : 3d array of size (k0,k1,k2)
    """
    
    v0, v1, v2 = vec3d.shape
    
    k0 = kmat[0].shape[0]
    k1 = kmat[1].shape[0]
    k2 = kmat[2].shape[0]
    
    res = ((kmat[2].dot(((kmat[1].dot(((kmat[0].dot(vec3d.reshape(v0, v1*v2))).T).reshape(v1, v2*k0))).T).reshape(v2, k0*k1))).T).reshape(k0, k1, k2)
    
    return res


def kron_lusolve_3d(kmatlu, rhs):
    """ 3D Kronecker LU solver.
        
    solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk

    implemented as three matrix-matrix solve with intermediate reshape and transpose.
    step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
    step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
    step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
    res <= reshaped_step3(r0,r1,r2)

    no overhead of numpy reshape command, as they do NOT copy the data.
    
    Parameters
    ----------
    kmatlu : 3 already LU decompositions of sparse matrices for each direction, 
              of size (r0,r0),(r1,r1),(r2,r2)

    rhs : 3d array of size (r0,r1,r2), right-hand size


    Returns
    -------
    res : 3d array of size (r0,r1,r2), solution 
    """
       
    r0, r1, r2 = rhs.shape
    
    res = ((kmatlu[2].solve(((kmatlu[1].solve(((kmatlu[0].solve(rhs.reshape(r0, r1*r2))).T).reshape(r1, r2*r0))).T).reshape(r2, r0*r1))).T).reshape(r0, r1, r2)
        
    return res


def kron_solve_3d(kmat,rhs):
    """3D Kronecker solver.
        
    solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk

    implemented as three matrix-matrix solve with intermediate reshape and transpose.
    step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
    step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
    step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
    res <= reshaped_step3(r0,r1,r2)

    no overhead of numpy reshape command, as they do NOT copy the data.
    
    Parameters
    ----------
    kmat : 3 sparse matrices for each direction, 
              of size (r0,r0),(r1,r1),(r2,r2)

    rhs : 3d array of size (r0,r1,r2), right-hand size


    Returns
    -------
    res : 3d array of size (r0,r1,r2), solution 
    """
        
    r0, r1, r2 = rhs.shape
    
    res = ((splu(kmat[2]).solve(((splu(kmat[1]).solve(((splu(kmat[0]).solve(rhs.reshape(r0, r1*r2))).T).reshape(r1, r2*r0))).T).reshape(r2, r0*r1))).T).reshape(r0, r1, r2)
    
    return res


def kron_fftsolve_3d(cvec,rhs):
    '''3D Kronecker fft solver for circulant matrices.
    
    solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk
    
    implemented as three matrix-matrix solve with intermediate reshape and transpose.
    step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
    step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
    step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
    res <= reshaped_step3(r0,r1,r2)
    
    no overhead of numpy reshape command, as they do NOT copy the data.

    Parameters
        ----------
        cvec   : 3 vectors of size (r0),(r1),(r2) defining 3 circulant matrices for each direction,       
            
        rhs   : 3d array of size (r0,r1,r2), right-hand size
            

        Returns
        -------
        res : 3d array of size (r0,r1,r2), solution 

    '''    
    r0,r1,r2 = rhs.shape
    res=(
            (
                solve_circulant(cvec[2], 
                    (
                        (
                            solve_circulant(cvec[1], 
                                (
                                    (
                                        solve_circulant(cvec[0],
                                            rhs.reshape(r0,r1*r2)
                                        )
                                    ).T
                                ).reshape(r1,r2*r0)
                            )
                        ).T
                    ).reshape(r2,r0*r1)
                )
            ).T
        ).reshape(r0,r1,r2)
    return res