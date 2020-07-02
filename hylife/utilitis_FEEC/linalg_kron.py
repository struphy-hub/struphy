import scipy.sparse as sparse
from scipy.sparse.linalg import splu

def kron_matvec_3d(kmat,vec3d):
    ''' matrix-vector product with 3d kronecker matrix with 3d vectors
    
        res_ijk = (A_im * B_jn * C_ko) * vec3d_mno
        
        implemented as three matrix-matrix multiplications with intermediate reshape and transpose.
        step1(v1*v2,k0) <= ( kmat[0](k0,v0) * reshaped_vec3d(v0,v1*v2) )^T
        step2(v2*k0,k1) <= ( kmat[1](k1,v1) * reshaped_step1(v1,v2*k0) )^T
        step3(k0*k1*k2) <= ( kmat[2](k2,v2) * reshaped_step2(v2,k0*k1) )^T
        res <= reshaped_step3(k0,k1,k2)
        
        no overhead of numpy reshape command, as they do NOT copy the data.
    Parameters
        ----------
        kmat  : 3 sparse matrices for each direction, 
                  of size (k0,v0),(k1,v1),(k2,v2)
            
        vec3d : 3d array of size (v0,v1,v2)   
            

        Returns
        -------
        res : 3d array of size (k0,k1,k2)

    '''
    v0,v1,v2=vec3d.shape
    k0=kmat[0].shape[0]
    k1=kmat[1].shape[0]
    k2=kmat[2].shape[0]
    res=((kmat[2].dot(((kmat[1].dot(((kmat[0].dot(vec3d.reshape(v0,v1*v2))).T).reshape(v1,v2*k0))).T).reshape(v2,k0*k1))).T).reshape(k0,k1,k2)
    return res

def kron_lusolve_3d(kmatlu,rhs):
    ''' Solve for 3d vector, matrix would be a 3d kronecker matrix, 
        but LU is only solved in each direction.
        
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
            
        rhs   : 3d array of size (r0,r1,r2), right-hand size
            

        Returns
        -------
        res : 3d array of size (r0,r1,r2), solution 

    '''    
    r0,r1,r2 = rhs.shape
    res=((kmatlu[2].solve(((kmatlu[1].solve(((kmatlu[0].solve(rhs.reshape(r0,r1*r2))).T).reshape(r1,r2*r0))).T).reshape(r2,r0*r1))).T).reshape(r0,r1,r2)
        
    return res


def kron_solve_3d(kmat,rhs):
    ''' Solve for 3d vector, matrix would be a 3d kronecker matrix, 
        but system is only solved in each direction.
        
        solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk
        
        implemented as three matrix-matrix solve with intermediate reshape and transpose.
        step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
        step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
        step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
        res <= reshaped_step3(r0,r1,r2)
        
        no overhead of numpy reshape command, as they do NOT copy the data.
    Parameters
        ----------
        kmat   : 3 sparse matrices for each direction, 
                  of size (r0,r0),(r1,r1),(r2,r2)
            
        rhs   : 3d array of size (r0,r1,r2), right-hand size
            

        Returns
        -------
        res : 3d array of size (r0,r1,r2), solution 

    '''    
    r0,r1,r2 = rhs.shape
    res=((splu(kmat[2]).solve(((splu(kmat[1]).solve(((splu(kmat[0]).solve(rhs.reshape(r0,r1*r2))).T).reshape(r1,r2*r0))).T).reshape(r2,r0*r1))).T).reshape(r0,r1,r2)
    return res
