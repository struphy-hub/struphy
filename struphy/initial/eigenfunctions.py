import os

import numpy as np

from sympde.topology import Line, Derham

from psydac.api.discretization import discretize

from struphy.initial.base import InitialMHD


class InitialMHDAxisymHdivEigFun:
    r"""
    Defines the initial condition via a 2-form MHD velocity field eigenfunction on the logical domain and setting the magnetic field and pressure to zero.
    
    Parameters
    ----------
        params : dict
            Parameters for loading and selecting the desired eigenfunction.
            
            * spec_name : str, relative path to the .npy eigenspectrum
            * eig_freq_upper : float, upper search limit of squared eigenfrequency
            * eig_freq_lower : float, lower search limit of squared eigenfrequency
            * kind : str, whether to use real (r) or imaginary (i) part of eigenfunction
            * scaling : float, scaling factor that is multiplied with the eigenfunction
        
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
    """
    
    def __init__(self, params, derham):
        
        # load eigenvector for velocity field
        omega2, U2_eig = np.split(np.load(os.path.abspath(os.getcwd()) + '/' + params['spec_name']), [1], axis=1)
        omega2 = omega2.flatten()

        # find eigenvector corresponding to given squared eigenfrequency range
        mode = np.where((np.real(omega2) < params['eig_freq_upper']) & 
                        (np.real(omega2) > params['eig_freq_lower']))[0]
        
        assert mode.size == 1
        mode = mode[0]
        
        print('Load eigenfunction with global mode number ' + str(mode) + ' and squared eigenfrequency ' + str(np.real(omega2)[mode]) + ' ...')

        nbasis_v2_0 = [derham.nbasis_v2[0][:2], 
                       derham.nbasis_v2[1][:2], 
                       derham.nbasis_v2[2][:2]] 
        
        if derham.bc[0][0] == 'd': nbasis_v2_0[0][0] -= 1
        if derham.bc[0][1] == 'd': nbasis_v2_0[0][0] -= 1
            
        if derham.bc[1][0] == 'd': nbasis_v2_0[1][1] -= 1
        if derham.bc[1][1] == 'd': nbasis_v2_0[1][1] -= 1
            
        n_v2_0_flat = [nbasis_v2_0[0][0]*nbasis_v2_0[0][1],
                       nbasis_v2_0[1][0]*nbasis_v2_0[1][1],
                       nbasis_v2_0[2][0]*nbasis_v2_0[2][1]]
        
        eig_vec_1 = U2_eig[:n_v2_0_flat[0], mode]
        eig_vec_2 = U2_eig[n_v2_0_flat[0]:n_v2_0_flat[0] + n_v2_0_flat[1], mode]
        eig_vec_3 = U2_eig[n_v2_0_flat[0] + n_v2_0_flat[1]:, mode]

        del omega2, U2_eig

        # project toroidal Fourier modes
        domain_log = Line('L', bounds=(0, 1))
        derham_sym = Derham(domain_log)
        
        domain_log_h = discretize(domain_log, ncells=[derham.Nel[2]])
        derham_1d = discretize(derham_sym, domain_log_h, degree=[derham.p[2]], periodic=[True], quad_order=[derham.quad_order[2]])
        
        p0, p1 = derham_1d.projectors(nquads=[derham.nq_pr[2]])
        
        n_tor = int(params['spec_name'][-13:-11])
        
        N_cos = p0(lambda phi : np.cos(2*np.pi*n_tor*phi)).coeffs.toarray()
        N_sin = p0(lambda phi : np.sin(2*np.pi*n_tor*phi)).coeffs.toarray()

        D_cos = p1(lambda phi : np.cos(2*np.pi*n_tor*phi)).coeffs.toarray()
        D_sin = p1(lambda phi : np.sin(2*np.pi*n_tor*phi)).coeffs.toarray()

        # select real part or imaginary part
        assert params['kind'] == 'r' or params['kind'] == 'i'
        
        if params['kind'] == 'r':
            eig_vec_1 = np.outer(np.real(eig_vec_1), D_cos) - np.outer(np.imag(eig_vec_1), D_sin)
            eig_vec_2 = np.outer(np.real(eig_vec_2), D_cos) - np.outer(np.imag(eig_vec_2), D_sin)
            eig_vec_3 = np.outer(np.real(eig_vec_3), N_cos) - np.outer(np.imag(eig_vec_3), N_sin)
        else:
            eig_vec_1 = np.outer(np.imag(eig_vec_1), D_cos) + np.outer(np.real(eig_vec_1), D_sin)
            eig_vec_2 = np.outer(np.imag(eig_vec_2), D_cos) + np.outer(np.real(eig_vec_2), D_sin)
            eig_vec_3 = np.outer(np.imag(eig_vec_3), N_cos) + np.outer(np.real(eig_vec_3), N_sin)
            
        # set coefficients in full space
        n3 = N_cos.size
        d3 = D_cos.size
        
        nbasis_v2_0[0] += [d3]
        nbasis_v2_0[1] += [d3]
        nbasis_v2_0[2] += [n3]
        
        self._eigvec_1 = np.zeros(derham.nbasis_v2[0], dtype=float)
        self._eigvec_2 = np.zeros(derham.nbasis_v2[1], dtype=float)
        self._eigvec_3 = np.zeros(derham.nbasis_v2[2], dtype=float)
        
        if derham.bc[0][0] == 'd':
            bc1_1 = 1 
        else: 
            bc1_1 = 0
        
        if derham.bc[0][1] == 'd': 
            bc1_2 = 1 
        else: 
            bc1_2 = 0
                
        if derham.bc[1][0] == 'd': 
            bc2_1 = 1 
        else: 
            bc2_1 = 0
            
        if derham.bc[1][1] == 'd':
            bc2_2 = 1 
        else: 
            bc2_2 = 0
        
        self._eigvec_1[bc1_1:derham.nbasis_v2[0][0] - bc1_2, :, :] = eig_vec_1.reshape(nbasis_v2_0[0])*params['scaling']
        self._eigvec_2[:, bc2_1:derham.nbasis_v2[1][1] - bc2_2, :] = eig_vec_2.reshape(nbasis_v2_0[1])*params['scaling']

        self._eigvec_3[:, :, :] = eig_vec_3.reshape(nbasis_v2_0[2])*params['scaling']
    
    @property
    def u2(self):
        """ List of eigenvectors
        """
        return self._eigvec_1, self._eigvec_2, self._eigvec_3
    