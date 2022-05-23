import numpy as np

import struphy.feec.bsplines as bsp


def get_mhd_continua_2d(space, domain, omega2, U_eig, m_range, omega_A, div_tol, comp_sound):
    """
    Get the eigenfrequencies omega^2/omega_A^2 in the range (0, 1) sorted by shear Alfvén modes and slow sound modes.
    
    Parameters
    ----------
    space : struphy object
        2d finite element B-spline space.
        
    domain : struphy object
        the domain in which the eigenvalue problem has been solved.
        
    omega2 : 1d array
        eigenfrequencies obtained from eigenvalue solver.
        
    U_eig : 2d array
        eigenvectors obtained from eigenvalue solver.
        
    m_range : list
        the range of poloidal mode numbers that shall be identified.
        
    omega_A : float 
        Alfvén frequency B0/R0.
    
    div_tol : float
        threshold for the maximum divergence of an eigenmode below which it is considered to be an Alfvénic mode.
        
    comp_sound : int
        the component that is used for the slow sound mode analysis (2 : 2nd component or 3 : third component).
        
    Returns
    -------
        a_spec : list of 2d numpy arrays
            the radial location a_spec[m][0], squared eigenfrequencis a_spec[m][1] and global mode index a_spec[m][2] corresponding to shear Alfvén modes for each poloidal mode number m in m_range.
            
        s_spec : list of 2d numpy arrays
            the radial location s_spec[m][0], squared eigenfrequencis s_spec[m][1] and global mode index s_spec[m][2] corresponding to slow sound modes for each poloidal mode number m in m_range.
    """
    
    # greville points in radial direction
    gN_1 = bsp.greville(space.T[0], space.p[0]    , space.spl_kind[0])
    gD_1 = bsp.greville(space.t[0], space.p[0] - 1, space.spl_kind[0])
    
    # greville points in angular direction
    gN_2 = bsp.greville(space.T[1], space.p[1]    , space.spl_kind[1])
    gD_2 = bsp.greville(space.t[1], space.p[1] - 1, space.spl_kind[1])
    
    # poloidal mode numbers
    ms = np.arange(m_range[1] - m_range[0] + 1) + m_range[0]
    
    # grid for normalized Jacobian determinant
    det_df = domain.evaluate(gD_1, gD_2, 0., 'det_df')
    
    # remove singularity for polar domains
    if domain.pole:
        det_df = det_df[1:, :]
    
    det_df_norm = det_df/det_df.max()
    
    # Alfvén and sound spectra (location, squared frequency, mode number)
    a_spec = [[[], [], []] for m in ms]
    s_spec = [[[], [], []] for m in ms]
    
    # only consider eigenmodes in range omega^2/omega_A^2 = [0, 1]
    modes_ind = np.where((np.real(omega2)/omega_A**2 < 1.0) & (np.real(omega2)/omega_A**2 > 0.0))[0]
    
    for i in range(modes_ind.size):
        
        # determine whether it's an Alfvén branch or sound branch by checking DIV(U)
        if space.ck == 0:
            divU = space.D0.dot(U_eig[:, modes_ind[i]])[space.NbaseD[1]:]
        else:
            divU = space.D0.dot(U_eig[:, modes_ind[i]])
            
        
        # Alfvén branch
        if abs(divU/det_df_norm.flatten()).max() < div_tol:
            
            # get FEM coefficients (1st component)
            U2_1_coeff = space.extract_2(U_eig[:, modes_ind[i]])[0]
            
            if space.basis_tor == 'i':
                U2_1_coeff =  U2_1_coeff[:, :, 0]
            else:
                U2_1_coeff = (U2_1_coeff[:, :, 0] - 1j*U2_1_coeff[:, :, 1])/2
        
            # determine radial location of singularity by looking for a peak in eigenfunction U2_1
            r_ind = np.unravel_index(np.argmax(abs(U2_1_coeff)), U2_1_coeff.shape)[0]
            r = gN_1[r_ind]
            
            # perform fft to determine m
            U2_1_fft = np.fft.fft(U2_1_coeff)
            
            # determine m by looking for peak in Fourier spectrum at singularity
            m = np.argmax(abs(U2_1_fft[r_ind]))
            
            # perform shift for negative m
            if m >= (space.Nel[1] + 1)//2:
                m -= space.Nel[1]
            
            # add to spectrum if found m is inside m_range
            for j in range(ms.size):
                if ms[j] == m:
                    a_spec[j][0].append(r)
                    a_spec[j][1].append(np.real(omega2[modes_ind[i]]))
                    a_spec[j][2].append(modes_ind[i])
        
        # Sound branch
        else:
            
            # get FEM coefficients (2nd component or 3rd component)
            U2_coeff = space.extract_2(U_eig[:, modes_ind[i]])[comp_sound - 1]
            
            if space.basis_tor == 'i':
                U2_coeff =  U2_coeff[:, :, 0]
            else:
                U2_coeff = (U2_coeff[:, :, 0] - 1j*U2_coeff[:, :, 1])/2
            
            # determine radial location of singularity by looking for a peak in eigenfunction (U2_2 or U2_3)
            r_ind = np.unravel_index(np.argmax(abs(U2_coeff)), U2_coeff.shape)[0]
            r = gD_1[r_ind]
            
            # perform fft to determine m
            U2_fft = np.fft.fft(U2_coeff)
            
            # determine m by looking for peak in Fourier spectrum at singularity
            m = np.argmax(abs(U2_fft[r_ind]))
            
            # perform shift for negative m
            if m >= (space.Nel[1] + 1)//2:
                m -= space.Nel[1]
            
            # add to spectrum if found m is inside m_range
            for j in range(ms.size):
                if ms[j] == m:
                    s_spec[j][0].append(r)
                    s_spec[j][1].append(np.real(omega2[modes_ind[i]]))
                    s_spec[j][2].append(modes_ind[i])
                 
    
    # convert to array
    for j in range(ms.size):
        a_spec[j] = np.array(a_spec[j])
        s_spec[j] = np.array(s_spec[j])
    
    return a_spec, s_spec