import numpy as np

import hylife.utilitis_FEEC.bsplines as bsp


def get_continua(space_2d, omega2, U_eig, omega_A, DIV, div_tol, m_range):
    
    #r_grid = np.linspace(0., 1., 401)
    
    # greville points
    gN = bsp.greville(space_2d.T[0], space_2d.p[0]    , False)
    gD = bsp.greville(space_2d.t[0], space_2d.p[0] - 1, False)
    
    ms = np.arange(m_range[1] - m_range[0] + 1) + m_range[0]
    
    # Alfvén and sound spectra (location, squared frequency, mode number)
    a_spec = [[[], [], []] for m in ms]
    s_spec = [[[], [], []] for m in ms]
    
    # only consider eigenmodes in range omega^2/omega_A^2 = [0, 1]
    modes_ind = np.where((np.real(omega2)/omega_A**2 < 1.0) & (np.real(omega2)/omega_A**2 > 0.0))[0]
    
    for i in range(modes_ind.size):
        
        # determine whether it's an Alfvén branch or sound branch by checking DIV(U)
        if abs(DIV.dot(U_eig[:, modes_ind[i]])).max() < div_tol:
        
            # determine radial location of singularity by looking for a peak in eigenfunction U2_1
            coeff = abs(space_2d.extract_2form(U_eig[:, modes_ind[i]])[0])
            
            r_ind = np.unravel_index(np.argmax(coeff), coeff.shape)[0]
            r = gN[r_ind]
            
            # perform fft to determine m
            U2_1_fft = np.fft.fft(space_2d.extract_2form(U_eig[:, modes_ind[i]])[0])
            
            # determine m by looking for peak in Fourier spectrum at singularity
            m = np.argmax(abs(U2_1_fft[r_ind]))
            
            if m >= space_2d.Nel[1]//2:
                m -= space_2d.Nel[1]
            
            for j in range(ms.size):
                if ms[j] == m:
                    a_spec[j][0].append(r)
                    a_spec[j][1].append(np.real(omega2[modes_ind[i]]))
                    a_spec[j][2].append(modes_ind[i])
                       
        else:
            
            # determine radial location of singularity by looking for a peak in eigenfunction U2_2
            coeff = abs(space_2d.extract_2form(U_eig[:, modes_ind[i]])[1])
            
            r_ind = np.unravel_index(np.argmax(coeff), coeff.shape)[0]
            r = gD[r_ind]
            
            # perform fft to determine m
            U2_2_fft = np.fft.fft(space_2d.extract_2form(U_eig[:, modes_ind[i]])[1])
            
            # determine m by looking for peak in Fourier spectrum at singularity
            m = np.argmax(abs(U2_2_fft[r_ind]))
            
            if m >= space_2d.Nel[1]//2:
                m -= space_2d.Nel[1]
            
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