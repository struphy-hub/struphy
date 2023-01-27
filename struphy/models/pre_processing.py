import numpy as np

def plasma_params(Z, M, kBT, beta, size_params):
    '''Compute plasma parameters for one species in a magnetized plasma.
    
    Parameters
    ----------
        Z : float
            Charge number Z = q/e, in units of elementary charge, can have a sign.

        M : float
            Mass ratio m/m_p, in units of proton mass.

        kBT : float
            Thermal energy in units keV.

        beta : float
            Plasma beta = p*2*mu_0/|B|^2.

        size_params : dict
            Holding info about plasma size and magnetic field strength.
            
    Returns
    -------
        pparams : dict
            Various plasma parameters of species.
    '''

    pparams = {}

    # physics constants
    e = 1.602176634e-19 # elementary charge (C)
    m_p = 1.67262192369e-27 # proton mass (kg)
    mu0 = 1.25663706212e-6 # magnetic constant (N*A^-2)
    eps0 = 8.8541878128e-12 # vacuum permittivity (F*m^-1)
    kB = 1.380649e-23 # Boltzmann constant (J*K^-1)

    # input
    pparams['charge [e]'] = Z
    pparams['mass [m_p]'] = M
    pparams['kBT [keV]'] = kBT
    pparams['beta'] = beta

    # thermal velocity (10^6 m/s)
    pparams['v_th [10^6 m/s]'] = np.sqrt(kBT*1000*e/(M*m_p))*1e-6

    # cyclotron frequency (MHz)
    pparams['omega_c/(2*pi) [MHz]'] = Z*e*size_params['B_abs [T]']/(M*m_p)/(2*np.pi)*1e-6

    # transit frequency (MHz)
    pparams['omega_th/(2*pi) [MHz]'] = pparams['v_th [10^6 m/s]']*size_params['transit k [1/m]']/(2*np.pi)

    # epsilon = omega_th/omega_c = k*rho = rhostar
    pparams['epsilon'] = pparams['omega_th/(2*pi) [MHz]'] / pparams['omega_c/(2*pi) [MHz]']

    # pressure (bar)
    pparams['p [bar]'] = beta*size_params['B_abs [T]']**2/(2*mu0)*1e-5

    # density (10^20/m^3)
    pparams['n [10^20/m^3]'] = pparams['p [bar]']*1e5/(kBT*1000*e)*1e-20

    # plasma frequency
    pparams['omega_p/(2*pi) [MHz]'] = np.sqrt( pparams['n [10^20/m^3]']*1e20*(Z*e)**2/(eps0*M*m_p) )/(2*np.pi)*1e-6

    # alpha = omega_p/omega_c
    pparams['alpha'] = pparams['omega_p/(2*pi) [MHz]'] / pparams['omega_c/(2*pi) [MHz]']
    
    # Alfvén velocity vA = B/sqrt(M*n*mu0)
    pparams['v_A [10^6 m/s]'] = size_params['B_abs [T]']/np.sqrt(mu0*M*pparams['n [10^20/m^3]']*1e20*m_p)*1e-6
    
    # kappa = e/(m_p*vA)
    pparams['kappa'] = e/(m_p*pparams['v_A [10^6 m/s]']*1e6)

    return pparams

