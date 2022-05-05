from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.special as sp


class DispersionRelations1D( metaclass=ABCMeta ):
    '''The base class for analytic 1d dispersion relations.
    
    Parameters
    ..........
        branch_names : list
            Strings denoting the branches in the spectrum.
        
        kwargs : 
            Keyword arguments describing the physical parameters necessary to compute the disp. rel., e.g. c=299792458.
    '''

    def __init__(self, branch_names, **kwargs):

        self._branches = branch_names
        self._nbranches = len(branch_names)
        self._params = kwargs

    @property
    def branches( self ):
        '''List of branch names in the spectrum.'''
        return self._branches

    @property
    def nbranches( self ):
        '''Integer: number of branches.'''
        return self._nbranches

    @property
    def params( self ):
        '''Dictionary of parameters necessary to compute the dispersion relation.'''
        return self._params

    @abstractmethod
    def spectrum(self, kvec, kperp=None):
        '''The calculation of all branches of a 1d dispersion relation.
        
        Parameters
        ----------
            kvec : np.array
                Wave numbers.
                
            kperp : np.array
                Optional: perpendicular wave numbers (w.r.t to background magnetic field).
                kperp.size=kvec.size
                
        Returns
        -------
            A dictionary with key=name_of_branch and value=np.array(omega_values_of_branch).
            value.size=kvec.size. 
            value can be complex-valued.'''


class Maxwell1D( DispersionRelations1D ):
    '''Dispersion relation for Maxwell's equation in vacuum, in Struphy normalization (c=1).'''

    def __init__(self):
        super().__init__(['light wave', 'test branch'], c=1., test_param=.5) 

    def spectrum(self, kvec, kperp=None):

        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [np.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################
        # first branch
        tmps[0][:] = self.params['c']*kvec
        # second branch (for testing)
        tmps[1][:] = self.params['test_param']*kvec
        ##############################################################

        # fill output dictionary
        dict = {}
        for name, tmp in zip(self.branches, tmps):
            dict[name] = tmp

        return dict


def maxwell_1d(k):
    '''Dispersion relation for the Struphy normalization of Maxwell's equation.
    
    Parameters
    ----------
        k : np.array
            1d array of wave vector values.
            
    Returns
    -------
        A list of np.arrays of same size as k, each entry corresponding to one branch of the spectrum.'''

    c = 1.



    return [c*k, .5*c*k]

# = dispersion relation for fast and slow magnetosonic waves propagating in the x-y-plane (B = B0x)=
def omegaM_xy(kx, ky, pol, B0x, p0, rho0, gamma):
                                           
    # speed of sound
    cS    = np.sqrt(gamma*p0/rho0) 
    
    # Alfvén velocity
    vA    = np.sqrt(B0x**2/rho0) 
    
    delta = (4*kx**2*cS**2*vA**2)/((cS**2 + vA**2)**2*(kx**2 + ky**2))
    
    return np.sqrt(1/2*(kx**2 + ky**2)*(cS**2 + vA**2)*(1 + pol*np.sqrt(1 - delta)))


# ====== dispersion relation for shear Alfvén waves propagating in the x-y-plane (B = B0x) =========
def omegaS_xy(kx, B0x, rho0):
    
    # Alfvén velocity
    vA    = np.sqrt(B0x**2/rho0) 
    
    return vA*kx


# ==== dispersion relation for fast and slow magnetosonic waves propagating along the x-axis =======
def omegaM(k, pol, B0, p0, rho0, gamma):
    
    B0x   = B0[0]
    B0y   = B0[1]
    B0z   = B0[2]
                                           
    # speed of sound
    cS    = np.sqrt(gamma*p0/rho0) 
    
    # Alfvén velocity
    vA    = np.sqrt((B0x**2 + B0y**2 + B0z**2)/rho0) 
    
    delta = (4*B0x**2*cS**2*vA**2)/((cS**2 + vA**2)**2*(B0x**2 + B0y**2 + B0z**2))
    
    return np.sqrt(1/2*k**2*(cS**2 + vA**2)*(1 + pol*np.sqrt(1 - delta)))


# ====== dispersion relation for shear Alfvén waves propagating along the x-axis ===================
def omegaS(k, B0, rho0):
    
    B0x   = B0[0]
    B0y   = B0[1]
    B0z   = B0[2]
    
    # Alfvén velocity
    vA    = np.sqrt((B0x**2 + B0y**2 + B0z**2)/rho0) 
    
    return vA*k*B0x/np.sqrt(B0x**2 + B0y**2 + B0z**2)


# ========= dispersion relation for shear Alfvén waves + full-orbit energetic ions ==================
def solveDispersionFullOrbit(k, pol, wch, vA, vth, v0, nuh, Ah, Zh, AMHD, initial_guess, tol, max_it=100):
    
    # plasma dispersion function
    def Z(xi):
        return np.sqrt(np.pi)*np.exp(-xi**2)*(1j - sp.erfi(xi))

    # derivative of plasma dispersion function
    def Zprime(xi):
        return -2*(1 + xi*Z(xi))
    
    # dispersion relation D(k, w) = 0
    def D(k, w, pol):
        xi = (w - k*v0 + pol*wch)/(k*vth)
        
        return 1 - vA**2*k**2/w**2 + pol*Zh*nuh*wch/(AMHD*w) + nuh*wch**2*Zh**2/(Ah*AMHD*w**2)*(w - k*v0)/(k*vth)*Z(xi)
    
    # derivative of dispersion relation with respect to w
    def Dprime(k, w, pol):
        xi  = (w - k*v0 + pol*wch)/(k*vth)
        xip = 1/(k*vth)
        
        return 2*vA**2*k**2/w**3 - pol*Zh*nuh*wch/(AMHD*w**2) - 2*nuh*wch**2*Zh**2/(Ah*AMHD*w**3)*(w - k*v0)/(k*vth)*Z(xi) + nuh*wch**2*Zh**2/(Ah*AMHD*w**2)*1/(k*vth)*Z(xi) + nuh*wch**2*Zh**2/(Ah*AMHD*w**2)*(w - k*v0)/(k*vth)*Zprime(xi)*xip
    
    
    # solve dispersion relation with Newton method
    w = initial_guess
    counter = 0
    
    while np.abs(D(k, w, pol)) > tol or counter == max_it:
        
        w = w - D(k, w, pol)/Dprime(k, w, pol)
        counter += 1

    return w, counter


# analytical eigenfrequencies for fast (+) and slow (-) modes with mode numbers (m, n) in a periodic cylinder with radius a and length 2*pi*R0 with homogeneous equilibrium profiles
def omega_cylinder_FS(l, pol, m, n, a, R0, gamma, p0, rho0, B0):
    
    # axial wavenumber
    k = n/R0
    
    # lth zero of the first derivative of the mth Bessel function
    alpha_ml = sp.jnp_zeros(m, l)[-1]
    
    # squared eigenfrequency (solution of a quadratic equation)
    b = rho0*gamma*p0*k**2 + rho0*k**2*B0**2 + alpha_ml**2*rho0*B0**2/a**2 + alpha_ml**2*rho0*gamma*p0/a**2
    c = k**4*B0**2*gamma*p0 + alpha_ml**2*gamma*p0*k**2*B0**2/a**2 
    
    omega = 1/(2*rho0**2)*b + pol/(2*rho0**2)*np.sqrt(b**2 - 4*rho0**2*c)
    
    return np.sqrt(omega)


# analytical radial eigenfunctions for fast (+) and slow (-) modes with mode numbers (m, n) in a periodic cylinder with radius a and length 2*pi*R0 with homogeneous equilibrium profiles
def xi_r_cylinder_FS(r, l, pol, m, n, a, R0, gamma, p0, rho0, B0):
    
    # axial wavenumber
    k = n/R0
    
    # eigenfrequency
    omega = omega_cylinder_FS(l, pol, m, n, a, R0, gamma, p0, rho0, B0)
    
    # eigenfunction (derivative of mth Bessel function)
    A = rho0*omega**2 - k**2*B0**2
    S = rho0*omega**2*(B0**2 + gamma*p0) - gamma*p0*k**2*B0**2
    
    K = np.sqrt(A*(rho0*omega**2 - gamma*p0*k**2)/S)
    
    xi_r = sp.jvp(m, K*r)
    
    return xi_r


# analytical eigenfrequencies for continuum modes with mode numbers (m, n) in a periodic cylinder with radius a and length 2*pi*R0 and a specified q-profile
def omega_A_cylinder(r, m, n, a, R0, gamma, pr, rho, B_z, B_phi, kind):
    
    # q-profile
    q = lambda r : r*B_z(r)/(R0*B_phi(r))
    
    # Alfvén continuum
    if B_phi(0.5) != 0.:
        omegaA2 = B_z(r)**2/(R0**2*rho(r))*(n + m/q(r))**2
    else:
        omegaA2 = B_z(r)**2/(R0**2*rho(r))*n**2
    
    if   kind == 'A':
        omega2 = omegaA2
    elif kind == 'S':
        omega2 = gamma*pr(r)/(B_phi(r)**2 + B_z(r)**2 + gamma*pr(r))*omegaA2
    
    return np.sqrt(omega2)