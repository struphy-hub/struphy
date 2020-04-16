import numpy as np
import scipy.special as sp


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
    
    while True:
        wnew = w - D(k, w, pol)/Dprime(k, w, pol)
        
        if np.abs(wnew - w) < tol or counter == max_it:
            w = wnew
            break

        w = wnew
        counter += 1

    return w, counter