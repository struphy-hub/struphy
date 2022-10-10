import numpy as np
import scipy.special as sp
from scipy.optimize import fsolve, root

from struphy.dispersion_relations.base import DispersionRelations1D


class Maxwell1D(DispersionRelations1D):
    r'''Dispersion relation for Maxwell's equation in vacuum in Struphy units (see ``Maxwell`` in :ref:`models`):
    
    .. math::
    
        \omega^2 = k^2 \,.
    '''

    def __init__(self, **params):
        super().__init__('light wave') 

    def __call__(self, kvec, kperp=None):

        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [np.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################
        # first branch
        tmps[0][:] = kvec
        ##############################################################

        # fill output dictionary
        dict_disp = {}
        for name, tmp in zip(self.branches, tmps):
            dict_disp[name] = tmp

        return dict_disp
    
    
class Mhd1D(DispersionRelations1D):
    r'''Dispersion relation for linear MHD equations for homogeneous background :math:`(n_0,p_0,\mathbf B_0)` and wave propagation along z-axis in Struphy units (see ``LinearMHD`` in :ref:`models`):
    
    .. math::
    
        \textnormal{shear Alfvén}:\quad &\omega^2 = c_\textnormal{A}^2 k^2\frac{B_{0z}^2}{|\mathbf B_0|^2}\,,
        
        \textnormal{fast (+) and slow (-) magnetosonic}:\quad &\omega^2 =\frac{1}{2}(c_\textnormal{S}^2+c_\textnormal{A}^2)k^2(1\pm\sqrt{1-\delta}\,)\,,\quad\delta=\frac{4B_{0z}^2c_\textnormal{S}^2c_\textnormal{A}^2}{(c_\textnormal{S}^2+c_\textnormal{A}^2)^2|\mathbf B_0|^2}\,,
        
    where :math:`c_\textnormal{A}^2=|\mathbf B_0|^2/n_0` is the Alfvén velocity and :math:`c_\textnormal{S}^2=\gamma\,p_0/n_0` is the speed of sound.
    '''

    def __init__(self, **params):
        super().__init__('shear Alfvén', 'slow magnetosonic', 'fast magnetosonic', **params) 

    def __call__(self, kvec, kperp=None):

        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [np.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################
        
        # Alfvén velocity and speed of sound
        cA = np.sqrt((self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2)/self.params['n0'])
        cS = np.sqrt(self.params['gamma']*self.params['p0']/self.params['n0']) 
        
        # shear Alfvén branch
        tmps[0][:] = cA * kvec * self.params['B0z']/np.sqrt(self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2)
        
        # slow/fast magnetosonic branch
        delta = (4*self.params['B0z']**2*cS**2*cA**2)/((cS**2 + cA**2)**2*(self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2))
        
        tmps[1][:] = np.sqrt(1/2*kvec**2*(cS**2 + cA**2)*(1 - np.sqrt(1 - delta)))
        tmps[2][:] = np.sqrt(1/2*kvec**2*(cS**2 + cA**2)*(1 + np.sqrt(1 - delta)))
        
        ##############################################################

        # fill output dictionary
        dict_disp = {}
        for name, tmp in zip(self.branches, tmps):
            dict_disp[name] = tmp

        return dict_disp


class PC_LinMHD_6d_full1D(DispersionRelations1D):
    r'''Dispersion relation for linear MHD equations coupled to the Vlasov equation with Full Pressure Coupling scheme
    for homogeneous background :math:`(n_0,p_0,\mathbf B_0)`, wave propagation along z-axis in Struphy units and space-homogeneous shifted Maxwellian energetic particles distribution :math:`f_h = f_{h,0} + \tilde{f_h}` 
    where :math:`f_{h,0}(v_{\parallel}, v_{\perp}) = n_0 \frac{1}{\sqrt{\pi}} \frac{1}{\hat{v_{\parallel}}} e^{- (v_{\parallel} - u_0)^2 / \hat{v}^2_{\parallel} } \frac{1}{\pi} \frac{1}{\hat{v^2_{\perp}}} e^{- v^2_{\perp} / \hat{v}^2_{\perp}}`
    here, :math:`u_0` is a velocity shift in the parallel direction (see ``PC_LinMHD_6d_full`` in :ref:`models`):
    

    :math:`\textnormal{shear Alfvén (R) and (L) wave}` :

    .. math::

        \omega^2 = c_\textnormal{A}^2 k^2\frac{B_{0z}^2}{|\mathbf B_0|^2} + \omega k \nu_h &\left[ \frac{\omega_c}{\omega} \left\{ \left( 1 - \frac{\hat{v}^2_\perp}{\hat{v}^2_\parallel}\right) \hat{v}_\parallel \left( \pm Y_3 \mp \frac{\omega - \omega_c}{\hat{v}_\parallel k_\parallel} Y_2 \right) + u_0 \frac{\hat{v}^2_\perp}{\hat{v}^2_\parallel} \left( \pm Y_2 \mp \frac{\omega - \omega_c}{\hat{v}_\parallel k_\parallel} Y_2 \right) \right\} \right.

        &- \left. \frac{\hat{v}^2_\perp}{\hat{v}^2_\parallel} \left( Y_3 - \frac{\omega \mp \omega_c}{\hat{v}_\parallel k_\parallel} Y_2 - \frac{u_0}{\hat{v}_\parallel} Y_2 + \frac{\omega \mp \omega_c}{\hat{v}^2_\parallel k_\parallel} u_0 Y_1 \right)\right]\,,
        
    :math:`\textnormal{sonic wave}` :

    .. math::

        \omega^2 =c_\textnormal{A}^2 k^2 - 2 \omega k_\parallel \nu_h \hat{v}_\parallel X_4 \,
        
    where :math:`c_\textnormal{A}^2=|\mathbf B_0|^2/n_0` is the Alfvén velocity and :math:`c_\textnormal{S}^2=\gamma\,p_0/n_0` is the speed of sound.

    Variaous integrals are defined as follows

    .. math::

        X_4(\xi_0, a_0):= \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{(t+a)^3 t }{t - \xi_0} e^{- t^2} dt \, \quad \qquad &= \frac{5}{4} \xi_0 + \frac{3}{2} a_0 + (\xi_0 + a_0)^3 [1 + \xi_0 Z(\xi_0)] \,,

        Y_1(\xi_-, \xi_+, a_0) := \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{t+a_0}{(t-\xi_-)(t-\xi_+)} e^{-t^2} dt &= Z(\xi_-) + (\xi_+ + a_0) \frac{Z(\xi_-) - Z(\xi_+)}{\xi_- - \xi_+} \,,

        Y_2(\xi_-, \xi_+, a_0) := \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{(t+a)^2}{(t-\xi_-)(t-\xi_+)} e^{-t^2} dt &= 1 + (\xi_- + \xi_+ + 2a_0) Z(\xi_-) 
        
        &+ (\xi_+ + a_0)^2 \frac{Z(\xi_-) - Z(\xi_+)}{\xi_- - \xi_+} \,,

        Y_3(\xi_-, \xi_+, a_0) := \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{(t+a)^3}{(t-\xi_-)(t-\xi_+)} e^{-t^2} dt &= \xi_- + \xi_+ + 3a_0 
        
        &+ [\xi_-^2 + \xi_- \xi_+ + \xi_+^2 + 3a_0(\xi_- + \xi_+) + 3a_0^2] Z(\xi_-) 
        
        &+ (\xi_+ + a_0)^3 \frac{Z(\xi_-) - Z(\xi_+)}{\xi_- - \xi_+} \,,

    where :math:`\xi_0 = \frac{\omega / k_\parallel - u_0}{\hat{v}_\parallel}, \quad \xi_\pm = \frac{(\omega \pm \omega_c) / k_\parallel - u_0}{\hat{v}_\parallel}, \quad a_0 = \frac{u_0}{\hat{v}_\parallel}`
    and :math:`Z(\xi) = \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{e^{- t^2}}{t - \xi} dt = i \sqrt{\pi} e^{- \xi^2} ( 1 + \text{erf}(i\xi))` is the plasma dispersion function. 

    '''

    def __init__(self, params):
        super().__init__('shear Alfvén_R', 'shear Alfvén_L', 'sonic', **params) 

    def __call__(self, kvec, kperp=None):
        
        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [np.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################
        
        # Alfvén velocity and speed of sound
        # TODO: call the parameters from the yml file.
        wc = 1.
        u0 = 2.5 #TODO
        vpara = 1. #TODO
        vperp = 1. #TODO
        vth = 1.

        cA = np.sqrt((self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2)/self.params['n0'])
        # cS = np.sqrt(self.params['beta']*cA)
        cS = 1. 
        
        a0 = u0 / vpara #TODO
        nu = 0.05 #TODO

        # initially asign k
        k = 1.

        # define plasma dispersion function
        def pdf(xi):
            return np.sqrt(np.pi)*np.e**(-xi**2)*(1j - sp.erfi(xi))

        # defin integrals and functions
        def X4(xi, a):
            return 5/4*xi + 3/2*a + (xi + a)**3 * (1 + xi*pdf(xi))

        def Y1(xi, eta, a):
            return pdf(xi) + (eta + a) * (pdf(xi) - pdf(eta)) / (xi - eta)

        def Y2(xi, eta, a):
            return 1 + (xi + eta + 2*a)*pdf(xi) + (eta + a)**2 * (pdf(xi) - pdf(eta)) / (xi - eta) 

        def Y3(xi, eta, a):
            c1 = xi + eta + 3*a + (xi**2 + xi*eta + eta**2 + 3*a*(xi + eta) + 3*a**2) * pdf(xi)
            c2 = (eta + a)**3 * (pdf(xi) - pdf(eta)) / (xi - eta)
            return c1 + c2

        def zeta0(w):
            return (w/k - u0)/vpara

        def zetap(w):
            return ((w+wc)/k - u0)/vpara

        def zetam(w):
            return ((w-wc)/k - u0)/vpara

        def sonic(w):
            c1 = (w[0] + 1j*w[1])**2 - k**2 * cS**2 + 2*(w[0] + 1j*w[1])*k*nu*vpara*X4(zeta0((w[0] + 1j*w[1])),a0)
            return np.real(c1), np.imag(c1)

        def shearAlfvén_R(w):
            c1 = (w[0] + 1j*w[1])**2 - k**2 * cA**2 - (w[0] + 1j*w[1])*nu*k*(wc/(w[0] + 1j*w[1])*u0* \
            (Y2(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) - (w[0] + 1j*w[1] - wc)/k/vpara* \
            Y1(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0)) - (vperp**2/vpara) * \
            (Y3(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) - (w[0] + 1j*w[1] - wc)/k/vpara* \
            Y2(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) - u0 / vpara * \
            Y2(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) + \
            (w[0] + 1j*w[1] - wc)/k/vpara**2 * u0 * Y1(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0)))
    
            return np.real(c1), np.imag(c1)

        def shearAlfvén_L(w):
            c1 = (w[0] + 1j*w[1])**2 - k**2 * cA**2 - (w[0] + 1j*w[1])*nu*k*(wc/(w[0] + 1j*w[1])*u0* \
            (-Y2(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) + (w[0] + 1j*w[1] + wc)/k/vpara* \
            Y1(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0)) - (vperp**2/vpara) * \
            (Y3(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) - (w[0] + 1j*w[1] + wc)/k/vpara* \
            Y2(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) - u0 / vpara * \
            Y2(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0) + \
            (w[0] + 1j*w[1] + wc)/k/vpara**2 * u0 * Y1(zetam(w[0] + 1j*w[1]), zetap(w[0] + 1j*w[1]), a0)))

            return np.real(c1), np.imag(c1)

        # solve omega
        for i, val in enumerate(kvec):
            k = val
            initial_guess = k * cA

            # R/L shearAlfvén wave
            sol_R = fsolve(shearAlfvén_R, [initial_guess, 0])
            sol_L = fsolve(shearAlfvén_L, [initial_guess, 0])

            tmps[0][i] = sol_R[0] + 1j*sol_R[1]
            tmps[1][i] = sol_L[0] + 1j*sol_L[1]

            # sonic wave
            sol_S = fsolve(sonic, [initial_guess, 0])
            tmps[2][i] = sol_S[0] + i*sol_S[1]
        
        ##############################################################

        # fill output dictionary
        dict_disp = {}
        for name, tmp in zip(self.branches, tmps):
            dict_disp[name] = tmp

        return dict_disp