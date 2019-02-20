import numpy as np
import scipy.special as sp
from scipy.interpolate import splev



def borisPush(particles, dt, B, E, qe, me, Lz, bcs = 1):
    '''Pushes particles by a time step dt in the electromagnetic fields E and B by using the Boris method.
    
        Parameters:
            particles : ndarray
                2D-arrray (Np x 5) containing the particle information (z,vx,vy,vz,wk).
            dt : float
                The size of the time step.
            B : ndarray
                2D-array (Np x 3) with the magnetic field (Bx, By, Bz) at the particle positions.
            E : ndarray
                2D-array (Np x 3) with the electric field (Ex, Ey, Ez) at the particle positions.
            qe : float
                The particles' electric charge.
            me : float
                The particles' mass.
            Lz : float
                The length of the compuational domain (important for boundary treatment).
            bcs : int
                The boundary conditions. 1: periodic, 2: reflecting
                
        Returns:
            znew : ndarray
                1D-array (Np) with the updated particle positions.
            vnew : ndarray
                2D-array (Np x 3) with the updated particle velocities.
    '''
    
    if bcs == 1:
        
        qprime = dt*qe/(2*me)
        H = qprime*B
        S = 2*H/(1 + np.linalg.norm(H, axis = 1)**2)[:, None]
        u = particles[:, 1:4] + qprime*E
        uprime = u + np.cross(u + np.cross(u, H), S)
        vnew = uprime + qprime*E
        znew = (particles[:, 0] + dt*vnew[:, 2])%Lz

        return znew, vnew

    elif bcs == 2:

        qprime = dt*qe/(2*me)
        H = qprime*B
        S = 2*H/(1 + np.linalg.norm(H, axis = 1)**2)[:, None]
        u = particles[:, 1:4] + qprime*E
        uprime = u + np.cross(u + np.cross(u, H), S)
        vnew = uprime + qprime*E
        znew = particles[:, 0] + dt*vnew[:, 2]

        indices_right = np.where(znew > Lz)[0]
        indices_left = np.where(znew < 0)[0]

        
        H_right = qprime*B[indices_right]
        S_right = 2*H_right/(1 + np.linalg.norm(H_right, axis = 1)**2)[:, None]

        H_left = qprime*B[indices_left]
        S_left = 2*H_left/(1 + np.linalg.norm(H_left, axis = 1)**2)[:, None]

        u_right = -particles[indices_right, 1:4] + qprime*E[indices_right]
        u_left = -particles[indices_left, 1:4] + qprime*E[indices_left]

        uprime_right = u_right + np.cross(u_right + np.cross(u_right, H_right), S_right)
        uprime_left = u_left + np.cross(u_left + np.cross(u_left, H_left), S_left)

        vnew[indices_right] = uprime_right + qprime*E[indices_right]
        vnew[indices_left] = uprime_left + qprime*E[indices_left]

        znew[indices_right] = particles[indices_right, 0] + dt*vnew[indices_right, 2]
        znew[indices_left] = particles[indices_left, 0] + dt*vnew[indices_left, 2]
       
        return znew, vnew
    

    
    
def borisPushRelativistic(particles, dt, B, E, qe, me, Lz, c, bcs = 1):
    '''Pushes relativistic particles by a time step dt in the electromagnetic fields E and B using the Boris method.
    
        Parameters:
            particles : ndarray
                2D-arrray (Np x 5) containing the particle information (z,ux,uy,uz,wk).
            dt : float
                The size of the time step.
            B : ndarray
                2D-array (Np x 3) with the magnetic field (Bx, By, Bz) at the particle positions.
            E : ndarray
                2D-array (Np x 3) with the electric field (Ex, Ey, Ez) at the particle positions.
            qe : float
                The particles' electric charge.
            me : float
                The particles' mass.
            Lz : float
                The length of the compuational domain (important for boundary treatment).
            c : float
                The speed of light.
            bcs : int
                The boundary conditions. 1: periodic, 2: reflecting
                
        Returns:
            znew : ndarray
                1D-array (Np) with the updated particle positions.
            unew : ndarray
                2D-array (Np x 3) with the updated particle momenta.
    '''
    
    if bcs == 1:
        
        qprime = dt*qe/(2*me)
        u = particles[:, 1:4] + qprime*E
        H = qprime*B/np.sqrt(1 + np.linalg.norm(u, axis = 1)**2/c**2)[:, None]
        S = 2*H/(1 + np.linalg.norm(H, axis = 1)**2)[:, None]
        uprime = u + np.cross(u + np.cross(u, H), S)
        unew = uprime + qprime*E
        vnew = unew[:, 2]/np.sqrt(1 + np.linalg.norm(unew, axis = 1)**2/c**2)
        znew = (particles[:, 0] + dt*vnew)%Lz

        return znew, unew
    
    elif bcs == 2:
        
        qprime = dt*qe/(2*me)
        u = particles[:, 1:4] + qprime*E
        H = qprime*B/np.sqrt(1 + np.linalg.norm(u, axis = 1)**2/c**2)[:, None]
        S = 2*H/(1 + np.linalg.norm(H, axis = 1)**2)[:, None]
        uprime = u + np.cross(u + np.cross(u, H), S)
        unew = uprime + qprime*E
        vnew = unew[:, 2]/np.sqrt(1 + np.linalg.norm(unew, axis = 1)**2/c**2)
        znew = particles[:, 0] + dt*vnew
        
        indices_right = np.where(znew > Lz)[0]
        indices_left = np.where(znew < 0)[0]
        
        u_right = particles[indices_right, 1:4] + qprime*E[indices_right]
        H_right = qprime*B[indices_right]/np.sqrt(1 + np.linalg.norm(u_right, axis = 1)**2/c**2)[:, None]
        S_right = 2*H_right/(1 + np.linalg.norm(H_right, axis = 1)**2)[:, None]
        
        u_left = particles[indices_left, 1:4] + qprime*E[indices_left]
        H_left = qprime*B[indices_left]/np.sqrt(1 + np.linalg.norm(u_left, axis = 1)**2/c**2)[:, None]
        S_left = 2*H_left/(1 + np.linalg.norm(H_left, axis = 1)**2)[:, None]
        
        uprime_right = u_right + np.cross(u_right + np.cross(u_right, H_right), S_right)
        uprime_left = u_left + np.cross(u_left + np.cross(u_left, H_left), S_left)
        
        unew[indices_right] = uprime_right + qprime*E[indices_right]
        unew[indices_left] = uprime_left + qprime*E[indices_left]
        
        vnew[indices_right] = unew[indices_right, 2]/np.sqrt(1 + np.linalg.norm(unew[indices_right], axis = 1)**2/c**2)
        vnew[indices_left] = unew[indices_left, 2]/np.sqrt(1 + np.linalg.norm(unew[indices_left], axis = 1)**2/c**2)
        
        znew[indices_right] = particles[indices_right, 0] + dt*vnew[indices_right]
        znew[indices_left] = particles[indices_left, 0] + dt*vnew[indices_left]
        
        return znew, unew
    
    

class Bspline(object):
    def __init__(self, T, p):
        """
        initialize splines for given knot sequence t and degree p
        """
        self.T = T
        self.p = p
        self.N = len(T) - p - 1
        self.c = np.zeros(self.N)

    def __call__(self, x, i = None, n_deriv = 0):
        """
        evaluate b-spline starting at node i at x
        """
        if i is not None:
            c = np.zeros_like(self.T)
            if i < 0:
                c[self.N + i] = 1.
            else:
                c[i] = 1.
        else:
            c = self.c

        tck = (self.T, c, self.p)
        return splev(x, tck, der = n_deriv)

    def grevilleIga(self):
        """
        Returns the Greville points
        """
        p = self.p
        T = self.T

        # TODO implement a pure python function and not use igakit
        #from igakit import igalib
        #return igalib.bsp.Greville(p, T)
    
    def greville(self):
        """
        Returns the Greville points
        """
        p = self.p
        T = self.T
        N = self.N

        grev = np.zeros(N)
        
        for i in range(N):
            grev[i] = 1/p*sum(T[i+1:p+i+1])
            
        return grev

    def plot(self, nx = 100):
        """
        Plots all splines constructed from a knot sequence
        """
        T = self.T
        p = self.p
        N = self.N
        x = np.linspace(0.0, 1.0, nx)

        y = np.zeros((N, nx), dtype = np.double)
        for i in range(0, N):
            y[i] = self(x, i = i)
            plt.plot(x, y[i])

            
def createBasis(L, Nel, p, bcs = 1):
    '''Creates a B-spline basis and the corresponding quadrature/weights grid for exact Gauss-Legendre quadrature.
    
        Parameters:
            L : float
                Length of the domain.
            Nel : int
                The number of elements.
            p : int
                The degree of the basis.
            bcs : int
                The boundary conditions. 1: periodic, 2: Dirichlet.
                
        Returns:
            bsp : B-spline object.
                The basis functions. They can be called via bsp(x,j,d), where x is the evaluation point, j the j-th B-spline and d the d-th derivative.
            N : int
                The number of basis functions.
            quad_points : ndarray
                1D- array with the quadrature points.
            weights : ndarray
                1D- array with the corresponding weights.
        '''

    if bcs == 1:

        dz = L/Nel
        zj = np.linspace(0, L, Nel + 1)

        left = np.linspace(-p*dz, -dz, p)
        right = np.linspace(L + dz, L + p*dz, p)
        Tbsp = np.array(list(left)  + list(zj) + list(right))
        bsp = Bspline(Tbsp, p)
        N = len(Tbsp) - p - 1
        xi, wi = np.polynomial.legendre.leggauss(p + 1)
        quad_points = np.zeros((p + 1)*Nel)
        weights = np.zeros((p + 1)*Nel)

        for i in range(0, Nel):
            a1 = zj[i]
            a2 = zj[i + 1]
            xis = (a2 - a1)/2*xi + (a1 + a2)/2
            quad_points[(p + 1)*i:(p + 1)*i + (p + 1)] = xis
            wis = (a2 - a1)/2*wi
            weights[(p + 1)*i:(p + 1)*i + (p + 1)] = wis

        return bsp, N, quad_points, weights

    elif bcs == 2:

        dz = L/Nel
        zj = np.linspace(0, L, Nel + 1)

        Tbsp = np.array([0]*p + list(zj) + [L]*p)
        bsp = Bspline(Tbsp, p)
        N = len(Tbsp) - p - 1
        xi,wi = np.polynomial.legendre.leggauss(p + 1)
        quad_points = np.zeros((p + 1)*Nel)
        weights = np.zeros((p + 1)*Nel)

        for i in range(0, Nel):
            a1 = zj[i]
            a2 = zj[i + 1]
            xis = (a2 - a1)/2*xi + (a1 + a2)/2
            quad_points[(p + 1)*i:(p + 1)*i + (p + 1)] = xis
            wis = (a2 - a1)/2*wi
            weights[(p + 1)*i:(p + 1)*i + (p + 1)] = wis

        return bsp, N, quad_points, weights

    
    
def fieldInterpolation(particles_pos, nodes, basis, uj, bcs = 1):
    '''Computes the electromagnetic fields E and B at the particle positions using the basis functions.
    
        Parameters:
            particles_pos : ndarray
                1D-array containing the particles positions.
            nodes : ndarray
                1D-array containing the element boundaries.
            basis : B-spline object
                The B-spline basis functions.
            uj : ndarray
                1D-array with the FEM coefficients.
            bcs : int
                The boundary conditions. 1: periodic, 2: homogeneous Dirichlet.
                
        Returns:
            Ep : ndarray
                2D-array (Np x 2) with the electric fields at the particle positions.
            Bp : ndarray
                2D-array (Np x 2) with the magnetic fields at the particle positions.
    '''
            
    

    if bcs == 1:
        
        Nel = len(nodes) - 1  
        p = basis.p
        Nb = Nel
        
        Ep = np.zeros((len(particles_pos), 2))
        Bp = np.zeros((len(particles_pos), 2))
    
        Zbin = np.digitize(particles_pos, nodes) - 1

        ex = uj[0::6]
        ey = uj[1::6]
        bx = uj[2::6]
        by = uj[3::6]
   
    
        for ie in range(0, Nel):   
       
            indices = np.where(Zbin == ie)[0]
              
            for il in range(0, p + 1):
            
                i = il + ie
                bi = basis(particles_pos[indices], i)
            
                Ep[indices, 0] += ex[i%Nb]*bi
                Ep[indices, 1] += ey[i%Nb]*bi
                Bp[indices, 0] += bx[i%Nb]*bi
                Bp[indices, 1] += by[i%Nb]*bi
       
        return Ep, Bp

    elif bcs == 2:
        
        Nel = len(nodes) - 1
        p = basis.p
        
        Ep = np.zeros((len(particles_pos), 2))
        Bp = np.zeros((len(particles_pos), 2))
    
        Zbin = np.digitize(particles_pos, nodes) - 1

        ex = np.array([0] + list(uj[0::6]) + [0]) 
        ey = np.array([0] + list(uj[1::6]) + [0]) 
        bx = np.array([0] + list(uj[2::6]) + [0]) 
        by = np.array([0] + list(uj[3::6]) + [0]) 


        for ie in range(0, Nel):

            indices = np.where(Zbin == ie)[0]

            for il in range(0, p + 1):

                i = il + ie
                bi = basis(particles_pos[indices], i)
            
                Ep[indices, 0] += ex[i]*bi
                Ep[indices, 1] += ey[i]*bi
                Bp[indices, 0] += bx[i]*bi
                Bp[indices, 1] += by[i]*bi
       
        return Ep,Bp

    
def hotCurrent(particles_vel, particles_pos, particles_wk, nodes, basis, qe, c, bcs = 1, rel = 1):
    '''Computes the hot current density on terms of the weak formulation.
    
        Parameters:
            particles_vel : ndarray
                2D-array (Np x 2) with the particle velocities (vx,vy).
            particles_pos : ndarray
                1D-array (Np x 1) with the particle positions (z).
            particles_wk : ndarray
                1D-array (Np x 1) with the particle weights.
            nodes : ndarray
                1D-array containing the element boundaries.
            basis : B-spline object
                B-spline basis functions.
            qe : float
                The particles' charge.
            c : foat
                The speed of light
            bcs : int
                The boundary conditions. 1: periodic, 2: homogeneous Dirichlet.
            rel : int
                Nonrelativistic (1) or relativistic computation (2).
                
        Returns:
            jh : ndarray
                1D-array with the hot current densities (jhx,jhy).
    '''

    if bcs == 1:
    
        Nel = len(nodes) - 1
        p = basis.p
        Nb = Nel
        Np = len(particles_pos)
    
        jh = np.zeros(2*Nb)
        
        if rel == 2:
            gamma = np.sqrt(1 + np.sqrt(1 + np.linalg.norm(particles_vel, axis = 1)**2/c**2))
            particles_vel[:, 0] = particles_vel[:, 0]/gamma
            particles_vel[:, 1] = particles_vel[:, 1]/gamma
    
        Zbin = np.digitize(particles_pos, nodes) - 1
    
        for ie in range(0, Nel):
        
            indices = np.where(Zbin == ie)[0]
            wk = particles_wk[indices]
            vx = particles_vel[indices, 0]
            vy = particles_vel[indices, 1]
        
            for il in range(0, p + 1):
            
                i = il + ie
                bi = basis(particles_pos[indices], i)         
         
            
                jh[2*(i%Nb)] += np.einsum('i,i,i', vx, wk, bi) 
                jh[2*(i%Nb) + 1] += np.einsum('i,i,i', vy, wk, bi)         

        return qe*1/Np*jh

    elif bcs == 2:

        Nel = len(nodes) - 1
        p = basis.p
        Nb = Nel + p
        Np = len(particles_pos)
    
        jh = np.zeros(2*Nb)
        
        if rel == 2:
            gamma = np.sqrt(1 + np.sqrt(1 + np.linalg.norm(particles_vel, axis = 1)**2/c**2))
            particles_vel[:, 0] = particles_vel[:, 0]/gamma
            particles_vel[:, 1] = particles_vel[:, 1]/gamma
    
        Zbin = np.digitize(particles_pos, nodes) - 1
    
        for ie in range(0, Nel):
        
            indices = np.where(Zbin == ie)[0]
            wk = particles_wk[indices]
            vx = particles_vel[indices, 0]
            vy = particles_vel[indices, 1]
        
            for il in range(0, p + 1):
            
                i = il + ie
                bi = basis(particles_pos[indices], i)         
         
            
                jh[2*i] += np.einsum('i,i,i', vx, wk, bi) 
                jh[2*i + 1] += np.einsum('i,i,i', vy, wk, bi)         

        return qe*1/Np*jh[2:2*(Nb - 1)]
    
    
def IC(z,ini,amp,k,omega):
    '''Defines the initial conditions of the simulation.
    
        Parameters: 
            z : ndarray
                Positions to be evaluated.
            ini : int
                Type of inital conditions.
            amp : float
                Amplitude of the initial perturbations.
            k : float
                Wavenumber of initial perturbations.
            omega : float
                Frequency of initial perturbations.
                
        Returns:
            initial : ndarray
                2D-array (6 x len(z)) with the initial values.
    '''
    
    if ini == 1:
 
        eps0 = 1.0
        wce = -1.0
        wpe = 2.0

        Ex0 = +amp*np.cos(k*z)
        Ey0 = -amp*np.sin(k*z)
        
        Bx0 = -Ey0*k/omega
        By0 = +Ex0*k/omega
        
        Dj = eps0*wpe**2*(omega - wce)/(wce**2 - omega**2)
        
        jx0 = -Ey0*Dj
        jy0 = +Ex0*Dj
            
        return np.array([Ex0, Ey0, Bx0, By0, jx0, jy0])
    
    elif ini == 2:
        
        Ex0 = +amp*np.real(np.exp(1j*k*z))
        Ey0 = -amp*np.imag(np.exp(1j*k*z))
        
        Bx0 = k*amp*np.imag(1/omega*np.exp(1j*k*z))
        By0 = k*amp*np.real(1/omega*np.exp(1j*k*z))
        
        Dj = eps0*wpe**2*(omega - wce)/(wce**2 - omega**2)
        
        jx0 = amp*np.imag(Dj*np.exp(1j*k*z))
        jy0 = amp*np.real(Dj*np.exp(1j*k*z))
            
        return np.array([Ex0, Ey0, Bx0, By0, Bz0, jx0, jy0])
    
    elif ini == 3:
        
        Ex0 = 0*z
        Ey0 = 0*z
        
        Bx0 = amp*np.sin(k*z)
        By0 = 0*z
        
        
        jx0 = 0*z
        jy0 = 0*z
            
        return np.array([Ex0, Ey0, Bx0, By0, jx0, jy0])
    
    elif ini == 4:
        
        Ex0 = 0*z
        Ey0 = 0*z
        
        Bx0 = amp*np.random.randn()
        By0 = amp*np.random.randn()
        
        jx0 = 0*z
        jy0 = 0*z
            
        return np.array([Ex0, Ey0, Bx0, By0, jx0, jy0])
      
    
    elif ini == 5:
        
        Ex0 = amp*np.random.randn()
        Ey0 = amp*np.random.randn()
        
        Bx0 = amp*np.random.randn()
        By0 = amp*np.random.randn()
        
        jx0 = amp*np.random.randn()
        jy0 = amp*np.random.randn()
        
        return np.array([Ex0, Ey0, Bx0, By0, jx0, jy0])
        
    elif ini == 6:
        
        Ex0 = 0*z
        Ey0 = 0*z
        
        Bx0 = 0*z
        By0 = 0*z
        
        jx0 = 0*z
        jy0 = 0*z
        
        return np.array([Ex0, Ey0, Bx0, By0, jx0, jy0])
    
    
def L2proj(basis, L, quad_points, weights, mass, fun, bcs = 1):
    '''Computes the coefficients of some given function in the B-spline basis using the L2-projection.
    
        Parameters:
            basis : B-spline object
                The B-spline basis functions.
            L : float
                The length of the computational domain.
            quad_points : ndarray
                Gauss-Legendre quadrature points.
            weights : ndarray
                The corresponding weights.
            mass : ndarry
                The mass matrix of the B-spline basis.
            fun : function
                Function to be projected.
            bcs : int
                Boundary conditions. 1: periodic, 2: homogeneous Dirichlet.
                
        Returns:
            fj : ndarray
                The coefficients of the projected function.
            f : ndarray
                The L2 scalar product.
    '''

    if bcs == 1:
        
        p = basis.p
        Nel = basis.N - p
        Nb = Nel
        
        f = np.zeros(Nb)

        for ie in range(0, Nel):
            for il in range(0, p + 1):

                i = il + ie

                value_f = 0.0

                for g in range(0, p + 1):
                    gl = ie*(p + 1) + g

                    value_f += weights[gl]*fun(quad_points[gl])*basis(quad_points[gl], i, 0)

                f[i%Nb] += value_f

        fj = np.linalg.solve(mass, f)

        return fj, f

    elif bcs == 2:
        
        p = basis.p
        Nel = basis.N - p
        Nb = basis.N

        Ua = fun(0)
        Ub = fun(L)

        f = np.zeros(Nb)
        fj = np.zeros(Nb)

        for ie in range(0, Nel):
            for il in range(0, p + 1):

                i = ie + il

                value_f = 0.0

                for g in range(0, p + 1):
                    gl = ie*(p + 1) + g

                    value_f += weights[gl]*(fun(quad_points[gl])*basis(quad_points[gl], i, 0) - Ua*basis(quad_points[gl], 0, 0)*basis(quad_points[gl], i, 0) - Ub*basis(quad_points[gl], Nb - 1, 0)*basis(quad_points[gl], i, 0))

                f[i] += value_f

        fj[1:Nb - 1] = np.linalg.solve(mass[1:Nb - 1, 1:Nb - 1],f[1:Nb - 1])
        fj[0] = Ua
        fj[-1] = Ub 

        return fj, f
    
    
    
def matrixAssembly(basis, weights, quad_points, B0, bcs):
    '''Assembles the mass, convection and field matrix of a given B-spline Finite Element basis.
    
        Parameters:
            basis : B-spline object
                The B-spline basis functions.
            weights: ndarray
                1D-array containing the weights of the Gauss-Legendre quadrature.
            quad_points : ndarray
                1D-array containing the evaluation points of the Gauss-Legendre quadrature.
            B0 : function
                The background magnetic field.
            bcs : int
                Boundary conditions. 1: periodic, 2: homogeneous Dirichlet.
                
        Returns:
            M : ndarray
                The mass matrix phi_i*phi_j (2D-array).
            C : ndarray
                The convection matrix phi_i*phi_j^' (2D-array).
            K : ndarray
                The stiffness matrix phi_i^'*phi_j^' (2D-array).
            D : ndarray
                The field matrix B0*phi_i*phi_j (2D-array).
    '''

    if bcs == 1:

        N = basis.N
        p = basis.p
        Nb = N - p
        Nel = Nb
        
        M = np.zeros((Nb, Nb))
        C = np.zeros((Nb, Nb))
        K = np.zeros((Nb, Nb))
        D = np.zeros((Nb, Nb))


        for ie in range(0, Nel):
            for il in range(0, p + 1):
                for jl in range(0, p + 1):
                
                    i = il + ie  
                    j = jl + ie
                
                    value_m = 0.0
                    value_c = 0.0
                    value_k = 0.0
                    value_d = 0.0
                
                    for g in range(0, p + 1):
                        gl = ie*(p + 1) + g
                    
                        value_m += weights[gl]*basis(quad_points[gl], i, 0)*basis(quad_points[gl], j ,0)
                        value_c += weights[gl]*basis(quad_points[gl], i, 0)*basis(quad_points[gl], j, 1)
                        value_k += weights[gl]*basis(quad_points[gl], i, 1)*basis(quad_points[gl], j, 1)
                        value_d += weights[gl]*basis(quad_points[gl], i, 0)*basis(quad_points[gl], j, 0)*B0(quad_points[gl])
                    
                    M[i%Nb, j%Nb] += value_m
                    C[i%Nb, j%Nb] += value_c
                    K[i%Nb, j%Nb] += value_k
                    D[i%Nb, j%Nb] += value_d

        return M, C, K, D

    elif bcs == 2:

        N = basis.N
        p = basis.p
        Nel = N - p
        
        M = np.zeros((N, N))
        C = np.zeros((N, N))
        K = np.zeros((N, N))
        D = np.zeros((N, N))

        
        for ie in range(0, Nel):
            for il in range(0, p + 1):
                for jl in range(0, p + 1):
                
                    i = il + ie  
                    j = jl + ie
                
                    value_m = 0.0
                    value_c = 0.0
                    value_k = 0.0
                    value_d = 0.0

                    for g in range(0, p + 1):
                        gl = ie*(p + 1) + g
                    
                        value_m += weights[gl]*basis(quad_points[gl], i, 0)*basis(quad_points[gl], j, 0)
                        value_c += weights[gl]*basis(quad_points[gl], i, 0)*basis(quad_points[gl], j, 1)
                        value_k += weights[gl]*basis(quad_points[gl], i, 1)*basis(quad_points[gl], j, 1)
                        value_d += weights[gl]*basis(quad_points[gl], i, 0)*basis(quad_points[gl], j, 0)*B0(quad_points[gl])
                        
                    M[i, j] += value_m
                    C[i, j] += value_c
                    K[i, j] += value_k
                    D[i, j] += value_d
                    
                    
        return M, C, K, D


def dampingAssembly(basis, damp, meth):
    '''Assembles the damping matrix.
    
        Parameters:
            basis : B-spline object
                The B-spline basis functions.
            damp : function
                The masking function.
            meth : int
                The used method for the assembly. 1: only evaluation on support. 2: conventional method.
                
        Returns:
            DAMP : ndarray
                The damping matrix (2D-array).
    '''

    if meth == 1:

        p = basis.p
        Nbase = basis.N
        grev = basis.greville()

        gi = np.zeros(Nbase)
        Bij = np.zeros((Nbase,Nbase))

        p_boundary_l = p
        p_boundary_r = p + 3
        counter = 2

        for ie in range(p+1):
            for il in range(p_boundary_l):
                Bij[il,ie] = basis(grev[il],ie)

            p_boundary_l += 1

        for ie in range(p+1,Nbase-p-1):
            for il in range(p+2):
                i = il + counter
                Bij[i,ie] = basis(grev[i],ie)

        for ie in range(Nbase-p-1,Nbase):
            for il in range(p_boundary_r):
                i = Nbase - p_boundary_r + il
                Bij[i,ie] = basis(grev[i],ie)

            p_boundary_r -= 1

        for i in range(Nbase):
            gi[i] = damp(grev[i])

        G = np.diag(gi[1:Nbase-1])
        Bijinv = np.linalg.inv(Bij[1:Nbase-1,1:Nbase-1])
        DAMP = np.dot(np.dot(Bijinv,G),Bij[1:Nbase-1,1:Nbase-1])

        return DAMP

    elif meth == 2:

        p = basis.p
        Nbase = basis.N
        grev = basis.greville()

        gi = np.zeros(Nbase)
        Bij = np.zeros((Nbase,Nbase))

        for i in range(Nbase):
            for j in range(Nbase):
                Bij[i,j] = basis(grev[i],j)

        for i in range(Nbase):
            gi[i] = damp(grev[i])

        G = np.diag(gi[1:Nbase-1])
        Bijinv = np.linalg.inv(Bij[1:Nbase-1,1:Nbase-1])
        DAMP = np.dot(np.dot(Bijinv,G),Bij[1:Nbase-1,1:Nbase-1])

        return DAMP

    
    
    
    
    
    
    
    
def evaluation(uj, basis , nodes, x, bcs = 1):
    ''' Given a coefficient vector, this function computes the value at some position in space spanned by a B-spline basis
    
        Parameters:
            uj : ndarray
                Coefficient vector.
            basis : B-spline object
                Object of B-spline basis functions. 
            nodes : ndarray.
                The element boundaries.
            x : ndarray
                Positions to be evaluated.
            bcs : int
                Boundary conditions. DEFAULT = 1 (periodic), bcs = 2 (Dirichlet).
                
                
        Returns:
            u : ndarray
                Vector with values at positions x.
    '''
    
    
    
    if bcs == 1:
        
        p = basis.p
        Nb = basis.N - p
        Nel = Nb
 
        u = np.zeros(len(x))
    
        Xbin = np.digitize(x, nodes) - 1 

        for ie in range(Nel):
            indices = np.where(ie == Xbin)[0]
            
            if len(indices) != 0:
            
                for il in range(0, p + 1):
                    i = ie + il
                    u[indices] += uj[i%Nb]*basis(x[indices], i)
        
        return u

    elif bcs == 2:
        
        p = basis.p
        N = basis.N
        Nel = N - p 
        
        u = np.zeros(len(x))
    
        Xbin = np.digitize(x, nodes) - 1 
        
        for ie in range(Nel):
            indices = np.where(ie == Xbin)[0]
            
            if len(indices) != 0:
                
                for il in range(0, p + 1):
                    i = ie + il
                    u[indices] += uj[i]*basis(x[indices], i)

        return u
    
    
    

def solveDispersionHybrid(k, pol, c, wce, wpe, wpar, wperp, nuh, initial_guess, tol, max_it = 100):

    Taniso = 1 - wperp**2/wpar**2


    def Z(xi):
        return np.sqrt(np.pi)*np.exp(-xi**2)*(1j - sp.erfi(xi))

    def Zprime(xi):
        return -2*(1 + xi*Z(xi))


    def Dhybrid(k, w, pol):
        xi = (w + pol*wce)/(k*np.sqrt(2)*wpar)

        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce)) + nuh*wpe**2/w**2*(w/(k*np.sqrt(2)*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi)))

   
    def Dhybridprime(k, w, pol):
        xi = (w + pol*wce)/(k*np.sqrt(2)*wpar)
        xip = 1/(k*np.sqrt(2)*wpar)

        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2) - 2*nuh*wpe**2/w**3*(w/(np.sqrt(2)*k*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi))) + nuh*wpe**2/w**2*(1/(np.sqrt(2)*k*wpar)*Z(xi) + w/(np.sqrt(2)*k*wpar)*Zprime(xi)*xip - Taniso*(xip*Z(xi) + xi*Zprime(xi)*xip)) 

    w = initial_guess
    counter = 0

    while True:
        wnew = w - Dhybrid(k, w, pol)/Dhybridprime(k, w, pol)

        if np.abs(wnew - w) < tol or counter == max_it:
            w = wnew
            break

        w = wnew
        counter += 1

    return w, counter


def solveDispersionHybridExplicit(k, pol, c, wce, wpe, wpar, wperp, nuh, initial_guess, tol, max_it = 100):

    def Dcold(k, w, pol):
        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce))

    def Dcoldprime(k, w, pol):
        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2)

    wr = initial_guess
    counter = 0

    while True:
        wnew = wr - Dcold(k, wr, pol)/Dcoldprime(k, wr, pol)

        if np.abs(wnew - wr) < tol or counter == max_it:
            wr = wnew
            break

        wr = wnew
        counter += 1

    vR = (wr + pol*wce)/k

    wi = 1/(2*wr - pol*wpe**2*wce/(wr + pol*wce)**2)*np.sqrt(2*np.pi)*wpe**2*nuh*vR/wpar*np.exp(-vR**2/(2*wpar**2))*(wr/(2*(-pol*wce - wr)) + 1/2*(1 - wperp**2/wpar**2))

    return wr, wi, counter




def solveDispersionCold(k, pol, c, wce, wpe, initial_guess, tol, max_it = 100):

    def Dcold(k, w, pol):
        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce))

    def Dcoldprime(k, w, pol):
        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2)

    wr = initial_guess
    counter = 0

    while True:
        wnew = wr - Dcold(k, wr, pol)/Dcoldprime(k, wr, pol)

        if np.abs(wnew - wr) < tol or counter == max_it:
            wr = wnew
            break

        wr = wnew
        counter += 1

    return wr, counter


def solveDispersionArtificial(k, pol, c, wce, wpe, c1, c2, mu0, initial_guess, tol, max_it = 100):

    def Dart(k, w, pol):
        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce)) + 1/w*(1j*c1 - pol*c2)
    
    def Dartprime(k, w, pol):
        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2) - 1/w**2*(1j*c1 - pol*c2)

    w = initial_guess
    counter = 0

    while True:
        wnew = w - Dart(k, w, pol)/Dartprime(k, w, pol)

        if np.abs(wnew - w) < tol or counter == max_it:
            w = wnew
            break

        w = wnew
        counter += 1

    return w, counter
