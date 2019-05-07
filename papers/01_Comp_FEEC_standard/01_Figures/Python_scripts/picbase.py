'''Basic functions for Particle-In-Cell (PIC) programming.
    Written by Florian Holderied
    florian.holderied@tum.de

   Contains:
       borisPush : function
           Pushes the particles' velocities and positions by a time step dt.
       fieldInterpolation : fuction
           Computes the fields at the particle positions.
'''


import numpy as np
import scipy as sc

def borisPush(particles, dt, Bp, Ep, q, m, L, bcs = 2):
    '''Pushes the particles' velocities and positions by a time step dt.
        
    Parameters:
        particles : ndarray
            2D-array (N_p x 7) containing the positions (x,y,z), velocities (vx,vy,vz) and weights (w) of N_p particles.
        dt: float
            The time step.
        Bp: ndarray
            2D-array (N_p x 3) containing the magnetic field at the particle positions. 
        Ep: ndarray
            2D-array (N_p x 3) containing the electric field at the particle positions.
        q : float
            The electric charge of the particles.
        m : float
            The mass of the particles.
        L : float
            The length of the computational domain.
        bcs : int
            The boundary conditions. DEFAULT = 2 (periodic), 1 (reflecting).

    Returns:
        xnew : ndarray
            2D-array (N_p x 3) with the updated particle positions.
        vnew : ndarray
            2D-array (N_p x 3) with the updated particle velocities.
    '''
    
    if bcs == 2:
        
        qprime = dt*q/(2*m)
        H = qprime*Bp
        S = 2*H/(1 + np.linalg.norm(H, axis = 1)**2)[:,None]
        u = particles[:, 3:6] + qprime*Ep
        uprime = u + np.cross(u + np.cross(u, H), S)
        vnew = uprime + qprime*Ep
        xnew = (particles[:, 0:3] + dt*vnew)%L

        return xnew,vnew
    
    elif bcs == 1:
        
        print('Not yet implemented!')
        
        return 
    
    
    
def fieldInterpolation(xk, el_b, shapefun, ex, ey, ez, bx, by, bz, bcs = 1):
    '''Computes the fields at the particle positions from a FEM Lagrange basis.
    
    Parameters:
        xk : ndarray
            1D-array containing the x-positions of all particles.
        el_b : ndarray
            The element boundaries.
        shapefun: LagrangeShape object
            List with the Lagrange shape functions.
        ex: ndarray
            The coefficients of the x-component of the electric field.
        ey: ndarray
            The coefficients of the y-component of the electric field.
        ez: ndarray
            The coefficients of the z-component of the electric field.
        bx : ndarray
            The coefficients of the x-component of magnetic field.
        by : ndarray
            The coefficients of the y-component of magnetic field.
        bz : ndarray
            The coefficients of the z-component of magnetic field.
        bcs : int
            The boundary conditions. DEFAULT = 1 (Dirichlet), 2 (periodic).
            
    Returns:
        Ep : ndarray
            2D-array (N_p x 3) containing the electric field at the Np particle positions.
        Bp : ndarray
            2D-array (N_p x 3) containing the magnetic field at the Np particle positions.
    '''
    
    if bcs == 1:
        
        N_el = len(el_b) - 1
        N_p = len(xk)
        d = shapefun.d
        
        Ep = np.zeros((N_p, 3))
        Bp = np.zeros((N_p, 3))
        
        exj = np.array([0] + list(ex) + [0])
        eyj = np.array([0] + list(ey) + [0])
        ezj = np.array([0] + list(ez) + [0])
        bxj = np.array([0] + list(bx) + [0])
        byj = np.array([0] + list(by) + [0])
        bzj = np.array([0] + list(bz) + [0])
        
        Xbin = np.digitize(xk, el_b) - 1
        
        for ie in range(0, N_el):
            
            indices = np.where(Xbin == ie)[0]
            s = 2*(xk[indices] - el_b[ie])/(el_b[ie + 1] - el_b[ie])
            
            for il in range(0, d + 1):
                
                i = d*ie + il
                bi = np.polyval(shapefun.eta[il],s)
                
                Ep[indices, 0] += exj[i]*bi
                Ep[indices, 1] += eyj[i]*bi
                Ep[indices, 2] += ezj[i]*bi
                Bp[indices, 0] += bxj[i]*bi
                Bp[indices, 1] += byj[i]*bi
                Bp[indices, 2] += bzj[i]*bi
                
        return Ep,Bp
    
    elif bcs == 2:
        
        N_el = len(el_b) - 1
        N_p = len(xk)
        d = shapefun.d
        Nbase = N_el*d
        
        Ep = np.zeros((N_p, 3))
        Bp = np.zeros((N_p, 3))
        
        Xbin = np.digitize(xk, el_b) - 1
        
        for ie in range(0, N_el):
            
            indices == np.where(Xbin == ie)[0]
            s = 2*(xk[indices] - el_b[ie])/(el_b[ie + 1] - el_b[ie])
            
            for il in range(0, d + 1):
                
                i = d*ie + il
                bi = np.polyval(shapefun.eta[il],s)
                
                Ep[indices, 0] += exj[i%Nbase]*bi
                Ep[indices, 1] += eyj[i%Nbase]*bi
                Ep[indices, 2] += ezj[i%Nbase]*bi
                Bp[indices, 0] += bxj[i%Nbase]*bi
                Bp[indices, 1] += byj[i%Nbase]*bi
                Bp[indices, 2] += bzj[i%Nbase]*bi
                
        return Ep,Bp


    
def assemb_S(x_p, kernel, kernel_supp, el_b, s, bcs = 1, siz = 0):
    '''Assembles the S-matrix with elements S_ip = S(x_i - x_p), where S is a smoothing kernel.

    Parameters:
        x_p : ndarray
            1D-array with the particle positions.
        kernel : function
            The smoothing kernel. Symmetric, positive and normalized to 1.
        kernel_supp : float
            The size of the kernel support.
        el_b : ndarray
            The element boundaries including the domain boundaries.
        s : ndarray
            The knot vector on the reference element [-1,1].
        bcs : int
            Boundary conditions: DEFAULT = 1 which stands for periodic boundary conditions
        siz : int
            Switch for V1-projection: DEFAULT = 0 (no projection), 2 means S is prepared with 2 additional lines for the V1-projection.
            
    Returns:
        S : sparse matrix
            The matrix with entries S_ip.
        x_vec : ndarray
            The global knot vector.
    '''
    
    
    Nel = len(el_b) - 1
    # number of elements
    
    L = el_b[-1] - el_b[0]
    # domain length
    
    Np = len(x_p)
    # number of particles
    
    d = len(s) - 1
    # degree of basis functions
            
    Nknots = Nel*d + 1 + siz
    x_vec = np.zeros(Nknots)
    # global knot vector
    
    if siz == 0:
        
        for ie in range(Nel):
            for il in range(d + 1):

                i = ie*d + il
                x_vec[i] = el_b[ie] + (s[il] + 1)/2*(el_b[ie + 1] - el_b[ie])
                # assemble global knot vector

    elif siz == 2:
        
        xvec[0] = el_b[0] - kernel_supp/2
        xvec[-1] = el_b[-1] + kernel_supp/2
        
        for ie in range(Nel):
            for il in range(d + 1):

                i = ie*d + il + 1
                x_vec[i] = el_b[ie] + (s[il] + 1)/2*(el_b[ie + 1] - el_b[ie])
                # assemble global knot vector
                
                
    col = np.array([])
    row = np.array([])
    data = np.array([])
    # initialize global col, row and data

    for i in range(Nknots - bcs):

        col_i = np.where(np.abs(x_p - x_vec[i]) < kernel_supp/2)[0]
        data_i = kernel(x_p[col_i] - x_vec[i])
        
        if np.abs(x_vec[i] - x_vec[0]) <= kernel_supp/2:
            col_i2 = np.where(np.abs(x_p - L - x_vec[i]) < kernel_supp/2)[0]
            data_i2 = kernel(x_p[col_i2] - L - x_vec[i])
            
            col_i = np.append(col_i, col_i2)
            data_i = np.append(data_i, data_i2)
            
        elif np.abs(x_vec[i] - x_vec[-1]) <= kernel_supp/2:
            col_i3 = np.where(np.abs(x_p + L - x_vec[i]) < kernel_supp/2)[0]
            data_i3 = kernel(x_p[col_i3] + L - x_vec[i])
            
            col_i = np.append(col_i, col_i3)
            data_i = np.append(data_i, data_i3)
            
            
        
       
        row_i = np.ones(len(col_i))*i
        

        col = np.append(col, col_i)
        row = np.append(row, row_i)
        data = np.append(data, data_i)



    S = sc.sparse.csr_matrix((data, (row, col)), shape = (Nknots - bcs, Np))

    return S, x_vec
    
    
    
    
def computeDensity(particles, q, el_b, kernel, s):
    '''Parameters:
        particles : ndarray
            2D-array (Np x 4) containing the particle information (x,vx,vy,w)
        q : float
            The charge of the particles
        el_b : ndarray
            1D-array specifying the element boundaries
        kernel : function
            The smoothing kernel
        s : ndarray
            The Lagrange interpolation points on the reference element [-1,1]
            
        Returns:
            rho : ndarray
                The coefficients of the charge density
    '''
    
    Nel = len(el_b) - 1
    # number of elements
    
    Np = len(particles[:, 0])
    # number of particles
    
    d = len(s) - 1
    # degree of basis functions
    
    p = d + 1
    # degree of Gauss-Legendre quadrature
    
    Nknots = Nel*d + 1
    glob_s = np.zeros(Nknots)
    # global knot vector
    
    rho = np.zeros(Nel*d)
    # initialize charge density
    
    for ie in range(Nel):
        for il in range(d + 1):
            
            i = ie*d + il
            glob_s[i] = el_b[ie] + (s[il] + 1)/2*(el_b[ie + 1] - el_b[ie])
            # assemble global knot vector
    
    xi,wi = np.polynomial.legendre.leggauss(p)
    # weights and quadrature points on reference element [-1,1]
    
    quad_points = np.zeros(p*(Nknots - 1))
    weights = np.zeros(p*(Nknots - 1))
    # global quadrature points and weights

    for i in range(Nknots - 1):
        a1 = glob_s[i]
        a2 = glob_s[i+1]
        xis = (a2 - a1)/2*xi + (a1 + a2)/2
        quad_points[p*i:p*i + p] = xis
        wis = (a2 - a1)/2*wi
        weights[p*i:p*i + p] = wis
        # assemble global quad_points and weights
                       
    bins = np.digitize(particles[:, 0], glob_s) - 1
    # particle binning in global knot vector
    
    dx = el_b[1] - el_b[0]
    
    for i in range(Nel*d):
        for j in range(Np):
            
            fun = lambda x: kernel(particles[j,0] - x)
            rho[i] += 2*q/dx*particles[j,4]*sc.integrate.quad(fun,glob_s[i],glob_s[i+1])
            
    return rho
        

    
def assemb_Q(x_p, shapefun, el_b, bcs = 2, basis = 0):
    '''Assembles the Q matrix Q_ip = phi_i(x_p) for given particle positions x_p and basis functions phi_i.
    
        Parameters:
            x_p : ndarray
                1D-array with the particle positions.
            shapefun : Lagrangeshape object
                The Lagrange shape functions.
            el_b : ndarray
                The element boundaries.
            bcs : int
                Boundary conditions. DEFAULT = 2 (Dirichlet), 1 (periodic)
            basis : int
                The type of basis functions. DEFAULT = 0 which stands for Lagrange interpolation polynomials. 1 stands for Lagrange histopolation polynomials.
                
        Returns:
            Q : sparse matrix
                The matrix with entries Q_ip.
    '''
    
    Nel = len(el_b) - 1
    # number of elements
    
    Np = len(x_p)
    # number of particles
    
    d = shapefun.d
    # degree of Lagrange interpolation shape functions
    
    Xbin = np.digitize(x_p, el_b) - 1
    # particle binning in elements
    
    col = np.array([])
    row = np.array([])
    data = np.array([])
    # initialize global col, row and data
    
    if basis == 0:
    
        if bcs == 2:

            Nbase = Nel*d - 1
            # number of basis functions

            # left boundary
            col_i = np.where(Xbin == 0)[0]
            s_p = 2*(x_p[col_i] - el_b[0])/(el_b[1] - el_b[0]) - 1

            for il in range(d):
                row_i = np.ones(len(col_i))*il
                data_i = np.polyval(shapefun.eta[1 + il], s_p)

                col = np.append(col, col_i)
                row = np.append(row, row_i)
                data = np.append(data, data_i)

            # bulk
            for ie in range(1, Nel - 1):
                col_i = np.where(Xbin == ie)[0]
                s_p = 2*(x_p[col_i] - el_b[ie])/(el_b[ie + 1] - el_b[ie]) - 1

                for il in range(d + 1):
                    i = ie*d + il - 1
                    row_i = np.ones(len(col_i))*i
                    data_i = np.polyval(shapefun.eta[il], s_p)

                    col = np.append(col, col_i)
                    row = np.append(row, row_i)
                    data = np.append(data, data_i)

            # right boundary
            col_i = np.where(Xbin == Nel - 1)[0]
            s_p = 2*(x_p[col_i] - el_b[Nel - 1])/(el_b[Nel] - el_b[Nel - 1]) - 1

            for il in range(d):
                i = Nbase - d + il
                row_i = np.ones(len(col_i))*i
                data_i = np.polyval(shapefun.eta[il], s_p)

                col = np.append(col, col_i)
                row = np.append(row, row_i)
                data = np.append(data, data_i)

            Q = sc.sparse.csr_matrix((data, (row, col)), shape = (Nbase, Np))

            return Q
        
        if bcs == 1:
            
            Nbase = Nel*d
            # number of basis functions
            
            for ie in range(Nel):
                col_i = np.where(Xbin == ie)[0]
                s_p = 2*(x_p[col_i] - el_b[ie])/(el_b[ie + 1] - el_b[ie]) - 1
                
                for il in range(d + 1):
                    i = (ie*d + il)%Nbase
                    row_i = np.ones(len(col_i))*i
                    data_i = np.polyval(shapefun.eta[il], s_p)
                    
                    col = np.append(col, col_i)
                    row = np.append(row, row_i)
                    data = np.append(data, data_i)
                    
            Q = sc.sparse.csr_matrix((data, (row, col)), shape = (Nbase, Np))
            
            return Q
        
    elif basis == 1:
        
        Nbase = Nel*d
        # number of basis functions
        
        for ie in range(Nel):
            col_i = np.where(Xbin == ie)[0]
            s_p = 2*(x_p[col_i] - el_b[ie])/(el_b[ie + 1] - el_b[ie]) - 1
                
            for il in range(d):
                i = ie*d + il
                row_i = np.ones(len(col_i))*i
                data_i = np.polyval(shapefun.chi[il], s_p)
                    
                col = np.append(col, col_i)
                row = np.append(row, row_i)
                data = np.append(data, data_i)
                    
        Q = sc.sparse.csr_matrix((data, (row, col)), shape = (Nbase, Np))
            
        return Q
    
def spline_parts(p, dS):
    '''Returns the piecewise polynomials of a B-spline of degree p as a list of callable functions on the intervall [0, dS].
    
        Parameters:
            p : int
                Spline degree.
            dS : float
                Width of one knot interval. The spline support is [-(p + 1)/2*dS, (p + 1)/2*dS].
                
        Returns:
            poly_list : list
                The piecewise polynomials of a B-spline of degree p as a list of callable functions on the intervall [0, dS].
            knots : ndarray
                Knot vector.
            kernel : callable
                The B-spline defined on the entire support.
    '''
    
    
    poly_list = []
    knots = np.linspace(-(p + 1)/2*dS, (p + 1)/2*dS, p + 2)
    
    if p == 1:
        poly_list.append(lambda x : (x/dS)/dS)
        poly_list.append(lambda x : (-x/dS + 1)/dS)
        
        
    elif p == 2:
        poly_list.append(lambda x : ((x/dS)**2/2)/dS)
        poly_list.append(lambda x : (-(x/dS)**2 + x/dS + 1/2)/dS)
        poly_list.append(lambda x : ((x/dS)**2/2 - x/dS + 1/2)/dS)
        
    elif p == 3:
        poly_list.append(lambda x : ((x/dS)**3/6)/dS)
        poly_list.append(lambda x : (-(x/dS)**3/2 + (x/dS)**2/2 + x/dS/2 + 1/6)/dS)
        poly_list.append(lambda x : ((x/dS)**3/2 - (x/dS)**2 + 2/3)/dS)
        poly_list.append(lambda x : (-(x/dS)**3/6 + (x/dS)**2/2 - x/dS/2 + 1/6)/dS)
     
    
    def kernel(x):
        bins = np.digitize(x, knots) - 1
        
        val = np.zeros(len(x))
        
        for ie in range(p + 1):
            ind_x = np.where(ie == bins)[0]
            val[ind_x] = poly_list[ie](x[ind_x] - knots[ie])
            
        return val
        
        
    return poly_list, knots, kernel
                