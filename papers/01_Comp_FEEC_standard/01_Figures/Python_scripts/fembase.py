'''Basic classes and functions for finite element (FEM) programming.
    Written by Stefan Possanner
    stefan.possanner@ma.tum.de

   Contains:
       LagrangeShape : class
           The class for 1D Lagrange shape functions on the interval [-1,1].
       lag_assemb : function
           Computes the global mass and stiffness matrices from Lagrange basis functions.
       lag_L2prod : function
           Computes the L2 scalar product of a given function with each element 
           of a basis defined from shape functions.
       lag_fun : function
           Given a coefficient vector, returns a function in the space spanned by a Lagrange basis.
'''

import numpy as np
from scipy.integrate import fixed_quad as gauss_int
from scipy.linalg import block_diag


class LagrangeShape:
    '''The class for 1D Lagrange shape functions on the interval [-1,1].
    
    Parameters: 
        pts : ndarray
            1D array of increasing values in [-1, 1] defining the Lagrange polynomials.   
                    
    Returns:
        self.kind : string
            Is set to 'lagrange'.
        self.d : int
            Polynomial degree.
        self.s : ndarray
            The input array pts (knot sequence).
        self.eta : list
            List elements are the Lagrange polynomials (LPs) in 'poly1d' format.
        self.Deta : list
            List elements are the derivatives of the LPs in 'poly1d' format.  
        self.mass0 : ndarray
            Local mass matrix of the LPs.  
        self.stiff0 : ndarray
            Local stiffness matrix of the LPs.
        self.chi : list
            List elements are the Lagrange histopolation polynomials (LHPs) in 'poly1d' format.
        self.Dchi : list
            List elements are the derivatives of LHPs in 'poly1d' format.  
        self.mass1 : ndarray
            Local mass matrix of the LHPs.  
        self.stiff1 : ndarray
            Local stiffness matrix of the LHPs.
    '''
    
    kind = 'lagrange'
    
    def __init__(self, pts):
        
        self.d = len(pts) - 1
        # polynomial degree
        
        self.s = pts
        # knot sequence
        
        ### Lagrange polynomials (LPs):
        self.eta = [] 
        for i in range(self.d + 1):
            condition = self.s != self.s[i]
            roots = np.compress(condition, self.s) 
            self.eta.append(np.poly1d(roots, r=True)) 
            # Numerator of LP
            for j in range(len(roots)):
                self.eta[i] /= self.s[i] - roots[j] 
                # Denominator of LP
                
        # derivatives of LPs
        self.Deta = []
        for i in range(self.d + 1):
            self.Deta.append(np.polyder(self.eta[i]))
                
        # mass and stiffness matrix:
        self.mass0 = np.zeros((self.d + 1, self.d + 1))
        self.stiff0 = np.zeros((self.d + 1, self.d + 1))
        for i in range(self.d + 1):
            for j in range(self.d + 1): 
                antider = np.polyint(self.eta[i]*self.eta[j])
                self.mass0[i, j] = antider(1) - antider(-1)
                antider_D = np.polyint(self.Deta[i]*self.Deta[j])
                self.stiff0[i, j] = antider_D(1) - antider_D(-1)
                
        ### Lagrange histopolation polynomials (LHPs):
        self.chi = []
        for i in range(self.d):
            
            chisum = 0
            for j in range(i + 1, self.d + 1):
                chisum += self.Deta[j]
            # sum of derivatives of LPs

            self.chi.append(chisum) 
            
        # derivatives of LHPs
        self.Dchi = []
        for i in range(self.d):
            self.Dchi.append(np.polyder(self.chi[i]))
                
        # mass and stiffness matrix:
        self.mass1 = np.zeros((self.d, self.d))
        self.stiff1 = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d): 
                antider = np.polyint(self.chi[i]*self.chi[j])
                self.mass1[i, j] = antider(1) - antider(-1)
                antider_D = np.polyint(self.Dchi[i]*self.Dchi[j])
                self.stiff1[i, j] = antider_D(1) - antider_D(-1)
                
                
def lag_assemb(el_b, mass_eta, stiff_eta, basis = 1, bcs = 2):
    ''' Computes the global mass and stiffness matrices from Lagrange basis functions.
    
    Parameters:
        el_b : ndarray
            1D array of element interfaces from left to right including the boundaries. 
        mass_eta : ndarray
            The mass matrix (2D) of the shape functions defined on [-1, 1].
        stiff_eta : ndarray
            The stiffness matrix (2D) of the shape functions defined on [-1, 1].
        bcs : int
            Specifies the boundary conditions. DEFAULT = 2 which stands for Dirichlet.
            1 stands for periodic.
        basis: int
            The type of basis functions. DEFAULT = 1 which stands for Langrange interpolation polynomials. 2 stands for Lagrange histoplation polynomials.
            
    Returns:
        Nel : int
            The number of elements, Nel = len(el_b) - 1.
        Nbase : int
            Number of basis functions, Nbase = np.size(fun_bar)
        mass : ndarray
            The mass matrix. If m = np.size(mass_eta[:, 0]) denotes the size of the local mass matrix, 
            then np.size(mass[:, 0]) = Nel*m - (Nel - 1) - bcs.
        stiff : ndarray
            The stiffness matrix. Same size as mass.
    '''
    
    if basis == 1:
    
        Nel = len(el_b) - 1
        # number of elements

        m = mass_eta[:, 0].size 
        # size of local mass matrix (of the shape functions)

        d = m - 1
        # polynomial degree

        Ntot = Nel*m
        # number of degrees of freedom (including the boundary) 
        Ntot -= (Nel - 1)
        # subtracted the number of shared degrees of freedom (interior interfaces)
        Nbase = Ntot - bcs
        # number of basis functions (for the respective boundary conditions)

        mass = np.zeros((Nbase, Nbase))
        stiff = np.zeros((Nbase, Nbase))
        # initiate mass and stiffness matrix

        if bcs == 2:
            # left boundary:
            mass[:d, :d] = (el_b[1] - el_b[0])/2*mass_eta[1:, 1:]
            stiff[:d, :d] = 2/(el_b[1] - el_b[0])*stiff_eta[1:, 1:]
            index = d - 1

            # bulk:
            for i in range(1, Nel - 1):
                mass[index:index + d + 1, index:index + d + 1] += (el_b[i + 1] - el_b[i])/2*mass_eta[:, :] 
                stiff[index:index + d + 1, index:index + d + 1] += 2/(el_b[i + 1] - el_b[i])*stiff_eta[:, :] 
                index += d
                # remark the '+=' in mass (stiff) for the cumulative sum for overlapping degrees of freedom

            # right boundary
            if bcs == 2:
                mass[index:index + d, index:index + d] += (el_b[-1] - el_b[-2])/2*mass_eta[:-1, :-1] 
                stiff[index:index + d, index:index + d] += 2/(el_b[-1] - el_b[-2])*stiff_eta[:-1, :-1] 


            return Nel, Nbase, mass, stiff

        elif bcs == 1:
            # bulk
            for ie in range(0, Nel):
                for il in range(0, d + 1):
                    for jl in range(0, d + 1):

                        i = d*ie + il
                        j = d*ie + jl

                        mass[i%Nbase, j%Nbase] += (el_b[ie + 1] - el_b[ie])/2*mass_eta[il, jl]
                        stiff[i%Nbase, j%Nbase] += 2/(el_b[ie + 1] - el_b[ie])*stiff_eta[il, jl]

            return Nel, Nbase, mass, stiff

        else:
            print('boundary conditions not yet implemented!')
            return
        
    elif basis == 2:
        
        Nel = len(el_b) - 1
        # number of elements

        d = mass_eta[:, 0].size 
        # size of local mass matrix (of the shape functions)

        Nbase = Nel*d
        # number of basis functions (for the respective boundary conditions)
        
        mass = np.zeros((Nbase, Nbase))
        stiff = np.zeros((Nbase, Nbase))
        # initiate mass and stiffness matrix

        for ie in range(Nel):
            i = ie*d
            
            mass[i:(i + d), i:(i + d)] = (el_b[ie + 1] - el_b[ie])/2*mass_eta
            stiff[i:(i + d), i:(i + d)] = (el_b[ie + 1] - el_b[ie])/2*stiff_eta
        
        return Nel, Nbase, mass, stiff
            
    
def lag_L2prod(fun, eta, el_b, basis = 1, bcs = 2):
    '''Computes the L2 scalar product of a function with each element of a Lagrange basis
    defined from shape functions.
    
    Parameters:
        fun : function
            The input function, for example a 'lambda'-function.
        eta : list
            List elements are the shape functions in 'poly1d' format.
        el_b : ndarray
            1D array of element interfaces from left to right including the boundaries. 
        bcs : int
            Specifies the boundary conditions. DEFAULT = 2 which stands for Dirichlet.
            1 stands for periodic.
        basis : int
            The type of basis functions. DEFAULT = 1 which stands for Langrange interpolation polynomials. 2 stands for Lagrange histoplation polynomials.
            
    Returns:
        Nel : int
            The number of elements, Nel = len(el_b) - 1. 
        Nbase : int
            Number of basis functions, Nbase = np.size(fun_bar)
        funbar : ndarray
            1D array of scalar products of fun with the basis functions.
    '''
    
    from scipy.integrate import fixed_quad 
    
    if basis == 1:
    
        Nel = len(el_b) - 1
        # number of elements

        m = len(eta)
        # number of shape functions

        d = m - 1
        # polynomial degree

        Ntot = Nel*m
        # number of degrees of freedom (including the boundary) 
        Ntot -= (Nel - 1)
        # subtracted the number of shared degrees of freedom (interior interfaces)
        Nbase = Ntot - bcs
        # number of basis functions (for the respective boundary conditions)

        funbar = np.zeros(Nbase)
        # initiate output vector

        index = 0
        # index of the basis function

        if bcs == 2:
            # left boundary:
            i = 0
            for j in range(1, m):

                fun1 = lambda s: fun( el_b[i] + (s + 1)/2*(el_b[i + 1] - el_b[i]) )
                # function fun transformed to the reference element [-1, 1]
                fun2 = lambda s: np.polyval(eta[j], s)
                # shape function

                fun12 = lambda s: fun1(s)*fun2(s)
                intval, foo = fixed_quad(fun12, -1, 1)
                funbar[index] += (el_b[i + 1] - el_b[i])/2*intval
                # integral

                if j != d:
                    index += 1
                # If it is the last shape function (j = d), the index rests the same
                # and the subsequent integral is added at the same position in funbar.

            # bulk:
            for i in range(1, Nel - 1): 
                for j in range(m):

                    fun1 = lambda s: fun( el_b[i] + (s + 1)/2*(el_b[i + 1] - el_b[i]) )
                    # function fun transformed to the reference element [-1, 1]
                    fun2 = lambda s: np.polyval(eta[j], s)
                    # shape function

                    fun12 = lambda s: fun1(s)*fun2(s)
                    intval, foo = fixed_quad(fun12, -1, 1)
                    funbar[index] += (el_b[i + 1] - el_b[i])/2*intval
                    # integral
                    if j != d:
                        index += 1

            # right boundary:
            i = Nel - 1
            for j in range(d):

                fun1 = lambda s: fun( el_b[i] + (s + 1)/2*(el_b[i + 1] - el_b[i]) )
                # function fun transformed to the reference element [-1, 1]
                fun2 = lambda s: np.polyval(eta[j], s)
                # shape function

                fun12 = lambda s: fun1(s)*fun2(s)
                intval, foo = fixed_quad(fun12, -1, 1)
                funbar[index] += (el_b[i + 1] - el_b[i])/2*intval
                # integral
                if j != d:
                        index += 1

            return Nel, Nbase, funbar

        elif bcs == 1:
            for ie in range(0, Nel):
                for il in range(0, d + 1):

                    fun1 = lambda s: fun( el_b[ie] + (s + 1)/2*(el_b[ie + 1] - el_b[ie]) )
                    # function fun transformed to the reference element [-1, 1]
                    fun2 = lambda s: np.polyval(eta[il], s)
                    # shape function

                    fun12 = lambda s: fun1(s)*fun2(s)
                    intval, foo = fixed_quad(fun12, -1, 1)

                    i = d*ie + il
                    funbar[i%Nbase] += (el_b[ie + 1] - el_b[ie])/2*intval 

            return Nel, Nbase, funbar

        else:
            print('boundary conditions not yet implemented')
            
    elif basis == 2:
        
        Nel = len(el_b) - 1
        # number of elements

        d = len(eta)
        # polynomial degree

        Nbase = Nel*d
        # number of basis functions (for the respective boundary conditions)

        funbar = np.zeros(Nbase)
        # initiate output vector

        for ie in range(0, Nel):
            for il in range(0, d):

                fun1 = lambda s: fun( el_b[ie] + (s + 1)/2*(el_b[ie + 1] - el_b[ie]) )
                # function fun transformed to the reference element [-1, 1]
                fun2 = lambda s: np.polyval(eta[il], s)
                # shape function

                fun12 = lambda s: fun1(s)*fun2(s)
                intval, foo = fixed_quad(fun12, -1, 1)

                i = d*ie + il
                    
                funbar[i] += (el_b[ie + 1] - el_b[ie])/2*intval 

        return Nel, Nbase, funbar


def lag_fun(cvec, eta, el_b, basis = 1, bcs = 2):
    '''Given a coefficient vector, returns a function in the space spanned by a Lagrange basis.
    
        Parameters:
            cvec : ndarray
                Coefficient vector.
            eta : list
                List elements are the shape functions in 'poly1d' format.
            el_b : ndarray
                1D array of element interfaces from left to right including the boundaries. 
            bcs : int
                Specifies the boundary conditions. DEFAULT = 2 which stands for Dirichlet.
                1 stands for periodic.
            basis : int
                The type of basis functions. DEFAULT = 1 which stands for Langrange interpolation polynomials. 2 stands for Lagrange histoplation polynomials.
                
        Returns:
            Nel : int
                The number of elements, Nel = len(el_b) - 1. 
            Nbase : int
                Number of basis functions, Nbase = np.size(fun_bar).
            fun : function
                Function defined on [el_b[0], el_b[-1]].    
            
    '''
    if basis == 1:
    
        Nel = len(el_b) - 1
        # number of elements

        m = len(eta)
        # number of shape functions

        d = m - 1
        # polynomial degree

        Ntot = Nel*m
        # number of degrees of freedom (including the boundary) 
        Ntot -= (Nel - 1)
        # subtracted the number of shared degrees of freedom (interior interfaces)
        Nbase = Ntot - bcs
        # number of basis functions (for the respective boundary conditions)

        el_b[-1] += 1e-8
        # important for binning, see np.digitize

        def fun(x):
            '''Function in a finite dimensional space spanned by Lagrange basis functions,
            created with fembase.lag_fun.

            Parameters:
                x : ndarray
                    Array of arguments passed to the function f.

            Returns:
                funval : ndarray
                    f(x), same size as x.
            '''

            binnr = np.digitize(x, el_b)
            # binning the input arguments
            il = binnr == 1
            ir = binnr == Nel 
            ii = np.logical_not(np.logical_or(il, ir))
            # logicals to treat boundary conditions 

            funval = np.zeros(np.size(x))
            index = np.int_(np.zeros(np.size(x)))
            index[ii] = (binnr[ii] - 1)*d - 1 
            index[ir] = (binnr[ir] - 1)*d - 1  
            # the starting index for each bin to identify the coefficient of the basis function

            if bcs == 2:
                for i in range(len(x)):

                    # left boundary:
                    if il[i] == True:

                        for j in range(1, m): 
                            funval[i] += ( cvec[index[i]]*np.polyval( eta[j], 
                                        2*(x[i] - el_b[binnr[i] - 1])
                                        /(el_b[binnr[i]] - el_b[binnr[i] - 1]) - 1. ) )
                            index[i] += 1
                     # right boundary:
                    elif ir[i] == True:

                        for j in range(d): 
                            funval[i] += ( cvec[index[i]]*np.polyval( eta[j], 
                                        2*(x[i] - el_b[binnr[i] - 1])
                                        /(el_b[binnr[i]] - el_b[binnr[i] - 1]) - 1. ) )
                            index[i] += 1
                    # bulk:
                    else:

                        for j in range(m): 
                            funval[i] += ( cvec[index[i]]*np.polyval( eta[j], 
                                            2*(x[i] - el_b[binnr[i] - 1])
                                            /(el_b[binnr[i]] - el_b[binnr[i] - 1]) - 1. ) )
                            index[i] += 1  

                return funval

            elif bcs == 1:
                for i in range(len(x)):

                    for j in range(0,m):

                        index = (binnr[i] - 1)*d + j

                        funval[i] += ( cvec[index%Nbase]*np.polyval( eta[j], 
                                            2*(x[i] - el_b[binnr[i] - 1])
                                            /(el_b[binnr[i]] - el_b[binnr[i] - 1]) - 1. ) )
                return funval


        return Nel, Nbase, fun
    
    elif basis == 2:
        
        Nel = len(el_b) - 1
        # number of elements

        d = len(eta)
        # polynomial degree

        Nbase = Nel*d
        # number of basis functions (for the respective boundary conditions)
        
        el_b[-1] += 1e-8
        # important for binning, see np.digitize
        
        def fun(x):
            funval = np.zeros(np.size(x))
            binnr = np.digitize(x, el_b)
            
            for i in range(len(x)):
                
                for j in range(d):
                    
                    index = (binnr[i] - 1)*d + j
                    funval[i] += ( cvec[index]*np.polyval( eta[j], 
                                            2*(x[i] - el_b[binnr[i] - 1])
                                            /(el_b[binnr[i]] - el_b[binnr[i] - 1]) - 1. ) )
            return funval
        
        return Nel, Nbase, fun
    
def lag_proj(s, el_b, fun):
    '''Projects the given function fun on the Lagrange histopolation basis.
    
        Parameters:
            s : ndarray
                local knot vector on reference element [-1,1] defining the Lagrange polynomials.
            el_b : ndarray
                The element boundaries.
            fun : function
                The function to be projected.
                
        Returns:
            xvec : ndarray
                Global knot vector
            funvec : ndarray
                Coefficient vector of the projected function
    '''
    
    Nel = len(el_b) - 1
    d = len(s) - 1
    Nknots = Nel*d + 1
    xvec = np.zeros(Nknots)
    funvec = np.zeros(Nel*d)
    
    
    for ie in range(Nel):
        for il in range(d + 1):

            i = ie*d + il
            xvec[i] = el_b[ie] + (s[il] + 1)/2*(el_b[ie + 1] - el_b[ie])
            # assemble global knot vector
    
    for ie in range(Nel):
        jac = (el_b[ie + 1] - el_b[ie])/2
        for il in range(d):
            i = ie*d + il
            funvec[i] = 1/(2*jac)*(fun(xvec[i + 1]) + fun(xvec[i]))*(xvec[i + 1] - xvec[i])
            
    return xvec, funvec
        
            