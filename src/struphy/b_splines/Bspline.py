import numpy as np
from scipy.interpolate import splev



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