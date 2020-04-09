import numpy        as np
import scipy.sparse as sparse
import hylife.utilitis_FEEC.bsplines as bsp



# =======================================================
class _0form():
    
    def __init__(self, T, p, bc, coeff):
        
        self.T     = T
        self.p     = p
        self.bc    = bc
        self.coeff = coeff
        
    def evaluate(self, xi):
        
        N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, xi, bc)) for T, p, xi, bc in zip(self.T, self.p, xi, self.bc)]

        return sparse.kron(sparse.kron(N[0], N[1]), N[2]).dot(self.coeff.flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
    
    
# =======================================================
class _1form():
    
    def __init__(self, T, p, bc, coeff):
        
        self.T     = T
        self.p     = p
        self.bc    = bc
        self.coeff = coeff
        
        self.t     = [T[1:-1] for T in self.T]
        
    def evaluate_1component(self, xi):
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.t[0], p[0] - 1, xi[0], bc[0], normalize=True))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], p[1],     xi[1], bc[1]))
        N3 = sparse.csr_matrix(bsp.collocation_matrix(self.T[2], p[2],     xi[2], bc[2]))

        return sparse.kron(sparse.kron(D1, N2), N3).dot(self.coeff[0].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
    
    def evaluate_2component(self, xi):
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], p[0],     xi[0], bc[0]))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.t[1], p[1] - 1, xi[1], bc[1], normalize=True))
        N3 = sparse.csr_matrix(bsp.collocation_matrix(self.T[2], p[2],     xi[2], bc[2]))

        return sparse.kron(sparse.kron(N1, D2), N3).dot(self.coeff[1].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
    
    def evaluate_3component(self, xi):
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], p[0],     xi[0], bc[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], p[1],     xi[1], bc[1]))
        D3 = sparse.csr_matrix(bsp.collocation_matrix(self.t[2], p[2] - 1, xi[2], bc[2], normalize=True))

        return sparse.kron(sparse.kron(N1, N2), D3).dot(self.coeff[2].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
    
    
# =======================================================
class _2form():
    
    def __init__(self, T, p, bc, coeff):
        
        self.T     = T
        self.p     = p
        self.bc    = bc
        self.coeff = coeff
        
        self.t     = [T[1:-1] for T in self.T]
        
    def evaluate_1component(self, xi):
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], p[0],     xi[0], bc[0]))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.t[1], p[1] - 1, xi[1], bc[1], normalize=True))
        D3 = sparse.csr_matrix(bsp.collocation_matrix(self.t[2], p[2] - 1, xi[2], bc[2], normalize=True))

        return sparse.kron(sparse.kron(N1, D2), D3).dot(self.coeff[0].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
    
    def evaluate_2component(self, xi):
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.t[0], p[0] - 1, xi[0], bc[0], normalize=True))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], p[1],     xi[1], bc[1]))
        D3 = sparse.csr_matrix(bsp.collocation_matrix(self.t[2], p[2] - 1, xi[2], bc[2], normalize=True))

        return sparse.kron(sparse.kron(D1, N2), D3).dot(self.coeff[1].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
    
    def evaluate_3component(self, xi):
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.t[0], p[0] - 1, xi[0], bc[0], normalize=True))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.t[1], p[1] - 1, xi[1], bc[1], normalize=True))
        N3 = sparse.csr_matrix(bsp.collocation_matrix(self.T[2], p[2],     xi[2], bc[2]))

        return sparse.kron(sparse.kron(D1, D2), N3).dot(self.coeff[2].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
    
    
# =======================================================
class _3form():
    
    def __init__(self, T, p, bc, coeff):
        
        self.T     = T
        self.p     = p
        self.bc    = bc
        self.coeff = coeff
        
        self.t     = [T[1:-1] for T in self.T]
        
    def evaluate(self, xi):
        
        D = [sparse.csr_matrix(bsp.collocation_matrix(t, p - 1, xi, bc, normalize=True)) for t, p, xi, bc in zip(self.t, self.p, xi, self.bc)]

        return sparse.kron(sparse.kron(D[0], D[1]), D[2]).dot(self.coeff.flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))