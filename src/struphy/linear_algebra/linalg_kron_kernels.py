

def kernel_kron_matvec_2d(A : 'float[:,:]', B : 'float[:,:]', v : 'float[:,:]', res : 'float[:,:]'):

    n_i, n_j = res.shape
    n_k, n_l = v.shape

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                for l in range(n_l):
                    res[i,j]+=A[i,k]*B[j,l]*v[k,l]

    