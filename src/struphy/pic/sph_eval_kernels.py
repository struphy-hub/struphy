from numpy import sqrt

def naive_evaluation(eta1 : 'float[:]', 
                     eta2 : 'float[:]', 
                     eta3 : 'float[:]', 
                     markers : 'float[:,:]', 
                     index : 'int', 
                     out : 'float[:]'):
    n_eval = len(eta1)
    n_particles = len(markers)
    for i in range(n_eval):
        for p in range(n_particles):
            r = sqrt((eta1[i]-markers[p,0])**2+(eta2[i]-markers[p,1])**2+(eta3[i]-markers[p,2])**2)
            out[i] += markers[p,index]*smoothing_kernel(r)

def smoothing_kernel(r : 'float'):
    if r>1.:
        return 0.
    else:
        return 1.-r