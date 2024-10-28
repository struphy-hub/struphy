from numpy import sqrt

def smoothing_kernel(r : 'float', h : 'float'):
    if r/h>1.:
        return 0.
    else:
        return (1.-r/h)/(1.0471975512*h**3) #normalization

def naive_evaluation(eta1 : 'float[:]', 
                     eta2 : 'float[:]', 
                     eta3 : 'float[:]', 
                     markers : 'float[:,:]',
                     holes : 'bool[:]', 
                     index : 'int', 
                     out : 'float[:]'):
    n_eval = len(eta1)
    n_particles = len(markers)
    out[:] = 0.
    for i in range(n_eval):
        for p in range(n_particles):
            if not holes[p]:
                r = sqrt((eta1[i]-markers[p,0])**2+(eta2[i]-markers[p,1])**2+(eta3[i]-markers[p,2])**2)
                #print(r, smoothing_kernel(r), markers[p,index]) 
                out[i] += markers[p,index]*smoothing_kernel(r, 0.5)

