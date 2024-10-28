from numpy import sqrt

def smoothing_kernel(r : 'float', h : 'float'):
    if r/h>1.:
        return 0.
    else:
        return (1.-r/h)/(1.0471975512*h**3) #normalization
    
def periodic_distance(x : 'float',y : 'float'):
    d = x-y
    if d>0.5:
        while d>0.5:
            d-=1.
    elif d<-0.5:
        while d<-0.5:
            d+=1.
    return d

def naive_evaluation(eta1 : 'float[:]', 
                     eta2 : 'float[:]', 
                     eta3 : 'float[:]', 
                     markers : 'float[:,:]',
                     holes : 'bool[:]', 
                     index : 'int', 
                     h : 'float',
                     out : 'float[:]'):
    n_eval = len(eta1)
    n_particles = len(markers)
    out[:] = 0.
    for i in range(n_eval):
        for p in range(n_particles):
            if not holes[p]:
                r = sqrt((eta1[i]-markers[p,0])**2+(eta2[i]-markers[p,1])**2+(eta3[i]-markers[p,2])**2)
                #print(r, smoothing_kernel(r), markers[p,index]) 
                out[i] += markers[p,index]*smoothing_kernel(r, h)

def naive_evaluation_3d(eta1 : 'float[:,:,:]', 
                     eta2 : 'float[:,:,:]', 
                     eta3 : 'float[:,:,:]', 
                     markers : 'float[:,:]',
                     holes : 'bool[:]', 
                     index : 'int', 
                     h : 'float',
                     out : 'float[:,:,:]'):
    n_eval_1 = eta1.shape[0]
    n_eval_2 = eta1.shape[1]
    n_eval_3 = eta1.shape[2]
    n_particles = len(markers)
    out[:] = 0.
    for i in range(n_eval_1):
        for j in range(n_eval_2):
            for k in range(n_eval_3):
                for p in range(n_particles):
                    if not holes[p]:
                        r = sqrt(periodic_distance(eta1[i,j,k],markers[p,0])**2 \
                                 +periodic_distance(eta2[i,j,k],markers[p,1])**2 \
                                 +periodic_distance(eta3[i,j,k],markers[p,2])**2)
                        #print(r, smoothing_kernel(r), markers[p,index]) 
                        out[i,j,k] += markers[p,index]*smoothing_kernel(r, h)

