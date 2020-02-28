from pyccel.decorators import types
from pyccel.decorators import external_call

# ==============================================================================
@external_call
@types('double[:,:]','double[:,:](order=F)')
def set_particles_symmetric(numbers, particles):
    
    from numpy import zeros
    
    q  = zeros(3, dtype=float)
    v  = zeros(3, dtype=float)
    np = len(particles[:, 0])
    
    for i_part in range(np):
        ip = i_part%64
        
        if ip == 0:
            q = numbers[int(i_part/64), 0:3]
            v = numbers[int(i_part/64), 3:6]
            
        elif ip%32 == 0:
            v[2] = 1 - v[2]
            
        elif ip%16 == 0:
            v[1] = 1 - v[1]
            
        elif ip%8 == 0:
            v[0] = 1 - v[0]
            
        elif ip%4 == 0:
            q[2] = 1 - q[2] 
             
        elif ip%2 == 0:
            q[1] = 1 - q[1]
            
        else:
            q[0] = 1 - q[0]
        
        particles[i_part, 0:3] = q
        particles[i_part, 3:6] = v  
        
    
    ierr = 0
# ==============================================================================