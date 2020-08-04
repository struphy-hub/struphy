# import pyccel decorators
from pyccel.decorators import types

# import input files for simulation setup
import input_run.equilibrium_PIC        as eq_pic
import input_run.initial_conditions_PIC as ini_pic

# import modules for mapping related quantities
import hylife.geometry.mappings_analytical as mapping


# ==============================================================================
@types('double[:,:]','double[:,:]')
def set_particles_symmetric(numbers, particles):
    
    from numpy import zeros
    
    q  = zeros(3, dtype=float)
    v  = zeros(3, dtype=float)
    np = len(particles[:, 0])
    
    for i_part in range(np):
        ip = i_part%64
        
        if ip == 0:
            q[:] = numbers[int(i_part/64), 0:3]
            v[:] = numbers[int(i_part/64), 3:6]
            
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
        
    
# ==============================================================================
@types('double[:,:]','double[:]','double[:]','int','double[:]')
def compute_weights_ini(particles, w0, s0, kind_map, params_map):
    
    np = len(particles[:, 0])
    
    for ip in range(np):
        
        xi1 = particles[ip, 0]
        xi2 = particles[ip, 1]
        xi3 = particles[ip, 2]
        
        vx  = particles[ip, 3]
        vy  = particles[ip, 4]
        vz  = particles[ip, 5]
        
        s0[ip] = ini_pic.sh(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map)
        w0[ip] = ini_pic.fh_ini(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map)/s0[ip]
    
    
# ==============================================================================
@types('double[:,:]','double[:]','double[:]','int','double[:]')
def update_weights(particles, w0, s0, kind_map, params_map):
    
    np = len(particles[:, 0])
    
    for ip in range(np):
        
        xi1 = particles[ip, 0]
        xi2 = particles[ip, 1]
        xi3 = particles[ip, 2]
        
        vx  = particles[ip, 3]
        vy  = particles[ip, 4]
        vz  = particles[ip, 5]
         
        particles[ip, 6] = w0[ip] - eq_pic.fh_eq(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map)/s0[ip]