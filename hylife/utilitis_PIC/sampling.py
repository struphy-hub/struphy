# import pyccel decorators
from pyccel.decorators import types

# import modules for mapping related quantities
import hylife.geometry.mappings_analytical as mapping

# import input files for simulation setup
import input_run.equilibrium_PIC        as eq_pic
import input_run.initial_conditions_PIC as ini_pic


# ==============================================================================
@types('double[:,:]','double[:,:]','int')
def set_particles_symmetric(numbers, particles, np):
    
    from numpy import zeros
    
    eta  = zeros(3, dtype=float)
    v    = zeros(3, dtype=float)
    
    for i_part in range(np):
        ip = i_part%64
        
        if ip == 0:
            eta[:] = numbers[int(i_part/64), 0:3]
            v[:]   = numbers[int(i_part/64), 3:6]
            
        elif ip%32 == 0:
            v[2] = 1 - v[2]
            
        elif ip%16 == 0:
            v[1] = 1 - v[1]
            
        elif ip%8 == 0:
            v[0] = 1 - v[0]
            
        elif ip%4 == 0:
            eta[2] = 1 - eta[2] 
             
        elif ip%2 == 0:
            eta[1] = 1 - eta[1]
            
        else:
            eta[0] = 1 - eta[0]
        
        particles[0:3, i_part] = eta
        particles[3:6, i_part] = v  
        
    
# ==============================================================================
@types('double[:,:]','int','double[:]','double[:]','int','double[:]')
def compute_weights_ini(particles, np, w0, s0, kind_map, params_map):
    
    #$ omp parallel
    #$ omp do private (ip)
    for ip in range(np):
        s0[ip] = ini_pic.sh(particles[0, ip], particles[1, ip], particles[2, ip], particles[3, ip], particles[4, ip], particles[5, ip], kind_map, params_map)
        w0[ip] = ini_pic.fh_ini(particles[0, ip], particles[1, ip], particles[2, ip], particles[3, ip], particles[4, ip], particles[5, ip], kind_map, params_map)/s0[ip]
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==============================================================================
@types('double[:,:]','int','double[:]','double[:]','int','double[:]')
def update_weights(particles, np, w0, s0, kind_map, params_map):
    
    #$ omp parallel
    #$ omp do private (ip)
    for ip in range(np):
        particles[6, ip] = w0[ip] - eq_pic.fh_eq(particles[0, ip], particles[1, ip], particles[2, ip], particles[3, ip], particles[4, ip], particles[5, ip], kind_map, params_map)/s0[ip]
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0