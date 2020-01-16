from pyccel.decorators import types
from pyccel.decorators import pure
from pyccel.decorators import external_call



#==========================================================================================================
@external_call
@types('double[:,:](order=F)','int[:]','int[:,:](order=F)','int[:]','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:]','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)')
def evaluate_1form(particles_pos, p0, spans0, Nbase, Np, u1, u2, u3, Ueq, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, U_part):
    
    p0_1   = p0[0]
    p0_2   = p0[1]
    p0_3   = p0[2]
    
    p1_1   = p0_1 - 1
    p1_2   = p0_2 - 1
    p1_3   = p0_3 - 1
    
    delta1 = 1/Nbase[0]
    delta2 = 1/Nbase[1]
    delta3 = 1/Nbase[2]
    
    
    for ip in range(Np):
        
        U_part[ip, 0] = Ueq[0]
        U_part[ip, 1] = Ueq[1]
        U_part[ip, 2] = Ueq[2]
        
        span0_1  = spans0[ip, 0]
        span0_2  = spans0[ip, 1]
        span0_3  = spans0[ip, 2]
        
        span1_1  = span0_1 - 1
        span1_2  = span0_2 - 1
        span1_3  = span0_3 - 1
        
        posloc_1 = particles_pos[ip, 0] - (span0_1 - p0_1)*delta1
        posloc_2 = particles_pos[ip, 1] - (span0_2 - p0_2)*delta2
        posloc_3 = particles_pos[ip, 2] - (span0_3 - p0_3)*delta3
        
        # evaluation of 1 - component (DNN)
        for jl3 in range(p0_3 + 1):
            pow3 = posloc_3**jl3
            for jl2 in range(p0_2 + 1):
                pow2 = posloc_2**jl2
                for jl1 in range(p1_1 + 1):
                    pow1 = posloc_1**jl1
                    
                    for il3 in range(p0_3 + 1):
                        i3 = (span0_3 - il3)%Nbase[2]
                        N3 = pp0_3[p0_3 - il3, jl3] * pow3
                        for il2 in range(p0_2 + 1):
                            i2 = (span0_2 - il2)%Nbase[1]
                            N2 = pp0_2[p0_2 - il2, jl2] * pow2
                            for il1 in range(p1_1 + 1):
                                i1 = (span1_1 - il1)%Nbase[0]
                                D1 = pp1_1[p1_1 - il1, jl1] * pow1

                                U_part[ip, 0] += u1[i1, i2, i3] * D1 * N2 * N3
        
        
        # evaluation of 2 - component (NDN)
        for jl3 in range(p0_3 + 1):
            pow3 = posloc_3**jl3
            for jl2 in range(p1_2 + 1):
                pow2 = posloc_2**jl2
                for jl1 in range(p0_1 + 1):
                    pow1 = posloc_1**jl1
                    
                    for il3 in range(p0_3 + 1):
                        i3 = (span0_3 - il3)%Nbase[2]
                        N3 = pp0_3[p0_3 - il3, jl3] * pow3
                        for il2 in range(p1_2 + 1):
                            i2 = (span1_2 - il2)%Nbase[1]
                            D2 = pp1_2[p1_2 - il2, jl2] * pow2
                            for il1 in range(p0_1 + 1):
                                i1 = (span0_1 - il1)%Nbase[0]
                                N1 = pp0_1[p0_1 - il1, jl1] * pow1
                                
                                U_part[ip, 1] += u2[i1, i2, i3] * N1 * D2 * N3
                                
        
        # evaluation of 3 - component (NND)
        for jl3 in range(p1_3 + 1):
            pow3 = posloc_3**jl3
            for jl2 in range(p0_2 + 1):
                pow2 = posloc_2**jl2
                for jl1 in range(p0_1 + 1):
                    pow1 = posloc_1**jl1
                    
                    for il3 in range(p1_3 + 1):
                        i3 = (span1_3 - il3)%Nbase[2]
                        D3 = pp1_3[p1_3 - il3, jl3] * pow3
                        for il2 in range(p0_2 + 1):
                            i2 = (span0_2 - il2)%Nbase[1]
                            N2 = pp0_2[p0_2 - il2, jl2] * pow2
                            for il1 in range(p0_1 + 1):
                                i1 = (span0_1 - il1)%Nbase[0]
                                N1 = pp0_1[p0_1 - il1, jl1] * pow1
                                
                                U_part[ip, 2] += u3[i1, i2, i3] * N1 * N2 * D3
                                 
    ierr = 0
#==========================================================================================================




#==========================================================================================================
@external_call
@types('double[:,:](order=F)','int[:]','int[:,:](order=F)','int[:]','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:]','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)')
def evaluate_2form(particles_pos, p0, spans0, Nbase, Np, b1, b2, b3, Beq, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, B_part):
    
    p0_1   = p0[0]
    p0_2   = p0[1]
    p0_3   = p0[2]
    
    p1_1   = p0_1 - 1
    p1_2   = p0_2 - 1
    p1_3   = p0_3 - 1
    
    delta1 = 1/Nbase[0]
    delta2 = 1/Nbase[1]
    delta3 = 1/Nbase[2]
    
    
    for ip in range(Np):
        
        B_part[ip, 0] = Beq[0]
        B_part[ip, 1] = Beq[1]
        B_part[ip, 2] = Beq[2]
        
        span0_1  = spans0[ip, 0]
        span0_2  = spans0[ip, 1]
        span0_3  = spans0[ip, 2]
        
        span1_1  = span0_1 - 1
        span1_2  = span0_2 - 1
        span1_3  = span0_3 - 1
        
        posloc_1 = particles_pos[ip, 0] - (span0_1 - p0_1)*delta1
        posloc_2 = particles_pos[ip, 1] - (span0_2 - p0_2)*delta2
        posloc_3 = particles_pos[ip, 2] - (span0_3 - p0_3)*delta3
        
        # evaluation of 1 - component (NDD)
        for jl3 in range(p1_3 + 1):
            pow3 = posloc_3**jl3
            for jl2 in range(p1_2 + 1):
                pow2 = posloc_2**jl2
                for jl1 in range(p0_1 + 1):
                    pow1 = posloc_1**jl1
                    
                    for il3 in range(p1_3 + 1):
                        i3 = (span1_3 - il3)%Nbase[2]
                        D3 = pp1_3[p1_3 - il3, jl3] * pow3
                        for il2 in range(p1_2 + 1):
                            i2 = (span1_2 - il2)%Nbase[1]
                            D2 = pp1_2[p1_2 - il2, jl2] * pow2 
                            for il1 in range(p0_1 + 1):
                                i1 = (span0_1 - il1)%Nbase[0]
                                N1 = pp0_1[p0_1 - il1, jl1] * pow1 

                                B_part[ip, 0] += b1[i1, i2, i3] * N1 * D2 * D3
        
        
        # evaluation of 2 - component (DND)
        for jl3 in range(p1_3 + 1):
            pow3 = posloc_3**jl3
            for jl2 in range(p0_2 + 1):
                pow2 = posloc_2**jl2
                for jl1 in range(p1_1 + 1):
                    pow1 = posloc_1**jl1
                    
                    for il3 in range(p1_3 + 1):
                        i3 = (span1_3 - il3)%Nbase[2]
                        D3 = pp1_3[p1_3 - il3, jl3] * pow3
                        for il2 in range(p0_2 + 1):
                            i2 = (span0_2 - il2)%Nbase[1]
                            N2 = pp0_2[p0_2 - il2, jl2] * pow2
                            for il1 in range(p1_1 + 1):
                                i1 = (span1_1 - il1)%Nbase[0]
                                D1 = pp1_1[p1_1 - il1, jl1] * pow1
                                
                                B_part[ip, 1] += b2[i1, i2, i3] * D1 * N2 * D3
                                
        
        # evaluation of 3 - component (DDN)
        for jl3 in range(p0_3 + 1):
            pow3 = posloc_3**jl3
            for jl2 in range(p1_2 + 1):
                pow2 = posloc_2**jl2
                for jl1 in range(p1_1 + 1):
                    pow1 = posloc_1**jl1
                    
                    for il3 in range(p0_3 + 1):
                        i3 = (span0_3 - il3)%Nbase[2]
                        N3 = pp0_3[p0_3 - il3, jl3] * pow3
                        for il2 in range(p1_2 + 1):
                            i2 = (span1_2 - il2)%Nbase[1]
                            D2 = pp1_2[p1_2 - il2, jl2] * pow2
                            for il1 in range(p1_1 + 1):
                                i1 = (span1_1 - il1)%Nbase[0]
                                D1 = pp1_1[p1_1 - il1, jl1] * pow1
                                
                                B_part[ip, 2] += b3[i1, i2, i3] * D1 * D2 * N3
                            
        
    ierr = 0
#==========================================================================================================