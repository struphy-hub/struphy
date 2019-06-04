from pyccel.decorators import types
from pyccel.decorators import external_call, pure



@external_call
@types('int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel0(px, py, pz, bx, by, bz, wx, wy, wz, ggs, mat):
    mat[:, :, :, :, :, :] = 0.
    
    for ilx in range(px + 1):
        for ily in range(py + 1):
            for ilz in range(pz + 1):
                for jlx in range(px + 1):
                    for jly in range(py + 1):
                        for jlz in range(pz + 1):
                            
                            value = 0.
                       
                            for gx in range(px + 1):
                                for gy in range(py + 1):
                                    for gz in range(pz + 1):

                                        wvol = wx[gx]*wy[gy]*wz[gz]
                                        bi = bx[ilx, gx]*by[ily, gy]*bz[ilz, gz]
                                        bj = bx[jlx, gx]*by[jly, gy]*bz[jlz, gz]
                                        value += wvol*bi*bj*ggs[gx, gy, gz]
                                        
                            mat[ilx, ily, ilz, jlx, jly, jlz] = value



@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','int','int','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel1(nx, ny, nz, px, py, pz, nix, niy, niz, njx, njy, njz, bix, biy, biz, bjx, bjy, bjz, p1, p2, t1, t2, wx, wy, wz, ggs, ginvs, mat):
    mat[:, :, :, :, :, :] = 0.
    
    for ilx in range(px + 1 - nix):
        for ily in range(py + 1 - niy):
            for ilz in range(pz + 1 - niz):
                for jlx in range(px + 1 - njx):
                    for jly in range(py + 1 - njy):
                        for jlz in range(pz + 1 - njz):
                            
                            ix = (nx + ilx)*nix 
                            iy = (ny + ily)*niy 
                            iz = (nz + ilz)*niz
                            
                            jx = (nx + jlx)*njx 
                            jy = (ny + jly)*njy 
                            jz = (nz + jlz)*njz
                            
                            value = 0.
                       
                            for gx in range(px + 1):
                                for gy in range(py + 1):
                                    for gz in range(pz + 1):

                                        wvol = wx[gx]*wy[gy]*wz[gz]
                                        bi = bix[ilx, gx]*biy[ily, gy]*biz[ilz, gz]
                                        bj = bjx[jlx, gx]*bjy[jly, gy]*bjz[jlz, gz]
                                        gvol = ggs[gx, gy, gz]*ginvs[gx, gy, gz]
                                        value += wvol*bi*bj*gvol
                                        
                            mat[ilx, ily, ilz, jlx, jly, jlz] = value*p1/(t1[ix + iy + iz + p1] - t1[ix + iy + iz])*p2/(t2[jx + jy + jz + p2] - t2[jx + jy + jz])
                            
                            

@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','int','int','int','int','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel2(nx, ny, nz, px, py, pz, nix, niy, niz, njx, njy, njz, bix, biy, biz, bjx, bjy, bjz, pi1, pi2, pj1, pj2, ti1, ti2, tj1, tj2, wx, wy, wz, ggs, gs, mat):
    mat[:, :, :, :, :, :] = 0.
    
    for ilx in range(px + 1 - nix):
        for ily in range(py + 1 - niy):
            for ilz in range(pz + 1 - niz):
                for jlx in range(px + 1 - njx):
                    for jly in range(py + 1 - njy):
                        for jlz in range(pz + 1 - njz):
                            
                            
                            ix = nx + ilx
                            iy = ny + ily 
                            iz = nz + ilz
                            
                            jx = nx + jlx 
                            jy = ny + jly 
                            jz = nz + jlz
                            
                            i1 = ix*nix + iy*niy - iy*nix*niy
                            i2 = iy*niy + iz*niz - iy*niy*niz
                            
                            j1 = jx*njx + jy*njy - jy*njx*njy
                            j2 = jy*njy + jz*njz - jy*njy*njz 
                            
                            
                            value = 0.
                       
                            for gx in range(px + 1):
                                for gy in range(py + 1):
                                    for gz in range(pz + 1):

                                        wvol = wx[gx]*wy[gy]*wz[gz]
                                        bi = bix[ilx, gx]*biy[ily, gy]*biz[ilz, gz]
                                        bj = bjx[jlx, gx]*bjy[jly, gy]*bjz[jlz, gz]
                                        gvol = gs[gx, gy, gz]/ggs[gx, gy, gz]
                                        value += wvol*bi*bj*gvol
                                        
                            di1 = pi1/(ti1[i1 + pi1] - ti1[i1])
                            di2 = pi2/(ti2[i2 + pi2] - ti2[i2])
                            
                            dj1 = pj1/(tj1[j1 + pj1] - tj1[j1])
                            dj2 = pj2/(tj2[j2 + pj2] - tj2[j2])
                            
                            mat[ilx, ily, ilz, jlx, jly, jlz] = value*di1*di2*dj1*dj2
                        
                            
                            

@external_call
@types('int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernelL0(px, py, pz, bix, biy, biz, wx, wy, wz, ggs, ffs, mat):
    mat[:, :, :] = 0.
    
    for ilx in range(px + 1):
        for ily in range(py + 1):
            for ilz in range(pz + 1):       
                            
                value = 0.

                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):

                            wvol = wx[gx]*wy[gy]*wz[gz]
                            bi = bix[ilx, gx]*biy[ily, gy]*biz[ilz, gz]
                            value += wvol*bi*ggs[gx, gy, gz]*ffs[gx, gy, gz]

                mat[ilx, ily, ilz] = value
                
                
                
@external_call
@types('int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','int','double[:]','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernelL1(nx, ny, nz, px, py, pz, nix, niy, niz, bix, biy, biz, p1, t1, wx, wy, wz, ggs, ginvs, ffs, mat):
    mat[:, :, :] = 0.
    
    for ilx in range(px + 1 - nix):
        for ily in range(py + 1 - niy):
            for ilz in range(pz + 1 - niz):  
                
                ix = (nx + ilx)*nix 
                iy = (ny + ily)*niy 
                iz = (nz + ilz)*niz
                            
                value = 0.

                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):

                            wvol = wx[gx]*wy[gy]*wz[gz]
                            bi = bix[ilx, gx]*biy[ily, gy]*biz[ilz, gz]
                            gvol = ggs[gx, gy, gz]*ginvs[gx, gy, gz]
                            value += wvol*bi*gvol*ffs[gx, gy, gz]

                mat[ilx, ily, ilz] = value*p1/(t1[ix + iy + iz + p1] - t1[ix + iy + iz])