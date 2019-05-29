from pyccel.decorators import types
from pyccel.decorators import external_call, pure


@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','int','int','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:](order=F)')
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