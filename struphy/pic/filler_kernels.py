"""
Pyccel functions to add one particle to a block matrix and a vector in marker accumulation/deposition step.

__all__ = [ 'fill_mat',
            'fill_vec',
            'fill_mat_vec',
            'fill_mat_pressure_full',
            'fill_mat_vec_pressure_full',
            'fill_mat_pressure',
            'fill_mat_vec_pressure'
            ]
"""

from pyccel.decorators import pure

@pure
def fill_mat(pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int, bi1 : 'float[:]', bi2 : 'float[:]', bi3 : 'float[:]', bj1 : 'float[:]', bj2 : 'float[:]', bj3 : 'float[:]', span1 : int, span2 : int, span3 : int, starts : 'int[:]', pads : 'int[:]', mat : 'float[:,:,:,:,:,:]', filling : float):
    """
    Computes the entries of a matrix block in an arbitrary space and fills it with corresponding basis functions times filling.

    Parameters
    ----------
        pi1, pi2, pi3 : int
            Spline degrees of the codomain (row indices).
            
        pj1, pj2, pj3 : int
            Spline degrees of the domain (column indices).

        bi1, bi2, bi3 : array[float]
            Contain the values of non-vanishing N/D-splines corresponding to the codomain.

        bj1, bj2, bj3 : array[float]
            Contains the values of non-vanishing N/D-splines corresponding to the domain.
            
        span1, span2, span3 : int
            Knot span index in each direction.
        
        starts : array[int]
            Start indices of the codomain.
        
        pads : array[int]
            Paddings of the codomain.
        
        mat : array[float]
            Matrix in which the basis functions times filling is to be written.
        
        filling : float
            Number which will be multiplied by the basis functions and written into mat.
    """
                            
    for il1 in range(pi1 + 1):
        i1  = span1 + il1 - starts[0]
        b1 = bi1[il1] * filling
        for il2 in range(pi2 + 1):
            i2  = span2 + il2 - starts[1]
            b2 = b1 * bi2[il2]
            for il3 in range(pi3 + 1):
                i3  = span3 + il3 - starts[2]
                b3 = b2 * bi3[il3]

                for jl1 in range(pj1 + 1):
                    j1 = pads[0] + jl1 - il1
                    b4 = b3 * bj1[jl1]
                    for jl2 in range(pj2 + 1):
                        j2 = pads[1] + jl2 - il2
                        b5 = b4 * bj2[jl2]
                        for jl3 in range(pj3 + 1):
                            j3 = pads[2] + jl3 - il3
                            b6 = b5 * bj3[jl3]

                            mat[i1, i2, i3, j1, j2, j3] += b6
                            
                            
@pure
def fill_vec(pi1 : int, pi2 : int, pi3 : int, bi1 : 'float[:]', bi2 : 'float[:]', bi3 : 'float[:]', span1 : int, span2 : int, span3 : int, starts : 'int[:]', vec : 'float[:,:,:]', filling : float):
    """
    Computes the entries of a matrix block in an arbitrary space and fills it with corresponding basis functions times filling.

    Parameters
    ----------
        pi1, pi2, pi3 : int
            Spline degrees.

        bi1, bi2, bi3 : array[float]
            Contain the values of non-vanishing N/D-splines.
            
        span1, span2, span3 : int
            Knot span index in each direction.
        
        starts : array[int]
            Start indices of the codomain.
            
        vec : array[float]
            Vector in which the basis functions times filling is to be written.
        
        filling : float
            Number which will be multiplied by the basis functions and written into mat.
    """
    
    for il1 in range(pi1 + 1):
        i1  = span1 + il1 - starts[0]
        b1 = bi1[il1] * filling
        for il2 in range(pi2 + 1):
            i2  = span2 + il2 - starts[1]
            b2 = b1 * bi2[il2]
            for il3 in range(pi3 + 1):
                i3  = span3 + il3 - starts[2]
                b3 = b2 * bi3[il3]
                
                vec[i1, i2, i3] += b3
                            
                            
@pure
def fill_mat_vec(pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int, bi1 : 'float[:]', bi2 : 'float[:]', bi3 : 'float[:]', bj1 : 'float[:]', bj2 : 'float[:]', bj3 : 'float[:]', span1 : int, span2 : int, span3 : int, starts : 'int[:]', pads : 'int[:]', mat : 'float[:,:,:,:,:,:]', filling_mat : float, vec : 'float[:,:,:]', filling_vec : float):
    """
    Computes the entries of a matrix block in an arbitrary space and fills it with corresponding basis functions times filling.

    Parameters
    ----------
        pi1, pi2, pi3 : int
            Spline degrees of the codomain (row indices).
            
        pj1, pj2, pj3 : int
            Spline degrees of the domain (column indices).

        bi1, bi2, bi3 : array[float]
            Contain the values of non-vanishing N/D-splines corresponding to the codomain.

        bj1, bj2, bj3 : array[float]
            Contains the values of non-vanishing N/D-splines corresponding to the domain.
            
        span1, span2, span3 : int
            Knot span index in each direction.
        
        starts : array[int]
            Start indices of the codomain.
        
        pads : array[int]
            Paddings of the codomain.
        
        mat : array[float]
            Matrix in which the basis functions times filling is to be written.
            
        filling_mat : float
            Number which will be multiplied by the basis functions and written into mat.
            
        vec : array[float]
            Vector in which the basis functions times filling is to be written.
        
        filling_vec : float
            Number which will be multiplied by the basis functions and written into vec.
    """

    for il1 in range(pi1 + 1):
        i1  = span1 + il1 - starts[0]
        b1 = bi1[il1]
        for il2 in range(pi2 + 1):
            i2  = span2 + il2 - starts[1]
            b2 = b1 * bi2[il2]
            for il3 in range(pi3 + 1):
                i3  = span3 + il3 - starts[2]
                b3 = b2 * bi3[il3]
                
                vec[i1, i2, i3] += b3 * filling_vec

                for jl1 in range(pj1 + 1):
                    j1 = pads[0] + jl1 - il1
                    b4 = b3 * bj1[jl1] * filling_mat
                    for jl2 in range(pj2 + 1):
                        j2 = pads[1] + jl2 - il2
                        b5 = b4 * bj2[jl2]
                        for jl3 in range(pj3 + 1):
                            j3 = pads[2] + jl3 - il3
                            b6 = b5 * bj3[jl3]

                            mat[i1, i2, i3, j1, j2, j3] += b6

                            
@pure
def fill_mat_pressure_full(pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int, bi1 : 'float[:]', bi2 : 'float[:]', bi3 : 'float[:]', bj1 : 'float[:]', bj2 : 'float[:]', bj3 : 'float[:]', span1 : int, span2 : int, span3 : int, starts : 'int[:]', pads : 'int[:]', mat_11 : 'float[:,:,:,:,:,:]', mat_12 : 'float[:,:,:,:,:,:]', mat_13 : 'float[:,:,:,:,:,:]', mat_22 : 'float[:,:,:,:,:,:]', mat_23 : 'float[:,:,:,:,:,:]', mat_33 : 'float[:,:,:,:,:,:]', filling_mat : float, vx : float, vy : float, vz : float):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pi1, pi2, pi3 : int
            Spline degrees of the codomain (row indices).
            
        pj1, pj2, pj3 : int
            Spline degrees of the domain (column indices).

        bi1, bi2, bi3 : array[float]
            Contain the values of non-vanishing N/D-splines corresponding to the codomain.

        bj1, bj2, bj3 : array[float]
            Contains the values of non-vanishing N/D-splines corresponding to the domain.
            
        span1, span2, span3 : int
            Knot span index in each direction.
        
        starts : array[int]
            Start indices of the codomain.
        
        pads : array[int]
            Paddings of the codomain.
        
        mat_.. : array[float]
            Matrices in which the basis functions times filling times velocity components v_a*v_b are to be written.
        
        filling_mat : float
            Number which will be multiplied by the basis functions and written into mat.
        
        vx, vy, vz : float
            Component of the particle velocity.
    """

    for il1 in range(pi1 + 1):
        i1  = span1 + il1 - starts[0]
        b1 = bi1[il1]
        for il2 in range(pi2 + 1):
            i2  = span2 + il2 - starts[1]
            b2 = b1 * bi2[il2]
            for il3 in range(pi3 + 1):
                i3  = span3 + il3 - starts[2]
                b3 = b2 * bi3[il3]

                for jl1 in range(pj1 + 1):
                    j1 = pads[0] + jl1 - il1
                    b4 = b3 * bj1[jl1] * filling_mat
                    for jl2 in range(pj2 + 1):
                        j2 = pads[1] + jl2 - il2
                        b5 = b4 * bj2[jl2]
                        for jl3 in range(pj3 + 1):
                            j3 = pads[2] + jl3 - il3
                            b6 = b5 * bj3[jl3]

                            mat_11[i1, i2, i3, j1, j2, j3] += b6 * vx * vx
                            mat_12[i1, i2, i3, j1, j2, j3] += b6 * vx * vy
                            mat_13[i1, i2, i3, j1, j2, j3] += b6 * vx * vz
                            mat_22[i1, i2, i3, j1, j2, j3] += b6 * vy * vy
                            mat_23[i1, i2, i3, j1, j2, j3] += b6 * vy * vz
                            mat_33[i1, i2, i3, j1, j2, j3] += b6 * vz * vz                            
                            
                                                                              
@pure
def fill_mat_vec_pressure_full(pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int, bi1 : 'float[:]', bi2 : 'float[:]', bi3 : 'float[:]', bj1 : 'float[:]', bj2 : 'float[:]', bj3 : 'float[:]', span1 : int, span2 : int, span3 : int, starts : 'int[:]', pads : 'int[:]', mat_11 : 'float[:,:,:,:,:,:]', mat_12 : 'float[:,:,:,:,:,:]', mat_13 : 'float[:,:,:,:,:,:]', mat_22 : 'float[:,:,:,:,:,:]', mat_23 : 'float[:,:,:,:,:,:]', mat_33 : 'float[:,:,:,:,:,:]', filling_mat : float, vec_1 : 'float[:,:,:]', vec_2 : 'float[:,:,:]', vec_3 : 'float[:,:,:]', filling_vec : float, vx : float, vy : float, vz : float):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pi1, pi2, pi3 : int
            Spline degrees of the codomain (row indices).
            
        pj1, pj2, pj3 : int
            Spline degrees of the domain (column indices).

        bi1, bi2, bi3 : array[float]
            Contain the values of non-vanishing N/D-splines corresponding to the codomain.

        bj1, bj2, bj3 : array[float]
            Contains the values of non-vanishing N/D-splines corresponding to the domain.
            
        span1, span2, span3 : int
            Knot span index in each direction.
        
        starts : array[int]
            Start indices of the codomain.
        
        pads : array[int]
            Paddings of the codomain.
        
        mat_.. : array[float]
            Matrices in which the basis functions times filling times velocity components v_a*v_b are to be written.
        
        filling_mat : float
            Number which will be multiplied by the basis functions and written into mat.
        
        vec_. : array[float]
            Vectors in which the basis functions times filling times velocity components v_a are to be written.
        
        filling_vec : float
            Number which will be multiplied by the basis functions and written into vec.
        
        vx, vy, vz : float
            Component of the particle velocity.
    """

    for il1 in range(pi1 + 1):
        i1  = span1 + il1 - starts[0]
        b1 = bi1[il1]
        for il2 in range(pi2 + 1):
            i2  = span2 + il2 - starts[1]
            b2 = b1 * bi2[il2]
            for il3 in range(pi3 + 1):
                i3  = span3 + il3 - starts[2]
                b3 = b2 * bi3[il3]

                vec_1[i1, i2, i3] += b3 * filling_vec * vx
                vec_2[i1, i2, i3] += b3 * filling_vec * vy
                vec_3[i1, i2, i3] += b3 * filling_vec * vz

                for jl1 in range(pj1 + 1):
                    j1 = pads[0] + jl1 - il1
                    b4 = b3 * bj1[jl1] * filling_mat
                    for jl2 in range(pj2 + 1):
                        j2 = pads[1] + jl2 - il2
                        b5 = b4 * bj2[jl2]
                        for jl3 in range(pj3 + 1):
                            j3 = pads[2] + jl3 - il3
                            b6 = b5 * bj3[jl3]

                            mat_11[i1, i2, i3, j1, j2, j3] += b6 * vx * vx
                            mat_12[i1, i2, i3, j1, j2, j3] += b6 * vx * vy
                            mat_13[i1, i2, i3, j1, j2, j3] += b6 * vx * vz
                            mat_22[i1, i2, i3, j1, j2, j3] += b6 * vy * vy
                            mat_23[i1, i2, i3, j1, j2, j3] += b6 * vy * vz
                            mat_33[i1, i2, i3, j1, j2, j3] += b6 * vz * vz                            


@pure
def fill_mat_pressure(pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int, bi1 : 'float[:]', bi2 : 'float[:]', bi3 : 'float[:]', bj1 : 'float[:]', bj2 : 'float[:]', bj3 : 'float[:]', span1 : int, span2 : int, span3 : int, starts : 'int[:]', pads : 'int[:]', mat_11 : 'float[:,:,:,:,:,:]', mat_12 : 'float[:,:,:,:,:,:]', mat_22 : 'float[:,:,:,:,:,:]', filling_mat : float, vx : float, vy : float):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pi1, pi2, pi3 : int
            Spline degrees of the codomain (row indices).
            
        pj1, pj2, pj3 : int
            Spline degrees of the domain (column indices).

        bi1, bi2, bi3 : array[float]
            Contain the values of non-vanishing N/D-splines corresponding to the codomain.

        bj1, bj2, bj3 : array[float]
            Contains the values of non-vanishing N/D-splines corresponding to the domain.
            
        span1, span2, span3 : int
            Knot span index in each direction.
        
        starts : array[int]
            Start indices of the codomain.
        
        pads : array[int]
            Paddings of the codomain.
        
        mat_.. : array[float]
            Matrices in which the basis functions times filling times velocity components v_a*v_b are to be written.
        
        filling_mat : float
            Number which will be multiplied by the basis functions and written into mat.
        
        vx, vy, vz : float
            Component of the particle velocity.
    """

    for il1 in range(pi1 + 1):
        i1  = span1 + il1 - starts[0]
        b1 = bi1[il1]
        for il2 in range(pi2 + 1):
            i2  = span2 + il2 - starts[1]
            b2 = b1 * bi2[il2]
            for il3 in range(pi3 + 1):
                i3  = span3 + il3 - starts[2]
                b3 = b2 * bi3[il3]

                for jl1 in range(pj1 + 1):
                    j1 = pads[0] + jl1 - il1
                    b4 = b3 * bj1[jl1] * filling_mat
                    for jl2 in range(pj2 + 1):
                        j2 = pads[1] + jl2 - il2
                        b5 = b4 * bj2[jl2]
                        for jl3 in range(pj3 + 1):
                            j3 = pads[2] + jl3 - il3
                            b6 = b5 * bj3[jl3]

                            mat_11[i1, i2, i3, j1, j2, j3] += b6 * vx * vx
                            mat_12[i1, i2, i3, j1, j2, j3] += b6 * vx * vy
                            mat_22[i1, i2, i3, j1, j2, j3] += b6 * vy * vy                        
                            
                                                                              
@pure
def fill_mat_vec_pressure(pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int, bi1 : 'float[:]', bi2 : 'float[:]', bi3 : 'float[:]', bj1 : 'float[:]', bj2 : 'float[:]', bj3 : 'float[:]', span1 : int, span2 : int, span3 : int, starts : 'int[:]', pads : 'int[:]', mat_11 : 'float[:,:,:,:,:,:]', mat_12 : 'float[:,:,:,:,:,:]', mat_22 : 'float[:,:,:,:,:,:]', filling_mat : float, vec_1 : 'float[:,:,:]', vec_2 : 'float[:,:,:]', filling_vec : float, vx : float, vy : float):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pi1, pi2, pi3 : int
            Spline degrees of the codomain (row indices).
            
        pj1, pj2, pj3 : int
            Spline degrees of the domain (column indices).

        bi1, bi2, bi3 : array[float]
            Contain the values of non-vanishing N/D-splines corresponding to the codomain.

        bj1, bj2, bj3 : array[float]
            Contains the values of non-vanishing N/D-splines corresponding to the domain.
            
        span1, span2, span3 : int
            Knot span index in each direction.
        
        starts : array[int]
            Start indices of the codomain.
        
        pads : array[int]
            Paddings of the codomain.
        
        mat_.. : array[float]
            Matrices in which the basis functions times filling times velocity components v_a*v_b are to be written.
        
        filling_mat : float
            Number which will be multiplied by the basis functions and written into mat.
        
        vec_. : array[float]
            Vectors in which the basis functions times filling times velocity components v_a are to be written.
        
        filling_vec : float
            Number which will be multiplied by the basis functions and written into vec.
        
        vx, vy, vz : float
            Component of the particle velocity.
    """

    for il1 in range(pi1 + 1):
        i1  = span1 + il1 - starts[0]
        b1 = bi1[il1]
        for il2 in range(pi2 + 1):
            i2  = span2 + il2 - starts[1]
            b2 = b1 * bi2[il2]
            for il3 in range(pi3 + 1):
                i3  = span3 + il3 - starts[2]
                b3 = b2 * bi3[il3]

                vec_1[i1, i2, i3] += b3 * filling_vec * vx
                vec_2[i1, i2, i3] += b3 * filling_vec * vy

                for jl1 in range(pj1 + 1):
                    j1 = pads[0] + jl1 - il1
                    b4 = b3 * bj1[jl1] * filling_mat
                    for jl2 in range(pj2 + 1):
                        j2 = pads[1] + jl2 - il2
                        b5 = b4 * bj2[jl2]
                        for jl3 in range(pj3 + 1):
                            j3 = pads[2] + jl3 - il3
                            b6 = b5 * bj3[jl3]

                            mat_11[i1, i2, i3, j1, j2, j3] += b6 * vx * vx
                            mat_12[i1, i2, i3, j1, j2, j3] += b6 * vx * vy
                            mat_22[i1, i2, i3, j1, j2, j3] += b6 * vy * vy                         
