"""
Pyccel functions to add one particle to a block matrix and a vector in marker accumulation/deposition step.

__all__ = ['fill_mat11_v1',
            'fill_mat12_v1',
            'fill_mat13_v1',
            'fill_mat21_v1',
            'fill_mat22_v1',
            'fill_mat23_v1',
            'fill_mat31_v1',
            'fill_mat32_v1',
            'fill_mat33_v1',
            'fill_mat12_v2',
            'fill_mat13_v2',
            'fill_mat21_v2',
            'fill_mat22_v2',
            'fill_mat23_v2',
            'fill_mat31_v2',
            'fill_mat32_v2',
            'fill_mat33_v2',
            'fill_mat11_vec1_v1',
            'fill_mat12_vec1_v1',
            'fill_mat13_vec1_v1',
            'fill_mat21_vec2_v1',
            'fill_mat22_vec2_v1',
            'fill_mat23_vec2_v1',
            'fill_mat31_vec3_v1',
            'fill_mat32_vec3_v1',
            'fill_mat33_vec3_v1',
            'fill_mat11_vec1_v2',
            'fill_mat12_vec1_v2',
            'fill_mat13_vec1_v2',
            'fill_mat21_vec2_v2',
            'fill_mat22_vec2_v2',
            'fill_mat23_vec2_v2',
            'fill_mat31_vec3_v2',
            'fill_mat32_vec3_v2',
            'fill_mat33_vec3_v2',
            'fill_vec1_v1',
            'fill_vec2_v1',
            'fill_vec3_v1',
            'fill_vec1_v2',
            'fill_vec2_v2',
            'fill_vec3_v2',
            ]
"""

# =====================================================================================================
def fill_mat11_v1(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V1 is to be written
        
        filling11 : float
            number which will be multiplied by the basis functions of V1 and written into mat11
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]

    # (DNN DNN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling11
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat11[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V1 is to be written
        
        filling12 : float
            number which will be multiplied by the basis functions of V1 and written into mat12
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DNN NDN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling12
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat12[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling13 : float
            number which will be multiplied by the basis functions of V1 and written into mat13
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DNN NND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling12
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat12[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V1 is to be written
        
        filling21 : float
            number which will be multiplied by the basis functions of V1 and written into mat21
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN DNN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling21
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat21[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_v1(p : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V1 is to be written
        
        filling22 : float
            number which will be multiplied by the basis functions of V1 and written into mat22
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN NDN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling22
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat22[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_v1(p : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat23 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling23 : float
            number which will be multiplied by the basis functions of V1 and written into mat23
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDN NND)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling23
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat23[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat31 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling31 : float
            number which will be multiplied by the basis functions of V1 and written into mat31
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NND DNN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling31
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat31[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_v1(p : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling32 : float
            number which will be multiplied by the basis functions of V1 and written into mat32
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NND NDN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling32
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat32[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_v1(p : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V1 is to be written
        
        filling33 : float
            number which will be multiplied by the basis functions of V1 and written into mat33
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NND NND)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling33
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat33[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat11_v2(p : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V2 is to be written
        
        filling11 : float
            number which will be multiplied by the basis functions of V2 and written into mat11
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDD NDD)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling11
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat11[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V2 is to be written
        
        filling12 : float
            number which will be multiplied by the basis functions of V2 and written into mat12
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDD DND)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling12
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat12[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat13 : 'float[:,:,:,:,:,:]', filling13 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V2 is to be written
        
        filling13 : float
            number which will be multiplied by the basis functions of V2 and written into mat13
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDD DDN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling13
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat13[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V2 is to be written
        
        filling21 : float
            number which will be multiplied by the basis functions of V2 and written into mat21
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND NDD)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling21
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat21[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_v2(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V2 is to be written
        
        filling22 : float
            number which will be multiplied by the basis functions of V2 and written into mat22
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND DND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling22
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat22[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_v2(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat23 : array
            matrix in which the filling23 times the basis functions of V2 is to be written
        
        filling23 : float
            number which will be multiplied by the basis functions of V2 and written into mat23
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND DDN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling23
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat23[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat31 : array
            matrix in which the filling31 times the basis functions of V2 is to be written
        
        filling31 : float
            number which will be multiplied by the basis functions of V2 and written into mat31
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DDN NDD)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling31
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat31[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_v2(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V2 is to be written
        
        filling32 : float
            number which will be multiplied by the basis functions of V2 and written into mat32
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DDN DND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling32
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat32[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_v2(p : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V2 is to be written
        
        filling33 : float
            number which will be multiplied by the basis functions of V2 and written into mat33
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DDN DDN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling33
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat33[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat11_vec1_v1(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V1 is to be written
        
        filling11 : float
            number which will be multiplied by the basis functions of V1 and written into mat11
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V1 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]

    # (DNN DNN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling11
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat11[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_vec1_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V1 is to be written
        
        filling12 : float
            number which will be multiplied by the basis functions of V1 and written into mat12
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V1 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DNN NDN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling12
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat12[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_vec1_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat13 : 'float[:,:,:,:,:,:]', filling13 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling13 : float
            number which will be multiplied by the basis functions of V1 and written into mat13
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V1 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DNN NND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling13
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat13[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_vec2_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V1 is to be written
        
        filling21 : float
            number which will be multiplied by the basis functions of V1 and written into mat21
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V1 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN DNN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling21
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat21[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_vec2_v1(p : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V1 is to be written
        
        filling22 : float
            number which will be multiplied by the basis functions of V1 and written into mat22
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V1 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN NDN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling22
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat22[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_vec2_v1(p : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat23 : array
            matrix in which the filling23 times the basis functions of V1 is to be written
        
        filling23 : float
            number which will be multiplied by the basis functions of V1 and written into mat23
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V1 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDN NND)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling23
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat23[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_vec3_v1(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat31 : array
            matrix in which the filling31 times the basis functions of V1 is to be written
        
        filling31 : float
            number which will be multiplied by the basis functions of V1 and written into mat31
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V1 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NND DNN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling31
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat31[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_vec3_v1(p : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling32 : float
            number which will be multiplied by the basis functions of V1 and written into mat32
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V1 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NND NDN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling32
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat32[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_vec3_v1(p : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V1 is to be written
        
        filling33 : float
            number which will be multiplied by the basis functions of V1 and written into mat33
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V1 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NND NND)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling33
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat33[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat11_vec1_v2(p : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V2 is to be written
        
        filling11 : float
            number which will be multiplied by the basis functions of V2 and written into mat11
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDD NDD)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling11
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat11[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_vec1_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V2 is to be written
        
        filling12 : float
            number which will be multiplied by the basis functions of V2 and written into mat12
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDD DND)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling12
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat12[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_vec1_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat13 : 'float[:,:,:,:,:,:]', filling13 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V2 is to be written
        
        filling13 : float
            number which will be multiplied by the basis functions of V2 and written into mat13
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (NDD DDN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling13
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat13[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_vec2_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V2 is to be written
        
        filling21 : float
            number which will be multiplied by the basis functions of V2 and written into mat21
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND NDD)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling21
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat21[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_vec2_v2(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V2 is to be written
        
        filling22 : float
            number which will be multiplied by the basis functions of V2 and written into mat22
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND DND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling22
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat22[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_vec2_v2(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat23 : array
            matrix in which the filling23 times the basis functions of V2 is to be written
        
        filling23 : float
            number which will be multiplied by the basis functions of V2 and written into mat23
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND DDN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling23
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat23[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_vec3_v2(p : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat31 : array
            matrix in which the filling31 times the basis functions of V2 is to be written
        
        filling31 : float
            number which will be multiplied by the basis functions of V2 and written into mat31
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DDN NDD)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling31
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat31[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_vec3_v2(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V2 is to be written
        
        filling32 : float
            number which will be multiplied by the basis functions of V2 and written into mat32
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DDN DND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling32
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat32[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_vec3_v2(p : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V2 is to be written
        
        filling33 : float
            number which will be multiplied by the basis functions of V2 and written into mat33
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DDN DDN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling33
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat33[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_vec1_v1(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the mu=1 element of a vector in V1 and fills it with basis functions times filling1

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V1 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]

    # (DNN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

# =====================================================================================================
def fill_vec2_v1(p : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the mu=2 element of a vector in V1 and fills it with basis functions times filling2

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V1 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

# =====================================================================================================
def fill_vec3_v1(p : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the mu=3 element of a vector in V1 and fills it with basis functions times filling3

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V1 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd3 = p[2] - 1

    # (NND)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

# =====================================================================================================
def fill_vec1_v2(p : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the mu=1 element of a vector in V2 and fills it with basis functions times filling1

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (NDD)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec1[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling1

# =====================================================================================================
def fill_vec2_v2(p : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the mu=2 element of a vector in V2 and fills it with basis functions times filling2

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd3 = p[2] - 1

    # (DND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec2[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling2

# =====================================================================================================
def fill_vec3_v2(p : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', start1 : 'int', start2 : 'int', start3 : 'int', pad1 : 'int', pad2 : 'int', pad3 : 'int', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the mu=3 element of a vector in V2 and fills it with basis functions times filling3

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DDN)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec3[i1 - start1 + pad1, i2 - start2 + pad2, i3 - start3 + pad3] += bi3 * filling3

