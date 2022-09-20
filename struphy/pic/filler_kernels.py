"""
Pyccel functions to add one particle to a block matrix and a vector in marker accumulation/deposition step.

__all__ = [ 'fill_mat11_v1',
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
            'fill_mat_u0',
            'fill_mat_u3',
            'fill_mat_vec_u0',
            'fill_mat_vec_u3',
            'fill_vec_u0',
            'fill_vec_u3',
            'fill_mat11_vec1_v1_pressure',
            'fill_mat22_vec2_v1_pressure',
            'fill_mat33_vec3_v1_pressure',
            'fill_mat12_v1_pressure',
            'fill_mat13_v1_pressure',
            'fill_mat23_v1_pressure',
            ]
"""


def fill_mat11_v1(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts : 'int[:]', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V1 is to be written
        
        filling11 : float
            number which will be multiplied by the basis functions of V1 and written into mat11
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]

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

                            mat11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat12_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V1 is to be written
        
        filling12 : float
            number which will be multiplied by the basis functions of V1 and written into mat12
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                            mat12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat13_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', filling13 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling13 : float
            number which will be multiplied by the basis functions of V1 and written into mat13
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

    # (DNN NND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling13
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

                            mat12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat21_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V1 is to be written
        
        filling21 : float
            number which will be multiplied by the basis functions of V1 and written into mat21
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                            mat21[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat22_v1(pn : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V1 is to be written
        
        filling22 : float
            number which will be multiplied by the basis functions of V1 and written into mat22
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                            mat22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat23_v1(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat23 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling23 : float
            number which will be multiplied by the basis functions of V1 and written into mat23
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat31_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat31 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling31 : float
            number which will be multiplied by the basis functions of V1 and written into mat31
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat31[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat32_v1(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling32 : float
            number which will be multiplied by the basis functions of V1 and written into mat32
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat32[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat33_v1(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V1 is to be written
        
        filling33 : float
            number which will be multiplied by the basis functions of V1 and written into mat33
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat11_v2(pn : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V2 is to be written
        
        filling11 : float
            number which will be multiplied by the basis functions of V2 and written into mat11
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat12_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V2 is to be written
        
        filling12 : float
            number which will be multiplied by the basis functions of V2 and written into mat12
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat13_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat13 : 'float[:,:,:,:,:,:]', filling13 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V2 is to be written
        
        filling13 : float
            number which will be multiplied by the basis functions of V2 and written into mat13
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd1 = pn[0] - 1
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat21_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V2 is to be written
        
        filling21 : float
            number which will be multiplied by the basis functions of V2 and written into mat21
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat21[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat22_v2(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V2 is to be written
        
        filling22 : float
            number which will be multiplied by the basis functions of V2 and written into mat22
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat23_v2(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat23 : array
            matrix in which the filling23 times the basis functions of V2 is to be written
        
        filling23 : float
            number which will be multiplied by the basis functions of V2 and written into mat23
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat31_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat31 : array
            matrix in which the filling31 times the basis functions of V2 is to be written
        
        filling31 : float
            number which will be multiplied by the basis functions of V2 and written into mat31
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd1 = pn[0] - 1
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat31[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat32_v2(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V2 is to be written
        
        filling32 : float
            number which will be multiplied by the basis functions of V2 and written into mat32
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat32[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat33_v2(pn : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V2 is to be written
        
        filling33 : float
            number which will be multiplied by the basis functions of V2 and written into mat33
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                            mat33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat11_vec1_v1(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]

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

                vec1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling11
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat12_vec1_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                vec1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling12
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat13_vec1_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat13 : 'float[:,:,:,:,:,:]', filling13 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling13
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat21_vec2_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                vec2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling21
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat21[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat22_vec2_v1(pn : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                vec2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling22
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat23_vec2_v1(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling23
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat31_vec3_v1(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling31
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat31[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat32_vec3_v1(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling32
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat32[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat33_vec3_v1(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling33
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat11_vec1_v2(pn : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', filling11 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling11
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat12_vec1_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', filling12 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling12
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat13_vec1_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat13 : 'float[:,:,:,:,:,:]', filling13 : 'float', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling13
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat21_vec2_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat21 : 'float[:,:,:,:,:,:]', filling21 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling21
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat21[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat22_vec2_v2(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat22 : 'float[:,:,:,:,:,:]', filling22 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling22
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat23_vec2_v2(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat23 : 'float[:,:,:,:,:,:]', filling23 : 'float', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling23
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat31_vec3_v2(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat31 : 'float[:,:,:,:,:,:]', filling31 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=1 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling31
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat31[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat32_vec3_v2(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat32 : 'float[:,:,:,:,:,:]', filling32 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=2 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling32
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat32[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat33_vec3_v2(pn : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat33 : 'float[:,:,:,:,:,:]', filling33 : 'float', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V2 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
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
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                vec3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling33
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_vec1_v1(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the mu=1 element of a vector in V1 and fills it with basis functions times filling1

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V1 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]

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

                vec1[i1 - starts[0] + pn[0], i2 - starts[1] + pn[1], i3 - starts[2] + pn[2]] += bi3 * filling1


def fill_vec2_v1(pn : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the mu=2 element of a vector in V1 and fills it with basis functions times filling2

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V1 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                vec2[i1 - starts[0] + pn[0], i2 - starts[1] + pn[1], i3 - starts[2] + pn[2]] += bi3 * filling2


def fill_vec3_v1(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the mu=3 element of a vector in V1 and fills it with basis functions times filling3

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V1 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd3 = pn[2] - 1

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

                vec3[i1 - starts[0] + pn[0], i2 - starts[1] + pn[1], i3 - starts[2] + pn[2]] += bi3 * filling3


def fill_vec1_v2(pn : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec1 : 'float[:,:,:]', filling1 : 'float'):
    """
    Computes the mu=1 element of a vector in V2 and fills it with basis functions times filling1

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd2 = pn[1] - 1
    pd3 = pn[2] - 1

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

                vec1[i1 - starts[0] + pn[0], i2 - starts[1] + pn[1], i3 - starts[2] + pn[2]] += bi3 * filling1


def fill_vec2_v2(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec2 : 'float[:,:,:]', filling2 : 'float'):
    """
    Computes the mu=2 element of a vector in V2 and fills it with basis functions times filling2

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd3 = pn[2] - 1

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

                vec2[i1 - starts[0] + pn[0], i2 - starts[1] + pn[1], i3 - starts[2] + pn[2]] += bi3 * filling2


def fill_vec3_v2(pn : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec3 : 'float[:,:,:]', filling3 : 'float'):
    """
    Computes the mu=3 element of a vector in V2 and fills it with basis functions times filling3

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = pn[0] - 1
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                vec3[i1 - starts[0] + pn[0], i2 - starts[1] + pn[1], i3 - starts[2] + pn[2]] += bi3 * filling3


def fill_mat_u0(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts : 'int[:]', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    Computes the entries of the matrix for three-vectors in V0 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat : array
            matrix in which the filling11 times the basis functions of V0 is to be written
        
        filling : float
            number which will be multiplied by the basis functions of V0 and written into mat
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # (NNN NNN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1] * filling
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
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat_u3(pn : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts : 'int[:]', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    Computes the entries of the matrix for three-vectors in V3 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat : array
            matrix in which the filling11 times the basis functions of V3 is to be written
        
        filling : float
            number which will be multiplied by the basis functions of V3 and written into mat
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # (DDD DDD)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling
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
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat_vec_u0(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    Computes the entries of the matrix and of the three-vector in V0 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat : array
            matrix in which the filling_m times the basis functions of V0 is to be written
        
        filling_m : float
            number which will be multiplied by the basis functions of V0 and written into mat
        
        vec : array
            component of the vector in which the filling_v times the basis functions of V0 is to be written
        
        filling_v : float
            number which will be multiplied times the basis functions in V0 and written into vec
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # (NNN NNN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling_v

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling_m
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_mat_vec_u3(pn : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    Computes the entries of the matrix and of the three-vector in V3 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat : array
            matrix in which the filling_m times the basis functions of V3 is to be written
        
        filling_m : float
            number which will be multiplied by the basis functions of V3 and written into mat
        
        vec : array
            component of the vector in which the filling_v times the basis functions of V3 is to be written
        
        filling_v : float
            number which will be multiplied times the basis functions in V3 and written into vec
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # (DDD DDD)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling_v

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling_m
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


def fill_vec_u0(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec : 'float[:,:,:]', filling : 'float'):
    """
    Computes an element of a thre-vector in V0 and fills it with basis functions times filling

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec : array
            component of the vector in which the filling times the basis functions of V0 is to be written
        
        filling : float
            number which will be multiplied times the basis functions in V0 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # (NNN)
    for il1 in range(pn1 + 1):
        i1  = ie1 + il1
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bn3[il3]

                vec[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling


def fill_vec_u3(pn : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', vec : 'float[:,:,:]', filling : 'float'):
    """
    Computes an element of a three-vector in V3 and fills it with basis functions times filling1

    Parameters : 
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices.
        
        pads : array[int]
            Paddings.
        
        vec : array
            component of the vector in which the filling times the basis functions of V3 is to be written
        
        filling : float
            number which will be multiplied times the basis functions in V3 and written into vec
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # (DDD)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = ie2 + il2
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = ie3 + il3
                bi3 = bi2 * bd3[il3]

                vec[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling


def fill_mat11_vec1_v1_pressure(pn : 'int[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', 
                                mat11_11 : 'float[:,:,:,:,:,:]', mat11_12 : 'float[:,:,:,:,:,:]', mat11_13 : 'float[:,:,:,:,:,:]', mat11_22 : 'float[:,:,:,:,:,:]', mat11_23 : 'float[:,:,:,:,:,:]', mat11_33 : 'float[:,:,:,:,:,:]',
                                filling11 : 'float', 
                                vec1_1 : 'float[:,:,:]', vec1_2 : 'float[:,:,:]', vec1_3 : 'float[:,:,:]', 
                                filling1 : 'float',
                                v1 : 'float', v2 : 'float', v3 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=1 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat11_xy : array
            matrix in which the filling11 times the basis functions of V1 with the velocity components v_x and v_y is to be written
        
        filling11 : float
            number which will be multiplied by the basis functions of V1 and written into mat11
        
        vec1_x : array
            component 1 of the vector in which the filling1 times the basis functions of V1 with the velocity component v_x is to be written
        
        filling1 : float
            number which will be multiplied times the basis functions in V1 and written into vec1
        
        v1 : float
            x=1 component of the particle velocity

        v2 : float
            x=2 component of the particle velocity

        v3 : float
            x=3 component of the particle velocity
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]

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

                vec1_1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1 * v1
                vec1_2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1 * v2
                vec1_3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling1 * v3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling11
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat11_11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v1
                            mat11_12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v2
                            mat11_13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v3
                            mat11_22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v2
                            mat11_23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v3
                            mat11_33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v3 * v3


def fill_mat22_vec2_v1_pressure(pn : 'int[:]', bn1 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', 
                                mat22_11 : 'float[:,:,:,:,:,:]', mat22_12 : 'float[:,:,:,:,:,:]', mat22_13 : 'float[:,:,:,:,:,:]', mat22_22 : 'float[:,:,:,:,:,:]', mat22_23 : 'float[:,:,:,:,:,:]', mat22_33 : 'float[:,:,:,:,:,:]', 
                                filling22 : 'float', 
                                vec2_1 : 'float[:,:,:]', vec2_2 : 'float[:,:,:]', vec2_3 : 'float[:,:,:]',
                                filling2 : 'float',
                                v1 : 'float', v2 : 'float', v3 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat22_xy : array
            matrix in which the filling22 times the basis functions of V1 with the velocity components v_x and v_y is to be written
        
        filling22 : float
            number which will be multiplied by the basis functions of V1 and written into mat22
        
        vec2_x : array
            component 2 of the vector in which the filling1 times the basis functions of V1 with the velocity component v_x is to be written
        
        filling2 : float
            number which will be multiplied times the basis functions in V1 and written into vec2
    
        v1 : float
            x=1 component of the particle velocity

        v2 : float
            x=2 component of the particle velocity

        v3 : float
            x=3 component of the particle velocity
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                vec2_1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2 * v1
                vec2_2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2 * v2
                vec2_3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling2 * v3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling22
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat22_11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v1
                            mat22_12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v2
                            mat22_13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v3
                            mat22_22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v2
                            mat22_23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v3
                            mat22_33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v3 * v3


def fill_mat33_vec3_v1_pressure(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', 
                                mat33_11 : 'float[:,:,:,:,:,:]', mat33_12 : 'float[:,:,:,:,:,:]', mat33_13 : 'float[:,:,:,:,:,:]', mat33_22 : 'float[:,:,:,:,:,:]', mat33_23 : 'float[:,:,:,:,:,:]', mat33_33 : 'float[:,:,:,:,:,:]',
                                filling33 : 'float', 
                                vec3_1 : 'float[:,:,:]', vec3_2 : 'float[:,:,:]', vec3_3 : 'float[:,:,:]',
                                filling3 : 'float',
                                v1 : 'float', v2 : 'float', v3 : 'float'):
    """
    Computes the entries of the matrix mu=3,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
            contains 3 values of the degrees of the B-splines in each direction

        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        ie1, ie2, ie3 : int
            particle's element index in each direction
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat23_xy : array
            matrix in which the filling33 times the basis functions of V1 with the velocity components v_x and v_y is to be written
        
        filling33 : float
            number which will be multiplied by the basis functions of V1 and written into mat33
        
        vec3_x : array
            component 3 of the vector in which the filling1 times the basis functions of V1 with the velocity component v_x is to be written
        
        filling3 : float
            number which will be multiplied times the basis functions in V1 and written into vec3
    
        v1 : float
            x=1 component of the particle velocity

        v2 : float
            x=2 component of the particle velocity

        v3 : float
            x=3 component of the particle velocity
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                vec3_1[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3 * v1
                vec3_2[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3 * v2
                vec3_3[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3] += bi3 * filling3 * v3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling33
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat33_11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v1
                            mat33_12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v2
                            mat33_13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v3
                            mat33_22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v2
                            mat33_23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v3
                            mat33_33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v3 * v3


def fill_mat12_v1_pressure(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', 
                           mat12_11 : 'float[:,:,:,:,:,:]', mat12_12 : 'float[:,:,:,:,:,:]', mat12_13 : 'float[:,:,:,:,:,:]', mat12_22 : 'float[:,:,:,:,:,:]', mat12_23 : 'float[:,:,:,:,:,:]', mat12_33 : 'float[:,:,:,:,:,:]',
                           filling12 : 'float',
                           v1 : 'float', v2 : 'float', v3 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=2 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat12_xy : array
            matrix in which the filling12 times the basis functions of V1 with the velocity components v_x and v_y is to be written
        
        filling12 : float
            number which will be multiplied by the basis functions of V1 and written into mat12

        v1 : float
            x=1 component of the particle velocity

        v2 : float
            x=2 component of the particle velocity

        v3 : float
            x=3 component of the particle velocity
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]

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

                            mat12_11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v1
                            mat12_12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v2 
                            mat12_13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v3
                            mat12_22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v2
                            mat12_23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v3
                            mat12_33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v3 * v3


def fill_mat13_v1_pressure(pn : 'int[:]', bn1 : 'float[:]', bd1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]',
                           mat13_11 : 'float[:,:,:,:,:,:]', mat13_12 : 'float[:,:,:,:,:,:]', mat13_13 : 'float[:,:,:,:,:,:]', mat13_22 : 'float[:,:,:,:,:,:]', mat13_23 : 'float[:,:,:,:,:,:]', mat13_33 : 'float[:,:,:,:,:,:]',
                           filling13 : 'float',
                           v1 : 'float', v2 : 'float', v3 : 'float'):
    """
    Computes the entries of the matrix mu=1,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat13_xy : array
            matrix in which the filling13 times the basis functions of V1 with the velocity components v_x and v_y is to be written
        
        filling13 : float
            number which will be multiplied by the basis functions of V1 and written into mat13
    
        v1 : float
            x=1 component of the particle velocity

        v2 : float
            x=2 component of the particle velocity

        v3 : float
            x=3 component of the particle velocity
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pd1 = pn[0] - 1
    pn2 = pn[1]
    pn3 = pn[2]
    pd3 = pn[2] - 1

    # (DNN NND)
    for il1 in range(pd1 + 1):
        i1  = ie1 + il1
        bi1 = bd1[il1] * filling13
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

                            mat13_11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v1
                            mat13_12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v2
                            mat13_13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v3
                            mat13_22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v2
                            mat13_23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v3
                            mat13_33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v3 * v3


def fill_mat23_v1_pressure(pn : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bd2 : 'float[:]', bn3 : 'float[:]', bd3 : 'float[:]', ie1 : 'int', ie2 : 'int', ie3 : 'int', starts: 'int[:]', 
                           mat23_11 : 'float[:,:,:,:,:,:]', mat23_12 : 'float[:,:,:,:,:,:]', mat23_13 : 'float[:,:,:,:,:,:]', mat23_22 : 'float[:,:,:,:,:,:]', mat23_23 : 'float[:,:,:,:,:,:]', mat23_33 : 'float[:,:,:,:,:,:]',
                           filling23 : 'float',
                           v1 : 'float', v2 : 'float', v3 : 'float'):
    """
    Computes the entries of the matrix mu=2,nu=3 in V1 and fills it with basis functions times filling

    Parameters :
    ------------
        pn : array of integers
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
        
        starts : array[int]
            Start indices of the codomain (row indices).
        
        pads : array[int]
            Paddings of the codomain (row indices).
        
        mat23_xy : array
            matrix in which the filling23 times the basis functions of V1 with the velocity components v_x and v_y is to be written
        
        filling23 : float
            number which will be multiplied by the basis functions of V1 and written into mat23
    
        v1 : float
            x=1 component of the particle velocity

        v2 : float
            x=2 component of the particle velocity

        v3 : float
            x=3 component of the particle velocity
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = pn[0]
    pn2 = pn[1]
    pd2 = pn[1] - 1
    pn3 = pn[2]
    pd3 = pn[2] - 1

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

                            mat23_11[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v1
                            mat23_12[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v2
                            mat23_13[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v1 * v3
                            mat23_22[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v2
                            mat23_23[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v2 * v3
                            mat23_33[i1 - starts[0] + pn1, i2 - starts[1] + pn2, i3 - starts[2] + pn3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * v3 * v3