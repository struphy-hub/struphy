"""
pyccel functions to accumulate the mu,nu-elements of a matrix and the mu-indices of a matrix vector product in accumulation step.
"""

# =====================================================================================================
def fill_mat11_v1(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indn3 : 'int[:]', mat11 : 'double[:,:,:,:,:,:]', filling11 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V1 is to be written
        
        filling11 : double
            number which will be multiplied by the basis functions of V1 and written into mat11
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]

    # (DNN DNN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling11
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat11[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indn3 : 'int[:]', mat12 : 'double[:,:,:,:,:,:]', filling12 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V1 is to be written
        
        filling12 : double
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
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling12
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat12[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indn3 : 'int[:]', mat12 : 'double[:,:,:,:,:,:]', filling12 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling13 : double
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
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling12
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat21 : 'double[:,:,:,:,:,:]', filling21 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V1 is to be written
        
        filling21 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling21
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat21[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_v1(p : 'int[:]', bn1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat22 : 'double[:,:,:,:,:,:]', filling22 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V1 is to be written
        
        filling22 : double
            number which will be multiplied by the basis functions of V1 and written into mat22
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN NDN)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling22
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat22[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_v1(p : 'int[:]', bn1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat23 : 'double[:,:,:,:,:,:]', filling23 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat23 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling23 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling23
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat31 : 'double[:,:,:,:,:,:]', filling31 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat31 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling31 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling31
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat31[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_v1(p : 'int[:]', bn1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat32 : 'double[:,:,:,:,:,:]', filling32 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling32 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling32
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat32[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_v1(p : 'int[:]', bn1 : 'double[:]', bn2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indn2 : 'int[:]',   indd3 : 'int[:]', mat33 : 'double[:,:,:,:,:,:]', filling33 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V1 is to be written
        
        filling33 : double
            number which will be multiplied by the basis functions of V1 and written into mat33
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd3 = p[2] - 1

    # (NND NND)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling33
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat33[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat11_v2(p : 'int[:]', bn1 : 'double[:]', bd2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indd3 : 'int[:]', mat11 : 'double[:,:,:,:,:,:]', filling11 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V2 is to be written
        
        filling11 : double
            number which will be multiplied by the basis functions of V2 and written into mat11
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (NDD NDD)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling11
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat11[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indd3 : 'int[:]', mat12 : 'double[:,:,:,:,:,:]', filling12 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V2 is to be written
        
        filling12 : double
            number which will be multiplied by the basis functions of V2 and written into mat12
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (NDD DND)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling12
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat12[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indd3 : 'int[:]', mat13 : 'double[:,:,:,:,:,:]', filling13 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V2 is to be written
        
        filling13 : double
            number which will be multiplied by the basis functions of V2 and written into mat13
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2] - 1
    pd3 = p[2] - 1

    # (NDD DDN)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1] * filling13
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat13[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat21 : 'double[:,:,:,:,:,:]', filling21 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V2 is to be written
        
        filling21 : double
            number which will be multiplied by the basis functions of V2 and written into mat21
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (DND NDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling21
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat21[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_v2(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat22 : 'double[:,:,:,:,:,:]', filling22 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V2 is to be written
        
        filling22 : double
            number which will be multiplied by the basis functions of V2 and written into mat22
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd3 = p[2] - 1

    # (DND DND)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling22
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat22[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_v2(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat23 : 'double[:,:,:,:,:,:]', filling23 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat23 : array
            matrix in which the filling23 times the basis functions of V2 is to be written
        
        filling23 : double
            number which will be multiplied by the basis functions of V2 and written into mat23
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND DDN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling23
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat23[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat31 : 'double[:,:,:,:,:,:]', filling31 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat31 : array
            matrix in which the filling31 times the basis functions of V2 is to be written
        
        filling31 : double
            number which will be multiplied by the basis functions of V2 and written into mat31
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2] - 1
    pd3 = p[2] - 1

    # (DDN NDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling31
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat31[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_v2(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat32 : 'double[:,:,:,:,:,:]', filling32 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V2 is to be written
        
        filling32 : double
            number which will be multiplied by the basis functions of V2 and written into mat32
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DDN DND)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling32
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat32[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_v2(p : 'int[:]', bd1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat33 : 'double[:,:,:,:,:,:]', filling33 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V2 is to be written
        
        filling33 : double
            number which will be multiplied by the basis functions of V2 and written into mat33
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DDN DDN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1] * filling33
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat33[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat11_vec1_v1(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indn3 : 'int[:]', mat11 : 'double[:,:,:,:,:,:]', filling11 : 'double', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V1 is to be written
        
        filling11 : double
            number which will be multiplied by the basis functions of V1 and written into mat11
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : double
            number which will be multiplied times the basis functions in V1 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]

    # (DNN DNN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling11
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat11[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_vec1_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indn3 : 'int[:]', mat12 : 'double[:,:,:,:,:,:]', filling12 : 'double', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V1 is to be written
        
        filling12 : double
            number which will be multiplied by the basis functions of V1 and written into mat12
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : double
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
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling12
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat12[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_vec1_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indn3 : 'int[:]', mat13 : 'double[:,:,:,:,:,:]', filling13 : 'double', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V1 is to be written
        
        filling13 : double
            number which will be multiplied by the basis functions of V1 and written into mat13
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : double
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
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling13
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_vec2_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat21 : 'double[:,:,:,:,:,:]', filling21 : 'double', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V1 is to be written
        
        filling21 : double
            number which will be multiplied by the basis functions of V1 and written into mat21
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling21
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat21[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_vec2_v1(p : 'int[:]', bn1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat22 : 'double[:,:,:,:,:,:]', filling22 : 'double', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V1 is to be written
        
        filling22 : double
            number which will be multiplied by the basis functions of V1 and written into mat22
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : double
            number which will be multiplied times the basis functions in V1 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN NDN)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling22
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat22[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_vec2_v1(p : 'int[:]', bn1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat23 : 'double[:,:,:,:,:,:]', filling23 : 'double', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat23 : array
            matrix in which the filling23 times the basis functions of V1 is to be written
        
        filling23 : double
            number which will be multiplied by the basis functions of V1 and written into mat23
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling23
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_vec3_v1(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat31 : 'double[:,:,:,:,:,:]', filling31 : 'double', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat31 : array
            matrix in which the filling31 times the basis functions of V1 is to be written
        
        filling31 : double
            number which will be multiplied by the basis functions of V1 and written into mat31
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling31
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat31[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_vec3_v1(p : 'int[:]', bn1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat32 : 'double[:,:,:,:,:,:]', filling32 : 'double', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V1 is to be written
        
        filling32 : double
            number which will be multiplied by the basis functions of V1 and written into mat32
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : double
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
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling32
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat32[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_vec3_v1(p : 'int[:]', bn1 : 'double[:]', bn2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat33 : 'double[:,:,:,:,:,:]', filling33 : 'double', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V1 is to be written
        
        filling33 : double
            number which will be multiplied by the basis functions of V1 and written into mat33
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : double
            number which will be multiplied times the basis functions in V1 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd3 = p[2] - 1

    # (NND NND)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling33
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat33[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat11_vec1_v2(p : 'int[:]', bn1 : 'double[:]', bd2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indd3 : 'int[:]', mat11 : 'double[:,:,:,:,:,:]', filling11 : 'double', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat11 : array
            matrix in which the filling11 times the basis functions of V2 is to be written
        
        filling11 : double
            number which will be multiplied by the basis functions of V2 and written into mat11
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : double
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (NDD NDD)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling11
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat11[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat12_vec1_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indd3 : 'int[:]', mat12 : 'double[:,:,:,:,:,:]', filling12 : 'double', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat12 : array
            matrix in which the filling12 times the basis functions of V2 is to be written
        
        filling12 : double
            number which will be multiplied by the basis functions of V2 and written into mat12
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : double
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (NDD DND)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling12
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat12[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat13_vec1_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indd3 : 'int[:]', mat13 : 'double[:,:,:,:,:,:]', filling13 : 'double', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat13 : array
            matrix in which the filling13 times the basis functions of V2 is to be written
        
        filling13 : double
            number which will be multiplied by the basis functions of V2 and written into mat13
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : double
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2] - 1
    pd3 = p[2] - 1

    # (NDD DDN)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling13
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat13[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat21_vec2_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat21 : 'double[:,:,:,:,:,:]', filling21 : 'double', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat21 : array
            matrix in which the filling21 times the basis functions of V2 is to be written
        
        filling21 : double
            number which will be multiplied by the basis functions of V2 and written into mat21
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : double
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (DND NDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling21
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat21[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat22_vec2_v2(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat22 : 'double[:,:,:,:,:,:]', filling22 : 'double', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat22 : array
            matrix in which the filling22 times the basis functions of V2 is to be written
        
        filling22 : double
            number which will be multiplied by the basis functions of V2 and written into mat22
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : double
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd3 = p[2] - 1

    # (DND DND)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling22
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat22[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat23_vec2_v2(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', mat23 : 'double[:,:,:,:,:,:]', filling23 : 'double', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        mat23 : array
            matrix in which the filling23 times the basis functions of V2 is to be written
        
        filling23 : double
            number which will be multiplied by the basis functions of V2 and written into mat23
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : double
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DND DDN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling23
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat23[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat31_vec3_v2(p : 'int[:]', bn1 : 'double[:]', bd1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat31 : 'double[:,:,:,:,:,:]', filling31 : 'double', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat31 : array
            matrix in which the filling31 times the basis functions of V2 is to be written
        
        filling31 : double
            number which will be multiplied by the basis functions of V2 and written into mat31
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : double
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2] - 1
    pd3 = p[2] - 1

    # (DDN NDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling31
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat31[i1, i2, i3, pn1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat32_vec3_v2(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat32 : 'double[:,:,:,:,:,:]', filling32 : 'double', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat32 : array
            matrix in which the filling32 times the basis functions of V2 is to be written
        
        filling32 : double
            number which will be multiplied by the basis functions of V2 and written into mat32
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : double
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd2 = p[1] - 1
    pn3 = p[2]
    pd3 = p[2] - 1

    # (DDN DND)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling32
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat32[i1, i2, i3, pd1 + jl1 - il1, pn2 + jl2 - il2, pd3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_mat33_vec3_v2(p : 'int[:]', bd1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', mat33 : 'double[:,:,:,:,:,:]', filling33 : 'double', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        mat33 : array
            matrix in which the filling33 times the basis functions of V2 is to be written
        
        filling33 : double
            number which will be multiplied by the basis functions of V2 and written into mat33
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : double
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DDN DDN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling33
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat33[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pn3 + jl3 - il3] += bj3

# =====================================================================================================
def fill_vec1_v1(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indn3 : 'int[:]', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V1 is to be written
        
        filling1 : double
            number which will be multiplied times the basis functions in V1 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pn3 = p[2]

    # (DNN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

# =====================================================================================================
def fill_vec2_v1(p : 'int[:]', bn1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V1 is to be written
        
        filling2 : double
            number which will be multiplied times the basis functions in V1 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pn3 = p[2]

    # (NDN)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

# =====================================================================================================
def fill_vec3_v1(p : 'int[:]', bn1 : 'double[:]', bn2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V1 is to be written
        
        filling3 : double
            number which will be multiplied times the basis functions in V1 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pn2 = p[1]
    pd3 = p[2] - 1

    # (NND)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

# =====================================================================================================
def fill_vec1_v2(p : 'int[:]', bn1 : 'double[:]', bd2 : 'double[:]', bd3 : 'double[:]', indn1 : 'int[:]', indd2 : 'int[:]', indd3 : 'int[:]', vec1 : 'double[:,:,:]', filling1 : 'double'):
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
        
        indn1 : array of integers
            contains the global indices of non-vanishing B-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        vec1 : array
            component 1 of the vector in which the filling1 times the basis functions of V2 is to be written
        
        filling1 : double
            number which will be multiplied times the basis functions in V2 and written into vec1
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pn1 = p[0]
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # (NDD)
    for il1 in range(pn1 + 1):
        i1  = indn1[il1]
        bi1 = bn1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec1[i1, i2, i3] += bi3 * filling1

# =====================================================================================================
def fill_vec2_v2(p : 'int[:]', bd1 : 'double[:]', bn2 : 'double[:]', bd3 : 'double[:]', indd1 : 'int[:]', indn2 : 'int[:]', indd3 : 'int[:]', vec2 : 'double[:,:,:]', filling2 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indn2 : array of integers
            contains the global indices of non-vanishing B-splines in direction 2
        
        indd3 : array of integers
            contains the global indices of non-vanishing D-splines in direction 3
        
        vec2 : array
            component 2 of the vector in which the filling2 times the basis functions of V2 is to be written
        
        filling2 : double
            number which will be multiplied times the basis functions in V2 and written into vec2
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pn2 = p[1]
    pd3 = p[2] - 1

    # (DND)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[il3]
                bi3 = bi2 * bd3[il3]

                vec2[i1, i2, i3] += bi3 * filling2

# =====================================================================================================
def fill_vec3_v2(p : 'int[:]', bd1 : 'double[:]', bd2 : 'double[:]', bn3 : 'double[:]', indd1 : 'int[:]', indd2 : 'int[:]', indn3 : 'int[:]', vec3 : 'double[:,:,:]', filling3 : 'double'):
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
        
        indd1 : array of integers
            contains the global indices of non-vanishing D-splines in direction 1
        
        indd2 : array of integers
            contains the global indices of non-vanishing D-splines in direction 2
        
        indn3 : array of integers
            contains the global indices of non-vanishing B-splines in direction 3
        
        vec3 : array
            component 3 of the vector in which the filling3 times the basis functions of V2 is to be written
        
        filling3 : double
            number which will be multiplied times the basis functions in V2 and written into vec3
    """

    # total number of basis functions : B-splines (pn) and D-splines(pd), only the needed ones are being computed
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pn3 = p[2]

    # (DDN)
    for il1 in range(pd1 + 1):
        i1  = indd1[il1]
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[il3]
                bi3 = bi2 * bn3[il3]

                vec3[i1, i2, i3] += bi3 * filling3

