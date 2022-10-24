import pytest
import numpy as np

from struphy.polar.basic import PolarVector

@pytest.mark.parametrize('Nel', [[8, 5, 6], [5, 4, 32]])
@pytest.mark.parametrize('p', [[3, 2, 1], [4, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
def test_spaces(Nel, p, spl_kind):

    from struphy.psydac_api.psydac_derham import Derham
    from struphy.polar.basic import PolarDerhamSpace 

    derham = Derham(Nel, p, spl_kind)

    print('polar V0:')
    V = PolarDerhamSpace(derham, 'H1')
    print('dimensions (parent, polar):', derham.V0.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.tp[:] = 1.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V1:')
    V = PolarDerhamSpace(derham, 'Hcurl')
    print('dimensions (parent, polar):', derham.V1.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.pol[1][:] = 2.
    a.pol[2][:] = 3.
    a.tp[0][:] = 1.
    a.tp[1][:] = 2.
    a.tp[2][:] = 3.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V2:')
    V = PolarDerhamSpace(derham, 'Hdiv')
    print('dimensions (parent, polar):', derham.V2.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.pol[1][:] = 2.
    a.pol[2][:] = 3.
    a.tp[0][:] = 1.
    a.tp[1][:] = 2.
    a.tp[2][:] = 3.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V3:')
    V = PolarDerhamSpace(derham, 'L2')
    print('dimensions (parent, polar):', derham.V3.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.tp[:] = 1.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V0vec:')
    V = PolarDerhamSpace(derham, 'H1vec')
    print('dimensions (parent, polar):', derham.V0vec.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.pol[1][:] = 2.
    a.pol[2][:] = 3.
    a.tp[0][:] = 1.
    a.tp[1][:] = 2.
    a.tp[2][:] = 3.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

@pytest.mark.parametrize('Nel', [[8, 9, 6], [5, 6, 16]])
@pytest.mark.parametrize('p', [[3, 2, 1], [4, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
def test_extraction_ops(Nel, p, spl_kind):

    from scipy.sparse import csr_matrix
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.geometry.domains import PoloidalSplineCylinder
    from struphy.polar.basic import PolarDerhamSpace, PolarExtractionOperator, set_tp_rings_to_zero
    from struphy.polar.extraction_operators import PolarExtractionBlocksC1
    from struphy.geometry.domains import Cuboid
    from struphy.psydac_api.mass import WeightedMass
    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    derham = Derham(Nel, p, spl_kind, comm=comm)

    params_map = {'Nel' : Nel[:2], 'p' : p[:2], 'spl_kind' : spl_kind[:2], 'a' : 1., 'R0' : 3.}
    domain = PoloidalSplineCylinder(params_map)
    c1_blocks = PolarExtractionBlocksC1(domain.cx[:, :, 0], domain.cy[:, :, 0])

    #########################################################
    print('polar V0:')
    V = PolarDerhamSpace(derham, 'H1')
    print('dimensions (parent, polar):', derham.V0.nbasis, V.dimension)
    
    domain = derham.V0.vector_space
    
    E = PolarExtractionOperator(domain, V)
    print(E.pol_blocks, E.pol_blocks_shapes)
    E.pol_blocks = [[np.ones(E.pol_blocks_shapes[0][0], dtype=float)]]
    print(E.pol_blocks)

    print(E.tp_blocks, E.tp_blocks_shapes)
    data = np.ones(E.tp_blocks_shapes[0][0]).flatten()
    indices = np.indices(E.tp_blocks_shapes[0][0])
    row = indices[0].flatten()
    col = indices[1].flatten()
    E.tp_blocks = [[csr_matrix((data, (row, col)), shape=E.tp_blocks_shapes[0][0])]]
    print(E.tp_blocks[0][0].toarray())

    w = StencilVector(domain)
    w[:] = 1.

    out = E.dot(w)
    print(out.toarray_tp())

    ET = E.transpose()
    w2 = PolarVector(V)
    w2.pol[0][:] = 1.
    w2.tp = w
    print(w2.toarray_tp())
    
    out = ET.dot(w2)
    print(out.toarray())

    #########################################################
    print('polar V1:')
    V = PolarDerhamSpace(derham, 'Hcurl')
    print('dimensions (parent, polar):', derham.V1.nbasis, V.dimension)
    
    domain = derham.V1.vector_space
    
    E = PolarExtractionOperator(domain, V)
    print(E.pol_blocks, E.pol_blocks_shapes)
    temp = []
    for n in range(3):
        temp += [[]]
        for m in range(3):
            if (n == 0 or n == 1) and m == 2:
                temp[-1] += [None]
            elif n == 2 and (m == 0 or m == 1):
                temp[-1] += [None]
            else:
                temp[-1] += [np.ones(E.pol_blocks_shapes[n][m])]
    E.pol_blocks = temp 
    print(E.pol_blocks)

    print(E.tp_blocks, E.tp_blocks_shapes)
    temp = []
    for n in range(3):
        temp += [[]]
        for m in range(3):
            if (n == 0 or n == 1) and m == 2:
                temp[-1] += [None]
            elif n == 2 and (m == 0 or m == 1):
                temp[-1] += [None]
            else:
                data = np.ones(E.tp_blocks_shapes[n][m]).flatten()
                indices = np.indices(E.tp_blocks_shapes[n][m])
                row = indices[0].flatten()
                col = indices[1].flatten()
                temp[-1] += [csr_matrix((data, (row, col)), shape=E.tp_blocks_shapes[n][m])]
    E.tp_blocks = temp 
    print(E.tp_blocks)

    w = BlockVector(domain)
    w[0][:] = 1.
    w[1][:] = 2.
    w[2][:] = 3.

    out = E.dot(w)
    print(out.toarray_tp())

    ET = E.transpose()
    w2 = PolarVector(V)
    w2.pol[0][:] = 1.
    w2.pol[1][:] = 1.
    w2.pol[2][:] = 1.
    w2.tp = w
    
    out = ET.dot(w2)
    print(out.toarray())
    
    #########################################################
    print('polar V2:')
    V = PolarDerhamSpace(derham, 'Hdiv')
    print('dimensions (parent, polar):', derham.V2.nbasis, V.dimension)
    
    domain = derham.V2.vector_space
    
    E = PolarExtractionOperator(domain, V)
    print(E.pol_blocks, E.pol_blocks_shapes)
    temp = []
    for n in range(3):
        temp += [[]]
        for m in range(3):
            if (n == 0 or n == 1) and m == 2:
                temp[-1] += [None]
            elif n == 2 and (m == 0 or m == 1):
                temp[-1] += [None]
            else:
                temp[-1] += [np.ones(E.pol_blocks_shapes[n][m])]
    E.pol_blocks = temp 
    print(E.pol_blocks)

    print(E.tp_blocks, E.tp_blocks_shapes)
    temp = []
    for n in range(3):
        temp += [[]]
        for m in range(3):
            if (n == 0 or n == 1) and m == 2:
                temp[-1] += [None]
            elif n == 2 and (m == 0 or m == 1):
                temp[-1] += [None]
            else:
                data = np.ones(E.tp_blocks_shapes[n][m]).flatten()
                indices = np.indices(E.tp_blocks_shapes[n][m])
                row = indices[0].flatten()
                col = indices[1].flatten()
                temp[-1] += [csr_matrix((data, (row, col)), shape=E.tp_blocks_shapes[n][m])]
    E.tp_blocks = temp 
    print(E.tp_blocks)

    w = BlockVector(domain)
    w[0][:] = 1.
    w[1][:] = 2.
    w[2][:] = 3.

    out = E.dot(w)
    print(out.toarray_tp())

    ET = E.transpose()
    w2 = PolarVector(V)
    w2.pol[0][:] = 1.
    w2.pol[1][:] = 1.
    w2.pol[2][:] = 1.
    w2.tp = w
    
    out = ET.dot(w2)
    print(out.toarray())
    
    #########################################################
    print('polar V3:')
    V = PolarDerhamSpace(derham, 'L2')
    print('dimensions (parent, polar):', derham.V3.nbasis, V.dimension)
    
    domain = derham.V3.vector_space
    
    E = PolarExtractionOperator(domain, V)
    print(E.pol_blocks, E.pol_blocks_shapes)
    E.pol_blocks = [[np.ones(E.pol_blocks_shapes[0][0], dtype=float)]]
    print(E.pol_blocks)

    print(E.tp_blocks, E.tp_blocks_shapes)
    data = np.ones(E.tp_blocks_shapes[0][0]).flatten()
    indices = np.indices(E.tp_blocks_shapes[0][0])
    row = indices[0].flatten()
    col = indices[1].flatten()
    E.tp_blocks = [[csr_matrix((data, (row, col)), shape=E.tp_blocks_shapes[0][0])]]
    print(E.tp_blocks[0][0].toarray())

    w = StencilVector(domain)
    w[:] = 1.

    out = E.dot(w)
    print(out.toarray_tp())

    ET = E.transpose()
    w2 = PolarVector(V)
    w2.pol[0][:] = 1.
    w2.tp = w
    print(w2.toarray_tp())
    
    out = ET.dot(w2)
    print(out.toarray())
    
    #########################################################
    print('polar V0vec:')
    V = PolarDerhamSpace(derham, 'H1vec')
    print('dimensions (parent, polar):', derham.V0vec.nbasis, V.dimension)
    
    domain = derham.V0vec.vector_space
    
    E = PolarExtractionOperator(domain, V)
    print(E.pol_blocks, E.pol_blocks_shapes)
    temp = []
    for n in range(3):
        temp += [[]]
        for m in range(3):
            temp[-1] += [np.ones(E.pol_blocks_shapes[n][m])]
    E.pol_blocks = temp 
    print(E.pol_blocks)

    print(E.tp_blocks, E.tp_blocks_shapes)
    temp = []
    for n in range(3):
        temp += [[]]
        for m in range(3):
            data = np.ones(E.tp_blocks_shapes[n][m]).flatten()
            indices = np.indices(E.tp_blocks_shapes[n][m])
            row = indices[0].flatten()
            col = indices[1].flatten()
            temp[-1] += [csr_matrix((data, (row, col)), shape=E.tp_blocks_shapes[n][m])]
    E.tp_blocks = temp 
    print(E.tp_blocks)

    w = BlockVector(domain)
    w[0][:] = 1.
    w[1][:] = 2.
    w[2][:] = 3.

    out = E.dot(w)
    print(out.toarray_tp())

    ET = E.transpose()
    w2 = PolarVector(V)
    w2.pol[0][:] = 1.
    w2.pol[1][:] = 1.
    w2.pol[2][:] = 1.
    w2.tp = w
    
    out = ET.dot(w2)
    print(out.toarray())



    
if __name__ == '__main__':
    #test_spaces([6, 5, 4], [2, 2, 2], [False, True, True])
    test_extraction_ops([6, 9, 4], [2, 2, 2], [False, True, True])