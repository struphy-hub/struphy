import pytest


@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['orthogonal', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}],
    ['colella', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}],
    ['hollow_cyl', {
        'a1': 1., 'a2': 2., 'R0': 3.}],
    ['hollow_torus', {
        'a1': 1., 'a2': 2., 'R0': 3.}],
    ['ellipse', {
        'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6.}],
    ['rotated_ellipse', {
        'x0': 1., 'y0': 2., 'z0': 3., 'r1': 4., 'r2': 5., 'Lz': 6., 'th': 7.}],
    ['shafranov_shift', {
        'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6., 'delta': 7.}],
    ['shafranov_sqrt', {
        'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6., 'delta': 7.}],
    ['shafranov_dshaped', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
def test_psydac_mapping(mapping):

    from struphy.geometry.domain_3d import Domain

    import numpy as np

    print('\n===== test_psydac_mapping() =====')

    ########################
    ### TEST EVALUATIONS ###
    ########################
    map = mapping[0]
    params = mapping[1]

    print(map)
    print(params)

    # Struphy domain object
    DOMAIN = Domain(map, params)
    Mapping_psydac = DOMAIN.Psydac_mapping('F', **params)

    print(Mapping_psydac._expressions, '\n')

    # Psydac mapping
    F_PSY = Mapping_psydac.get_callable_mapping()

    # Comparisons at random logical point
    eta = np.random.rand(3)

    # Mapping
    assert np.allclose(F_PSY(*eta)[0], DOMAIN.evaluate(*eta, 'x'))
    assert np.allclose(F_PSY(*eta)[1], DOMAIN.evaluate(*eta, 'y'))
    assert np.allclose(F_PSY(*eta)[2], DOMAIN.evaluate(*eta, 'z'))

    # Absolute value of Jacobian determinant
    assert np.allclose(np.sqrt(F_PSY.metric_det(*eta)),
                       np.abs(DOMAIN.evaluate(*eta, 'det_df')))

    # Jacobian
    assert np.allclose(F_PSY.jacobian(
        *eta)[0, 0], DOMAIN.evaluate(*eta, 'df_11'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[0, 1], DOMAIN.evaluate(*eta, 'df_12'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[0, 2], DOMAIN.evaluate(*eta, 'df_13'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[1, 0], DOMAIN.evaluate(*eta, 'df_21'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[1, 1], DOMAIN.evaluate(*eta, 'df_22'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[1, 2], DOMAIN.evaluate(*eta, 'df_23'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[2, 0], DOMAIN.evaluate(*eta, 'df_31'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[2, 1], DOMAIN.evaluate(*eta, 'df_32'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[2, 2], DOMAIN.evaluate(*eta, 'df_33'))

    # Inverse Jacobian
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[0, 0], DOMAIN.evaluate(*eta, 'df_inv_11'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[0, 1], DOMAIN.evaluate(*eta, 'df_inv_12'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[0, 2], DOMAIN.evaluate(*eta, 'df_inv_13'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[1, 0], DOMAIN.evaluate(*eta, 'df_inv_21'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[1, 1], DOMAIN.evaluate(*eta, 'df_inv_22'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[1, 2], DOMAIN.evaluate(*eta, 'df_inv_23'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[2, 0], DOMAIN.evaluate(*eta, 'df_inv_31'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[2, 1], DOMAIN.evaluate(*eta, 'df_inv_32'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[2, 2], DOMAIN.evaluate(*eta, 'df_inv_33'))

    # Metric tensor
    assert np.allclose(F_PSY.metric(*eta)[0, 0], DOMAIN.evaluate(*eta, 'g_11'))
    assert np.allclose(F_PSY.metric(*eta)[0, 1], DOMAIN.evaluate(*eta, 'g_12'))
    assert np.allclose(F_PSY.metric(*eta)[0, 2], DOMAIN.evaluate(*eta, 'g_13'))
    assert np.allclose(F_PSY.metric(*eta)[1, 0], DOMAIN.evaluate(*eta, 'g_21'))
    assert np.allclose(F_PSY.metric(*eta)[1, 1], DOMAIN.evaluate(*eta, 'g_22'))
    assert np.allclose(F_PSY.metric(*eta)[1, 2], DOMAIN.evaluate(*eta, 'g_23'))
    assert np.allclose(F_PSY.metric(*eta)[2, 0], DOMAIN.evaluate(*eta, 'g_31'))
    assert np.allclose(F_PSY.metric(*eta)[2, 1], DOMAIN.evaluate(*eta, 'g_32'))
    assert np.allclose(F_PSY.metric(*eta)[2, 2], DOMAIN.evaluate(*eta, 'g_33'))

    # Inverse metric tensor
    metric_inv_PSY = np.matmul(F_PSY.jacobian_inv(
        *eta), F_PSY.jacobian_inv(*eta).T)  # missing in psydac
    assert np.allclose(metric_inv_PSY[0, 0], DOMAIN.evaluate(*eta, 'g_inv_11'))
    assert np.allclose(metric_inv_PSY[0, 1], DOMAIN.evaluate(*eta, 'g_inv_12'))
    assert np.allclose(metric_inv_PSY[0, 2], DOMAIN.evaluate(*eta, 'g_inv_13'))
    assert np.allclose(metric_inv_PSY[1, 0], DOMAIN.evaluate(*eta, 'g_inv_21'))
    assert np.allclose(metric_inv_PSY[1, 1], DOMAIN.evaluate(*eta, 'g_inv_22'))
    assert np.allclose(metric_inv_PSY[1, 2], DOMAIN.evaluate(*eta, 'g_inv_23'))
    assert np.allclose(metric_inv_PSY[2, 0], DOMAIN.evaluate(*eta, 'g_inv_31'))
    assert np.allclose(metric_inv_PSY[2, 1], DOMAIN.evaluate(*eta, 'g_inv_32'))
    assert np.allclose(metric_inv_PSY[2, 2], DOMAIN.evaluate(*eta, 'g_inv_33'))

    print(map + ' done.\n')


@pytest.mark.parametrize('Nel', [[6, 6, 4]])
@pytest.mark.parametrize('p', [[3, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['shafranov_dshaped', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
def test_psydac_derham(Nel, p, spl_kind, mapping):
    '''Requires pyccel==1.4.0, install manually (pyccel=0.10.1 installed by struphy).

    Remark: p=even projectors yield slightly different results, pass with atol=1e-3.'''

    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import Derham_build
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space

    from psydac.fem.tensor import TensorFemSpace
    from psydac.fem.vector import ProductFemSpace
    from psydac.fem.basic import FemField
    from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
    from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix

    from mpi4py import MPI
    import numpy as np

    print('\n===== test_psydac_FemSpace() =====')

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # Domain object
    map = mapping[0]
    params_map = mapping[1]

    DOMAIN = Domain(map, params_map)
    # create psydac mapping for mass matrices only
    F_psy = DOMAIN.Psydac_mapping('F', **params_map)

    # Psydac discrete Derham sequence
    DR = Derham_build(Nel, p, spl_kind, F=F_psy, comm=MPI_COMM)

    print('Discrete Derham set set.')
    print()

    assert isinstance(DR.V0, TensorFemSpace)
    assert isinstance(DR.V1, ProductFemSpace)
    assert isinstance(DR.V2, ProductFemSpace)
    assert isinstance(DR.V3, TensorFemSpace)

    assert isinstance(DR.V0.vector_space, StencilVectorSpace)
    assert isinstance(DR.V1.vector_space, BlockVectorSpace)
    assert isinstance(DR.V2.vector_space, BlockVectorSpace)
    assert isinstance(DR.V3.vector_space, StencilVectorSpace)

    u0_fem = FemField(DR.V0)
    print('\nV0 FemField:')
    print(u0_fem)
    print('\n.space:')
    print(u0_fem.space)
    print('\ntype(coeffs):')
    print(type(u0_fem.coeffs))
    # FemField can be avaluated
    print(u0_fem)
    assert u0_fem(0, 0, 0) == 0.

    # see struphy/examples/example_psydac_parallel.py for more details
    u0 = StencilVector(DR.V0.vector_space)
    # StencilVector can be assigned
    assert u0[0, 0, 0] == 0.

    # Overwrite id with new stencil vector
    v0 = StencilVector(DR.V0.vector_space)
    old_id = id(u0)
    u0[:] = v0[:]
    assert id(u0) == old_id

    # Stencil matrix
    A = StencilMatrix(DR.V3.vector_space, DR.V0.vector_space)
    assert A[0, 0, 0, 0, 0, 0] == 0.

    # Product spaces
    # Vector
    u1 = BlockVector(DR.V1.vector_space)
    assert isinstance(u1[0], StencilVector)
    assert isinstance(u1[1], StencilVector)
    assert isinstance(u1[2], StencilVector)
    assert u1[0][0, 0, 0] == 0.

    # Matrix
    # maps from 0 -> 0, thus first column x
    Bxx = StencilMatrix(
        DR.V1.spaces[0].vector_space, DR.V1.spaces[0].vector_space)
    # maps from 0 -> 1, thus first column x
    Byx = StencilMatrix(
        DR.V1.spaces[0].vector_space, DR.V1.spaces[1].vector_space)
    # maps from 0 -> 2, thus first column x
    Bzx = StencilMatrix(
        DR.V1.spaces[0].vector_space, DR.V1.spaces[2].vector_space)

    Bxy = StencilMatrix(
        DR.V1.spaces[1].vector_space, DR.V1.spaces[0].vector_space)
    Byy = StencilMatrix(
        DR.V1.spaces[1].vector_space, DR.V1.spaces[1].vector_space)
    Bzy = StencilMatrix(
        DR.V1.spaces[1].vector_space, DR.V1.spaces[2].vector_space)

    Bxz = StencilMatrix(
        DR.V1.spaces[2].vector_space, DR.V1.spaces[0].vector_space)
    Byz = StencilMatrix(
        DR.V1.spaces[2].vector_space, DR.V1.spaces[1].vector_space)
    Bzz = StencilMatrix(
        DR.V1.spaces[2].vector_space, DR.V1.spaces[2].vector_space)

    blocks = [[Bxx, Bxy, Bxz], [Byx, Byy, Byz], [Bzx, Bzy, Bzz]]

    A11 = BlockMatrix(DR.V1.vector_space, DR.V1.vector_space, blocks)

    assert A11[0, 0][0, 0, 0, 0, 0, 0] == 0.

    #########################
    ### Access dimensions ###
    #########################
    print('\nPsydac dimensions:')
    N0_tot = DR.V0.nbasis
    N1_tot = [space.nbasis for space in DR.V1.spaces]
    N2_tot = [space.nbasis for space in DR.V2.spaces]
    N3_tot = DR.V3.nbasis
    N0 = [space.nbasis for space in DR.V0.spaces]
    N1 = [[direction.nbasis for direction in space.spaces]
          for space in DR.V1.spaces]
    N2 = [[direction.nbasis for direction in space.spaces]
          for space in DR.V2.spaces]
    N3 = [space.nbasis for space in DR.V3.spaces]
    print('V0.nbasis:', N0_tot)
    print('V0.spaces.nbasis:',    N0)
    print('V1.spaces.nbasis:', N1_tot)
    print('V1.spaces.spaces.nbasis:',    N1)
    print('V2.spaces.nbasis:', N2_tot)
    print('V2.spaces.spaces.nbasis:',    N2)
    print('V3.nbasis:', N3_tot)
    print('V3.spaces.nbasis:',    N3, '\n')

    # Struphy Derham (deprecated)
    nq_el = [4, 4, 4]
    spaces = [Spline_space_1d(Nel_i, p_i, spl_kind_i, nq_el_i)
              for Nel_i, p_i, spl_kind_i, nq_el_i in zip(Nel, p, spl_kind, nq_el)]

    spaces[0].set_projectors(p[0] + 1)
    spaces[1].set_projectors(p[1] + 1)
    spaces[2].set_projectors(p[2] + 1)

    DR_STR = Tensor_spline_space(spaces)
    DR_STR.set_projectors('tensor')

    # Space dimensions
    N0_tot = DR_STR.Ntot_0form
    N1_tot = DR_STR.Ntot_1form
    N2_tot = DR_STR.Ntot_2form
    N3_tot = DR_STR.Ntot_3form

    # Random vectors for testing
    x0 = np.random.rand(N0_tot)
    x1 = np.random.rand(np.sum(N1_tot))
    x2 = np.random.rand(np.sum(N2_tot))
    x3 = np.random.rand(N3_tot)

    ############################
    ### TEST STENCIL VECTORS ###
    ############################
    # Stencil vectors for Psydac:
    x0_PSY = StencilVector(DR.V0.vector_space)
    print('0-form StencilVector:')
    print('starts:', x0_PSY.starts)
    print('ends  :', x0_PSY.ends)
    print('pads  :', x0_PSY.pads)
    print('shape (=dim):', x0_PSY.shape)
    print('[:].shape (=shape):', x0_PSY[:].shape)

    s0 = x0_PSY.starts
    e0 = x0_PSY.ends

    # Assign from start to end index + 1
    x0_PSY[s0[0]: e0[0] + 1, s0[1]: e0[1] + 1, s0[2]: e0[2] +
           1] = DR_STR.extract_0(x0)[s0[0]: e0[0] + 1, s0[1]: e0[1] + 1, s0[2]: e0[2] + 1]

    # toarray() eliminates the padding
    assert np.all(x0 == x0_PSY.toarray())
    print('Assertion x0.toarray() passed.')

    # Block of StencilVecttors
    x1_PSY = BlockVector(DR.V1.vector_space)
    print('\n1-form StencilVector:')
    print('starts:', [component.starts for component in x1_PSY])
    print('ends  :', [component.ends for component in x1_PSY])
    print('pads  :', [component.pads for component in x1_PSY])
    print('shape (=dim):', [component.shape for component in x1_PSY])
    print('[:].shape (=shape):', [component[:].shape for component in x1_PSY])

    s11, s12, s13 = [component.starts for component in x1_PSY]
    e11, e12, e13 = [component.ends for component in x1_PSY]

    x11, x12, x13 = DR_STR.extract_1(x1)
    x1_PSY[0][s11[0]: e11[0] + 1, s11[1]: e11[1] + 1, s11[2]: e11[2] +
              1] = x11[s11[0]: e11[0] + 1, s11[1]: e11[1] + 1, s11[2]: e11[2] + 1]
    x1_PSY[1][s12[0]: e12[0] + 1, s12[1]: e12[1] + 1, s12[2]: e12[2] +
              1] = x12[s12[0]: e12[0] + 1, s12[1]: e12[1] + 1, s12[2]: e12[2] + 1]
    x1_PSY[2][s13[0]: e13[0] + 1, s13[1]: e13[1] + 1, s13[2]: e13[2] +
              1] = x13[s13[0]: e13[0] + 1, s13[1]: e13[1] + 1, s13[2]: e13[2] + 1]

    # toarray() is flattened and concatenated
    assert np.all(x1 == x1_PSY.toarray())
    print('Assertion x1.toarray() passed.')

    x2_PSY = BlockVector(DR.V2.vector_space)
    print('\n2-form StencilVector:')
    print('starts:', [component.starts for component in x2_PSY])
    print('ends  :', [component.ends for component in x2_PSY])
    print('pads  :', [component.pads for component in x2_PSY])
    print('shape (=dim):', [component.shape for component in x2_PSY])
    print('[:].shape (=shape):', [component[:].shape for component in x2_PSY])

    s21, s22, s23 = [component.starts for component in x2_PSY]
    e21, e22, e23 = [component.ends for component in x2_PSY]

    x21, x22, x23 = DR_STR.extract_2(x2)
    x2_PSY[0][s21[0]: e21[0] + 1, s21[1]: e21[1] + 1, s21[2]: e21[2] +
              1] = x21[s21[0]: e21[0] + 1, s21[1]: e21[1] + 1, s21[2]: e21[2] + 1]
    x2_PSY[1][s22[0]: e22[0] + 1, s22[1]: e22[1] + 1, s22[2]: e22[2] +
              1] = x22[s22[0]: e22[0] + 1, s22[1]: e22[1] + 1, s22[2]: e22[2] + 1]
    x2_PSY[2][s23[0]: e23[0] + 1, s23[1]: e23[1] + 1, s23[2]: e23[2] +
              1] = x23[s23[0]: e23[0] + 1, s23[1]: e23[1] + 1, s23[2]: e23[2] + 1]

    assert np.all(x2 == x2_PSY.toarray())
    print('Assertion x2.toarray() passed.')

    x3_PSY = StencilVector(DR.V3.vector_space)
    print('\n3-form StencilVector:')
    print('starts:', x3_PSY.starts)
    print('ends  :', x3_PSY.ends)
    print('pads  :', x3_PSY.pads)
    print('shape (=dim):', x3_PSY.shape)
    print('[:].shape (=shape):', x3_PSY[:].shape)

    s3 = x3_PSY.starts
    e3 = x3_PSY.ends

    x3_PSY[s3[0]: e3[0] + 1, s3[1]: e3[1] + 1, s3[2]: e3[2] +
           1] = DR_STR.extract_3(x3)[s3[0]: e3[0] + 1, s3[1]: e3[1] + 1, s3[2]: e3[2] + 1]

    assert np.all(x3 == x3_PSY.toarray())
    print('Assertion x2.toarray() passed.')

    ########################
    ### TEST DERIVATIVES ###
    ########################
    # Struphy derivative operators
    grad_STR = DR_STR.G0
    curl_STR = DR_STR.C0
    div_STR = DR_STR.D0
    print('')
    print('Struphy derivatives operators type:')
    print(type(grad_STR), type(curl_STR), type(div_STR))

    # Psydac derivative operators
    print('Psydac derivatives operators type:')
    print(type(DR.grad), type(DR.curl), type(DR.div))

    # compare derivatives
    d1_STR = grad_STR.dot(x0)
    d1_PSY = DR.grad.dot(x0_PSY)

    zero1_STR = curl_STR.dot(d1_STR)
    zero1_PSY = DR.curl.dot(d1_PSY)

    assert np.all(d1_STR == d1_PSY.toarray())
    assert np.allclose(zero1_STR, np.zeros_like(zero1_STR))
    assert np.allclose(zero1_PSY.toarray(), np.zeros_like(zero1_STR))
    print('Assertion grad passed.')

    d2_STR = curl_STR.dot(x1)
    d2_PSY = DR.curl.dot(x1_PSY)

    zero2_STR = div_STR.dot(d2_STR)
    zero2_PSY = DR.div.dot(d2_PSY)

    assert np.allclose(d2_STR, d2_PSY.toarray())
    assert np.allclose(zero2_STR, np.zeros_like(zero2_STR))
    assert np.allclose(zero2_PSY.toarray(), np.zeros_like(zero2_STR))
    print('Assertion curl passed.')

    d3_STR = div_STR.dot(x2)
    d3_PSY = DR.div.dot(x2_PSY)

    assert np.allclose(d3_STR, d3_PSY.toarray())
    print('Assertion div passed.')

    ##########################
    ### TEST MASS MATRICES ###
    ##########################
    # Struphy mass matrices

    print('Struphy')
    DR_STR.assemble_Mk(DOMAIN, 'V0')
    DR_STR.assemble_Mk(DOMAIN, 'V1')
    DR_STR.assemble_Mk(DOMAIN, 'V2')
    DR_STR.assemble_Mk(DOMAIN, 'V3')

    print('Psydac')
    # Psydac mass matrices
    DR.assemble_M0()
    DR.assemble_M1()
    DR.assemble_M2()
    DR.assemble_M3()

    print('Psydac mass matrices type:')
    print(type(DR.M0), '\n')

    # compare mass matrices
    prod0 = DR_STR.M0(x0)
    prod0_STR = DR_STR.M0_mat.dot(x0)
    prod0_PSY = DR.M0.dot(x0_PSY)

    # print('rel 0 error STR-STR:', np.max(np.abs(prod0_STR - prod0)) /
    #       np.max(np.abs(prod0_STR)))
    # print('rel 0 error STR-PSY:', np.max(np.abs(prod0_STR -
    #       prod0_PSY.toarray())) / np.max(np.abs(prod0_STR)))

    assert np.allclose(prod0_STR, prod0)
    assert np.allclose(prod0_STR, prod0_PSY.toarray())
    print('Assertion M0 passed.')

    prod1 = DR_STR.M1(x1)
    prod1_STR = DR_STR.M1_mat.dot(x1)
    prod1_PSY = DR.M1.dot(x1_PSY)

    # print('rel 1 error STR-STR:', np.max(np.abs(prod1_STR - prod1)) /
    #       np.max(np.abs(prod1_STR)))
    # print('rel 1 error STR-PSY:', np.max(np.abs(prod1_STR -
    #       prod1_PSY.toarray())) / np.max(np.abs(prod1_STR)))

    assert np.allclose(prod1_STR, prod1)
    assert np.allclose(prod1_STR, prod1_PSY.toarray())
    print('Assertion M1 passed.')

    prod2 = DR_STR.M2(x2)
    prod2_STR = DR_STR.M2_mat.dot(x2)
    prod2_PSY = DR.M2.dot(x2_PSY)

    # print('rel 2 error STR-STR:', np.max(np.abs(prod2_STR - prod2)) /
    #       np.max(np.abs(prod2_STR)))
    # print('rel 2 error STR-PSY:', np.max(np.abs(prod2_STR -
    #       prod2_PSY.toarray())) / np.max(np.abs(prod2_STR)))

    assert np.allclose(prod2_STR, prod2)
    assert np.allclose(prod2_STR, prod2_PSY.toarray())
    print('Assertion M2 passed.')

    prod3 = DR_STR.M3(x3)
    prod3_STR = DR_STR.M3_mat.dot(x3)
    prod3_PSY = DR.M3.dot(x3_PSY)

    # print('rel 3 error STR-STR:', np.max(np.abs(prod3_STR - prod3)) /
    #       np.max(np.abs(prod3_STR)))
    # print('rel 3 error STR-PSY:', np.max(np.abs(prod3_STR -
    #       prod3_PSY.toarray())) / np.max(np.abs(prod3_STR)))

    assert np.allclose(prod3_STR, prod3)
    assert np.allclose(prod3_STR, prod3_PSY.toarray())
    print('Assertion M3 passed.')

    #######################
    ### TEST PROJECTORS ###
    #######################
    # Struphy projectors
    DR_STR.set_projectors()
    PI = DR_STR.projectors.PI     # callable as input
    PI_mat = DR_STR.projectors.PI_mat  # dofs as input (as 3d array)
    print('\nStruphy projectors type:')
    print(type(PI), type(PI_mat))

    # compare projectors
    def f(eta1, eta2, eta3): return np.sin(4*np.pi*eta1) * \
        np.cos(2*np.pi*eta2) + np.exp(np.cos(2*np.pi*eta3))

    fh0_STR = PI('0', f)
    fh0_PSY = DR.P0(f)
    # print('\nfh0 shapes:')
    # print('Struphy:', fh0_STR.shape, ', flattened:', fh0_STR.flatten().shape)
    # print('Psydac:', fh0_PSY.coeffs.toarray().shape)

    # print(fh0_STR.flatten()[:10])
    # print(fh0_PSY.coeffs.toarray()[:10])

    assert np.allclose(fh0_STR.flatten(), fh0_PSY.coeffs.toarray())
    print('Assertion P0 passed.')

    fh11_STR = PI('11', f)
    fh12_STR = PI('12', f)
    fh13_STR = PI('13', f)
    fh1_STR = np.concatenate(
        (fh11_STR.flatten(), fh12_STR.flatten(), fh13_STR.flatten()))
    fh1_PSY = DR.P1((f, f, f))

    # print('\nfh1 shapes:')
    # print('Struphy 1:', fh11_STR.shape, ', flattened:', fh11_STR.flatten().shape)
    # print('Struphy 2:', fh12_STR.shape, ', flattened:', fh12_STR.flatten().shape)
    # print('Struphy 3:', fh13_STR.shape, ', flattened:', fh13_STR.flatten().shape)
    # print('Struphy:', fh1_STR.shape)
    # print('Psydac:', fh1_PSY.coeffs.toarray().shape)

    print('rel 1 error STR-PSY:', np.max(np.abs(fh1_STR -
          fh1_PSY.coeffs.toarray())) / np.max(np.abs(fh1_STR)))
    print(fh1_STR.flatten()[: 10])
    print(fh1_PSY.coeffs.toarray()[: 10])

    assert np.allclose(fh1_STR, fh1_PSY.coeffs.toarray(), atol=1e-3)
    print('Assertion P1 passed.')

    fh21_STR = PI('21', f)
    fh22_STR = PI('22', f)
    fh23_STR = PI('23', f)
    fh2_STR = np.concatenate(
        (fh21_STR.flatten(), fh22_STR.flatten(), fh23_STR.flatten()))
    fh2_PSY = DR.P2((f, f, f))

    # print('\nfh2 shapes:')
    # print('Struphy 1:', fh21_STR.shape, ', flattened:', fh21_STR.flatten().shape)
    # print('Struphy 2:', fh22_STR.shape, ', flattened:', fh22_STR.flatten().shape)
    # print('Struphy 3:', fh23_STR.shape, ', flattened:', fh23_STR.flatten().shape)
    # print('Struphy:', fh2_STR.shape)
    # print('Psydac:', fh2_PSY.coeffs.toarray().shape)

    assert np.allclose(fh2_STR, fh2_PSY.coeffs.toarray(), atol=1e-3)
    print('Assertion P2 passed.')

    fh3_STR = PI('3', f)
    fh3_PSY = DR.P3(f)
    # print('\nfh3 shapes:')
    # print('Struphy:', fh3_STR.shape, ', flattened:', fh3_STR.flatten().shape)
    # print('Psydac:', fh3_PSY.coeffs.toarray().shape)

    assert np.allclose(fh3_STR.flatten(), fh3_PSY.coeffs.toarray(), atol=1e-3)
    print('Assertion P3 passed.')


if __name__ == '__main__':
    test_psydac_mapping(['shafranov_dshaped', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}])
    test_psydac_derham([8, 8, 4], [2, 3, 2], [False, True, True], ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}])
