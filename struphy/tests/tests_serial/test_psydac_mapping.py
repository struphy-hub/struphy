import pytest


@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['Orthogonal', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}],
    ['Colella', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}],
    ['HollowCylinder', None],
    ['HollowTorus', {
        'a1': 1., 'a2': 2., 'R0': 3.}],
    ['EllipticCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6.}],
    ['RotatedEllipticCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'r1': 4., 'r2': 5., 'Lz': 6., 'th': 7.}],
    ['ShafranovShiftCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6., 'delta': 7.}],
    ['ShafranovSqrtCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6., 'delta': 7.}],
    ['ShafranovDshapedCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
def test_psydac_mapping(mapping):

    from struphy.geometry import domains

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
    domain_class = getattr(domains, map)
    domain = domain_class(params)

    print(domain.F_psy._expressions, '\n')

    # Psydac mapping
    F_PSY = domain.F_psy.get_callable_mapping()

    # Comparisons at random logical point
    eta = np.random.rand(3)

    # Mapping
    assert np.allclose(F_PSY(*eta)[0], domain.evaluate(*eta, 'x'))
    assert np.allclose(F_PSY(*eta)[1], domain.evaluate(*eta, 'y'))
    assert np.allclose(F_PSY(*eta)[2], domain.evaluate(*eta, 'z'))

    # Absolute value of Jacobian determinant
    assert np.allclose(np.sqrt(F_PSY.metric_det(*eta)),
                       np.abs(domain.evaluate(*eta, 'det_df')))

    # Jacobian
    assert np.allclose(F_PSY.jacobian(
        *eta)[0, 0], domain.evaluate(*eta, 'df_11'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[0, 1], domain.evaluate(*eta, 'df_12'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[0, 2], domain.evaluate(*eta, 'df_13'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[1, 0], domain.evaluate(*eta, 'df_21'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[1, 1], domain.evaluate(*eta, 'df_22'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[1, 2], domain.evaluate(*eta, 'df_23'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[2, 0], domain.evaluate(*eta, 'df_31'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[2, 1], domain.evaluate(*eta, 'df_32'))
    assert np.allclose(F_PSY.jacobian(
        *eta)[2, 2], domain.evaluate(*eta, 'df_33'))

    # Inverse Jacobian
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[0, 0], domain.evaluate(*eta, 'df_inv_11'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[0, 1], domain.evaluate(*eta, 'df_inv_12'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[0, 2], domain.evaluate(*eta, 'df_inv_13'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[1, 0], domain.evaluate(*eta, 'df_inv_21'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[1, 1], domain.evaluate(*eta, 'df_inv_22'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[1, 2], domain.evaluate(*eta, 'df_inv_23'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[2, 0], domain.evaluate(*eta, 'df_inv_31'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[2, 1], domain.evaluate(*eta, 'df_inv_32'))
    assert np.allclose(F_PSY.jacobian_inv(
        *eta)[2, 2], domain.evaluate(*eta, 'df_inv_33'))

    # Metric tensor
    assert np.allclose(F_PSY.metric(*eta)[0, 0], domain.evaluate(*eta, 'g_11'))
    assert np.allclose(F_PSY.metric(*eta)[0, 1], domain.evaluate(*eta, 'g_12'))
    assert np.allclose(F_PSY.metric(*eta)[0, 2], domain.evaluate(*eta, 'g_13'))
    assert np.allclose(F_PSY.metric(*eta)[1, 0], domain.evaluate(*eta, 'g_21'))
    assert np.allclose(F_PSY.metric(*eta)[1, 1], domain.evaluate(*eta, 'g_22'))
    assert np.allclose(F_PSY.metric(*eta)[1, 2], domain.evaluate(*eta, 'g_23'))
    assert np.allclose(F_PSY.metric(*eta)[2, 0], domain.evaluate(*eta, 'g_31'))
    assert np.allclose(F_PSY.metric(*eta)[2, 1], domain.evaluate(*eta, 'g_32'))
    assert np.allclose(F_PSY.metric(*eta)[2, 2], domain.evaluate(*eta, 'g_33'))

    # Inverse metric tensor
    metric_inv_PSY = np.matmul(F_PSY.jacobian_inv(
        *eta), F_PSY.jacobian_inv(*eta).T)  # missing in psydac
    assert np.allclose(metric_inv_PSY[0, 0], domain.evaluate(*eta, 'g_inv_11'))
    assert np.allclose(metric_inv_PSY[0, 1], domain.evaluate(*eta, 'g_inv_12'))
    assert np.allclose(metric_inv_PSY[0, 2], domain.evaluate(*eta, 'g_inv_13'))
    assert np.allclose(metric_inv_PSY[1, 0], domain.evaluate(*eta, 'g_inv_21'))
    assert np.allclose(metric_inv_PSY[1, 1], domain.evaluate(*eta, 'g_inv_22'))
    assert np.allclose(metric_inv_PSY[1, 2], domain.evaluate(*eta, 'g_inv_23'))
    assert np.allclose(metric_inv_PSY[2, 0], domain.evaluate(*eta, 'g_inv_31'))
    assert np.allclose(metric_inv_PSY[2, 1], domain.evaluate(*eta, 'g_inv_32'))
    assert np.allclose(metric_inv_PSY[2, 2], domain.evaluate(*eta, 'g_inv_33'))

    print(map + ' done.\n')


if __name__ == '__main__':
    test_psydac_mapping(['ShafranovDshapedCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}])
