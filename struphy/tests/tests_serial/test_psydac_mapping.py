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
    assert np.allclose(F_PSY(*eta), domain(*eta))

    # Absolute value of Jacobian determinant
    assert np.allclose(np.sqrt(F_PSY.metric_det(*eta)),
                       np.abs(domain.jacobian_det(*eta)))

    # Jacobian
    assert np.allclose(F_PSY.jacobian(
        *eta), domain.jacobian(*eta))

    # Inverse Jacobian
    assert np.allclose(F_PSY.jacobian_inv(
        *eta), domain.jacobian_inv(*eta))
    
    # Metric tensor
    assert np.allclose(F_PSY.metric(*eta), domain.metric(*eta))

    # Inverse metric tensor
    metric_inv_PSY = np.matmul(F_PSY.jacobian_inv(
        *eta), F_PSY.jacobian_inv(*eta).T)  # missing in psydac
    assert np.allclose(metric_inv_PSY, domain.metric_inv(*eta))

    print(map + ' done.\n')


if __name__ == '__main__':
    test_psydac_mapping(['ShafranovDshapedCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}])
