def test_cuboid():
    '''Test read-in of default parameter file, domain object creation and evaluation of cuboid mapping.'''

    import sysconfig
    import yaml 
    import numpy as np

    from struphy.geometry         import domain_3d

    file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
    print(file_in)

    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    DOMAIN   = domain_3d.Domain(params['geometry']['type'], 
                                params['geometry']['params_' + params['geometry']['type']])
    print('Domain object set.')

    print(DOMAIN.kind_map)
    print(DOMAIN.params_map)

    # cuboid
    assert DOMAIN.kind_map == 10

    # point-wise evaluation:
    print(DOMAIN.evaluate(.5, .5, .5, 'x'))
    print(DOMAIN.evaluate(.5, .5, .5, 'y'))
    print(DOMAIN.evaluate(.5, .5, .5, 'z'))
    assert DOMAIN.evaluate(.5, .5, .5, 'x') == 1.5
    assert DOMAIN.evaluate(.5, .5, .5, 'y') == 15.
    assert DOMAIN.evaluate(.5, .5, .5, 'z') == 150.

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)

    # eta1-array evaluation:
    print(DOMAIN.evaluate(arr1, .5, .5, 'x'))
    print(DOMAIN.evaluate(arr1, .5, .5, 'y'))
    print(DOMAIN.evaluate(arr1, .5, .5, 'z'))
    assert np.allclose(DOMAIN.evaluate(arr1, .5, .5, 'x'), np.linspace(1., 2., 4), 1e-14)
    assert np.allclose(DOMAIN.evaluate(arr1, .5, .5, 'y'), np.ones_like(arr1)*15.,   1e-14)
    assert np.allclose(DOMAIN.evaluate(arr1, .5, .5, 'z'), np.ones_like(arr1)*150.,  1e-14)

    # eta2-array evaluation:
    # TODO: fix this evaluation
    print(DOMAIN.evaluate(.5, arr2, .5, 'x'))
    print(DOMAIN.evaluate(.5, arr2, .5, 'y'))
    print(DOMAIN.evaluate(.5, arr2, .5, 'z'))
    # assert DOMAIN.evaluate(.5, arr, .5, 'x').shape == arr.shape
    # assert DOMAIN.evaluate(.5, arr, .5, 'y').shape == arr.shape
    # assert DOMAIN.evaluate(.5, arr, .5, 'z').shape == arr.shape

    # eta3-array evaluation:
    # TODO: fix this evaluation
    print(DOMAIN.evaluate(.5, .5, arr3, 'x'))
    print(DOMAIN.evaluate(.5, .5, arr3, 'y'))
    print(DOMAIN.evaluate(.5, .5, arr3, 'z'))
    # assert DOMAIN.evaluate(.5, .5, arr, 'x').shape == arr.shape
    # assert DOMAIN.evaluate(.5, .5, arr, 'y').shape == arr.shape
    # assert DOMAIN.evaluate(.5, .5, arr, 'z').shape == arr.shape

    # eta1-eta2-array evaluation:
    a = DOMAIN.evaluate(arr1, arr2, .5, 'x')
    b = DOMAIN.evaluate(arr1, arr2, .5, 'y')
    c = DOMAIN.evaluate(arr1, arr2, .5, 'z')
    print(a)
    print(b)
    print(c)
    assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
    assert b.shape[0] == arr1.size and b.shape[1] == arr2.size
    assert c.shape[0] == arr1.size and c.shape[1] == arr2.size

    # eta1-eta3-array evaluation:
    a = DOMAIN.evaluate(arr1, .5, arr3, 'x')
    b = DOMAIN.evaluate(arr1, .5, arr3, 'y')
    c = DOMAIN.evaluate(arr1, .5, arr3, 'z')
    print(a)
    print(b)
    print(c)
    assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
    assert b.shape[0] == arr1.size and b.shape[1] == arr3.size
    assert c.shape[0] == arr1.size and c.shape[1] == arr3.size

    # eta2-eta3-array evaluation:
    # TODO: fix this evaluation
    a = DOMAIN.evaluate(.5, arr2, arr3, 'x')
    b = DOMAIN.evaluate(.5, arr2, arr3, 'y')
    c = DOMAIN.evaluate(.5, arr2, arr3, 'z')
    print(a)
    print(b)
    print(c)
    # assert a.shape[0] == arr2.size and a.shape[1] == arr3.size
    # assert b.shape[0] == arr2.size and b.shape[1] == arr3.size
    # assert c.shape[0] == arr2.size and c.shape[1] == arr3.size

    # eta1-eta2-eta3-array evaluation:
    a = DOMAIN.evaluate(arr1, arr2, arr3, 'x')
    b = DOMAIN.evaluate(arr1, arr2, arr3, 'y')
    c = DOMAIN.evaluate(arr1, arr2, arr3, 'z')
    print(a)
    print(b)
    print(c)
    assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 
    assert b.shape[0] == arr1.size and b.shape[1] == arr2.size and a.shape[2] == arr3.size
    assert c.shape[0] == arr1.size and c.shape[1] == arr2.size and a.shape[2] == arr3.size

    # matrix evaluations
    # TODO: all possible combinations
    

if __name__ == '__main__':
    test_cuboid()