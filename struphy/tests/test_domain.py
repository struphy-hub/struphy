def test_evaluation_mappings():
    '''Test domain object creation and evaluation of mappings.'''

    import sysconfig
    import yaml 
    import numpy as np

    from struphy.geometry import domain_3d

    file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
    print(f'Path to parameters file: {file_in}')

    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Update path to sample spline coefficients.
    params['geometry']['params_spline']['file'] = sysconfig.get_path("platlib") + '/struphy/' + params['geometry']['params_spline']['file']
    print(f"Updated path to sample spline coefficients: {params['geometry']['params_spline']['file']}")

    kind_maps = [
        'cuboid',
        'orthogonal',
        'colella',
        'hollow_cyl',
        'hollow_torus',
        'ellipse',
        'rotated_ellipse',
        'shafranov_shift',
        'shafranov_sqrt',
        'shafranov_dshaped',
        'spline',
        'spline_cyl',
        'spline_torus',
    ]

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "evaluate"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    for kind_map in kind_maps:

        DOMAIN   = domain_3d.Domain(kind_map, params['geometry']['params_' + kind_map])
        print()
        print('Domain object set.')

        print('yaml\'s kind_map     :', kind_map)
        print('DOMAIN\'s kind_map   :', DOMAIN.kind_map)
        print('yaml\'s params_map   :', params['geometry']['params_' + kind_map])
        print('DOMAIN\'s params_map :', DOMAIN.params_map)

        # point-wise evaluation:
        print('pointwise evaluation, size:', DOMAIN.evaluate(.5, .5, .5, 'x').size)
        assert DOMAIN.evaluate(.5, .5, .5, 'x').size == 1
        assert DOMAIN.evaluate(.5, .5, .5, 'y').size == 1
        assert DOMAIN.evaluate(.5, .5, .5, 'z').size == 1

        # flat evaluation:
        print('flat evaluation, shape:', DOMAIN.evaluate(arr1, arr2[:-1], arr3[:-2], 'x', flat_eval=True).shape)
        assert DOMAIN.evaluate(arr1, arr2[:-1], arr3[:-2], 'x', flat_eval=True).shape == arr1.shape
        assert DOMAIN.evaluate(arr1, arr2[:-1], arr3[:-2], 'y', flat_eval=True).shape == arr1.shape
        assert DOMAIN.evaluate(arr1, arr2[:-1], arr3[:-2], 'z', flat_eval=True).shape == arr1.shape

        # eta1-array evaluation:
        print('eta1 array evaluation, shape:', DOMAIN.evaluate(arr1, .5, .5, 'x').shape)
        assert DOMAIN.evaluate(arr1, .5, .5, 'x').shape == arr1.shape
        assert DOMAIN.evaluate(arr1, .5, .5, 'y').shape == arr1.shape
        assert DOMAIN.evaluate(arr1, .5, .5, 'z').shape == arr1.shape
        # eta2-array evaluation:
        print('eta2 array evaluation, shape:', DOMAIN.evaluate(.5, arr2, .5, 'x').shape)
        assert DOMAIN.evaluate(.5, arr2, .5, 'x').shape == arr2.shape
        assert DOMAIN.evaluate(.5, arr2, .5, 'y').shape == arr2.shape
        assert DOMAIN.evaluate(.5, arr2, .5, 'z').shape == arr2.shape
        # eta3-array evaluation:
        print('eta3 array evaluation, shape:', DOMAIN.evaluate(.5, .5, arr3, 'x').shape)
        assert DOMAIN.evaluate(.5, .5, arr3, 'x').shape == arr3.shape
        assert DOMAIN.evaluate(.5, .5, arr3, 'y').shape == arr3.shape
        assert DOMAIN.evaluate(.5, .5, arr3, 'z').shape == arr3.shape

        # eta1-eta2-array evaluation:
        a = DOMAIN.evaluate(arr1, arr2, .5, 'x')
        b = DOMAIN.evaluate(arr1, arr2, .5, 'y')
        c = DOMAIN.evaluate(arr1, arr2, .5, 'z')
        print('eta1-eta2 array evaluation, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
        assert b.shape[0] == arr1.size and b.shape[1] == arr2.size
        assert c.shape[0] == arr1.size and c.shape[1] == arr2.size
        # eta1-eta3-array evaluation:
        a = DOMAIN.evaluate(arr1, .5, arr3, 'x')
        b = DOMAIN.evaluate(arr1, .5, arr3, 'y')
        c = DOMAIN.evaluate(arr1, .5, arr3, 'z')
        print('eta1-eta3 array evaluation, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
        assert b.shape[0] == arr1.size and b.shape[1] == arr3.size
        assert c.shape[0] == arr1.size and c.shape[1] == arr3.size
        # eta2-eta3-array evaluation:
        a = DOMAIN.evaluate(.5, arr2, arr3, 'x')
        b = DOMAIN.evaluate(.5, arr2, arr3, 'y')
        c = DOMAIN.evaluate(.5, arr2, arr3, 'z')
        print('eta2-eta3 array evaluation, shape:', a.shape)
        assert a.shape[0] == arr2.size and a.shape[1] == arr3.size
        assert b.shape[0] == arr2.size and b.shape[1] == arr3.size
        assert c.shape[0] == arr2.size and c.shape[1] == arr3.size

        # eta1-eta2-eta3 array evaluation:
        a = DOMAIN.evaluate(arr1, arr2, arr3, 'x')
        b = DOMAIN.evaluate(arr1, arr2, arr3, 'y')
        c = DOMAIN.evaluate(arr1, arr2, arr3, 'z')
        print('eta1-eta2-eta3-array evaluation, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 
        assert b.shape[0] == arr1.size and b.shape[1] == arr2.size and b.shape[2] == arr3.size
        assert c.shape[0] == arr1.size and c.shape[1] == arr2.size and c.shape[2] == arr3.size

        # matrix evaluations at one point in third direction
        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

        # eta1-eta2 matrix evaluation:
        a = DOMAIN.evaluate(mat12_x, mat12_y, .5, 'x')
        b = DOMAIN.evaluate(mat12_x, mat12_y, .5, 'y')
        c = DOMAIN.evaluate(mat12_x, mat12_y, .5, 'z')
        print('eta1-eta2 matrix evaluation, shape:', a.shape)
        assert a.shape == mat12_x.shape
        assert b.shape == mat12_x.shape
        assert c.shape == mat12_x.shape
        # eta1-eta3 matrix evaluation:
        a = DOMAIN.evaluate(mat13_x, .5, mat13_z, 'x')
        b = DOMAIN.evaluate(mat13_x, .5, mat13_z, 'y')
        c = DOMAIN.evaluate(mat13_x, .5, mat13_z, 'z')
        print('eta1-eta3 matrix evaluation, shape:', a.shape)
        assert a.shape == mat13_x.shape
        assert b.shape == mat13_x.shape
        assert c.shape == mat13_x.shape
        # eta2-eta3 matrix evaluation:
        a = DOMAIN.evaluate(.5, mat23_y, mat23_z, 'x')
        b = DOMAIN.evaluate(.5, mat23_y, mat23_z, 'y')
        c = DOMAIN.evaluate(.5, mat23_y, mat23_z, 'z')
        print('eta2-eta3 matrix evaluation, shape:', a.shape)
        assert a.shape == mat23_y.shape
        assert b.shape == mat23_y.shape
        assert c.shape == mat23_y.shape

        # matrix evaluations for sparse meshgrid
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
        a = DOMAIN.evaluate(mat_x, mat_y, mat_z, 'x')
        b = DOMAIN.evaluate(mat_x, mat_y, mat_z, 'y')
        c = DOMAIN.evaluate(mat_x, mat_y, mat_z, 'z')
        print('sparse meshgrid matrix evaluation, shape:', a.shape)
        assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]
        assert b.shape[0] == mat_x.shape[0] and b.shape[1] == mat_y.shape[1] and b.shape[2] == mat_z.shape[2]
        assert c.shape[0] == mat_x.shape[0] and c.shape[1] == mat_y.shape[1] and c.shape[2] == mat_z.shape[2]

        # matrix evaluations 
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
        a = DOMAIN.evaluate(mat_x, mat_y, mat_z, 'x')
        b = DOMAIN.evaluate(mat_x, mat_y, mat_z, 'y')
        c = DOMAIN.evaluate(mat_x, mat_y, mat_z, 'z')
        print('matrix evaluation, shape:', a.shape)
        assert a.shape == mat_x.shape 
        assert b.shape == mat_x.shape
        assert c.shape == mat_x.shape


def test_pullback():
    '''Test pullbacks of p-forms for different geometries.'''

    import sysconfig
    import yaml 
    import numpy as np

    from struphy.geometry import domain_3d

    file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
    print(f'Path to parameters file: {file_in}')

    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Update path to sample spline coefficients.
    params['geometry']['params_spline']['file'] = sysconfig.get_path("platlib") + '/struphy/' + params['geometry']['params_spline']['file']
    print(f"Updated path to sample spline coefficients: {params['geometry']['params_spline']['file']}")

    kind_maps = [
        'cuboid',
        'orthogonal',
        'colella',
        'hollow_cyl',
        'hollow_torus',
        'ellipse',
        'rotated_ellipse',
        'shafranov_shift',
        'shafranov_sqrt',
        'shafranov_dshaped',
        'spline',
        'spline_cyl',
        'spline_torus',
    ]

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "pull"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    # physical function to pull back (used as components of forms too):
    fun = lambda x, y, z: np.exp(x)*np.sin(y)*np.cos(z)

    for kind_map in kind_maps:

        DOMAIN   = domain_3d.Domain(kind_map, params['geometry']['params_' + kind_map])
        print()
        print('Domain object set.')

        print('yaml\'s kind_map     :', kind_map)
        print('DOMAIN\'s kind_map   :', DOMAIN.kind_map)
        print('yaml\'s params_map   :', params['geometry']['params_' + kind_map])
        print('DOMAIN\'s params_map :', DOMAIN.params_map)

        for p_str in DOMAIN.keys_pull:

            print('component:', p_str)

            if p_str=='0_form' or p_str=='3_form':
                fun_form = fun
            else:
                fun_form = [fun, fun, fun]

            # point-wise pullback:
            assert DOMAIN.pull(fun_form, .5, .5, .5, p_str).size == 1
            #print('pointwise pullback, size:', DOMAIN.pull(fun_form, .5, .5, .5, p_str).size)

            # flat pullback:
            assert DOMAIN.pull(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
            assert DOMAIN.pull(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
            assert DOMAIN.pull(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape

            # eta1-array pullback:
            #print('eta1 array pullback, shape:', DOMAIN.pull(fun_form, arr1, .5, .5, p_str).shape)
            assert DOMAIN.pull(fun_form, arr1, .5, .5, p_str).shape == arr1.shape
            # eta2-array pullback:
            #print('eta2 array pullback, shape:', DOMAIN.pull(fun_form, .5, arr2, .5, p_str).shape)
            assert DOMAIN.pull(fun_form, .5, arr2, .5, p_str).shape == arr2.shape
            # eta3-array pullback:
            #print('eta3 array pullback, shape:', DOMAIN.pull(fun_form, .5, .5, arr3, p_str).shape)
            assert DOMAIN.pull(fun_form, .5, .5, arr3, p_str).shape == arr3.shape

            # eta1-eta2-array pullback:
            a = DOMAIN.pull(fun_form, arr1, arr2, .5, p_str)
            #print('eta1-eta2 array pullback, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
            # eta1-eta3-array pullback:
            a = DOMAIN.pull(fun_form, arr1, .5, arr3, p_str)
            #print('eta1-eta3 array pullback, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
            # eta2-eta3-array pullback:
            a = DOMAIN.pull(fun_form, .5, arr2, arr3, p_str)
            #print('eta2-eta3 array pullback, shape:', a.shape)
            assert a.shape[0] == arr2.size and a.shape[1] == arr3.size

            # eta1-eta2-eta3 array pullback:
            a = DOMAIN.pull(fun_form, arr1, arr2, arr3, p_str)
            #print('eta1-eta2-eta3-array pullback, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 

            # matrix pullbacks at one point in third direction
            mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
            mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
            mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

            # eta1-eta2 matrix pullback:
            a = DOMAIN.pull(fun_form, mat12_x, mat12_y, .5, p_str)
            #print('eta1-eta2 matrix pullback, shape:', a.shape)
            assert a.shape == mat12_x.shape
            # eta1-eta3 matrix pullback:
            a = DOMAIN.pull(fun_form, mat13_x, .5, mat13_z, p_str)
            #print('eta1-eta3 matrix pullback, shape:', a.shape)
            assert a.shape == mat13_x.shape
            # eta2-eta3 matrix pullback:
            a = DOMAIN.pull(fun_form, .5, mat23_y, mat23_z, p_str)
            #print('eta2-eta3 matrix pullback, shape:', a.shape)
            assert a.shape == mat23_y.shape

            # matrix pullbacks for sparse meshgrid
            mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
            a = DOMAIN.pull(fun_form, mat_x, mat_y, mat_z, p_str)
            #print('sparse meshgrid matrix pullback, shape:', a.shape)
            assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]

            # matrix pullbacks 
            mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
            a = DOMAIN.pull(fun_form, mat_x, mat_y, mat_z, p_str)
            #print('matrix pullback, shape:', a.shape)
            assert a.shape == mat_x.shape 


def test_pushforward():
    '''Test pullbacks of p-forms for different geometries.'''

    import sysconfig
    import yaml 
    import numpy as np

    from struphy.geometry import domain_3d

    file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
    print(f'Path to parameters file: {file_in}')

    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Update path to sample spline coefficients.
    params['geometry']['params_spline']['file'] = sysconfig.get_path("platlib") + '/struphy/' + params['geometry']['params_spline']['file']
    print(f"Updated path to sample spline coefficients: {params['geometry']['params_spline']['file']}")

    kind_maps = [
        'cuboid',
        'orthogonal',
        'colella',
        'hollow_cyl',
        'hollow_torus',
        'ellipse',
        'rotated_ellipse',
        'shafranov_shift',
        'shafranov_sqrt',
        'shafranov_dshaped',
        'spline',
        'spline_cyl',
        'spline_torus',
    ]

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "push"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    # logical function to push forward (used as components of forms too):
    fun = lambda eta1, eta2, eta3: np.exp(eta1)*np.sin(eta2)*np.cos(eta3)

    for kind_map in kind_maps:

        DOMAIN   = domain_3d.Domain(kind_map, params['geometry']['params_' + kind_map])
        print()
        print('Domain object set.')

        print('yaml\'s kind_map     :', kind_map)
        print('DOMAIN\'s kind_map   :', DOMAIN.kind_map)
        print('yaml\'s params_map   :', params['geometry']['params_' + kind_map])
        print('DOMAIN\'s params_map :', DOMAIN.params_map)

        for p_str in DOMAIN.keys_push:

            print('component:', p_str)

            if p_str=='0_form' or p_str=='3_form':
                fun_form = fun
            else:
                fun_form = [fun, fun, fun]

            # point-wise pushforward:
            assert DOMAIN.push(fun_form, .5, .5, .5, p_str).size == 1
            #print('pointwise pushforward, size:', DOMAIN.push(fun_form, .5, .5, .5, p_str).size)

            # flat pushforward:
            assert DOMAIN.push(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
            assert DOMAIN.push(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
            assert DOMAIN.push(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape

            # eta1-array pushforward:
            #print('eta1 array pushforward, shape:', DOMAIN.push(fun_form, arr1, .5, .5, p_str).shape)
            assert DOMAIN.push(fun_form, arr1, .5, .5, p_str).shape == arr1.shape
            # eta2-array pushforward:
            #print('eta2 array pushforward, shape:', DOMAIN.push(fun_form, .5, arr2, .5, p_str).shape)
            assert DOMAIN.push(fun_form, .5, arr2, .5, p_str).shape == arr2.shape
            # eta3-array pushforward:
            #print('eta3 array pushforward, shape:', DOMAIN.push(fun_form, .5, .5, arr3, p_str).shape)
            assert DOMAIN.push(fun_form, .5, .5, arr3, p_str).shape == arr3.shape

            # eta1-eta2-array pushforward:
            a = DOMAIN.push(fun_form, arr1, arr2, .5, p_str)
            #print('eta1-eta2 array pushforward, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
            # eta1-eta3-array pushforward:
            a = DOMAIN.push(fun_form, arr1, .5, arr3, p_str)
            #print('eta1-eta3 array pushforward, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
            # eta2-eta3-array pushforward:
            a = DOMAIN.push(fun_form, .5, arr2, arr3, p_str)
            #print('eta2-eta3 array pushforward, shape:', a.shape)
            assert a.shape[0] == arr2.size and a.shape[1] == arr3.size

            # eta1-eta2-eta3 array pushforward:
            a = DOMAIN.push(fun_form, arr1, arr2, arr3, p_str)
            #print('eta1-eta2-eta3-array pushforward, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 

            # matrix pullbacks at one point in third direction
            mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
            mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
            mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

            # eta1-eta2 matrix pushforward:
            a = DOMAIN.push(fun_form, mat12_x, mat12_y, .5, p_str)
            #print('eta1-eta2 matrix pushforward, shape:', a.shape)
            assert a.shape == mat12_x.shape
            # eta1-eta3 matrix pushforward:
            a = DOMAIN.push(fun_form, mat13_x, .5, mat13_z, p_str)
            #print('eta1-eta3 matrix pushforward, shape:', a.shape)
            assert a.shape == mat13_x.shape
            # eta2-eta3 matrix pushforward:
            a = DOMAIN.push(fun_form, .5, mat23_y, mat23_z, p_str)
            #print('eta2-eta3 matrix pushforward, shape:', a.shape)
            assert a.shape == mat23_y.shape

            # matrix pullbacks for sparse meshgrid
            mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
            a = DOMAIN.push(fun_form, mat_x, mat_y, mat_z, p_str)
            #print('sparse meshgrid matrix pushforward, shape:', a.shape)
            assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]

            # matrix pullbacks 
            mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
            a = DOMAIN.push(fun_form, mat_x, mat_y, mat_z, p_str)
            #print('matrix pushforward, shape:', a.shape)
            assert a.shape == mat_x.shape 


def test_transformation():
    '''Test transformation between different geometries.'''

    import sysconfig
    import yaml 
    import numpy as np

    from struphy.geometry import domain_3d

    file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
    print(f'Path to parameters file: {file_in}')

    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Update path to sample spline coefficients.
    params['geometry']['params_spline']['file'] = sysconfig.get_path("platlib") + '/struphy/' + params['geometry']['params_spline']['file']
    print(f"Updated path to sample spline coefficients: {params['geometry']['params_spline']['file']}")

    kind_maps = [
        'cuboid',
        'orthogonal',
        'colella',
        'hollow_cyl',
        'hollow_torus',
        'ellipse',
        'rotated_ellipse',
        'shafranov_shift',
        'shafranov_sqrt',
        'shafranov_dshaped',
        'spline',
        'spline_cyl',
        'spline_torus',
    ]

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "transform"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    # logical function to tranform (used as components of forms too):
    fun = lambda eta1, eta2, eta3: np.exp(eta1)*np.sin(eta2)*np.cos(eta3)

    for kind_map in kind_maps:

        DOMAIN   = domain_3d.Domain(kind_map, params['geometry']['params_' + kind_map])
        print()
        print('Domain object set.')

        print('yaml\'s kind_map     :', kind_map)
        print('DOMAIN\'s kind_map   :', DOMAIN.kind_map)
        print('yaml\'s params_map   :', params['geometry']['params_' + kind_map])
        print('DOMAIN\'s params_map :', DOMAIN.params_map)

        for p_str in DOMAIN.keys_transform:

            print('component:', p_str)

            if p_str=='norm_to_0' or p_str=='norm_to_3' or p_str=='0_to_3' or p_str=='3_to_0':
                fun_form = fun
            else:
                fun_form = [fun, fun, fun]

            # point-wise transformation:
            assert DOMAIN.transformation(fun_form, .5, .5, .5, p_str).size == 1
            #print('pointwise transformation, size:', DOMAIN.transformation(fun_form, .5, .5, .5, p_str).size)

            # flat transformation:
            assert DOMAIN.transformation(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
            assert DOMAIN.transformation(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
            assert DOMAIN.transformation(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape

            # eta1-array transformation:
            #print('eta1 array transformation, shape:', DOMAIN.transformation(fun_form, arr1, .5, .5, p_str).shape)
            assert DOMAIN.transformation(fun_form, arr1, .5, .5, p_str).shape == arr1.shape
            # eta2-array transformation:
            #print('eta2 array transformation, shape:', DOMAIN.transformation(fun_form, .5, arr2, .5, p_str).shape)
            assert DOMAIN.transformation(fun_form, .5, arr2, .5, p_str).shape == arr2.shape
            # eta3-array transformation:
            #print('eta3 array transformation, shape:', DOMAIN.transformation(fun_form, .5, .5, arr3, p_str).shape)
            assert DOMAIN.transformation(fun_form, .5, .5, arr3, p_str).shape == arr3.shape

            # eta1-eta2-array transformation:
            a = DOMAIN.transformation(fun_form, arr1, arr2, .5, p_str)
            #print('eta1-eta2 array transformation, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
            # eta1-eta3-array transformation:
            a = DOMAIN.transformation(fun_form, arr1, .5, arr3, p_str)
            #print('eta1-eta3 array transformation, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
            # eta2-eta3-array transformation:
            a = DOMAIN.transformation(fun_form, .5, arr2, arr3, p_str)
            #print('eta2-eta3 array transformation, shape:', a.shape)
            assert a.shape[0] == arr2.size and a.shape[1] == arr3.size

            # eta1-eta2-eta3 array transformation:
            a = DOMAIN.transformation(fun_form, arr1, arr2, arr3, p_str)
            #print('eta1-eta2-eta3-array transformation, shape:', a.shape)
            assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 

            # matrix transformation at one point in third direction
            mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
            mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
            mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

            # eta1-eta2 matrix transformation:
            a = DOMAIN.transformation(fun_form, mat12_x, mat12_y, .5, p_str)
            #print('eta1-eta2 matrix transformation, shape:', a.shape)
            assert a.shape == mat12_x.shape
            # eta1-eta3 matrix transformation:
            a = DOMAIN.transformation(fun_form, mat13_x, .5, mat13_z, p_str)
            #print('eta1-eta3 matrix transformation, shape:', a.shape)
            assert a.shape == mat13_x.shape
            # eta2-eta3 matrix transformation:
            a = DOMAIN.transformation(fun_form, .5, mat23_y, mat23_z, p_str)
            #print('eta2-eta3 matrix transformation, shape:', a.shape)
            assert a.shape == mat23_y.shape

            # matrix transformation for sparse meshgrid
            mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
            a = DOMAIN.transformation(fun_form, mat_x, mat_y, mat_z, p_str)
            #print('sparse meshgrid matrix transformation, shape:', a.shape)
            assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]

            # matrix transformation 
            mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
            a = DOMAIN.transformation(fun_form, mat_x, mat_y, mat_z, p_str)
            #print('matrix transformation, shape:', a.shape)
            assert a.shape == mat_x.shape 
    

if __name__ == '__main__':
    test_evaluation_mappings()
    test_pullback()
    test_pushforward()
    test_transformation()
