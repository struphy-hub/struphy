import pytest

def test_prepare_arg():
    """ Tests prepare_arg static method in domain base class.
    """
    
    from struphy.geometry.base import Domain
    import numpy as np
    
    
    a1 = lambda e1, e2, e3 : e1*e2
    a2 = lambda e1, e2, e3 : e2*e3
    a3 = lambda e1, e2, e3 : e3*e1
    
    def a_vec(e1, e2, e3):
        
        a_1 = e1*e2
        a_2 = e2*e3
        a_3 = e3*e1
        
        return np.stack((a_1, a_2, a_3), axis=0)
    
    
    # ============== flat_eval == False =========================
    flat = False
    
    e1 = np.random.rand(4)
    e2 = np.random.rand(5)
    e3 = np.random.rand(6)
    
    E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(e1, e2, e3, flat_eval=flat)
    
    shape_scalar = (1, E1.shape[0], E2.shape[1], E3.shape[2])
    shape_vector = (3, E1.shape[0], E2.shape[1], E3.shape[2])
    
    # ======== callables ============
    
    # scalar function
    assert Domain.prepare_arg(a1, E1, E2, E3, flat_eval=flat).shape == shape_scalar 
    assert Domain.prepare_arg((a1,), E1, E2, E3, flat_eval=flat).shape == shape_scalar
    assert Domain.prepare_arg([a1,], E1, E2, E3, flat_eval=flat).shape == shape_scalar
    
    # vector-valued function
    assert Domain.prepare_arg(a_vec, E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg((a1, a2, a3), E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg([a1, a2, a3], E1, E2, E3, flat_eval=flat).shape == shape_vector
    
    
    # ======== arrays ===============
    
    A1 = a1(E1, E2, E3)
    A2 = a2(E1, E2, E3)
    A3 = a3(E1, E2, E3)
    
    A = a_vec(E1, E2, E3)
    
    # scalar function
    assert Domain.prepare_arg(A1, E1, E2, E3, flat_eval=flat).shape == shape_scalar
    assert Domain.prepare_arg((A1,), E1, E2, E3, flat_eval=flat).shape == shape_scalar
    assert Domain.prepare_arg([A1,], E1, E2, E3, flat_eval=flat).shape == shape_scalar
    
    # vector-valued function
    assert Domain.prepare_arg(A, E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg((A1, A2, A3), E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg([A1, A2, A3], E1, E2, E3, flat_eval=flat).shape == shape_vector
    
    # ============== flat_eval == True ==========================
    flat = True
    
    e1 = np.random.rand(4)
    e2 = np.random.rand(4)
    e3 = np.random.rand(4)
    
    E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(e1, e2, e3, flat_eval=flat)
    
    shape_scalar = (1, E1.shape[0], 1, 1)
    shape_vector = (3, E1.shape[0], 1, 1)
    
    # ======== callables ============
    
    # scalar function
    assert Domain.prepare_arg(a1, E1, E2, E3, flat_eval=flat).shape == shape_scalar 
    assert Domain.prepare_arg((a1,), E1, E2, E3, flat_eval=flat).shape == shape_scalar
    assert Domain.prepare_arg([a1,], E1, E2, E3, flat_eval=flat).shape == shape_scalar
    
    # vector-valued function
    assert Domain.prepare_arg(a_vec, E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg((a1, a2, a3), E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg([a1, a2, a3], E1, E2, E3, flat_eval=flat).shape == shape_vector
    
    # ======== arrays ===============
    
    A1 = a1(e1, e2, e3)
    A2 = a2(e1, e2, e3)
    A3 = a3(e1, e2, e3)
    
    A = a_vec(e1, e2, e3)
    
    # scalar function
    assert Domain.prepare_arg(A1, E1, E2, E3, flat_eval=flat).shape == shape_scalar
    assert Domain.prepare_arg((A1,), E1, E2, E3, flat_eval=flat).shape == shape_scalar
    assert Domain.prepare_arg([A1,], E1, E2, E3, flat_eval=flat).shape == shape_scalar
    
    # vector-valued function
    assert Domain.prepare_arg(A, E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg((A1, A2, A3), E1, E2, E3, flat_eval=flat).shape == shape_vector
    assert Domain.prepare_arg([A1, A2, A3], E1, E2, E3, flat_eval=flat).shape == shape_vector

    
@pytest.mark.parametrize('mapping', [
    'Cuboid',
    'HollowCylinder',
    'Colella',
    'Orthogonal',
    'HollowTorus',
    'EllipticCylinder',
    'RotatedEllipticCylinder',
    'PoweredEllipticCylinder',
    'ShafranovShiftCylinder',
    'ShafranovSqrtCylinder',
    'ShafranovDshapedCylinder',
    'ShafranovNonAxisSymmCylinder',
    'Spline',
    'PoloidalSplineCylinder',
    'PoloidalSplineTorus'])    
def test_evaluation_mappings(mapping):
    """ Tests domain object creation with default parameters and evaluation of mappings.
    """

    from struphy.geometry import domains
    import numpy as np
    

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "evaluate"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    domain_class = getattr(domains, mapping)
    domain = domain_class()
    print()
    print('Domain object set.')

    print('domain\'s kind_map   :', domain.kind_map)
    print('domain\'s params_map :', domain.params_map)

    # point-wise evaluation:
    print('pointwise evaluation:', domain.evaluate(.5, .5, .5, 'x'))
    assert isinstance(domain.evaluate(.5, .5, .5, 'x'), float)
    assert isinstance(domain.evaluate(.5, .5, .5, 'y'), float)
    assert isinstance(domain.evaluate(.5, .5, .5, 'z'), float)

    # flat evaluation:
    print('flat evaluation, shape:', domain.evaluate(arr1, arr2[:-1], arr3[:-2], 'x', flat_eval=True).shape)
    assert domain.evaluate(arr1, arr2[:-1], arr3[:-2], 'x', flat_eval=True).shape == arr1.shape
    assert domain.evaluate(arr1, arr2[:-1], arr3[:-2], 'y', flat_eval=True).shape == arr1.shape
    assert domain.evaluate(arr1, arr2[:-1], arr3[:-2], 'z', flat_eval=True).shape == arr1.shape

    # eta1-array evaluation:
    print('eta1 array evaluation, shape:', domain.evaluate(arr1, .5, .5, 'x').shape)
    assert domain.evaluate(arr1, .5, .5, 'x').shape == arr1.shape
    assert domain.evaluate(arr1, .5, .5, 'y').shape == arr1.shape
    assert domain.evaluate(arr1, .5, .5, 'z').shape == arr1.shape
    # eta2-array evaluation:
    print('eta2 array evaluation, shape:', domain.evaluate(.5, arr2, .5, 'x').shape)
    assert domain.evaluate(.5, arr2, .5, 'x').shape == arr2.shape
    assert domain.evaluate(.5, arr2, .5, 'y').shape == arr2.shape
    assert domain.evaluate(.5, arr2, .5, 'z').shape == arr2.shape
    # eta3-array evaluation:
    print('eta3 array evaluation, shape:', domain.evaluate(.5, .5, arr3, 'x').shape)
    assert domain.evaluate(.5, .5, arr3, 'x').shape == arr3.shape
    assert domain.evaluate(.5, .5, arr3, 'y').shape == arr3.shape
    assert domain.evaluate(.5, .5, arr3, 'z').shape == arr3.shape

    # eta1-eta2-array evaluation:
    a = domain.evaluate(arr1, arr2, .5, 'x')
    b = domain.evaluate(arr1, arr2, .5, 'y')
    c = domain.evaluate(arr1, arr2, .5, 'z')
    print('eta1-eta2 array evaluation, shape:', a.shape)
    assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
    assert b.shape[0] == arr1.size and b.shape[1] == arr2.size
    assert c.shape[0] == arr1.size and c.shape[1] == arr2.size
    # eta1-eta3-array evaluation:
    a = domain.evaluate(arr1, .5, arr3, 'x')
    b = domain.evaluate(arr1, .5, arr3, 'y')
    c = domain.evaluate(arr1, .5, arr3, 'z')
    print('eta1-eta3 array evaluation, shape:', a.shape)
    assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
    assert b.shape[0] == arr1.size and b.shape[1] == arr3.size
    assert c.shape[0] == arr1.size and c.shape[1] == arr3.size
    # eta2-eta3-array evaluation:
    a = domain.evaluate(.5, arr2, arr3, 'x')
    b = domain.evaluate(.5, arr2, arr3, 'y')
    c = domain.evaluate(.5, arr2, arr3, 'z')
    print('eta2-eta3 array evaluation, shape:', a.shape)
    assert a.shape[0] == arr2.size and a.shape[1] == arr3.size
    assert b.shape[0] == arr2.size and b.shape[1] == arr3.size
    assert c.shape[0] == arr2.size and c.shape[1] == arr3.size

    # eta1-eta2-eta3 array evaluation:
    a = domain.evaluate(arr1, arr2, arr3, 'x')
    b = domain.evaluate(arr1, arr2, arr3, 'y')
    c = domain.evaluate(arr1, arr2, arr3, 'z')
    print('eta1-eta2-eta3-array evaluation, shape:', a.shape)
    assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 
    assert b.shape[0] == arr1.size and b.shape[1] == arr2.size and b.shape[2] == arr3.size
    assert c.shape[0] == arr1.size and c.shape[1] == arr2.size and c.shape[2] == arr3.size

    # matrix evaluations at one point in third direction
    mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
    mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
    mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

    # eta1-eta2 matrix evaluation:
    a = domain.evaluate(mat12_x, mat12_y, .5, 'x')
    b = domain.evaluate(mat12_x, mat12_y, .5, 'y')
    c = domain.evaluate(mat12_x, mat12_y, .5, 'z')
    print('eta1-eta2 matrix evaluation, shape:', a.shape)
    assert a.shape == mat12_x.shape
    assert b.shape == mat12_x.shape
    assert c.shape == mat12_x.shape
    # eta1-eta3 matrix evaluation:
    a = domain.evaluate(mat13_x, .5, mat13_z, 'x')
    b = domain.evaluate(mat13_x, .5, mat13_z, 'y')
    c = domain.evaluate(mat13_x, .5, mat13_z, 'z')
    print('eta1-eta3 matrix evaluation, shape:', a.shape)
    assert a.shape == mat13_x.shape
    assert b.shape == mat13_x.shape
    assert c.shape == mat13_x.shape
    # eta2-eta3 matrix evaluation:
    a = domain.evaluate(.5, mat23_y, mat23_z, 'x')
    b = domain.evaluate(.5, mat23_y, mat23_z, 'y')
    c = domain.evaluate(.5, mat23_y, mat23_z, 'z')
    print('eta2-eta3 matrix evaluation, shape:', a.shape)
    assert a.shape == mat23_y.shape
    assert b.shape == mat23_y.shape
    assert c.shape == mat23_y.shape

    # matrix evaluations for sparse meshgrid
    mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
    a = domain.evaluate(mat_x, mat_y, mat_z, 'x')
    b = domain.evaluate(mat_x, mat_y, mat_z, 'y')
    c = domain.evaluate(mat_x, mat_y, mat_z, 'z')
    print('sparse meshgrid matrix evaluation, shape:', a.shape)
    assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]
    assert b.shape[0] == mat_x.shape[0] and b.shape[1] == mat_y.shape[1] and b.shape[2] == mat_z.shape[2]
    assert c.shape[0] == mat_x.shape[0] and c.shape[1] == mat_y.shape[1] and c.shape[2] == mat_z.shape[2]

    # matrix evaluations 
    mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
    a = domain.evaluate(mat_x, mat_y, mat_z, 'x')
    b = domain.evaluate(mat_x, mat_y, mat_z, 'y')
    c = domain.evaluate(mat_x, mat_y, mat_z, 'z')
    print('matrix evaluation, shape:', a.shape)
    assert a.shape == mat_x.shape 
    assert b.shape == mat_x.shape
    assert c.shape == mat_x.shape    
    
    
def test_pullback():
    """ Tests pullbacks to p-forms.
    """

    from struphy.geometry import domains
    import numpy as np

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "pull"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    # physical function to pull back (used as components of forms too):
    fun = lambda x, y, z: np.exp(x)*np.sin(y)*np.cos(z)

    domain_class = getattr(domains, 'Colella')
    domain = domain_class()
    print()
    print('Domain object set.')

    print('domain\'s kind_map   :', domain.kind_map)
    print('domain\'s params_map :', domain.params_map)

    for p_str in domain.keys_pull:

        print('component:', p_str)

        if p_str == '0_form' or p_str == '3_form':
            fun_form = fun
        else:
            fun_form = [fun, fun, fun]

        # point-wise pullback:
        assert isinstance(domain.pull(fun_form, .5, .5, .5, p_str), float)
        #print('pointwise pullback, size:', domain.pull(fun_form, .5, .5, .5, p_str).size)

        # flat pullback:
        assert domain.pull(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
        assert domain.pull(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
        assert domain.pull(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape

        # eta1-array pullback:
        #print('eta1 array pullback, shape:', domain.pull(fun_form, arr1, .5, .5, p_str).shape)
        assert domain.pull(fun_form, arr1, .5, .5, p_str).shape == arr1.shape
        # eta2-array pullback:
        #print('eta2 array pullback, shape:', domain.pull(fun_form, .5, arr2, .5, p_str).shape)
        assert domain.pull(fun_form, .5, arr2, .5, p_str).shape == arr2.shape
        # eta3-array pullback:
        #print('eta3 array pullback, shape:', domain.pull(fun_form, .5, .5, arr3, p_str).shape)
        assert domain.pull(fun_form, .5, .5, arr3, p_str).shape == arr3.shape

        # eta1-eta2-array pullback:
        a = domain.pull(fun_form, arr1, arr2, .5, p_str)
        #print('eta1-eta2 array pullback, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
        # eta1-eta3-array pullback:
        a = domain.pull(fun_form, arr1, .5, arr3, p_str)
        #print('eta1-eta3 array pullback, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
        # eta2-eta3-array pullback:
        a = domain.pull(fun_form, .5, arr2, arr3, p_str)
        #print('eta2-eta3 array pullback, shape:', a.shape)
        assert a.shape[0] == arr2.size and a.shape[1] == arr3.size

        # eta1-eta2-eta3 array pullback:
        a = domain.pull(fun_form, arr1, arr2, arr3, p_str)
        #print('eta1-eta2-eta3-array pullback, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 

        # matrix pullbacks at one point in third direction
        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

        # eta1-eta2 matrix pullback:
        a = domain.pull(fun_form, mat12_x, mat12_y, .5, p_str)
        #print('eta1-eta2 matrix pullback, shape:', a.shape)
        assert a.shape == mat12_x.shape
        # eta1-eta3 matrix pullback:
        a = domain.pull(fun_form, mat13_x, .5, mat13_z, p_str)
        #print('eta1-eta3 matrix pullback, shape:', a.shape)
        assert a.shape == mat13_x.shape
        # eta2-eta3 matrix pullback:
        a = domain.pull(fun_form, .5, mat23_y, mat23_z, p_str)
        #print('eta2-eta3 matrix pullback, shape:', a.shape)
        assert a.shape == mat23_y.shape

        # matrix pullbacks for sparse meshgrid
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
        a = domain.pull(fun_form, mat_x, mat_y, mat_z, p_str)
        #print('sparse meshgrid matrix pullback, shape:', a.shape)
        assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]

        # matrix pullbacks 
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
        a = domain.pull(fun_form, mat_x, mat_y, mat_z, p_str)
        #print('matrix pullback, shape:', a.shape)
        assert a.shape == mat_x.shape     
    

def test_pushforward():
    """ Tests pushforward of p-forms.
    """

    from struphy.geometry import domains
    import numpy as np

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "push"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    # logical function to push forward (used as components of forms too):
    fun = lambda x, y, z : np.exp(x)*np.sin(y)*np.cos(z)

    domain_class = getattr(domains, 'Colella')
    domain = domain_class()
    print()
    print('Domain object set.')

    print('domain\'s kind_map   :', domain.kind_map)
    print('domain\'s params_map :', domain.params_map)

    for p_str in domain.keys_push:

        print('component:', p_str)

        if p_str == '0_form' or p_str == '3_form':
            fun_form = fun
        else:
            fun_form = [fun, fun, fun]

        # point-wise pushforward:
        assert isinstance(domain.push(fun_form, .5, .5, .5, p_str), float)
        #print('pointwise pushforward, size:', domain.push(fun_form, .5, .5, .5, p_str).size)

        # flat pushforward:
        assert domain.push(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
        assert domain.push(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
        assert domain.push(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape

        # eta1-array pushforward:
        #print('eta1 array pushforward, shape:', domain.push(fun_form, arr1, .5, .5, p_str).shape)
        assert domain.push(fun_form, arr1, .5, .5, p_str).shape == arr1.shape
        # eta2-array pushforward:
        #print('eta2 array pushforward, shape:', domain.push(fun_form, .5, arr2, .5, p_str).shape)
        assert domain.push(fun_form, .5, arr2, .5, p_str).shape == arr2.shape
        # eta3-array pushforward:
        #print('eta3 array pushforward, shape:', domain.push(fun_form, .5, .5, arr3, p_str).shape)
        assert domain.push(fun_form, .5, .5, arr3, p_str).shape == arr3.shape

        # eta1-eta2-array pushforward:
        a = domain.push(fun_form, arr1, arr2, .5, p_str)
        #print('eta1-eta2 array pushforward, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
        # eta1-eta3-array pushforward:
        a = domain.push(fun_form, arr1, .5, arr3, p_str)
        #print('eta1-eta3 array pushforward, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
        # eta2-eta3-array pushforward:
        a = domain.push(fun_form, .5, arr2, arr3, p_str)
        #print('eta2-eta3 array pushforward, shape:', a.shape)
        assert a.shape[0] == arr2.size and a.shape[1] == arr3.size

        # eta1-eta2-eta3 array pushforward:
        a = domain.push(fun_form, arr1, arr2, arr3, p_str)
        #print('eta1-eta2-eta3-array pushforward, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 

        # matrix pullbacks at one point in third direction
        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

        # eta1-eta2 matrix pushforward:
        a = domain.push(fun_form, mat12_x, mat12_y, .5, p_str)
        #print('eta1-eta2 matrix pushforward, shape:', a.shape)
        assert a.shape == mat12_x.shape
        # eta1-eta3 matrix pushforward:
        a = domain.push(fun_form, mat13_x, .5, mat13_z, p_str)
        #print('eta1-eta3 matrix pushforward, shape:', a.shape)
        assert a.shape == mat13_x.shape
        # eta2-eta3 matrix pushforward:
        a = domain.push(fun_form, .5, mat23_y, mat23_z, p_str)
        #print('eta2-eta3 matrix pushforward, shape:', a.shape)
        assert a.shape == mat23_y.shape

        # matrix pullbacks for sparse meshgrid
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
        a = domain.push(fun_form, mat_x, mat_y, mat_z, p_str)
        #print('sparse meshgrid matrix pushforward, shape:', a.shape)
        assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]

        # matrix pullbacks 
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
        a = domain.push(fun_form, mat_x, mat_y, mat_z, p_str)
        #print('matrix pushforward, shape:', a.shape)
        assert a.shape == mat_x.shape 


def test_transform():
    """ Tests transformation of p-forms.
    """

    from struphy.geometry import domains
    import numpy as np

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    print()
    print('Testing "transform"...')
    print('array shapes:', arr1.shape, arr2.shape, arr3.shape)

    # logical function to tranform (used as components of forms too):
    fun = lambda eta1, eta2, eta3: np.exp(eta1)*np.sin(eta2)*np.cos(eta3)
    
    domain_class = getattr(domains, 'Colella')
    domain = domain_class()
    print()
    print('Domain object set.')

    print('domain\'s kind_map   :', domain.kind_map)
    print('domain\'s params_map :', domain.params_map)

    for p_str in domain.keys_transform:

        print('component:', p_str)

        if p_str == '0_to_3' or p_str == '3_to_0':
            fun_form = fun
        else:
            fun_form = [fun, fun, fun]

        # point-wise transformation:
        assert isinstance(domain.transform(fun_form, .5, .5, .5, p_str), float)
        #print('pointwise transformation, size:', domain.transform(fun_form, .5, .5, .5, p_str).size)

        # flat transformation:
        assert domain.transform(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
        assert domain.transform(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape
        assert domain.transform(fun_form, arr1, arr2[:-1], arr3[:-2], p_str, flat_eval=True).shape == arr1.shape

        # eta1-array transformation:
        #print('eta1 array transformation, shape:', domain.transform(fun_form, arr1, .5, .5, p_str).shape)
        assert domain.transform(fun_form, arr1, .5, .5, p_str).shape == arr1.shape
        # eta2-array transformation:
        #print('eta2 array transformation, shape:', domain.transform(fun_form, .5, arr2, .5, p_str).shape)
        assert domain.transform(fun_form, .5, arr2, .5, p_str).shape == arr2.shape
        # eta3-array transformation:
        #print('eta3 array transformation, shape:', domain.transform(fun_form, .5, .5, arr3, p_str).shape)
        assert domain.transform(fun_form, .5, .5, arr3, p_str).shape == arr3.shape

        # eta1-eta2-array transformation:
        a = domain.transform(fun_form, arr1, arr2, .5, p_str)
        #print('eta1-eta2 array transformation, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size
        # eta1-eta3-array transformation:
        a = domain.transform(fun_form, arr1, .5, arr3, p_str)
        #print('eta1-eta3 array transformation, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr3.size
        # eta2-eta3-array transformation:
        a = domain.transform(fun_form, .5, arr2, arr3, p_str)
        #print('eta2-eta3 array transformation, shape:', a.shape)
        assert a.shape[0] == arr2.size and a.shape[1] == arr3.size

        # eta1-eta2-eta3 array transformation:
        a = domain.transform(fun_form, arr1, arr2, arr3, p_str)
        #print('eta1-eta2-eta3-array transformation, shape:', a.shape)
        assert a.shape[0] == arr1.size and a.shape[1] == arr2.size and a.shape[2] == arr3.size 

        # matrix transformation at one point in third direction
        mat12_x, mat12_y = np.meshgrid(arr1, arr2, indexing='ij')
        mat13_x, mat13_z = np.meshgrid(arr1, arr3, indexing='ij')
        mat23_y, mat23_z = np.meshgrid(arr2, arr3, indexing='ij')

        # eta1-eta2 matrix transformation:
        a = domain.transform(fun_form, mat12_x, mat12_y, .5, p_str)
        #print('eta1-eta2 matrix transformation, shape:', a.shape)
        assert a.shape == mat12_x.shape
        # eta1-eta3 matrix transformation:
        a = domain.transform(fun_form, mat13_x, .5, mat13_z, p_str)
        #print('eta1-eta3 matrix transformation, shape:', a.shape)
        assert a.shape == mat13_x.shape
        # eta2-eta3 matrix transformation:
        a = domain.transform(fun_form, .5, mat23_y, mat23_z, p_str)
        #print('eta2-eta3 matrix transformation, shape:', a.shape)
        assert a.shape == mat23_y.shape

        # matrix transformation for sparse meshgrid
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij', sparse=True)
        a = domain.transform(fun_form, mat_x, mat_y, mat_z, p_str)
        #print('sparse meshgrid matrix transformation, shape:', a.shape)
        assert a.shape[0] == mat_x.shape[0] and a.shape[1] == mat_y.shape[1] and a.shape[2] == mat_z.shape[2]

        # matrix transformation 
        mat_x, mat_y, mat_z = np.meshgrid(arr1, arr2, arr3, indexing='ij')
        a = domain.transform(fun_form, mat_x, mat_y, mat_z, p_str)
        #print('matrix transformation, shape:', a.shape)
        assert a.shape == mat_x.shape     


if __name__ == '__main__':
    test_prepare_arg()
    test_evaluation_mappings('Colella')
    test_pullback()
    test_pushforward()
    test_transform()