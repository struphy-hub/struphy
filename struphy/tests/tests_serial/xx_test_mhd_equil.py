def test_slab():

    import sysconfig
    import yaml 
    import numpy as np

    from struphy.geometry         import domain_3d
    from struphy.mhd_equil        import mhd_equil_physical 
    from struphy.mhd_equil        import mhd_equil_logical 

    file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
    print(file_in)

    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Domain:
    DOMAIN = domain_3d.Domain(params['geometry']['type'], 
                              params['geometry'][params['geometry']['type']])
    print('Domain object set.')

    # MHD equilibirum (physical)
    EQ_MHD_P = mhd_equil_physical.Equilibrium_mhd_physical(params['mhd_equilibrium']['general']['type'], 
         params['mhd_equilibrium'][params['mhd_equilibrium']['general']['type']])
    print('MHD equilibrium (physical) set.')
    
    # MHD equilibrium (logical)
    EQ_MHD_L = mhd_equil_logical.Equilibrium_mhd_logical(DOMAIN, EQ_MHD_P)
    print('MHD equilibrium (logical) set.')

    # point-wise evaluation:
    print('point-wise evaluation:')
    b1, e1, b2, e2, b3, e3 = DOMAIN.params_map
    print(EQ_MHD_P.p_eq(b1, b2, b3))
    print(EQ_MHD_L.p0_eq(0., 0., 0.))

    # arrays:
    arr1 = np.linspace(0., 1., 4)
    arr2 = np.linspace(0., 1., 5)
    arr3 = np.linspace(0., 1., 6)
    
    arr_x = np.linspace(b1, e1, 4)
    arr_y = np.linspace(b2, e2, 5)
    arr_z = np.linspace(b3, e3, 6)

    # eta1-array evaluation:
    print('eta1-array evaluation:')
    print(EQ_MHD_P.p_eq(arr_x, b2, b3))
    print(EQ_MHD_L.p0_eq(arr1, 0., 0.))
    assert EQ_MHD_P.p_eq(arr_x, b2, b3).shape == arr_x.shape
    assert EQ_MHD_L.p0_eq(arr1, 0., 0.).shape == arr1.shape

    # eta2-array evaluation:
    print('eta2-array evaluation:')
    # TODO: fix this evaluation
    # print(EQ_MHD.p_eq(b1, arr_y, b3))
    # print(EQ_MHD.p0_eq(0., arr2, 0.))
    # assert EQ_MHD.p_eq(b1, arr_y, b3).shape == arr_y.shape
    # assert EQ_MHD.p0_eq(0., arr2, 0.).shape == arr2.shape

    # eta1-eta2-array evaluation:
    print('eta1-eta2-array evaluation:')
    # TODO: fix p_eq evaluation
    # a = EQ_MHD.p_eq(arr_x, arr_y, b3)
    # b = EQ_MHD.p0_eq(arr1, arr2, 0.)
    # print(a)
    # print(b)
    # assert a.shape[0] == arr_x.size and a.shape[1] == arr_y.size
    # assert b.shape[0] == arr1.size and b.shape[1] == arr2.size

if __name__ == '__main__':
    test_slab()