def test_physical_conditions():
    '''
    TEST3 : push forward of INIT function defined with 'physical' conditions == evaluation at the physical domain
    '''

    import sysconfig
    import yaml 
    import numpy as np

    from struphy.geometry import domain_3d 
    from struphy.feec     import spline_space 
    
    from  struphy.mhd_init import mhd_init

    
    file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
    
    print(file_in)

    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # domain
    DOMAIN   = domain_3d.Domain(params['geometry']['type'], 
                                params['geometry']['params_' + params['geometry']['type']])
    
    print('DOMAIN set')
    
    # spline spaces
    Nel      = params['grid']['Nel']
    p        = params['grid']['p']
    spl_kind = params['grid']['spl_kind']
    nq_el    = params['grid']['nq_el']
    nq_pr    = params['grid']['nq_pr']
    bc       = params['grid']['bc'] 

    spaces_FEM = [spline_space.Spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc) 
                                           for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, nq_el)]

    [space.set_projectors(params['grid']['nq_pr'][0]) for space in spaces_FEM]

    if params['grid']['polar']:
        SPACES = spline_space.Tensor_spline_space(spaces_FEM, ck=1, cx=DOMAIN.cx, cy=DOMAIN.cy)
    else:
        SPACES = spline_space.Tensor_spline_space(spaces_FEM)

    SPACES.set_projectors('general')
    print('FEEC spaces and projectors set.')
    print()
    
    # mhd_init
    INIT = mhd_init.Initialize_mhd(DOMAIN, SPACES, params['mhd_init']['general'],
                                                   params['mhd_init']['params_' + params['mhd_init']['general']['type']])
    print('INIT is defined.')
    
    print('test conditions')
    print('type : ',   INIT.init_type)
    print('coords : ', INIT.init_coords)
    print('target : ', INIT.target)
    print('kx : ',     INIT.modes_k[0])
    print('ky : ',     INIT.modes_k[1])
    print('kz : ',     INIT.modes_k[2])
    print('amp : ',    INIT.amp)
    print() 

    # define the callable functions of initial conditions at the logical domain
    R0_ini =  INIT.r0_ini
    P0_ini =  INIT.p0_ini
    R3_ini =  INIT.r3_ini
    P3_ini =  INIT.p3_ini
    B2_ini = [INIT.b2_ini_1, INIT.b2_ini_2, INIT.b2_ini_3]
    # Uv_ini = [INIT.uv_ini_1, INIT.uv_ini_2, INIT.uv_ini_3]
    U1_ini = [INIT.u1_ini_1, INIT.u1_ini_2, INIT.u1_ini_3]
    U2_ini = [INIT.u2_ini_1, INIT.u2_ini_2, INIT.u2_ini_3]
    
    # eval point set (logical)
    eta1 = np.linspace(0., 1., 30)
    eta2 = np.linspace(0., 1., 30)
    eta3 = np.linspace(0., 1., 30)

    # evaluation of the pushed quantities at eval point set
    pushed_R0_ini = DOMAIN.push(R0_ini, eta1, eta2, eta3, kind_fun = '0_form')
    pushed_P0_ini = DOMAIN.push(P0_ini, eta1, eta2, eta3, kind_fun = '0_form')
    pushed_R3_ini = DOMAIN.push(R3_ini, eta1, eta2, eta3, kind_fun = '3_form')
    pushed_P3_ini = DOMAIN.push(P3_ini, eta1, eta2, eta3, kind_fun = '3_form')

    # pushed_Uv_ini_1 = DOMAIN.push(Uv_ini, eta1, eta2, eta3, kind_fun = 'vector_1')
    # pushed_Uv_ini_2 = DOMAIN.push(Uv_ini, eta1, eta2, eta3, kind_fun = 'vector_2')
    # pushed_Uv_ini_3 = DOMAIN.push(Uv_ini, eta1, eta2, eta3, kind_fun = 'vector_3')

    pushed_U1_ini_1 = DOMAIN.push(U1_ini, eta1, eta2, eta3, kind_fun = '1_form_1')
    pushed_U1_ini_2 = DOMAIN.push(U1_ini, eta1, eta2, eta3, kind_fun = '1_form_2')
    pushed_U1_ini_3 = DOMAIN.push(U1_ini, eta1, eta2, eta3, kind_fun = '1_form_3')

    pushed_U2_ini_1 = DOMAIN.push(U2_ini, eta1, eta2, eta3, kind_fun = '2_form_1')
    pushed_U2_ini_2 = DOMAIN.push(U2_ini, eta1, eta2, eta3, kind_fun = '2_form_2')
    pushed_U2_ini_3 = DOMAIN.push(U2_ini, eta1, eta2, eta3, kind_fun = '2_form_3')

    pushed_B2_ini_1 = DOMAIN.push(B2_ini, eta1, eta2, eta3, kind_fun = '2_form_1')
    pushed_B2_ini_2 = DOMAIN.push(B2_ini, eta1, eta2, eta3, kind_fun = '2_form_2')
    pushed_B2_ini_3 = DOMAIN.push(B2_ini, eta1, eta2, eta3, kind_fun = '2_form_3')

    # eval point set (physical)
    X = DOMAIN.evaluate(eta1, eta2, eta3, 'x')
    Y = DOMAIN.evaluate(eta1, eta2, eta3, 'y')
    Z = DOMAIN.evaluate(eta1, eta2, eta3, 'z')

    # draw random indices
    i1 = np.random.randint(0, 29, 5)
    i2 = np.random.randint(0, 29, 5)
    i3 = np.random.randint(0, 29, 5)

    for j1 in i1:
        for j2 in i2:
            for j3 in i3:
                anal_value = INIT.amp[0] * np.sin(INIT.modes_k[0][0]*X[j1,j2,j3] + INIT.modes_k[1][0]*Y[j1,j2,j3] + INIT.modes_k[2][0]*Z[j1,j2,j3])

                # assert np.isclose(pushed_R0_ini[j1,j2,j3],   anal_value, 1e-14)
                # assert np.isclose(pushed_P0_ini[j1,j2,j3],   anal_value, 1e-14)
                # assert np.isclose(pushed_R3_ini[j1,j2,j3],   anal_value, 1e-14)
                # assert np.isclose(pushed_P3_ini[j1,j2,j3],   anal_value, 1e-14)
                # assert np.isclose(pushed_Uv_ini_1[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_Uv_ini_2[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_Uv_ini_3[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_U1_ini_1[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_U1_ini_2[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_U1_ini_3[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_U2_ini_1[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_U2_ini_2[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_U2_ini_3[j1,j2,j3], anal_value, 1e-14) 
                # assert np.isclose(pushed_B2_ini_1[j1,j2,j3], anal_value, 1e-14)
                # assert np.isclose(pushed_B2_ini_2[j1,j2,j3], anal_value, 1e-14)
                assert np.isclose(pushed_B2_ini_3[j1,j2,j3], anal_value, 1e-14)


if __name__ == '__main__':
    test_physical_conditions()