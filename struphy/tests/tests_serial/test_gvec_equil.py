import numpy as np

def test_gvec_equil():
    '''Test the workflow of creating a gvec mhd equilibirum and compares struphy with gvec_to_python evaluations.'''

    from struphy.fields_background.mhd_equil.numerical import GVECequilibrium
    from gvec_to_python.reader.gvec_reader import create_GVEC_json
    from gvec_to_python import GVEC
    from gvec_to_python.geometry.domain import GVEC_domain

    import numpy as np

    # struphy discrete equilibirum
    mhd_equil = GVECequilibrium()

    # gvec continuous equilibirum
    import struphy as _
    dat_file_in = _.__path__[0] + '/fields_background/mhd_equil/gvec' + mhd_equil.params['dat_file']
    json_file_out = dat_file_in[:-4] + '.json'

    print(json_file_out)

    create_GVEC_json(dat_file_in, json_file_out)
    gvec = GVEC(json_file_out, mapping='unit', unit_tor_domain="one-fp", use_pyccel=True)

    print(mhd_equil.params)

    e1 = np.linspace(0.001, 1, 10) 
    e2 = np.linspace(0, 1, 10) 
    e3 = np.linspace(0, 1, 10)

    # mapping 
    print('mapping:')
    #print(rel_err(gvec.f(e1, e2, e3)[0], mhd_equil.domain(e1, e2, e3)[0]))
    assert rel_err(gvec.f(e1, e2, e3)[0], mhd_equil.domain(e1, e2, e3)[0]) < 2e-4
    #print(rel_err(gvec.f(e1, e2, e3)[1], mhd_equil.domain(e1, e2, e3)[1]))
    assert rel_err(gvec.f(e1, e2, e3)[1], mhd_equil.domain(e1, e2, e3)[1]) < 4e-4
    #print(rel_err(gvec.f(e1, e2, e3)[2], mhd_equil.domain(e1, e2, e3)[2]))
    assert rel_err(gvec.f(e1, e2, e3)[2], mhd_equil.domain(e1, e2, e3)[2]) < 2e-3

    # Jacobian 
    struphy_df = GVEC_domain.swap_J_axes(mhd_equil.domain.jacobian(e1, e2, e3))
    #print(rel_err(gvec.df(e1, e2, e3), struphy_df))
    assert rel_err(gvec.df(e1, e2, e3), struphy_df) < 3e-3

    # Jacobian determinant 
    #print(rel_err(gvec.det_df(e1, e2, e3), mhd_equil.domain.jacobian_det(e1, e2, e3)))
    assert rel_err(gvec.det_df(e1, e2, e3), mhd_equil.domain.jacobian_det(e1, e2, e3)) < 4e-3

    # Inverse Jacobian 
    struphy_df_inv = GVEC_domain.swap_J_axes(mhd_equil.domain.jacobian_inv(e1, e2, e3))
    #print(rel_err(gvec.df_inv(e1, e2, e3), struphy_df_inv))
    assert rel_err(gvec.df_inv(e1, e2, e3), struphy_df_inv) < 4e-3

    # Metric tensor
    struphy_g = GVEC_domain.swap_J_axes(mhd_equil.domain.metric(e1, e2, e3))
    #print(rel_err(gvec.g(e1, e2, e3), struphy_g))
    assert rel_err(gvec.g(e1, e2, e3), struphy_g) < 3e-3

    # Inverse metric tensor
    struphy_g_inv = GVEC_domain.swap_J_axes(mhd_equil.domain.metric_inv(e1, e2, e3))
    #print(rel_err(gvec.g_inv(e1, e2, e3), struphy_g_inv))
    assert rel_err(gvec.g_inv(e1, e2, e3), struphy_g_inv) < 4e-3

    # equilibrium
    print('p:')
    #print(rel_err(gvec.p0(e1, e2, e3), mhd_equil.p0(e1, e2, e3)))
    assert rel_err(gvec.p0(e1, e2, e3), mhd_equil.p0(e1, e2, e3)) < 1e-16
    #print(rel_err(gvec.p3(e1, e2, e3), mhd_equil.p3(e1, e2, e3)))
    assert rel_err(gvec.p3(e1, e2, e3), mhd_equil.p3(e1, e2, e3)) < 3e-3

    print('bv:')
    assert np.all(gvec.bv(e1, e2, e3)[0] == 0.)
    assert np.all(mhd_equil.bv_1(e1, e2, e3) == 0.)
    #print(rel_err(gvec.bv(e1, e2, e3)[1], mhd_equil.bv_2(e1, e2, e3)))
    assert rel_err(gvec.bv(e1, e2, e3)[1], mhd_equil.bv_2(e1, e2, e3)) < 4e-3
    #print(rel_err(gvec.bv(e1, e2, e3)[2], mhd_equil.bv_3(e1, e2, e3)))
    assert rel_err(gvec.bv(e1, e2, e3)[2], mhd_equil.bv_3(e1, e2, e3)) < 3e-3

    print('b1:')
    #print(rel_err(gvec.b1(e1, e2, e3)[0], mhd_equil.b1_1(e1, e2, e3)))
    assert rel_err(gvec.b1(e1, e2, e3)[0], mhd_equil.b1_1(e1, e2, e3)) < 4e-2
    #print(rel_err(gvec.b1(e1, e2, e3)[1], mhd_equil.b1_2(e1, e2, e3)))
    assert rel_err(gvec.b1(e1, e2, e3)[1], mhd_equil.b1_2(e1, e2, e3)) < 9e-3
    #print(rel_err(gvec.b1(e1, e2, e3)[2], mhd_equil.b1_3(e1, e2, e3)))
    assert rel_err(gvec.b1(e1, e2, e3)[2], mhd_equil.b1_3(e1, e2, e3)) < 5e-3

    print('b2:')
    assert np.all(gvec.b2(e1, e2, e3)[0] == 0.)
    assert np.all(mhd_equil.b2_1(e1, e2, e3) == 0.)
    #print(rel_err(gvec.b2(e1, e2, e3)[1], mhd_equil.b2_2(e1, e2, e3)))
    assert rel_err(gvec.b2(e1, e2, e3)[1], mhd_equil.b2_2(e1, e2, e3)) < 1e-16
    #print(rel_err(gvec.b2(e1, e2, e3)[2], mhd_equil.b2_3(e1, e2, e3)))
    assert rel_err(gvec.b2(e1, e2, e3)[2], mhd_equil.b2_3(e1, e2, e3)) < 1e-16

    print('b_cart:')
    #print(rel_err(gvec.b_cart(e1, e2, e3)[0][0], mhd_equil.b_cart_1(e1, e2, e3)[0]))
    assert rel_err(gvec.b_cart(e1, e2, e3)[0][0], mhd_equil.b_cart_1(e1, e2, e3)[0]) < 4e-3
    #print(rel_err(gvec.b_cart(e1, e2, e3)[0][1], mhd_equil.b_cart_2(e1, e2, e3)[0]))
    assert rel_err(gvec.b_cart(e1, e2, e3)[0][1], mhd_equil.b_cart_2(e1, e2, e3)[0]) < 3e-3
    #print(rel_err(gvec.b_cart(e1, e2, e3)[0][2], mhd_equil.b_cart_3(e1, e2, e3)[0]))
    assert rel_err(gvec.b_cart(e1, e2, e3)[0][2], mhd_equil.b_cart_3(e1, e2, e3)[0]) < 6e-3

    print('jv:')
    #print(rel_err(gvec.jv(e1, e2, e3)[0], mhd_equil.jv_1(e1, e2, e3)))
    assert rel_err(gvec.jv(e1, e2, e3)[0], mhd_equil.jv_1(e1, e2, e3)) < 4e-1
    # print(rel_err(gvec.jv(e1, e2, e3)[1], mhd_equil.jv_2(e1, e2, e3)))
    # assert rel_err(gvec.jv(e1, e2, e3)[1], mhd_equil.jv_2(e1, e2, e3)) < 4e-1
    # print(rel_err(gvec.jv(e1, e2, e3)[2], mhd_equil.jv_3(e1, e2, e3)))
    # assert rel_err(gvec.jv(e1, e2, e3)[2], mhd_equil.jv_3(e1, e2, e3)) < 3e-1

    # print('j1:')
    # print(rel_err(gvec.j1(e1, e2, e3)[0], mhd_equil.j1_1(e1, e2, e3)))
    # assert rel_err(gvec.j1(e1, e2, e3)[0], mhd_equil.j1_1(e1, e2, e3)) < 4e-1
    # print(rel_err(gvec.j1(e1, e2, e3)[1], mhd_equil.j1_2(e1, e2, e3)))
    # assert rel_err(gvec.j1(e1, e2, e3)[1], mhd_equil.j1_2(e1, e2, e3)) < 9e-1
    # print(rel_err(gvec.j1(e1, e2, e3)[2], mhd_equil.j1_3(e1, e2, e3)))
    # assert rel_err(gvec.j1(e1, e2, e3)[2], mhd_equil.j1_3(e1, e2, e3)) < 5e-1

    # print('j2:')
    # print(rel_err(gvec.j2(e1, e2, e3)[0], mhd_equil.j2_1(e1, e2, e3)))
    # assert rel_err(gvec.j2(e1, e2, e3)[0], mhd_equil.j2_1(e1, e2, e3)) < 1e-16
    # print(rel_err(gvec.j2(e1, e2, e3)[1], mhd_equil.j2_2(e1, e2, e3)))
    # assert rel_err(gvec.j2(e1, e2, e3)[1], mhd_equil.j2_2(e1, e2, e3)) < 1e-16
    # print(rel_err(gvec.j2(e1, e2, e3)[2], mhd_equil.j2_3(e1, e2, e3)))
    # assert rel_err(gvec.j2(e1, e2, e3)[2], mhd_equil.j2_3(e1, e2, e3)) < 1e-16

    # print('j_cart:')
    # print(rel_err(gvec.j_cart(e1, e2, e3)[0][0], mhd_equil.j_cart_1(e1, e2, e3)[0]))
    # assert rel_err(gvec.j_cart(e1, e2, e3)[0][0], mhd_equil.j_cart_1(e1, e2, e3)[0]) < 4e-3
    # print(rel_err(gvec.j_cart(e1, e2, e3)[0][1], mhd_equil.j_cart_2(e1, e2, e3)[0]))
    # assert rel_err(gvec.j_cart(e1, e2, e3)[0][1], mhd_equil.j_cart_2(e1, e2, e3)[0]) < 3e-3
    # print(rel_err(gvec.j_cart(e1, e2, e3)[0][2], mhd_equil.j_cart_3(e1, e2, e3)[0]))
    # assert rel_err(gvec.j_cart(e1, e2, e3)[0][2], mhd_equil.j_cart_3(e1, e2, e3)[0]) < 6e-3

def rel_err(a, b):
    assert a.shape == b.shape
    return np.max(np.abs(a - b)) / np.max(np.abs(a)) 

if __name__ == '__main__':
    test_gvec_equil()
