import numpy as np


def test_gvec_equil():
    '''Test the workflow of creating a gvec mhd equilibirum and compares struphy with gvec_to_python evaluations.'''

    from struphy.fields_background.mhd_equil.equils import GVECequilibrium
    from gvec_to_python.reader.gvec_reader import create_GVEC_json
    from gvec_to_python import GVEC
    from gvec_to_python.geometry.domain import GVEC_domain

    import numpy as np
    import os
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD

    # struphy discrete equilibirum
    mhd_equil = GVECequilibrium()

    # gvec continuous equilibirum
    import struphy 
    dat_file_in = os.path.join(struphy.__path__[0], 'fields_background/mhd_equil/gvec', mhd_equil.params['dat_file'])
    json_file_out = dat_file_in[:-4] + '.json'

    print(json_file_out)

    create_GVEC_json(dat_file_in, json_file_out)
    
    # test only on 1 process because of json.load command
    if comm.Get_rank() == 0:
        gvec = GVEC(json_file_out, mapping='unit',
                    unit_tor_domain="one-fp", use_pyccel=True)

        print(mhd_equil.params)

        e1 = np.linspace(0.001, 1, 10)
        e2 = np.linspace(0, 1, 10)
        e3 = np.linspace(0, 1, 10)

        # mapping
        print('mapping:')
        #print(rel_err(gvec.f(e1, e2, e3)[0], mhd_equil.domain(e1, e2, e3)[0]))
        assert rel_err(gvec.f(e1, e2, e3)[0],
                    mhd_equil.domain(e1, e2, e3)[0]) < 2e-4
        #print(rel_err(gvec.f(e1, e2, e3)[1], mhd_equil.domain(e1, e2, e3)[1]))
        assert rel_err(gvec.f(e1, e2, e3)[1],
                    mhd_equil.domain(e1, e2, e3)[1]) < 4e-4
        #print(rel_err(gvec.f(e1, e2, e3)[2], mhd_equil.domain(e1, e2, e3)[2]))
        assert rel_err(gvec.f(e1, e2, e3)[2],
                    mhd_equil.domain(e1, e2, e3)[2]) < 2e-3

        # Jacobian
        struphy_df = GVEC_domain.swap_J_axes(mhd_equil.domain.jacobian(e1, e2, e3))
        #print(rel_err(gvec.df(e1, e2, e3), struphy_df))
        assert rel_err(gvec.df(e1, e2, e3), struphy_df) < 3e-3

        # Jacobian determinant
        #print(rel_err(gvec.det_df(e1, e2, e3), mhd_equil.domain.jacobian_det(e1, e2, e3)))
        assert rel_err(gvec.det_df(e1, e2, e3),
                    mhd_equil.domain.jacobian_det(e1, e2, e3)) < 4e-3

        # Inverse Jacobian
        struphy_df_inv = GVEC_domain.swap_J_axes(
            mhd_equil.domain.jacobian_inv(e1, e2, e3))
        #print(rel_err(gvec.df_inv(e1, e2, e3), struphy_df_inv))
        assert rel_err(gvec.df_inv(e1, e2, e3), struphy_df_inv) < 4e-3

        # Metric tensor
        struphy_g = GVEC_domain.swap_J_axes(mhd_equil.domain.metric(e1, e2, e3))
        #print(rel_err(gvec.g(e1, e2, e3), struphy_g))
        assert rel_err(gvec.g(e1, e2, e3), struphy_g) < 3e-3

        # Inverse metric tensor
        struphy_g_inv = GVEC_domain.swap_J_axes(
            mhd_equil.domain.metric_inv(e1, e2, e3))
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
        assert np.all(mhd_equil.bv(e1, e2, e3)[0] == 0.)
        #print(rel_err(gvec.bv(e1, e2, e3)[1], mhd_equil.bv_2(e1, e2, e3)))
        assert rel_err(gvec.bv(e1, e2, e3)[1], mhd_equil.bv(e1, e2, e3)[1]) < 4e-3
        #print(rel_err(gvec.bv(e1, e2, e3)[2], mhd_equil.bv_3(e1, e2, e3)))
        assert rel_err(gvec.bv(e1, e2, e3)[2], mhd_equil.bv(e1, e2, e3)[2]) < 3e-3

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
        assert rel_err(gvec.b_cart(e1, e2, e3)[0][0],
                    mhd_equil.b_cart(e1, e2, e3)[0][0]) < 4e-3
        #print(rel_err(gvec.b_cart(e1, e2, e3)[0][1], mhd_equil.b_cart_2(e1, e2, e3)[0]))
        assert rel_err(gvec.b_cart(e1, e2, e3)[0][1],
                    mhd_equil.b_cart(e1, e2, e3)[0][1]) < 3e-3
        #print(rel_err(gvec.b_cart(e1, e2, e3)[0][2], mhd_equil.b_cart_3(e1, e2, e3)[0]))
        assert rel_err(gvec.b_cart(e1, e2, e3)[0][2],
                    mhd_equil.b_cart(e1, e2, e3)[0][2]) < 6e-3

        print('jv:')
        jv_gvec = gvec.jv(e1, e2, e3)
        jv_struphy = mhd_equil.jv(e1, e2, e3)
        #print(rel_err(jv_gvec[0], jv_struphy[0]))
        assert rel_err(jv_gvec[0], jv_struphy[0]) < 4e-1
        #print(rel_err(jv_gvec[1], jv_struphy[1]))
        assert rel_err(jv_gvec[1], jv_struphy[1]) < 4e-1
        #print(rel_err(jv_gvec[2], jv_struphy[2]))
        assert rel_err(jv_gvec[2], jv_struphy[2]) < 3e-1

        print('j1:')
        j1_gvec = gvec.j1(e1, e2, e3)
        j1_struphy = mhd_equil.j1(e1, e2, e3)
        #print(rel_err(j1_gvec[0], j1_struphy[0]))
        assert rel_err(j1_gvec[0], j1_struphy[0]) < 4e-1
        #print(rel_err(j1_gvec[1], j1_struphy[1]))
        assert rel_err(j1_gvec[1], j1_struphy[1]) < 9e-1
        #print(rel_err(j1_gvec[2], j1_struphy[2]))
        assert rel_err(j1_gvec[2], j1_struphy[2]) < 5e-1

        print('j2:')
        j2_gvec = gvec.j2(e1, e2, e3)
        j2_struphy = mhd_equil.j2(e1, e2, e3)
        #print(rel_err(j2_gvec[0],  j2_struphy[0]))
        assert rel_err(j2_gvec[0], j2_struphy[0]) < 1e-16
        #print(rel_err(j2_gvec[1],  j2_struphy[1]))
        assert rel_err(j2_gvec[1], j2_struphy[1]) < 1e-16
        #print(rel_err(j2_gvec[2],  j2_struphy[2]))
        assert rel_err(j2_gvec[2], j2_struphy[2]) < 1e-16

        print('j_cart:')
        j_cart_gvec = gvec.jv(e1, e2, e3)[0]
        j_cart_struphy = mhd_equil.jv(e1, e2, e3)[0]
        #print(rel_err(j_cart_gvec[0], j_cart_struphy[0]))
        assert rel_err(j_cart_gvec[0], j_cart_struphy[0]) < 4e-3
        #print(rel_err(j_cart_gvec[1], j_cart_struphy[1]))
        assert rel_err(j_cart_gvec[1], j_cart_struphy[1]) < 3e-3
        #print(rel_err(j_cart_gvec[2], j_cart_struphy[2]))
        assert rel_err(j_cart_gvec[2], j_cart_struphy[2]) < 6e-3


def rel_err(a, b):
    assert a.shape == b.shape
    return np.max(np.abs(a - b)) / np.max(np.abs(a))


if __name__ == '__main__':
    test_gvec_equil()
