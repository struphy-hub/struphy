import pytest
import yaml

import struphy.geometry.mappings_3d      as mapping
import struphy.geometry.mappings_3d_fast as mapping_fast
import struphy.feec.bsplines_kernels     as bsp

from struphy.feec     import spline_space
from struphy.geometry import domain_3d 

import numpy as np

@pytest.mark.parametrize('Nel', [[8,5,6], [4, 4, 128]])
@pytest.mark.parametrize('p', [[2,3,2], [1,1,4]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, True]])
@pytest.mark.parametrize('mappings', [
    ['Cuboid', {'l1':0., 'r1':1., 'l2':0., 'r2':1., 'l3':0., 'r3':1.}],
    ['HollowCylinder', {'a1': np.random.rand(), 'a2': np.random.rand(), 'a3': np.random.rand()}],
    ['Colella', {'Lx':1., 'Ly':1., 'Lz':1., 'alpha':np.random.rand()}],
    ['Orthogonal', {'Lx':1., 'Ly':1., 'Lz':1., 'alpha':np.random.rand()}],
    ['HollowTorus', {'a1':np.random.rand(), 'a2':np.random.rand(), 'R0':np.random.rand()}],
    ['EllipticCylinder', {'x0':np.random.rand(), 'y0':np.random.rand(), 'z0':np.random.rand(), 'rx':np.random.rand(), 'ry':np.random.rand(), 'Lz':np.random.rand()}],
    ['RotatedEllipticCylinder', {'x0':np.random.rand(), 'y0':np.random.rand(), 'z0':np.random.rand(), 'r1':np.random.rand(), 'r2':np.random.rand(), 'Lz':np.random.rand(), 'th':np.random.rand()}],
    ['ShafranovSqrtCylinder', {'x0':np.random.rand(), 'y0':np.random.rand(), 'z0':np.random.rand(), 'rx':np.random.rand(), 'ry':np.random.rand(), 'Lz':np.random.rand(), 'delta':np.random.rand()}],
])
@pytest.mark.parametrize('N', [100])
def test_Mappings(Nel, p, spl_kind, mappings, N):
    # ========================================================================================= 
    # FEEC SPACES Object & related quantities
    # =========================================================================================

    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0]) 
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    t1 = SPACES.T[0]
    t2 = SPACES.T[1]
    t3 = SPACES.T[2]

    pn1 = SPACES.p[0]
    pn2 = SPACES.p[1]
    pn3 = SPACES.p[2]

    ind_n1 = SPACES.indN[0]
    ind_n2 = SPACES.indN[1]
    ind_n3 = SPACES.indN[2]
    
    decimals = 12

    map_type = mappings[0]
    params   = mappings[1]

    DOMAIN = domain_3d.Domain(map_type, params)

    # ========================================================================================= 
    # Test mappings_3d
    # =========================================================================================
    for j in range(N):
        eta = np.random.rand(3)

        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        x,y,z = np.random.rand(3)*10

        span1 = bsp.find_span(t1, pn1, eta1)
        span2 = bsp.find_span(t2, pn2, eta2)
        span3 = bsp.find_span(t3, pn3, eta3)

        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3


        # Test mapping vector
        vec_out = np.empty( 3, dtype=float )

        mapping.f_vec(eta1, eta2, eta3, DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), ind_n1[ie1,:], ind_n2[ie2,:], ind_n3[ie3,:], DOMAIN.cx, DOMAIN.cy, DOMAIN.cz, vec_out)

        assert np.isnan(vec_out).any() == False
        assert np.isinf(vec_out).any() == False

        vec_leg = np.empty( 3, dtype=float )

        for comp in range(3):
            vec_leg[comp] = mapping.f(eta1, eta2, eta3, comp+1, DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), np.array(SPACES.NbaseN), DOMAIN.cx, DOMAIN.cy, DOMAIN.cz)

        assert np.isnan(vec_leg).any() == False
        assert np.isinf(vec_leg).any() == False

        assert np.round(np.sum(np.abs( vec_out - vec_leg )), decimals) == 0.

        del(vec_out)
        del(vec_leg)


        # Test inverse mapping vector
        vec_out = np.empty( 3, dtype=float )

        mapping.f_inv_vec(x, y, z, DOMAIN.kind_map, DOMAIN.params_map, vec_out)

        assert np.isnan(vec_out).any() == False
        assert np.isinf(vec_out).any() == False

        vec_leg = np.empty( 3, dtype=float )

        for comp in range(3):
            vec_leg[comp] = mapping.f_inv(x, y, z, comp+1, DOMAIN.kind_map, DOMAIN.params_map)

        assert np.isnan(vec_leg).any() == False
        assert np.isinf(vec_leg).any() == False

        assert np.round(np.sum(np.abs( vec_out - vec_leg )), decimals) == 0.

        del(vec_out)
        del(vec_leg)


        # Test Jacobian matrix
        mat_out = np.empty( (3,3), dtype=float )

        mapping.df_mat(eta1, eta2, eta3, DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), ind_n1[ie1,:], ind_n2[ie2,:], ind_n3[ie3,:], DOMAIN.cx, DOMAIN.cy, DOMAIN.cz, mat_out)

        assert np.isnan(mat_out).any() == False
        assert np.isinf(mat_out).any() == False

        mat_leg = np.empty( (3,3), dtype=float )

        for comp_x in range(3):
            for comp_y in range(3):
                comp = int(str(comp_x+1) + str(comp_y+1))
                mat_leg[comp_x,comp_y] = mapping.df(eta1, eta2, eta3, comp, DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), np.array(SPACES.NbaseN), DOMAIN.cx, DOMAIN.cy, DOMAIN.cz)

        assert np.isnan(mat_leg).any() == False
        assert np.isinf(mat_leg).any() == False

        assert np.round(np.sum(np.abs( mat_out - mat_leg )), decimals) == 0.

        del(mat_out)
        del(mat_leg)


        # Test analytical Jacobian matrix
        mat_out = np.empty( (3,3), dtype=float )

        mapping.df_ana_mat(eta1, eta2, eta3, DOMAIN.kind_map, DOMAIN.params_map, mat_out)

        assert np.isnan(mat_out).any() == False
        assert np.isinf(mat_out).any() == False

        mat_leg = np.empty( (3,3), dtype=float )

        for comp_x in range(3):
            for comp_y in range(3):
                comp = int(str(comp_x+1) + str(comp_y+1))
                mat_leg[comp_x,comp_y] = mapping.df_ana(eta1, eta2, eta3, comp, DOMAIN.kind_map, DOMAIN.params_map)

        assert np.isnan(mat_leg).any() == False
        assert np.isinf(mat_leg).any() == False

        assert np.round(np.sum(np.abs( mat_out - mat_leg )), decimals) == 0.

        del(mat_out)
        del(mat_leg)


        # Test determinant
        value = mapping.det_df_mat(eta1, eta2, eta3, DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), ind_n1[ie1,:], ind_n2[ie2,:], ind_n3[ie3,:], DOMAIN.cx, DOMAIN.cy, DOMAIN.cz)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        value_leg = mapping.det_df(eta1, eta2, eta3, DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), np.array(SPACES.NbaseN), DOMAIN.cx, DOMAIN.cy, DOMAIN.cz)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        assert np.round( value - value_leg, decimals ) == 0.

        del(value)
        del(value_leg)

        # TODO: other functions also, but they are not really used since mappings_3d_fast provides better functions

        del(eta1)
        del(eta2)
        del(eta3)

        del(span1)
        del(span2)
        del(span3)



    # ========================================================================================= 
    # Test mappings_3d_fast
    # =========================================================================================
    for j in range(N):
        eta = np.random.rand(3)

        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        span1 = bsp.find_span(t1, pn1, eta1)
        span2 = bsp.find_span(t2, pn2, eta2)
        span3 = bsp.find_span(t3, pn3, eta3)

        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3

        for mat_or_vec in range(3):

            df_mat = np.empty( (3,3), dtype=float )
            df_vec = np.empty( 3    , dtype=float )
            mapping_fast.dl_all(DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), DOMAIN.cx, DOMAIN.cy, DOMAIN.cz, ind_n1, ind_n2, ind_n3, eta1, eta2, eta3, df_mat, df_vec, mat_or_vec)

            assert np.isnan(df_mat).any() == False
            assert np.isinf(df_mat).any() == False

            assert np.isnan(df_vec).any() == False
            assert np.isinf(df_vec).any() == False

            df_mat_leg = np.empty( (3,3), dtype=float )
            df_vec_leg = np.empty( 3    , dtype=float )
            left1      = np.empty( pn1  , dtype=float )
            left2      = np.empty( pn2  , dtype=float )
            left3      = np.empty( pn3  , dtype=float )
            right1     = np.empty( pn1  , dtype=float )
            right2     = np.empty( pn2  , dtype=float )
            right3     = np.empty( pn3  , dtype=float )
            bn1        = np.empty( (pn1+1,pn1+1), dtype=float )
            bn2        = np.empty( (pn2+1,pn2+1), dtype=float )
            bn3        = np.empty( (pn3+1,pn3+1), dtype=float )
            bd1        = np.empty( pn1  , dtype=float )
            bd2        = np.empty( pn2  , dtype=float )
            bd3        = np.empty( pn3  , dtype=float )
            diff1      = np.empty( pn1+1, dtype=float )
            diff2      = np.empty( pn2+1, dtype=float )
            diff3      = np.empty( pn3+1, dtype=float )
            der1       = np.empty( pn1+1, dtype=float )
            der2       = np.empty( pn2+1, dtype=float )
            der3       = np.empty( pn3+1, dtype=float )
            bsp.basis_funs_all(t1, pn1, eta1, span1, left1, right1, bn1, diff1)
            bsp.basis_funs_all(t2, pn2, eta2, span2, left2, right2, bn2, diff2)
            bsp.basis_funs_all(t3, pn3, eta3, span3, left3, right3, bn3, diff3)
            bsp.basis_funs_1st_der(t1, pn1, eta1, span1, left1, right1, der1)
            bsp.basis_funs_1st_der(t2, pn2, eta2, span2, left2, right2, der2)
            bsp.basis_funs_1st_der(t3, pn3, eta3, span3, left3, right3, der3)
            mapping_fast.df_all(DOMAIN.kind_map, DOMAIN.params_map, t1, t2, t3, np.array(SPACES.p), np.array(SPACES.NbaseN), span1, span2, span3, DOMAIN.cx, DOMAIN.cy, DOMAIN.cz, left1, left2, left3, right1, right2, right3, bn1, bn2, bn3, bd1, bd2, bd3, der1, der2, der3, eta1, eta2, eta3, df_mat_leg, df_vec_leg, mat_or_vec)

            assert np.isnan(df_mat_leg).any() == False
            assert np.isinf(df_mat_leg).any() == False

            assert np.isnan(df_vec_leg).any() == False
            assert np.isinf(df_vec_leg).any() == False

            assert np.isnan(left1).any() == False
            assert np.isinf(left1).any() == False
            assert np.isnan(left2).any() == False
            assert np.isinf(left2).any() == False
            assert np.isnan(left3).any() == False
            assert np.isinf(left3).any() == False

            assert np.isnan(right1).any() == False
            assert np.isinf(right1).any() == False
            assert np.isnan(right2).any() == False
            assert np.isinf(right2).any() == False
            assert np.isnan(right3).any() == False
            assert np.isinf(right3).any() == False

            assert np.isnan(bn1).any() == False
            assert np.isinf(bn1).any() == False
            assert np.isnan(bn2).any() == False
            assert np.isinf(bn2).any() == False
            assert np.isnan(bn3).any() == False
            assert np.isinf(bn3).any() == False

            assert np.isnan(bd1).any() == False
            assert np.isinf(bd1).any() == False
            assert np.isnan(bd2).any() == False
            assert np.isinf(bd2).any() == False
            assert np.isnan(bd3).any() == False
            assert np.isinf(bd3).any() == False

            assert np.isnan(der1).any() == False
            assert np.isinf(der1).any() == False
            assert np.isnan(der2).any() == False
            assert np.isinf(der2).any() == False
            assert np.isnan(der3).any() == False
            assert np.isinf(der3).any() == False

            # for mat_or_vec=1 mat_out is not written to
            if mat_or_vec != 1:
                assert np.round(np.sum(np.abs( df_mat - df_mat_leg )), decimals) == 0.

            # for mat_or_vec=0 vec_out is not written to
            if mat_or_vec != 0:
                assert np.round(np.sum(np.abs( df_vec - df_vec_leg )), decimals) == 0.
            
    print('test_mappings passed!')


if __name__ == '__main__':
    Nel         = [4,5,18]
    p           = [2,2,3]
    spl_kind    = [True, True, True]
    mappings    = 'Cuboid', {'l1':0., 'r1':1., 'l2':0., 'r2':1., 'l3':0., 'r3':1.}
    N           = 100
    test_Mappings(Nel, p, spl_kind, mappings, N)