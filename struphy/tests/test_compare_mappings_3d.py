import pytest

import numpy as np
from time import time

from pyccel import epyccel


@pytest.mark.parametrize('mapping', ['cuboid', 
                                    'orthogonal',  
                                    'colella', 
                                    'hollow_cyl', 
                                    'hollow_torus', 
                                    'ellipse',
                                    'rotated_ellipse',
                                    'shafranov_shift',
                                    'shafranov_sqrt',
                                    'shafranov_dshaped',
                                    'spline_cyl', 
                                    'spline_torus', 
                                    'spline'])
def test_compare_mappings_3d(mapping, n_markers=100):
    '''Compares old and new versions of mappings_3d.'''

    from struphy.geometry.domain_3d import Domain
    from struphy.geometry import mappings_3d, mappings_3d_new, map_eval

    # Domain object
    print(f'\nmapping: {mapping}\n')
    DOMAIN = Domain(mapping)

    # Markers
    eta1s = np.random.rand(n_markers)
    eta2s = np.random.rand(n_markers)
    eta3s = np.random.rand(n_markers)

    # Output
    det1 = np.zeros(n_markers, dtype=float)
    vec1 = np.zeros((n_markers, 3), dtype=float)
    mat1 = np.zeros((n_markers, 3, 3), dtype=float)

    det2 = np.zeros(n_markers, dtype=float)
    vec2 = np.zeros((n_markers, 3), dtype=float)
    mat2 = np.zeros((n_markers, 3, 3), dtype=float)

    det3 = np.zeros(n_markers, dtype=float)
    vec3 = np.zeros((n_markers, 3), dtype=float)
    mat3 = np.zeros((n_markers, 3, 3), dtype=float)

    vec4 = np.zeros((n_markers, 3), dtype=float)
    mat4 = np.zeros((n_markers, 3, 3), dtype=float)

    # Compare legacy with slim
    # ========================
    # Legacy loop
    st1 = time()
    mappings_3d.loop_legacy(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, vec1, mat1)
    en1 = time()

    # Slim loop
    st2 = time()
    mappings_3d.loop_slim(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, vec2, mat2)
    en2 = time()

    # f and df from mappings_3d_new
    st3 = time()
    mappings_3d_new.loop_f_and_df(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, vec3, mat3)
    en3 = time()

    # f and df from map_eval
    st4 = time()
    map_eval.loop_f_and_df(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, vec4, mat4)
    en4 = time()

    print(np.max(np.abs(vec4 - vec1)))
    print(np.max(np.abs(mat4 - mat1)))

    assert np.allclose(vec1, vec2)
    assert np.allclose(mat1, mat2)
    assert np.allclose(vec1, vec3)
    assert np.allclose(mat1, mat3)
    assert np.allclose(vec1, vec4)
    assert np.allclose(mat1, mat4)

    print(f'time f_df_pic_legacy           (n={n_markers}): {en1 - st1}')
    print(f'time f_df_pic_slim             (n={n_markers}): {en2 - st2}')
    print(f'time f, df separate (new)      (n={n_markers}): {en3 - st3}')
    print(f'time f, df separate (map_eval) (n={n_markers}): {en4 - st4}')

    # Compare mappings_3d (old) and mappings_3d_new
    # =============================================
    # old f
    st1 = time()
    mappings_3d.loop_f(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, vec1)
    en1 = time()

    # new f
    st2 = time()
    mappings_3d_new.loop_f(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, vec2)
    en2 = time()

    # f from map_eval
    st3 = time()
    map_eval.loop_f(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, vec3)
    en3 = time()

    assert np.allclose(vec1, vec2)
    assert np.allclose(vec1, vec3)

    print('')
    print(f'time f old      (n={n_markers}): {en1 - st1}')
    print(f'time f new      (n={n_markers}): {en2 - st2}')
    print(f'time f map_eval (n={n_markers}): {en3 - st3}')

    # old df
    st1 = time()
    mappings_3d.loop_df(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat1)
    en1 = time()

    # new df
    st2 = time()
    mappings_3d_new.loop_df(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat2)
    en2 = time()

    # df from map_eval
    st3 = time()
    map_eval.loop_df(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat3)
    en3 = time()

    assert np.allclose(mat1, mat2)
    assert np.allclose(mat1, mat3)

    print(f'time df old      (n={n_markers}): {en1 - st1}')
    print(f'time df new      (n={n_markers}): {en2 - st2}')
    print(f'time df map_eval (n={n_markers}): {en3 - st3}')

    # old detdf
    st1 = time()
    mappings_3d.loop_detdf(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, det1)
    en1 = time()

    # new detdf
    st2 = time()
    mappings_3d_new.loop_detdf(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, det2)
    en2 = time()

    # detdf from map_eval
    st3 = time()
    map_eval.loop_detdf(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, det3)
    en3 = time()

    assert np.allclose(det1, det2)
    assert np.allclose(det1, det3)

    print(f'time detdf old      (n={n_markers}): {en1 - st1}')
    print(f'time detdf new      (n={n_markers}): {en2 - st2}')
    print(f'time detdf map_eval (n={n_markers}): {en3 - st3}')

    # old dfinv
    st1 = time()
    mappings_3d.loop_dfinv(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat1)
    en1 = time()

    # new dfinv
    st2 = time()
    mappings_3d_new.loop_dfinv(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat2)
    en2 = time()

    # dfinv from map_eval
    st3 = time()
    map_eval.loop_dfinv(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat3)
    en3 = time()

    assert np.allclose(mat1, mat2)
    assert np.allclose(mat1, mat3)

    print(f'time dfinv old      (n={n_markers}): {en1 - st1}')
    print(f'time dfinv new      (n={n_markers}): {en2 - st2}')
    print(f'time dfinv map_eval (n={n_markers}): {en3 - st3}')

    # old g
    st1 = time()
    mappings_3d.loop_g(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat1)
    en1 = time()

    # new g
    st2 = time()
    mappings_3d_new.loop_g(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat2)
    en2 = time()

    # g from map_eval
    st3 = time()
    map_eval.loop_g(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat3)
    en3 = time()

    assert np.allclose(mat1, mat2)
    assert np.allclose(mat1, mat3)

    print(f'time g old      (n={n_markers}): {en1 - st1}')
    print(f'time g new      (n={n_markers}): {en2 - st2}')
    print(f'time g map_eval (n={n_markers}): {en3 - st3}')

    # old ginv
    st1 = time()
    mappings_3d.loop_ginv(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat1)
    en1 = time()

    # new ginv
    st2 = time()
    mappings_3d_new.loop_ginv(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat2)
    en2 = time()

    # ginv from map_eval
    st3 = time()
    map_eval.loop_ginv(DOMAIN.kind_map, DOMAIN.params_map,
                DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2], np.array(DOMAIN.p),
                DOMAIN.indN[0], DOMAIN.indN[1], DOMAIN.indN[2],
                DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                eta1s, eta2s, eta3s, mat3)
    en3 = time()

    atol = 1e-6

    try:
        assert np.allclose(mat1, mat2)
    except:
        assert np.allclose(mat1, mat2, atol=atol)
        print(f'Assertion for ginv passed with lower atol={atol}.')

    try:
        assert np.allclose(mat1, mat3)
    except:
        assert np.allclose(mat1, mat3, atol=atol)
        print(f'Assertion for ginv passed with lower atol={atol}.')
    

    print(f'time ginv old      (n={n_markers}): {en1 - st1}')
    print(f'time ginv new      (n={n_markers}): {en2 - st2}')
    print(f'time ginv map_eval (n={n_markers}): {en3 - st3}')


# def loop_slim():

if __name__ == '__main__':
    test_compare_mappings_3d('colella', n_markers=100000)
    test_compare_mappings_3d('spline_cyl', n_markers=100000)
    test_compare_mappings_3d('spline_torus', n_markers=100000)
    test_compare_mappings_3d('spline', n_markers=100000)
