import logging
import pytest
import cunumpy as xp
from struphy import set_logging_level
import logging

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)


def test_local_matrix_vector():
    from struphy.feec.utilities import LocalVector
    from struphy.feec.utilities import LocalRotationMatrix

    def fun0(e1, e2, e3):
        return e1 + 2 * e1**3 + xp.exp(e2)
    
    def fun1(e1, e2, e3):
        return xp.cos(xp.pi * e2) * xp.sin(2 * xp.pi * e3)
    
    def fun2(e1, e2, e3):
        return xp.tan(e1) * e2**2

    # create local vector
    local_S = LocalVector(fun0, fun1, fun2)
    # create local rotation matrix
    local_R = LocalRotationMatrix(fun0, fun1, fun2)

    # test that local scalar product is correct
    n1 = 5
    n2 = 6
    n3 = 7
    
    e1 = xp.linspace(0, 1, n1)
    e2 = xp.linspace(0, 1, n2)
    e3 = xp.linspace(0, 1, n3)
    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")
    
    v = 2*xp.random.rand(n1, n2, n3, 3) - 0.5
    mat = 2*xp.random.rand(n1, n2, n3, 3, 3) - 0.5
    
    # test np.vecdot
    logger.info(f"{local_S(ee1, ee2, ee3).shape=}, {v.shape=}")
    result_S = xp.vecdot(local_S(ee1, ee2, ee3), v)
    assert result_S.shape == (n1, n2, n3), f"Expected shape {(n1, n2, n3)}, got {result_S.shape}"
    
    # test np.matvec
    locR = local_R(ee1, ee2, ee3)
    logger.info(f"{locR.shape=}, {v.shape=}")
    result_R = xp.matvec(locR, v)
    assert result_R.shape == (n1, n2, n3, 3), f"Expected shape {(n1, n2, n3, 3)}, got {result_R.shape}"
    
    # test @ (matrix-matrix product)
    result_R2 = locR @ mat
    assert result_R2.shape == (n1, n2, n3, 3, 3), f"Expected shape {(n1, n2, n3, 3, 3)}, got {result_R2.shape}"

    # slow versions for comarison
    expected_R = xp.zeros(3, dtype=float)
    expected_R2 = xp.zeros((3, 3), dtype=float)
    for i in range(n1):
        ee1_i = ee1[i, 0, 0]
        for j in range(n2):
            ee2_j = ee2[0, j, 0]
            for k in range(n3):
                ee3_k = ee3[0, 0, k]
                
                expected_S = fun0(ee1_i, ee2_j, ee3_k) * v[i, j, k, 0] + fun1(ee1_i, ee2_j, ee3_k) * v[i, j, k, 1] + fun2(ee1_i, ee2_j, ee3_k) * v[i, j, k, 2]
                assert xp.isclose(result_S[i, j, k], expected_S), f"Expected {expected_S}, got {result_S[i, j, k]}"
                
                expected_R[0] = fun1(ee1_i, ee2_j, ee3_k) * v[i, j, k, 2] - fun2(ee1_i, ee2_j, ee3_k) * v[i, j, k, 1]
                expected_R[1] = - fun0(ee1_i, ee2_j, ee3_k) * v[i, j, k, 2] + fun2(ee1_i, ee2_j, ee3_k) * v[i, j, k, 0] 
                expected_R[2] = fun0(ee1_i, ee2_j, ee3_k) * v[i, j, k, 1] - fun1(ee1_i, ee2_j, ee3_k) * v[i, j, k, 0]
                assert xp.allclose(result_R[i, j, k], expected_R), f"Expected {expected_R}, got {result_R[i, j, k]}"
                
                expected_R2[0, 0] = fun1(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 2, 0] - fun2(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 1, 0]
                expected_R2[1, 0] = - fun0(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 2, 0] + fun2(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 0, 0]
                expected_R2[2, 0] = fun0(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 1, 0] - fun1(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 0, 0]
                
                expected_R2[0, 1] = fun1(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 2, 1] - fun2(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 1, 1]
                expected_R2[1, 1] = - fun0(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 2, 1] + fun2(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 0, 1]
                expected_R2[2, 1] = fun0(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 1, 1] - fun1(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 0, 1]
                
                expected_R2[0, 2] = fun1(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 2, 2] - fun2(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 1, 2]
                expected_R2[1, 2] = - fun0(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 2, 2] + fun2(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 0, 2]
                expected_R2[2, 2] = fun0(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 1, 2] - fun1(ee1_i, ee2_j, ee3_k) * mat[i, j, k, 0, 2]
                assert xp.allclose(result_R2[i, j, k], expected_R2), f"Expected {expected_R2}, got {result_R2[i, j, k]}"

    logger.info("test_local_matrix_vector passed")


if __name__ == "__main__":
    test_local_matrix_vector()