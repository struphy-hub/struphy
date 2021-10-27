# PyTest will consider all `.py` files that begin with `test_` as test files.
# Then it will run all functions that begin with `test_`, and look for `assert` statements.
# Therefore it is important to put all imports inside the function.
# Also because these test functions are located inside the `test/` directory, we need to update the import location with `../`.



def test_divB():
    """
    Test if divergence of B-field is zero, after B-field is pulled to 2-form and projected to spline basis.
    """

    # ============================================================
    # Imports.
    # ============================================================

    import os
    import sys
    sys.path.append('..') # Because we are inside './test/' directory.

    import json
    import numpy as np

    import hylife.geometry.domain_3d as dom
    import hylife.utilitis_FEEC.projectors.projectors_global as pro
    import hylife.utilitis_FEEC.spline_space as spl

    # from base.base import Base
    # from base.make_base import make_base
    # from reader.read_json import read_json
    # from hmap.suv_to_xyz import suv_to_xyz as MapFull



    # ============================================================
    # Primary parameters to control this test file.
    # ============================================================

    # Which test case to use. 1: Identity mapping, 2: torus mapping, 3: GVEC mapping.
    test_case = 1
    # Which B-field to use. 1: (0,0,1), 2:(1,1,1), 3: (1,2,3), 4: GVEC's B-field.
    kind_B_field = 3
    # Present each B-field component as a tuple of functions B=(Bx,By,Bz). If False, B itself is a function.
    use_B_components = True



    # ============================================================
    # Test cases.
    # ============================================================

    # Create splines to STRUPHY, unrelated to splines in GVEC.
    Nel      = [6, 6, 6] 
    p        = [2, 3, 3]
    spl_kind = None       # Set values below
    nq_el    = [4, 4, 6]  # Element integration
    nq_pr    = [4, 4, 6]  # Greville integration
    bc       = ['f', 'f'] # BC in s-direction
    X, Y, Z = None, None, None

    # Case 1: Identity mapping. (x,y,z) -> (x,y,z) = (s,u,v). It is not periodic.
    if test_case == 1:

        print('Test case {}: Identity mapping.'.format(test_case))
        X = lambda s,u,v: s * np.ones_like(u) * np.ones_like(v)
        Y = lambda s,u,v: u * np.ones_like(s) * np.ones_like(v)
        Z = lambda s,u,v: v * np.ones_like(s) * np.ones_like(u)
        spl_kind = [False, False, False] # Periodic or not.

    # Case 2: Torus mapping. It is periodic along angular directions.
    elif test_case == 2:

        print('Test case {}: Torus mapping.'.format(test_case))
        a1 = 0
        a2 = 1
        r0 = 10
        da = a2 - a1
        X = lambda eta1,eta2,eta3: ((a1 + eta1 * da) * np.cos(2*np.pi*eta2) + r0) * np.cos(2*np.pi*eta3)
        Y = lambda eta1,eta2,eta3:  (a1 + eta1 * da) * np.sin(2*np.pi*eta2)       * np.ones_like(eta3)
        Z = lambda eta1,eta2,eta3: ((a1 + eta1 * da) * np.cos(2*np.pi*eta2) + r0) * np.sin(2*np.pi*eta3)
        spl_kind = [False, True, True] # Periodic or not.

    # Case 3: GVEC mapping. It is periodic along angular directions. (emm... NFP?)
    elif test_case == 3:

        print('Test case {}: GVEC mapping.'.format(test_case))

        # filepath = '../GVEC/testcases/ellipstell/' # Because we are inside './test/' directory.
        # filename = 'GVEC_ellipStell_profile_update_State_0000_00010000.json'
        # data = read_json(filepath, filename)

        # X1_coef = np.array(data['X1']['coef'])
        # X2_coef = np.array(data['X2']['coef'])
        # LA_coef = np.array(data['LA']['coef'])
        # X1_base = make_base(data, 'X1')
        # X2_base = make_base(data, 'X2')
        # LA_base = make_base(data, 'LA')

        # mapfull = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef)
        # mapX    = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef, 'x')
        # mapY    = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef, 'y')
        # mapZ    = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef, 'z')
        # f = mapfull.mapto
        # X = mapX.mapto_x
        # Y = mapY.mapto_y
        # Z = mapZ.mapto_z
        # spl_kind = [False, True, True] # Periodic or not.
        raise NotImplementedError('Test case {} not implemented.'.format(test_case))

    # Case N: Mapping not found.
    else:

        raise ValueError('Test case {} not found.'.format(test_case))



    # ============================================================
    # Define B-field.
    # ============================================================

    # Case 1: Constant B-field of strength (0,0,1) in Cartesian (x,y,z) directions.
    if kind_B_field == 1:

        print('B-field type {}.'.format(kind_B_field))

        B = lambda x,y,z: (0,0,1) if not isinstance(x, np.ndarray) else (0 * np.ones_like(x), 0 * np.ones_like(y), 1 * np.ones_like(z)) 
        print('Cartesian B(x,y,z) -> (0,0,1):', B(0,0,1))

        Bx = lambda x,y,z: 0 if not isinstance(x, np.ndarray) else 0 * np.ones_like(x)
        By = lambda x,y,z: 0 if not isinstance(x, np.ndarray) else 0 * np.ones_like(y)
        Bz = lambda x,y,z: 1 if not isinstance(x, np.ndarray) else 1 * np.ones_like(z)
        print('Cartesian [Bx(x,y,z), By(x,y,z), Bz(x,y,z)] -> (0,0,1)', (Bx(0,0,1), By(0,0,1), Bz(0,0,1),))

    # Case 2: Constant B-field of strength (1,1,1) in Cartesian (x,y,z) directions.
    elif kind_B_field == 2:

        print('B-field type {}.'.format(kind_B_field))

        B = lambda x,y,z: (1,1,1) if not isinstance(x, np.ndarray) else (1 * np.ones_like(x), 1 * np.ones_like(y), 1 * np.ones_like(z))
        print('Cartesian B(x,y,z) -> (1,1,1):', B(1,1,1))

        Bx = lambda x,y,z: 1 if not isinstance(x, np.ndarray) else 1 * np.ones_like(x)
        By = lambda x,y,z: 1 if not isinstance(x, np.ndarray) else 1 * np.ones_like(y)
        Bz = lambda x,y,z: 1 if not isinstance(x, np.ndarray) else 1 * np.ones_like(z)
        print('Cartesian [Bx(x,y,z), By(x,y,z), Bz(x,y,z)] -> (1,1,1)', (Bx(1,1,1), By(1,1,1), Bz(1,1,1),))

    # Case 3: Constant B-field of strength (1,2,3) in Cartesian (x,y,z) directions.
    elif kind_B_field == 3:

        print('B-field type {}.'.format(kind_B_field))

        B = lambda x,y,z: (1,2,3) if not isinstance(x, np.ndarray) else (1 * np.ones_like(x), 2 * np.ones_like(y), 3 * np.ones_like(z))
        print('Cartesian B(x,y,z) -> (1,2,3):', B(1,2,3))

        Bx = lambda x,y,z: 1 if not isinstance(x, np.ndarray) else 1 * np.ones_like(x)
        By = lambda x,y,z: 2 if not isinstance(x, np.ndarray) else 2 * np.ones_like(y)
        Bz = lambda x,y,z: 3 if not isinstance(x, np.ndarray) else 3 * np.ones_like(z)
        print('Cartesian [Bx(x,y,z), By(x,y,z), Bz(x,y,z)] -> (1,2,3)', (Bx(1,2,3), By(1,2,3), Bz(1,2,3),))

    # Case 4: B-field from GVEC data.
    elif kind_B_field == 4:

        print('B-field type {}.'.format(kind_B_field))
        raise NotImplementedError('B-field type {} not implemented.'.format(kind_B_field))

    # Case N: B-field type not found.
    else:

        raise ValueError('B-field type {} not found.'.format(kind_B_field))

    # Supply a tuple of each B-field components B=(Bx,By,Bz).
    if use_B_components:

        B = [Bx, By, Bz] # Must be a list!



    # ============================================================
    # Create FEM space.
    # ============================================================

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]

    # 2d tensor-product B-spline space for polar splines (if used)
    tensor_space_pol = spl.tensor_spline_space(spaces_FEM[:2])

    # 3d tensor-product B-spline space for finite elements
    tensor_space_FEM = spl.tensor_spline_space(spaces_FEM)
    print('Tensor space set up done.')

    # Set extraction operators (is automatic now) and discrete derivatives 
    #polar_splines = None
    #tensor_space_FEM.set_extraction_operators(bc, polar_splines)
    #tensor_space_FEM.set_derivatives()
    #print('Set tensor space derivatives done.')



    # ============================================================
    # 3D projection using `projectors_tensor_3d`.
    # ============================================================

    # Create 3D projector.
    proj_eta1 = pro.projectors_global_1d(spaces_FEM[0])
    proj_eta2 = pro.projectors_global_1d(spaces_FEM[1])
    proj_eta3 = pro.projectors_global_1d(spaces_FEM[2])
    proj_3d   = pro.projectors_tensor_3d([proj_eta1, proj_eta2, proj_eta3])
    print('Create 3D projector done.')



    # ============================================================
    # Create splines (using PI_0 projector).
    # ============================================================

    # Calculate spline coefficients using PI_0 projector.
    cx = proj_3d.PI_0(X)
    cy = proj_3d.PI_0(Y)
    cz = proj_3d.PI_0(Z)

    domain = dom.domain('spline', params_map=None, Nel=Nel, p=p, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)



    # ============================================================
    # Convert pulled 2-form B-field into functions.
    # Because the projector needs it.
    # ============================================================

    def hat_B2_1(eta1, eta2, eta3):
        return domain.pull(B, eta1, eta2, eta3, '2_form_1')
    def hat_B2_2(eta1, eta2, eta3):
        return domain.pull(B, eta1, eta2, eta3, '2_form_2')
    def hat_B2_3(eta1, eta2, eta3):
        return domain.pull(B, eta1, eta2, eta3, '2_form_3')

    hat_B2 = [hat_B2_1, hat_B2_2, hat_B2_3]




    # ============================================================
    # Test push-pull.
    # ============================================================

    # Test 4 types of inputs.
    print('Test if push-pull to 2-form and back are identical.')
    num_pts = 20

    # To broadcast sparse meshgrids.
    broadcast = lambda x,y,z: np.ones_like(x) * np.ones_like(y) * np.ones_like(z)

    def test_push_pull(eta1, eta2, eta3):
        """Compare push-pulled output with original output."""

        is_array = isinstance(eta1, np.ndarray)
        is_sparse_meshgrid = None
        broadcast = None

        if is_array:

            # Tensor-product evaluation.
            if eta1.ndim == 1:

                # Don't handle meshgrid logic here. That's something we wanna test.
                # E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij', sparse=True)
                # is_sparse_meshgrid = True
                pass

            # General evaluation.
            else:

                # Distinguish if it is sparse or dense.
                # Sparse: eta1.shape = (n1,  1,  1)
                # Dense : eta.shape = (n1, n2, n3)

                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    is_sparse_meshgrid = True
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    is_sparse_meshgrid = False

        if use_B_components:
            B_orig = [B[0](eta1, eta2, eta3), B[1](eta1, eta2, eta3), B[2](eta1, eta2, eta3)]
        else:
            B_orig = list(B(eta1, eta2, eta3))

        if is_array:
            if is_sparse_meshgrid is None:
                # 1D arrays.
                broadcast = np.ones((eta1.size, eta2.size, eta3.size,))
                B_orig    = [B_o * broadcast for B_o in B_orig]
            elif is_sparse_meshgrid:
                # 3D arrays with unequal dimensions.
                broadcast = np.ones((eta1.size, eta2.size, eta3.size,))
                B_orig    = [B_o * broadcast for B_o in B_orig]
        B_orig = np.array(B_orig)
        if is_array:
            if broadcast is None:
                print('B_orig[0].shape', B_orig[0].shape)
            else:
                print('B_orig[0].shape', B_orig[0].shape, '(broadcasted)')
        # print('Original B-field evaluated at (eta1, eta2, eta3): {}'.format(B_orig))

        # Pull B(\eta) -> (Bx,By,Bz) to its 2-form representation.
        pulled_B = [hat_B2_i(eta1, eta2, eta3) for hat_B2_i in hat_B2]
        pulled_B = np.array(pulled_B)
        # print('B pulled to 2-form:', pulled_B)

        # Then push back to Cartesian to see if we get the same thing.
        pullpushed_Bx = domain.push(pulled_B, eta1, eta2, eta3, '2_form_1')
        pullpushed_By = domain.push(pulled_B, eta1, eta2, eta3, '2_form_2')
        pullpushed_Bz = domain.push(pulled_B, eta1, eta2, eta3, '2_form_3')
        if isinstance(eta1, np.ndarray):
            print('pullpushed_Bx.shape', pullpushed_Bx.shape)
        pullpushed_B  = [pullpushed_Bx, pullpushed_By, pullpushed_Bz]
        pullpushed_B  = np.array(pullpushed_B)
        # print('Push-pulled B(x,y,z):', pullpushed_B)

        print('Push-pull error:', np.max(np.abs(B_orig - pullpushed_B)))

        assert np.allclose(B_orig, pullpushed_B), 'Original B-field should be identical to the push-pulled one.'

        print(' ')

    # ============================================================
    # Case 1: Scalar.
    print('Case 1: Scalar.')
    eta1, eta2, eta3 = 1/2, 1/2, 1/2
    test_push_pull(eta1, eta2, eta3)

    # ============================================================
    # Case 2: 3x 1D arrays.
    print('Case 2: 3x 1D arrays.')
    eta1, eta2, eta3 = np.linspace(0,1,num_pts), np.linspace(0,1,num_pts), np.linspace(0,1,num_pts)
    print('Input shapes, (eta1,eta2,eta3):', eta1.shape, eta2.shape, eta3.shape)
    test_push_pull(eta1, eta2, eta3)

    # ============================================================
    # Case 3: 3x 3D arrays of dense meshgrid.
    print('Case 3: 3x 3D arrays of dense meshgrid.')
    eta1, eta2, eta3 = np.linspace(0,1,num_pts), np.linspace(0,1,num_pts), np.linspace(0,1,num_pts)
    eta1, eta2, eta3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
    print('Input shapes, (eta1,eta2,eta3):', eta1.shape, eta2.shape, eta3.shape)
    test_push_pull(eta1, eta2, eta3)

    # ============================================================
    # Case 4: 3x 3D arrays of sparse meshgrid.
    print('Case 4: 3x 3D arrays of sparse meshgrid.')
    eta1, eta2, eta3 = np.linspace(0,1,num_pts), np.linspace(0,1,num_pts), np.linspace(0,1,num_pts)
    eta1, eta2, eta3 = np.meshgrid(eta1, eta2, eta3, indexing='ij', sparse=True)
    print('Input shapes, (eta1,eta2,eta3):', eta1.shape, eta2.shape, eta3.shape)
    test_push_pull(eta1, eta2, eta3)



    # ============================================================
    # 3D Pi_2 projection.
    # ============================================================

    # Do the Pi_2 projection.
    b2_coeff = proj_3d.PI_2(*hat_B2) # Coefficients of projected 2-form.
    # print('b2_coeff[0].shape', b2_coeff[0].shape, 'b2_coeff[1].shape', b2_coeff[1].shape, 'b2_coeff[2].shape', b2_coeff[2].shape)
    print('Pi_2 projection done.')



    # ============================================================
    # Show Div B = 0 in both real space and 2-form.
    # ============================================================

    b2_coeff_concat = np.concatenate((b2_coeff[0].flatten(), b2_coeff[1].flatten(), b2_coeff[2].flatten()))
    div = tensor_space_FEM.D.dot(b2_coeff_concat)
    print('Maximum error (how close is Div B to 0):', np.max(np.abs(div)))

    assert tensor_space_FEM.D.shape[1] == b2_coeff[0].flatten().size + b2_coeff[1].flatten().size + b2_coeff[2].flatten().size, 'Matrix size should match for dicrete divergence.'
    assert np.max(np.abs(div)) < 1e-10, 'Divergence should be zero.'



if __name__ == "__main__":
    test_divB()
