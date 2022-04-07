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

    import h5py
    import tempfile
    temp_dir = tempfile.TemporaryDirectory(prefix='STRUPHY-')
    print(f'Created temp directory at: {temp_dir.name}')

    import numpy as np

    # Which diagnostics is run
    print('Run diagnostics:', sys.argv[0])

    basedir = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, os.path.join(basedir, '..'))

    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl
    from gvec_to_python import GVEC, Form, Variable
    from gvec_to_python.reader.gvec_reader import GVEC_Reader



    # ============================================================
    # Primary parameters to control this test file.
    # ============================================================

    # Which test case to use. 1: Identity mapping, 2: torus mapping, 3: GVEC mapping.
    test_case = 2
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

        raise NotImplementedError('Test case {} not implemented, because extraction/boundary operators in `spline_space` no longer support aperiodic 3rd dimension.'.format(test_case))

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

        # ============================================================
        # Convert GVEC .dat output to .json.
        # ============================================================

        read_filepath = 'mhd_equil/gvec/'
        read_filepath = os.path.join(basedir, '..', read_filepath)
        read_filename = 'GVEC_ellipStell_profile_update_State_0000_00010000.dat'
        save_filepath = temp_dir.name
        save_filename = 'GVEC_ellipStell_profile_update_State_0000_00010000.json'
        reader = GVEC_Reader(read_filepath, read_filename, save_filepath, save_filename, with_spl_coef=True)

        # ============================================================
        # Load GVEC mapping.
        # ============================================================

        filepath = temp_dir.name
        filename = 'GVEC_ellipStell_profile_update_State_0000_00010000.json'
        gvec = GVEC(filepath, filename)

        f = gvec.mapfull.f # Full mapping, (s,u,v) to (x,y,z).
        X = gvec.mapX.f    # Only x component of the mapping.
        Y = gvec.mapY.f    # Only y component of the mapping.
        Z = gvec.mapZ.f    # Only z component of the mapping.
        print('Loaded GVEC mapping.')

        spl_kind = [False, True, True] # Periodic or not.

    # Case N: Mapping not found.
    else:

        raise NotImplementedError('Test case {} not implemented.'.format(test_case))
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

        print('B-field type {}: GVEC.'.format(kind_B_field))

        B = gvec.B
        print('GVEC logical B(s,u,v)@(0.5,0.5,0.5) -> B(x,y,z):', B(0.5,0.5,0.5))
        print('GVEC logical B2(s,u,v)@(0.5,0.5,0.5) -> B2(x,y,z):', gvec.B_2(0.5,0.5,0.5))

        # This is a stupid idea. Every function call to `B` is evaluated 3 times.
        Bx = lambda s,u,v: gvec.B(s,u,v)[0]
        By = lambda s,u,v: gvec.B(s,u,v)[1]
        Bz = lambda s,u,v: gvec.B(s,u,v)[2]
        print('GVEC logical [Bx(s,u,v), By(s,u,v), Bz(s,u,v)] -> B(x,y,z)', (Bx(0.5,0.5,0.5), By(0.5,0.5,0.5), Bz(0.5,0.5,0.5),))

    # Case N: B-field type not found.
    else:

        raise NotImplementedError('B-field type {} not implemented.'.format(kind_B_field))
        raise ValueError('B-field type {} not found.'.format(kind_B_field))

    # Supply a tuple of each B-field components B=(Bx,By,Bz).
    if use_B_components:

        B = [Bx, By, Bz] # Must be a list!



    # ============================================================
    # Create FEM space.
    # ============================================================

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [spl.Spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]

    # 2d tensor-product B-spline space for polar splines (if used)
    tensor_space_pol = spl.Tensor_spline_space(spaces_FEM[:2])

    # 3d tensor-product B-spline space for finite elements
    tensor_space_FEM = spl.Tensor_spline_space(spaces_FEM)
    print('Tensor space set up done.')



    # ============================================================
    # 3D projection using `projectors_tensor_3d`.
    # ============================================================

    # Create 3D projector. It's not automatic.
    for space in spaces_FEM:
        if not hasattr(space, 'projectors'):
            space.set_projectors() # def set_projectors(self, nq=6):
    if not hasattr(tensor_space_FEM, 'projectors'):
        tensor_space_FEM.set_projectors() # def set_projectors(self, which='tensor', nq=[6, 6]):
    proj_3d = tensor_space_FEM.projectors
    print('Create 3D projector done.')



    # ============================================================
    # Create splines (using PI_0 projector).
    # ============================================================

    # Calculate spline coefficients using PI_0 projector.
    cx = proj_3d.PI_0(X)
    cy = proj_3d.PI_0(Y)
    cz = proj_3d.PI_0(Z)

    spline_coeffs_file = os.path.join(temp_dir.name, 'spline_coeffs.hdf5')

    with h5py.File(spline_coeffs_file, 'w') as handle:
        handle['cx'] = cx
        handle['cy'] = cy
        handle['cz'] = cz
        handle.attrs['whatis'] = 'These are 3D spline coefficients constructed from GVEC mapping.'

    params_map = {
        'file': spline_coeffs_file,
        'Nel': Nel,
        'p': p,
        'spl_kind': spl_kind,
    }

    domain = dom.Domain('spline', params_map=params_map)
    print('Computed spline coefficients.')

    temp_dir.cleanup()
    print('Removed temp directory.')



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

    if test_case == 3: # GVEC

        # Solution 1: Cheat by using 2-form from GVEC directly.
        # def hat_B2_1(eta1, eta2, eta3):
        #     return gvec.get_variable(eta1, eta2, eta3, variable=Variable.B, form=Form.TWO)[0]
        # def hat_B2_2(eta1, eta2, eta3):
        #     return gvec.get_variable(eta1, eta2, eta3, variable=Variable.B, form=Form.TWO)[1]
        # def hat_B2_3(eta1, eta2, eta3):
        #     return gvec.get_variable(eta1, eta2, eta3, variable=Variable.B, form=Form.TWO)[2]

        # Solution 2: Pre-evaluate GVEC B-field at logical coordinate, to bypass forced evaluation of B at Cartesian coordinate.
        def hat_B2_1(eta1, eta2, eta3):
            return domain.pull(gvec.get_variable(eta1, eta2, eta3, variable=Variable.B, form=Form.PHYSICAL), eta1, eta2, eta3, '2_form_1')
        def hat_B2_2(eta1, eta2, eta3):
            return domain.pull(gvec.get_variable(eta1, eta2, eta3, variable=Variable.B, form=Form.PHYSICAL), eta1, eta2, eta3, '2_form_2')
        def hat_B2_3(eta1, eta2, eta3):
            return domain.pull(gvec.get_variable(eta1, eta2, eta3, variable=Variable.B, form=Form.PHYSICAL), eta1, eta2, eta3, '2_form_3')

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

        is_array = isinstance(eta1, np.ndarray) and eta1.ndim != 0
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
                print('B_orig.shape', B_orig.shape)
                print('B_orig[0].shape', B_orig[0].shape)
            else:
                print('B_orig.shape', B_orig.shape)
                print('B_orig[0].shape', B_orig[0].shape, '(broadcasted)')
        # print('Original B-field evaluated at (eta1, eta2, eta3): {}'.format(B_orig))
        B_orig_mag = np.sqrt(B_orig[0]**2 + B_orig[1]**2 + B_orig[2]**2)
        print('B_orig_mag.shape', B_orig_mag.shape)
        print('Max |B_orig|:', np.max(B_orig_mag))
        print('Min |B_orig|:', np.min(B_orig_mag))

        # Pull B(\eta) -> (Bx,By,Bz) to its 2-form representation.
        pulled_B = [hat_B2_i(eta1, eta2, eta3) for hat_B2_i in hat_B2]
        pulled_B = np.array(pulled_B)
        # print('B pulled to 2-form:', pulled_B)
        B_pull_mag = np.sqrt(pulled_B[0]**2 + pulled_B[1]**2 + pulled_B[2]**2)
        print('B_pull_mag.shape', B_pull_mag.shape)
        print('Max |B_pull|:', np.max(B_pull_mag))
        print('Min |B_pull|:', np.min(B_pull_mag))

        # Then push back to Cartesian to see if we get the same thing.
        pullpushed_Bx = domain.push(pulled_B, eta1, eta2, eta3, '2_form_1')
        pullpushed_By = domain.push(pulled_B, eta1, eta2, eta3, '2_form_2')
        pullpushed_Bz = domain.push(pulled_B, eta1, eta2, eta3, '2_form_3')
        if isinstance(eta1, np.ndarray):
            print('pullpushed_Bx.shape', pullpushed_Bx.shape)
        pullpushed_B  = [pullpushed_Bx, pullpushed_By, pullpushed_Bz]
        pullpushed_B  = np.array(pullpushed_B)
        # print('Push-pulled B(x,y,z):', pullpushed_B)
        B_pped_mag = np.sqrt(pullpushed_B[0]**2 + pullpushed_B[1]**2 + pullpushed_B[2]**2)
        print('B_pped_mag.shape', B_pped_mag.shape)
        print('Max |B_pped|:', np.max(B_pped_mag))
        print('Min |B_pped|:', np.min(B_pped_mag))

        print('Push-pull error:', np.max(np.abs(B_orig - pullpushed_B)))

        assert np.allclose(B_orig, pullpushed_B), 'Original B-field should be identical to the push-pulled one.'

        print(' ')

    # ============================================================
    # Case 1: Scalar.
    print('Case 1: Scalar.')
    eta1, eta2, eta3 = 0.5, 0.5, 0.5
    test_push_pull(eta1, eta2, eta3)

    # ============================================================
    # Case 2: 3x 1D arrays.
    print('Case 2: 3x 1D arrays.')
    eta1, eta2, eta3 = np.linspace(1e-12,1,num_pts), np.linspace(0,1,num_pts), np.linspace(0,1,num_pts)
    print('Input shapes, (eta1,eta2,eta3):', eta1.shape, eta2.shape, eta3.shape)
    test_push_pull(eta1, eta2, eta3)

    # ============================================================
    # Case 3: 3x 3D arrays of dense meshgrid.
    print('Case 3: 3x 3D arrays of dense meshgrid.')
    eta1, eta2, eta3 = np.linspace(1e-12,1,num_pts), np.linspace(0,1,num_pts), np.linspace(0,1,num_pts)
    eta1, eta2, eta3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
    print('Input shapes, (eta1,eta2,eta3):', eta1.shape, eta2.shape, eta3.shape)
    test_push_pull(eta1, eta2, eta3)

    # ============================================================
    # Case 4: 3x 3D arrays of sparse meshgrid.
    print('Case 4: 3x 3D arrays of sparse meshgrid.')
    eta1, eta2, eta3 = np.linspace(1e-12,1,num_pts), np.linspace(0,1,num_pts), np.linspace(0,1,num_pts)
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
    # print(f'b2_coeff_concat.size {b2_coeff_concat.size}')
    div = tensor_space_FEM.D.dot(b2_coeff_concat)
    print('Maximum error (how close is Div B to 0):', np.max(np.abs(div)))

    assert tensor_space_FEM.D.shape[1] == b2_coeff[0].flatten().size + b2_coeff[1].flatten().size + b2_coeff[2].flatten().size, 'Matrix size should match for dicrete divergence.'
    assert np.max(np.abs(div)) < 1e-10, 'Divergence should be zero.'



if __name__ == "__main__":
    test_divB()
