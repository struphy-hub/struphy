"""Load GVEC equilibrium from data."""

def test_GVEC_equilibrium(plot=False):
    """
    Test if divergence of 2-form B-field is zero, and if 2-form equilibrium current and its divergence are zero. 
    Also test if discrete curl of 1-form A-vector-potential is equivalent to analytical 2-form B-field.
    """

    # ============================================================
    # Imports.
    # ============================================================

    import numpy             as np
    import matplotlib.pyplot as plt
    import matplotlib.tri    as tri

    import os
    import sys

    import h5py
    import tempfile
    temp_dir = tempfile.TemporaryDirectory(prefix='STRUPHY-')
    print(f'Created temp directory at: {temp_dir.name}')

    # Which diagnostics is run
    print('Run diagnostics:', sys.argv[0])

    basedir = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, os.path.join(basedir, '..'))

    # Import necessary struphy.modules.
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl
    from gvec_to_python import GVEC, Form, Variable
    from gvec_to_python.reader.gvec_reader import GVEC_Reader



    # ============================================================
    # Primary parameters to control this test file.
    # ============================================================

    # Present each B-field component as a tuple of functions B=(Bx,By,Bz). If False, B itself is a function.
    # Always True because that's what expected by STRUPHY's push/pull mechanisms.
    use_B_components = True



    # ============================================================
    # Configure STRUPHY splines.
    # ============================================================

    # Create splines to STRUPHY, unrelated to splines in GVEC.
    Nel      = [6, 6, 6] 
    p        = [2, 3, 3]
    spl_kind = None       # Set values below
    nq_el    = [4, 4, 6]  # Element integration
    nq_pr    = [4, 4, 6]  # Greville integration
    bc       = ['f', 'f'] # BC in s-direction
    spl_kind = [False, True, True] # Periodic or not.



    # ============================================================
    # Convert GVEC .dat output to .json.
    # ============================================================

    read_filepath = 'mhd_equil/gvec/'
    read_filepath = os.path.join(basedir, '..', read_filepath)
    read_filename = 'GVEC_ellipStell_profile_update_State_0000_00010000.dat'
    save_filepath = temp_dir.name
    save_filename = 'GVEC_ellipStell_profile_update_State_0000_00010000.json'
    reader = GVEC_Reader(read_filepath, read_filename, save_filepath, save_filename)



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



    # ============================================================
    # Setup plot.
    # ============================================================

    # Plot settings.
    row = 2
    col = 4
    dpi = 100
    width, height = (1920 / dpi, 1200 / dpi)
    fig = plt.figure(figsize=(width,height), dpi=dpi)
    gs  = fig.add_gridspec(row, col, width_ratios=[2,1]*2)
    fig.suptitle('Visualizion of GVEC equilibrium profiles', y=0.98)
    ax1_3D = fig.add_subplot(gs[0, 0], projection='3d')
    ax2_3D = fig.add_subplot(gs[0, 2], projection='3d')
    # ax3_3D = fig.add_subplot(gs[1, 0], projection='3d')
    ax3_3D = fig.add_subplot(gs[1, 0:2], projection='3d') # Combine two subplots.
    ax4_3D = fig.add_subplot(gs[1, 2], projection='3d')
    ax1_1D = fig.add_subplot(gs[0, 1])
    ax2_1D = fig.add_subplot(gs[0, 3])
    # ax3_1D = fig.add_subplot(gs[1, 1])
    ax4_1D = fig.add_subplot(gs[1, 3])
    axes_3D = [ax1_3D, ax2_3D, ax3_3D, ax4_3D]
    # axes_1D = [ax1_1D, ax2_1D, ax3_1D, ax4_1D]
    axes_1D = [ax1_1D, ax2_1D, ax4_1D]

    for ax in axes_3D:
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.zaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')

    for ax in axes_1D:
        ax.set_xlabel('$\eta^1$')



    # ============================================================
    # Draw surface of GVEC grid.
    # ============================================================

    num_pts = 10
    eta1_range, eta2_range, eta3_range = np.linspace(0,1,num_pts*2), np.linspace(0,1,num_pts*3), np.linspace(0,1,num_pts*5)
    eta2, eta3 = np.meshgrid(eta2_range, eta3_range, indexing='ij')
    eta2, eta3 = eta2.flatten(), eta3.flatten()

    # Get contour of outer surface.
    surface_contour_x = np.zeros((eta2_range.shape[0] * eta3_range.shape[0],))
    surface_contour_y = np.zeros((eta2_range.shape[0] * eta3_range.shape[0],))
    surface_contour_z = np.zeros((eta2_range.shape[0] * eta3_range.shape[0],))

    idx = 0
    for j, u in enumerate(eta2_range):
        for k, v in enumerate(eta3_range):
            (x, y, z) = f(eta1_range[-1], u, v)
            surface_contour_x[idx] = x
            surface_contour_y[idx] = y
            surface_contour_z[idx] = z
            idx += 1

    # Automatically generate unstructured triangles.
    # https://matplotlib.org/stable/api/tri_api.html#matplotlib.tri.Triangulation
    surface_tri = tri.Triangulation(eta2, eta3)

    for ax in axes_3D:
        ax.set_box_aspect((np.ptp(surface_contour_x), np.ptp(surface_contour_y), np.ptp(surface_contour_z)))
        ax.plot_trisurf(surface_contour_x, surface_contour_y, surface_contour_z, triangles=surface_tri.triangles, alpha=0.2)
        # ax.plot_trisurf(surface_contour_x, surface_contour_y, surface_contour_z, triangles=surface_tri.triangles, cmap=plt.cm.Spectral, alpha=0.5)



    # ============================================================
    # Draw 1D profiles.
    # ============================================================

    num_pts = 20
    eta1_range, eta2_range, eta3_range = np.linspace(0,1,num_pts), np.linspace(0,1,num_pts,endpoint=False), np.linspace(0,1,5,endpoint=False)
    eta1, eta2, eta3 = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=True)
    print('Shapes of eta1, eta2, eta3:', eta1.shape, eta2.shape, eta3.shape)

    xs, ys, zs = f(eta1, eta2, eta3)

    ax1_3D.set_title('Pressure profile')
    ax1_1D.set_title('Pressure profile')
    ax2_3D.set_title('Iota profile')
    ax2_1D.set_title('Iota profile')
    ax4_3D.set_title('q profile')
    ax4_1D.set_title('q profile')

    P = gvec.P(eta1, eta2, eta3)
    I = np.abs(gvec.IOTA(eta1, eta2, eta3))
    Q = 2 * np.pi / I

    img1 = ax1_3D.scatter(xs, ys, zs, c=P, cmap=plt.cm.plasma, marker='.')
    img2 = ax2_3D.scatter(xs, ys, zs, c=I, cmap=plt.cm.plasma, marker='.')
    img4 = ax4_3D.scatter(xs, ys, zs, c=Q, cmap=plt.cm.plasma, marker='.')

    cbar1 = fig.colorbar(img1, ax=ax1_3D, shrink=0.5, pad=0.1, label='Pressure $P(\eta^1)$')
    cbar2 = fig.colorbar(img2, ax=ax2_3D, shrink=0.5, pad=0.1, label='Absolute Iota $|\iota(\eta^1)|$')
    cbar4 = fig.colorbar(img4, ax=ax4_3D, shrink=0.5, pad=0.1, label='Absolute Safety factor $|q(\eta^1)|$')

    ax1_1D.plot(eta1[:,0,0], P[:,0,0], marker='.')
    ax2_1D.plot(eta1[:,0,0], I[:,0,0], marker='.')
    ax4_1D.plot(eta1[:,0,0], Q[:,0,0], marker='.')

    ax1_1D.set_ylabel('Pressure $P(\eta^1)$')
    ax2_1D.set_ylabel('Absolute Iota $|\iota(\eta^1)|$')
    ax4_1D.set_ylabel('Absolute Safety factor $|q(\eta^1)|$')

    print('Evaluate and plot 1D profiles done.')



    # ============================================================
    # Everything below is about B-field.
    # ============================================================



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
    # Define 2-form B-field as individual components.
    # ============================================================

    B2 = gvec.B_2

    # This is a stupid idea. Every function call to `B_2` is evaluated 3 times.
    B2_1 = lambda s,u,v: gvec.B_2(s,u,v)[0]
    B2_2 = lambda s,u,v: gvec.B_2(s,u,v)[1]
    B2_3 = lambda s,u,v: gvec.B_2(s,u,v)[2]

    # Supply a tuple of each B-field components B=(B1,B2,B3).
    if use_B_components:

        B2 = [B2_1, B2_2, B2_3] # Must be a list!

    # In this test case, B2 is already a function:
    hat_B2 = B2

    print('Loaded GVEC 2-form B(s,u,v).')



    # ============================================================
    # Push 2-form B-field to Cartesian.
    # ============================================================

    print('Pushing 2-form B-field to Cartesian...')

    num_pts = 12
    eta1_range, eta2_range, eta3_range = np.linspace(1e-12,1,num_pts), np.linspace(0,1,num_pts,endpoint=False), np.linspace(0,1,num_pts,endpoint=False)
    # eta1_range, eta2_range, eta3_range = np.linspace(0,1,num_pts+2)[2:], np.linspace(0,1,8,endpoint=False), np.linspace(0,0.5,8,endpoint=False)
    eta1, eta2, eta3 = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=True)
    print('Shapes of eta1, eta2, eta3:', eta1.shape, eta2.shape, eta3.shape)

    def flip_and_concat(a, eta2_pts, eta2_idx=0, eta3_idx=0):
        """Flip an array, then concatenate the flipped one with the original one.

        Normally along the radial direction, eta1 goes from 0 to 1.
        This here would return the MHD variables given eta2 corresponding to eta1 [-1, 1]."""

        return np.concatenate((np.flip(a[:,(eta2_idx+int(eta2_pts//2))%eta2_pts,eta3_idx]), a[:,eta2_idx,eta3_idx]))

    # To plot eta1 from -1 to 1 in a 1D profile.
    # eta1_axis = np.concatenate((np.flip(-eta1[:,0,0]), eta1[:,0,0]))
    # print('eta1_axis', eta1_axis)

    # Pull B(\eta) -> (Bx,By,Bz) to its 2-form representation.
    # evaled_B = [hat_B2_i(eta1, eta2, eta3) for hat_B2_i in hat_B2]
    # evaled_B = np.array(evaled_B)
    # print('2-form B evaluated at (eta1, eta2, eta3):', evaled_B)

    # Then push back to Cartesian to see if we get the same thing.
    pushed_Bx = domain.push(hat_B2, eta1, eta2, eta3, '2_form_1')
    pushed_By = domain.push(hat_B2, eta1, eta2, eta3, '2_form_2')
    pushed_Bz = domain.push(hat_B2, eta1, eta2, eta3, '2_form_3')
    if isinstance(eta1, np.ndarray):
        print('pushed_Bx.shape', pushed_Bx.shape)
    pushed_B  = [pushed_Bx, pushed_By, pushed_Bz]
    pushed_B  = np.array(pushed_B)
    print('pushed_B.shape', pushed_B.shape)
    print('Max B:', np.max(pushed_B))
    print('Min B:', np.min(pushed_B))
    # print('2-form B(eta1, eta2, eta3) pushed to Cartesian:', pushed_B)

    B_mag = np.sqrt(pushed_B[0]**2 + pushed_B[1]**2 + pushed_B[2]**2)
    print('B_mag.shape', B_mag.shape)
    print('Max |B|:', np.max(B_mag))
    print('Min |B|:', np.min(B_mag))

    print('Pushed 2-form B-field to Cartesian.')



    # ============================================================
    # Draw B-field.
    # ============================================================

    xs, ys, zs = f(eta1, eta2, eta3)
    print('Shapes of xs, ys, zs:', xs.shape, ys.shape, zs.shape)

    ax3_3D.set_title('B-field profile')
    # ax3_1D.set_title('B-field profile')

    # Flatten and normalize.
    c = (B_mag.ravel() - B_mag.min()) / B_mag.ptp()
    # Repeat for each body line and two head lines in a quiver.
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap.
    c = plt.cm.plasma(c)

    img3 = ax3_3D.quiver(xs.flatten(), ys.flatten(), zs.flatten(), pushed_B[0].flatten(), pushed_B[1].flatten(), pushed_B[2].flatten(), colors=c, length=1, arrow_length_ratio=.3, cmap=plt.cm.plasma)
    # img3.set_array(B_mag.flatten())
    img3.set_clim(np.min(B_mag), np.max(B_mag))
    cbar3 = fig.colorbar(img3, ax=ax3_3D, shrink=0.5, pad=0.1, label='B-field magnitude $|B|$')

    # Test different cross sections along eta1 direction, given eta3.
    # eta3_pt  = 0.5
    # # print(f'(x,y,z) coordinates along eta1 direction, given eta2=0 and eta3={eta3_pt}.')
    # # mapped = f(eta1_range, eta2_range, eta3_range)
    # eta3_idx = int(np.round(eta3_range.size * eta3_pt))
    # print(f'eta3_idx: {eta3_idx}')
    # # (x, y, z) = mapped[:, :, 0, eta3_idx]
    # # for i, s in enumerate(eta1_range):
    # #     print(eta1[:,0,0][i], eta1_range[i], (x[i], y[i], z[i]))

    # Plot 1D crossection of B_mag along eta1 direction, given eta3.
    # ax3_1D.plot(eta1_axis, flip_and_concat(B_mag, num_pts, eta3_idx=eta3_idx), marker='.', label='$\eta^3 = {}$'.format(eta3_pt))
    # # ax3_1D.plot(eta1[:,0,0], B_mag[:,0,0], marker='.')
    # # ax3_1D.plot(eta1[:,0,0], B_mag[:,0,eta3_idx], marker='.')

    # # for eta3_idx, v in enumerate(eta3_range):
    # #     if v < 0.5:
    # #         xs_axis = flip_and_concat(xs, num_pts, eta3_idx=eta3_idx)
    # #         ys_axis = flip_and_concat(ys, num_pts, eta3_idx=eta3_idx)
    # #         xys_axis = np.sqrt(xs_axis**2 + ys_axis**2)
    # #         # ax3_1D.plot(xys_axis, flip_and_concat(B_mag, num_pts, eta3_idx=eta3_idx), marker='.', label='$\eta^3 = {}$'.format(v))

    # ax3_1D.set_ylabel('B-field magnitude $|B|$')
    # ax3_1D.set_ylim(0.3,0.7)
    # ax3_1D.legend()

    print('Evaluate and plot 3D B-field done.')
    print(' ')



    # ============================================================
    # 3D Pi_2 projection.
    # ============================================================

    print('Test Div B = 0.')

    # Do the Pi_2 projection.
    b2_coeff = proj_3d.PI_2(*hat_B2) # Coefficients of projected 2-form.
    print('Shapes of each component of b2 coefficients. 2_1: {}, 2_2: {}, 2_3: {}'.format(b2_coeff[0].shape, b2_coeff[1].shape, b2_coeff[2].shape))
    print('Pi_2 projection of B-field done.')

    # Because discrete Div/Curl takes only 1D array.
    b2_coeff_concat = np.concatenate((b2_coeff[0].flatten(), b2_coeff[1].flatten(), b2_coeff[2].flatten()))
    print('Max b2 coeff: {}'.format(np.max(np.abs(b2_coeff_concat))))

    # Test `extract_2()`.
    b2_coeff2_1, b2_coeff2_2, b2_coeff2_3 = tensor_space_FEM.extract_2(b2_coeff_concat)
    # print('Shapes of each component of b2 coefficients. 2_1: {}, 2_2: {}, 2_3: {}'.format(b2_coeff2_1.shape, b2_coeff2_2.shape, b2_coeff2_3.shape))
    assert np.allclose(b2_coeff[0], b2_coeff2_1)
    assert np.allclose(b2_coeff[1], b2_coeff2_2)
    assert np.allclose(b2_coeff[2], b2_coeff2_3)



    # ============================================================
    # Show Div B = 0 in both real space and 2-form.
    # ============================================================

    # Take discrete Div.
    divB = tensor_space_FEM.D.dot(b2_coeff_concat)
    print('Shapes of discrete Div matrix: {}, flattened 2-form coeff: {}, and after taking discrete Div: {}.'.format(tensor_space_FEM.D.shape, b2_coeff_concat.shape, divB.shape))
    print('Maximum error (how close is Div B to 0): {}'.format(np.max(np.abs(divB))))

    assert tensor_space_FEM.D.shape[1] == b2_coeff[0].size + b2_coeff[1].size + b2_coeff[2].size, 'Matrix size should match for dicrete divergence.'
    assert np.max(np.abs(divB)) < 1e-9, 'Divergence of B should be zero.'

    print('Test Div B = 0 success.')
    print(' ')



    # ============================================================
    # Check if discrete Curl A1 == B2.
    # ============================================================

    print('Test Div Curl A = 0.')

    A1 = gvec.A_1

    # This is a stupid idea. Every function call to `A_1` is evaluated 3 times.
    A1_1 = lambda s,u,v: gvec.A_1(s,u,v)[0]
    A1_2 = lambda s,u,v: gvec.A_1(s,u,v)[1]
    A1_3 = lambda s,u,v: gvec.A_1(s,u,v)[2]

    # Supply a tuple of each A1 components A=(A1,A2,A3).
    if use_B_components:

        A1 = [A1_1, A1_2, A1_3] # Must be a list!

    hat_A1 = A1

    print('Loaded GVEC 1-form A(s,u,v).')

    # Do the Pi_1 projection.
    a1_coeff = proj_3d.PI_1(*hat_A1) # Coefficients of projected 1-form.
    print('Shapes of each component of a1 coefficients. 1_1: {}, 1_2: {}, 1_3: {}'.format(a1_coeff[0].shape, a1_coeff[1].shape, a1_coeff[2].shape))
    print('Pi_1 projection of A-field done.')

    # Because discrete Div/Curl takes only 1D array.
    a1_coeff_concat = np.concatenate((a1_coeff[0].flatten(), a1_coeff[1].flatten(), a1_coeff[2].flatten()))
    print('Max a1 coeff: {}'.format(np.max(np.abs(a1_coeff_concat))))

    # Take discrete Curl.
    curlA = tensor_space_FEM.C.dot(a1_coeff_concat)
    print('Shapes of discrete Curl matrix: {}, flattened 1-form coeff: {}, and after taking discrete Curl: {}.'.format(tensor_space_FEM.C.shape, a1_coeff_concat.shape, curlA.shape))
    curlA_1, curlA_2, curlA_3 = tensor_space_FEM.extract_2(curlA)
    print('Shapes of each component of Curl a1 coefficients. 2_1: {}, 2_2: {}, 2_3: {} (== shapes of b2 coefficients)'.format(curlA_1.shape, curlA_2.shape, curlA_3.shape))
    print('Maximum error between B2 and Curl A1: 2_1: {}, 2_2: {}, 2_3: {}'.format(np.max(np.abs(b2_coeff[0] - curlA_1)), np.max(np.abs(b2_coeff[1] - curlA_2)), np.max(np.abs(b2_coeff[2] - curlA_3))))
    assert np.allclose(b2_coeff[0], curlA_1, atol=1e-4)
    assert np.allclose(b2_coeff[1], curlA_2, atol=1e-4)
    assert np.allclose(b2_coeff[2], curlA_3, atol=1e-4)

    # Take discrete Div.
    divcurlA = tensor_space_FEM.D.dot(curlA)
    print('Shapes of discrete Div matrix: {}, flattened 2-form coeff: {}, and after taking discrete Div: {}.'.format(tensor_space_FEM.D.shape, curlA.shape, divcurlA.shape))
    print('Maximum error (how close is Div Curl A to 0): {}'.format(np.max(np.abs(divcurlA))))

    assert np.max(np.abs(divcurlA)) < 1e-9, 'Divergence of Curl A should be zero.'

    print('Test Div Curl A = 0 success.')
    print(' ')



    # ============================================================
    # Compute current Curl B1 == J.
    # ============================================================

    print('Test Curl B = J.')

    B1 = gvec.B_1

    # This is a stupid idea. Every function call to `B_1` is evaluated 3 times.
    B1_1 = lambda s,u,v: gvec.B_1(s,u,v)[0]
    B1_2 = lambda s,u,v: gvec.B_1(s,u,v)[1]
    B1_3 = lambda s,u,v: gvec.B_1(s,u,v)[2]

    # Supply a tuple of each B1 components B=(B1,B2,B3).
    if use_B_components:

        B1 = [B1_1, B1_2, B1_3] # Must be a list!

    hat_B1 = B1

    print('Loaded GVEC 1-form B(s,u,v).')

    # Do the Pi_1 projection.
    b1_coeff = proj_3d.PI_1(*hat_B1) # Coefficients of projected 1-form.
    print('Shapes of each component of b1 coefficients. 1_1: {}, 1_2: {}, 1_3: {}'.format(b1_coeff[0].shape, b1_coeff[1].shape, b1_coeff[2].shape))
    print('Pi_1 projection of B-field done.')

    # Because discrete Div/Curl takes only 1D array.
    b1_coeff_concat = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten()))
    print('Max b1 coeff: {}'.format(np.max(np.abs(b1_coeff_concat))))

    # Take discrete Curl.
    curlB = tensor_space_FEM.C.dot(b1_coeff_concat)
    print('Shapes of discrete Curl matrix: {}, flattened 1-form coeff: {}, and after taking discrete Curl: {}.'.format(tensor_space_FEM.C.shape, b1_coeff_concat.shape, curlB.shape))
    curlB_1, curlB_2, curlB_3 = tensor_space_FEM.extract_2(curlB)
    print('Shapes of each component of J = Curl b1 coefficients. 2_1: {}, 2_2: {}, 2_3: {} (== shapes of b2 coefficients)'.format(curlB_1.shape, curlB_2.shape, curlB_3.shape))
    print('Maximum error (how close is J to 0): {} (Not really an error.)'.format(np.max(np.abs(curlB))))
    # Florian: The current is normally non-zero in a Stellarator.
    # One can impose a zero toroidal current density profile, total current density (\mu_0 J = \nabla \times B\) is still non-zero.
    # In our simulations up to now in GVEC, we cannot impose that condition yet, so we use a given iota profile, which in general, produces also toroidal current.
    # If we were to use a w7x equilibrium with a known iota profile, the toroidal current would be small.

    # Take discrete Div.
    divJ = tensor_space_FEM.D.dot(curlB)
    print('Shapes of discrete Div matrix: {}, flattened 2-form coeff: {}, and after taking discrete Div: {}.'.format(tensor_space_FEM.D.shape, curlB.shape, divJ.shape))
    print('Maximum error (how close is Div J to 0): {}'.format(np.max(np.abs(divJ))))

    assert np.max(np.abs(divJ)) < 1e-9, 'Divergence of J (Curl B) should be zero.'

    print('Test Div Curl B = Div J = 0 success.')
    print(' ')



    # ============================================================
    # Show the figure.
    # ============================================================

    print('Before `tight_layout()`   | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.tight_layout()
    print('After `tight_layout()`    | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.subplots_adjust(wspace=fig.subplotpars.wspace * 0.6)
    print('After `subplots_adjust()` | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    if plot:
        plt.show()



if __name__ == "__main__":
    test_GVEC_equilibrium(plot=True)
