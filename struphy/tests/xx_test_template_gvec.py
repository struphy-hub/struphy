def test_template_gvec(num_s=21, num_u=4, num_v=5):
    """Test if `simulations/template_gvec/equilibrium_MHD.py` runs correctly."""

    # ============================================================
    # Imports.
    # ============================================================

    import numpy             as np

    import os
    import sys

    import h5py
    import tempfile
    temp_dir = tempfile.TemporaryDirectory(prefix='STRUPHY-')
    print(f'Created temp directory at: {temp_dir.name}')

    # which diagnostics is run
    print('Run diagnostics:', sys.argv[0])

    basedir = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, os.path.join(basedir, '..'))

    # Import necessary struphy.modules.
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl
    from gvec_to_python import GVEC, Form, Variable
    from gvec_to_python.reader.gvec_reader import GVEC_Reader

    gvec_dir = os.path.abspath(os.path.join(basedir, '..', 'mhd_equil', 'gvec'))
    print(f'Path to GVEC eq    : {gvec_dir}')
    gvec_files   = [f for f in os.listdir(gvec_dir) if os.path.isfile(os.path.join(gvec_dir, f))]
    gvec_folders = [f for f in os.listdir(gvec_dir) if os.path.isdir( os.path.join(gvec_dir, f))]
    print(f'Files in GVEC eq   : {gvec_files}')
    print(f'Folders in GVEC eq : {gvec_folders}')
    print(' ')

    from struphy.mhd_equil.gvec.mhd_equil_gvec import Equilibrium_mhd_gvec
    from struphy.mhd_equil.mhd_equil_physical  import Equilibrium_mhd_physical



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
    # Create fake input params.
    # ============================================================

    # Fake input params.
    params = {
        "mhd_equilibrium" : {
            "general" : {
                "type" : "gvec"
            },
            "params_gvec" : {
                "filepath" : 'struphy/mhd_equil/gvec/',
                "filename" : 'GVEC_ellipStell_profile_update_State_0000_00010000.dat', # .dat or .json
            },
        },
    }

    params['mhd_equilibrium']['params_gvec']['filepath'] = gvec_dir # Overwrite path, because this is a test file.
    params = params['mhd_equilibrium']['params_gvec']



    # ============================================================
    # Convert GVEC .dat output to .json.
    # ============================================================

    if params['filename'].endswith('.dat'):

        read_filepath = params['filepath']
        read_filename = params['filename']
        gvec_filepath = temp_dir.name
        gvec_filename = params['filename'][:-4] + '.json'
        reader = GVEC_Reader(read_filepath, read_filename, gvec_filepath, gvec_filename, with_spl_coef=True)

    elif params['filename'].endswith('.json'):

        gvec_filepath = params['filepath']
        gvec_filename = params['filename'][:-4] + '.json'



    # ============================================================
    # Load GVEC mapping.
    # ============================================================

    gvec = GVEC(gvec_filepath, gvec_filename)

    # f = gvec.mapfull.f # Full mapping, (s,u,v) to (x,y,z).
    # X = gvec.mapX.f    # Only x component of the mapping.
    # Y = gvec.mapY.f    # Only y component of the mapping.
    # Z = gvec.mapZ.f    # Only z component of the mapping.
    print('Loaded default GVEC mapping.')



    # ===============================================================
    # Map source domain
    # ===============================================================

    # Enable another layer of mapping, from STRUPHY's (eta1,eta2,eta3) to GVEC's (s,u,v).
    # bounds = [0.3,0.8,0.3,0.8,0.3,0.8]
    bounds = {'b1': 0.3, 'e1': 0.8, 'b2': 0.3, 'e2': 0.8, 'b3': 0.3, 'e3': 0.8}
    SOURCE_DOMAIN = dom.Domain('cuboid', params_map=bounds)

    def f(eta1, eta2, eta3):
        """Mapping that goes from (eta1,eta2,eta3) to (s,u,v) then to (x,y,z). All (x,y,z) components."""
        return gvec.mapfull.f(s(eta1,eta2,eta3), u(eta1,eta2,eta3), v(eta1,eta2,eta3))

    def X(eta1, eta2, eta3):
        """Mapping that goes from (eta1,eta2,eta3) to (s,u,v) then to (x,y,z). Only x-component."""
        return gvec.mapX.f(s(eta1,eta2,eta3), u(eta1,eta2,eta3), v(eta1,eta2,eta3))

    def Y(eta1, eta2, eta3):
        """Mapping that goes from (eta1,eta2,eta3) to (s,u,v) then to (x,y,z). Only y-component."""
        return gvec.mapY.f(s(eta1,eta2,eta3), u(eta1,eta2,eta3), v(eta1,eta2,eta3))

    def Z(eta1, eta2, eta3):
        """Mapping that goes from (eta1,eta2,eta3) to (s,u,v) then to (x,y,z). Only z-component."""
        return gvec.mapZ.f(s(eta1,eta2,eta3), u(eta1,eta2,eta3), v(eta1,eta2,eta3))

    def s(eta1, eta2, eta3):
        return SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'x')

    def u(eta1, eta2, eta3):
        return SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'y')

    def v(eta1, eta2, eta3):
        return SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'z')

    def eta123_to_suv(eta1, eta2, eta3):
        return s(eta1, eta2, eta3), u(eta1, eta2, eta3), v(eta1, eta2, eta3)



    # ============================================================
    # Create FEM space.
    # ============================================================

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [spl.Spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]

    # 2d tensor-product B-spline space for polar splines (if used)
    tensor_space_pol = spl.Tensor_spline_space(spaces_FEM[:2])

    # 3d tensor-product B-spline space for finite elements
    TENSOR_SPACE_FEM = spl.Tensor_spline_space(spaces_FEM)
    print('Tensor space set up done.')



    # ============================================================
    # 3D projection using `projectors_tensor_3d`.
    # ============================================================

    # Create 3D projector. It's not automatic.
    for space in spaces_FEM:
        if not hasattr(space, 'projectors'):
            space.set_projectors() # def set_projectors(self, nq=6):
    if not hasattr(TENSOR_SPACE_FEM, 'projectors'):
        TENSOR_SPACE_FEM.set_projectors() # def set_projectors(self, which='tensor'). Use 'general' for polar splines.
    PROJ_3D = TENSOR_SPACE_FEM.projectors
    print('Create 3D projector done.')



    # ============================================================
    # Create splines (using PI_0 projector).
    # ============================================================

    # Calculate spline coefficients using PI_0 projector.
    cx = PROJ_3D.PI_0(X)
    cy = PROJ_3D.PI_0(Y)
    cz = PROJ_3D.PI_0(Z)

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

    DOMAIN = dom.Domain('spline', params_map=params_map)
    print('Computed spline coefficients.')



    # ============================================================
    # Initialize the `Equilibrium_mhd_gvec` class.
    # ============================================================

    # Dummy. Physical equilibrium is not used.
    mhd_equil_type = 'slab'
    params_slab = {
        'B0x'         : 1.,   # magnetic field in Tesla (x)
        'B0y'         : 0.,   # magnetic field in Tesla (y)
        'B0z'         : 0.,   # magnetic field in Tesla (z)
        'rho0'        : 1.,   # equilibirum mass density
        'beta'        : 0.,   # plasma beta in %
    }
    EQ_MHD_P = Equilibrium_mhd_physical(mhd_equil_type, params_slab)

    # Actual initialization.
    EQ_MHD = Equilibrium_mhd_gvec(params, DOMAIN, EQ_MHD_P, TENSOR_SPACE_FEM, SOURCE_DOMAIN)
    print('Initialized the `Equilibrium_mhd_gvec` class.')

    temp_dir.cleanup()
    print('Removed temp directory.')



    # ============================================================
    # Draw 1D profiles.
    # ============================================================

    num_pts = 20
    eta1_range, eta2_range, eta3_range = np.linspace(0,1,num_pts), np.linspace(0,1,num_pts,endpoint=False), np.linspace(0,1,5,endpoint=False)
    eta1, eta2, eta3 = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=True)
    print('Shapes of eta1, eta2, eta3:', eta1.shape, eta2.shape, eta3.shape)

    j2_1 = EQ_MHD.j2_eq_1(eta1,eta2,eta3)
    print('j2_1.shape: {}'.format(j2_1.shape))

    j_x = EQ_MHD.j_eq_x(eta1,eta2,eta3)
    print('j_x.shape: {}'.format(j_x.shape))



    # ============================================================
    # Write to ParaView.
    # ============================================================

    import vtk
    from struphy.diagnostics.paraview.vtk_writer import vtkWriter
    import struphy.diagnostics.paraview.mesh_creator as MC

    print(f'Writing result to ParaView')
    print(f'VTK version: {vtk.vtkVersion.GetVTKVersion()}')

    # Output directory.
    vtk_dir = os.path.join(basedir, 'paraview_output')

    # Class implementation of a ParaView writer.
    writer = vtkWriter('vtu')

    # Sample points uniformly in (s, u, v) and convert them to (x, y, z).
    # s_range = np.arange(0, 1.0001, 0.1)
    # u_range = np.arange(0, 1.0000, 0.05)  # Skipping the last point because periodic.
    # v_range = np.arange(0, 1.0000, 0.025) # Skipping the last point because periodic.
    periodic = True
    if periodic:
        s_range = np.linspace(0, 1, num_s)
        u_range = np.linspace(0, 1, num_u+1)[:-1] # Skipping the last point because periodic.
        v_range = np.linspace(0, 1, num_v+1)[:-1] # Skipping the last point because periodic.
    else:
        s_range = np.linspace(0, 1, num_s)
        u_range = np.linspace(0, 1, num_u)
        v_range = np.linspace(0, 1, num_v)
    use_GVEC_grid = True
    if use_GVEC_grid:
        s_range = np.array(gvec.data['grid']['sGrid'])
    s_range[0] = 1e-12 # Bypass 0 if it blows up.

    # TODO: Generalize MC functions to accept a map and an Equilibrium_mhd_gvec class, instead of a GVEC class.
    filename = params['filename'][:-5]
    MC.make_ugrid_and_write_vtu(filename, writer, vtk_dir, gvec, s_range, u_range, v_range, periodic)
    gvec.mapfull.clear_cache()



if __name__ == "__main__":
    test_template_gvec()
