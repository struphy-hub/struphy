def test_polar_splines_3D():
    """
    Test constructing 3D polar splines for a non-axisymmetric GVEC equilibrium 
    via an intermediate axisymmetric torus mapping.
    """

    # ============================================================
    # Imports.
    # ============================================================

    import numpy             as np
    import matplotlib.pyplot as plt
    import matplotlib.tri    as tri

    import os
    import sys
    sys.path.append('..') # Because we are inside './test/' directory.
    sys.path.append('simulations/template_gvec/') # Because `template_gvec` is elsewhere.
    sys.path.append('../simulations/template_gvec/') # Because `template_gvec` is elsewhere.
    sys.path.append('../../simulations/template_gvec/') # Because `template_gvec` is elsewhere.
    sys.path.append('../../simulations/template_gvec/input_run/') # Because `template_gvec` is elsewhere.

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

    simdir = os.path.dirname(os.path.abspath(os.path.join(basedir, '../../simulations/template_gvec/')))
    print(f'Path to simulations    : {simdir}')
    simfiles   = [f for f in os.listdir(simdir) if os.path.isfile(os.path.join(simdir, f))]
    simfolders = [f for f in os.listdir(simdir) if os.path.isdir( os.path.join(simdir, f))]
    print(f'Files in simulations   : {simfiles}')
    print(f'Folders in simulations : {simfolders}')
    print(' ')

    templatedir = os.path.dirname(os.path.join(simdir, 'template_gvec/'))
    print(f'Path to GVEC template : {templatedir}')
    templatefiles   = [f for f in os.listdir(templatedir) if os.path.isfile(os.path.join(templatedir, f))]
    templatefolders = [f for f in os.listdir(templatedir) if os.path.isdir( os.path.join(templatedir, f))]
    print(f'Files in template     : {templatefiles}')
    print(f'Folders in template   : {templatefolders}')
    print(' ')

    from input_run.equilibrium_MHD import equilibrium_mhd



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

    # f = gvec.mapfull.f # Full mapping, (s,u,v) to (x,y,z).
    # X = gvec.mapX.f    # Only x component of the mapping.
    # Y = gvec.mapY.f    # Only y component of the mapping.
    # Z = gvec.mapZ.f    # Only z component of the mapping.
    print('Loaded default GVEC mapping.')



    # ===============================================================
    # Map source domain
    # ===============================================================

    # Enable another layer of mapping, from (eta1,eta2,eta3) to (s,u,v).
    # bounds = [0.3,0.8,0.3,0.8,0.3,0.8]
    bounds = {'b1': 0.3, 'e1': 0.8, 'b2': 0.3, 'e2': 0.8, 'b3': 0.3, 'e3': 0.8}
    source_domain = dom.Domain('cuboid', params_map=bounds)

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
        return source_domain.evaluate(eta1, eta2, eta3, 'x')

    def u(eta1, eta2, eta3):
        return source_domain.evaluate(eta1, eta2, eta3, 'y')

    def v(eta1, eta2, eta3):
        return source_domain.evaluate(eta1, eta2, eta3, 'z')

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
    # TODO: Add mapping from [0,1] to whatever section of mapping, and pass that to MHD init. ???
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

    # TODO: spline + params_map=None ???
    # TODO: Now we have spline + params_map=(begin, end)*3 ???
    domain = dom.Domain('spline', params_map=params_map)
    print('Computed spline coefficients.')



    # ============================================================
    # Initialize the `equilibrium_mhd` class.
    # ============================================================

    eq_mhd = equilibrium_mhd(tensor_space_FEM, domain, source_domain, filepath, filename)
    print('Initialized the `equilibrium_mhd` class.')

    temp_dir.cleanup()
    print('Removed temp directory.')



if __name__ == "__main__":
    test_polar_splines_3D()