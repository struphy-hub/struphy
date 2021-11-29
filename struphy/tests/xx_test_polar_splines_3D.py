def test_polar_splines_3D(plot=False):
    """Test constructing 3D polar splines for a non-axisymmetric GVEC equilibrium via an intermediate axisymmetric torus mapping.
    """

    # ============================================================
    # Imports.
    # ============================================================

    import os
    import sys
    import tempfile
    temp_dir = tempfile.TemporaryDirectory(prefix='STRUPHY-')
    print(f'Created temp directory at: {temp_dir.name}')

    # which diagnostics is run
    print('Run diagnostics:', sys.argv[0])

    basedir = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, os.path.join(basedir, '..'))



    # ============================================================
    # Configure STRUPHY splines.
    # ============================================================

    # Create splines to STRUPHY, unrelated to splines in GVEC.
    # Nel      = [8, 30, 6]
    Nel      = [8, 12, 6]
    # p        = [3, 1, 1]
    p        = [3, 3, 3]
    nq_el    = [4, 4, 4]  # Element integration, quadrature points per grid cell
    nq_pr    = [4, 4, 4]  # Greville integration, quadrature points per histopolation cell (for projection)
    bc       = ['f', 'f'] # BC in s-direction
    spl_kind = [False, True, True] # Spline type: True=periodic, False=clamped



    # ============================================================
    # Run test cases.
    # ============================================================

    print('Done testing 3D polar splines.')
    case_01_circle_identity(Nel, p, spl_kind, nq_el, nq_pr, bc, plot)
    # case_02_circle_scaled(Nel, p, spl_kind, nq_el, nq_pr, bc, plot)
    # case_03_ellipse(Nel, p, spl_kind, nq_el, nq_pr, bc, plot)
    # case_04_ellipse_rotated(Nel, p, spl_kind, nq_el, nq_pr, bc, plot)
    # case_05_circular_shifted(Nel, p, spl_kind, nq_el, nq_pr, bc, plot)
    # case_06_spline_3D(Nel, p, spl_kind, nq_el, nq_pr, bc, plot)
    print('Done testing 3D polar splines.')



def case_01_circle_identity(Nel, p, spl_kind, nq_el, nq_pr, bc, plot):

    import numpy as np
    import matplotlib.pyplot as plt
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test case 01: Identity map.')



    # ============================================================
    # Define mappings F = F2 \circ F1.
    # ============================================================

    # Map F:
    DOMAIN_F = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 2., 'R0': 10.})

    def F_x(eta1, eta2, eta3):
        return DOMAIN_F.evaluate(eta1, eta2, eta3, 'x')
    def F_y(eta1, eta2, eta3):
        return DOMAIN_F.evaluate(eta1, eta2, eta3, 'y')
    def F_z(eta1, eta2, eta3):
        return DOMAIN_F.evaluate(eta1, eta2, eta3, 'z')

    # Map F1: Canonical disk.
    DOMAIN_F1 = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 1., 'R0': 10.})

    def F1_x(eta1, eta2, eta3):
        return DOMAIN_F1.evaluate(eta1, eta2, eta3, 'x')
    def F1_y(eta1, eta2, eta3):
        return DOMAIN_F1.evaluate(eta1, eta2, eta3, 'y')
    def F1_z(eta1, eta2, eta3):
        return DOMAIN_F1.evaluate(eta1, eta2, eta3, 'z')

    print('Domain maps defined.')



    # ============================================================
    # Create FEM space and setup projectors.
    # ============================================================

    # 1D B-spline spline spaces for finite elements.
    spaces_FEM = [spl.Spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]
    [space.set_projectors(nq=nq) for space, nq in zip(spaces_FEM, nq_pr) if not hasattr(space, 'projectors')]

    # 3D tensor-product B-spline space for finite elements.
    # Independent of Domain.
    TENSOR_SPACE = spl.Tensor_spline_space(spaces_FEM, ck=-1)
    if not hasattr(TENSOR_SPACE, 'projectors'):
        TENSOR_SPACE.set_projectors('general') # def set_projectors(self, which='tensor'). Use 'general' for polar splines.
    print('Tensor space and projector set up done.')



    # ============================================================
    # Evaluate cx, cy coefficients for initializing polar splines.
    # ============================================================

    cx_F  = TENSOR_SPACE.projectors.pi_0(F_x)
    cy_F  = TENSOR_SPACE.projectors.pi_0(F_y)
    cx_F1 = TENSOR_SPACE.projectors.pi_0(F1_x)
    cy_F1 = TENSOR_SPACE.projectors.pi_0(F1_y)

    cx_F  = TENSOR_SPACE.extract_0(cx_F)
    cy_F  = TENSOR_SPACE.extract_0(cy_F)
    cx_F1 = TENSOR_SPACE.extract_0(cx_F1)
    cy_F1 = TENSOR_SPACE.extract_0(cy_F1)

    POLAR_SPACE_F  = spl.Tensor_spline_space(spaces_FEM, ck=1, cx=cx_F[:, :, 0], cy=cy_F[:, :, 0])
    POLAR_SPACE_F1 = spl.Tensor_spline_space(spaces_FEM, ck=1, cx=cx_F1[:, :, 0], cy=cy_F1[:, :, 0])

    POLAR_SPACE_F.set_projectors('general')
    POLAR_SPACE_F1.set_projectors('general')

    print('Polar space and projector set up done.')

    # Compare coefficients obtained from interp_mapping.
    # To obtain 2D coefficients, eta3 must be an optional parameter for the map itself (e.g. F_x).
    # But since we are using a Domain object, eta3 isn't optional in Domain.evaluate().
    cx_interp_F, cy_interp_F, cz_interp_F = dom.interp_mapping(Nel, p, spl_kind, F_x, F_y, F_z)
    print(f'Shape of coefficients from PI_0          : cx {cx_F.shape} cy {cy_F.shape}')
    print(f'Shape of coefficients from interp_mapping: cx {cx_interp_F.shape} cy {cy_interp_F.shape}')
    print(f'Are PI_0 and interp_mapping equivalent? cx:{np.allclose(cx_F, cx_interp_F)} cy:{np.allclose(cy_F, cy_interp_F)}')



    # ============================================================
    # Test functions.
    # ============================================================

    # TODO: take care of boundary conditions, shifts, etc. (-> define proper function)
    # The Gaussian should be shifted away from (0,0), 
    # otherwise its derivative at the pole is analytically zero, 
    # and would not be of interest to us,
    # because we want to test C1 continuity.
    # A jump is expected in TENSOR_SPACE.

    # # Shifted Gaussian.
    # shift_x = 0.02 + 10
    # shift_y = 0.04
    # hw = 0.16
    # fun = lambda x, y, z : np.exp(-((x - shift_x)**2 + (y - shift_y)**2) / hw**2)

    # # Gradient to the "test function".
    # pfpx = lambda x, y, z : np.exp(-((x - shift_x)**2 + (y - shift_y)**2) / hw**2) * (- 2 * (x - shift_x) / hw**2)
    # pfpy = lambda x, y, z : np.exp(-((x - shift_x)**2 + (y - shift_y)**2) / hw**2) * (- 2 * (y - shift_y) / hw**2)

    # # Shifted sin/cos.
    # shift_x = 0.5 + 10
    # shift_y = 0.5
    # k_x = 2 * np.pi
    # k_y = 2 * np.pi

    # fun = lambda x, y, z : np.sin(k_x * (x - shift_x)) * np.cos(k_y * (y - shift_y))

    # # Gradient to the "test function".
    # pfpx = lambda x, y, z :   k_x * np.cos(k_x * (x - shift_x)) * np.cos(k_y * (y - shift_y))
    # pfpy = lambda x, y, z : - k_y * np.sin(k_x * (x - shift_x)) * np.sin(k_y * (y - shift_y))

    # Rotated sin/cos.
    shift_x = 0.02 + 10
    shift_y = 0.1
    k_x = 2 * np.pi * 0.23 * 2
    k_y = 2 * np.pi * 0.13 * 2

    fun = lambda x, y, z : np.sin(k_x * (x - shift_x + y)) * np.cos(k_y * (y - shift_y - x))

    # Gradient to the "test function".
    pfpx = lambda x, y, z :   k_x * np.cos(k_x * (x - shift_x + y)) * np.cos(k_y * (y - shift_y - x)) + k_y * np.sin(k_x * (x - shift_x + y)) * np.sin(k_y * (y - shift_y - x))
    pfpy = lambda x, y, z : - k_y * np.sin(k_x * (x - shift_x + y)) * np.sin(k_y * (y - shift_y - x)) + k_x * np.cos(k_x * (x - shift_x + y)) * np.cos(k_y * (y - shift_y - x))



    # Use F for pullback!!!
    def fun_L(eta1, eta2, eta3):
        return DOMAIN_F.pull(fun, eta1, eta2, eta3, kind_fun='0_form')

    proj_tensor   = TENSOR_SPACE.projectors.pi_0(fun_L)
    proj_polar_F  = POLAR_SPACE_F.projectors.pi_0(fun_L)
    proj_polar_F1 = POLAR_SPACE_F1.projectors.pi_0(fun_L)
    print(f'Shape of proj_tensor   : {proj_tensor.shape}')
    print(f'Shape of proj_polar_F  : {proj_polar_F.shape}')
    print(f'Shape of proj_polar_F1 : {proj_polar_F1.shape}')

    # Apply discrete gradient -> 1-form. Are all 3 components continuous when pushed forward?
    proj_tensor_1form   = TENSOR_SPACE.G.dot(proj_tensor)
    proj_polar_F_1form  = POLAR_SPACE_F.G.dot(proj_polar_F)
    proj_polar_F1_1form = POLAR_SPACE_F1.G.dot(proj_polar_F1)

    f1_1_ten   , f1_2_ten   , f1_3_ten    = TENSOR_SPACE.extract_1(proj_tensor_1form)
    f1_1_pol_F , f1_2_pol_F , f1_3_pol_F  = POLAR_SPACE_F.extract_1(proj_polar_F_1form)
    f1_1_pol_F1, f1_2_pol_F1, f1_3_pol_F1 = POLAR_SPACE_F1.extract_1(proj_polar_F1_1form)

    # TODO: evaluate splines, push forward
    eta1_range = np.linspace(1e-10, 1, 101)
    eta2_range = np.linspace(0, 1, 101)
    eta3_range = np.linspace(0, 1, 3)

    eta1_sparse, eta2_sparse, eta3_sparse = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=True)
    eta1_dense,  eta2_dense,  eta3_dense  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)
    eta1,  eta2,  eta3  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)

    # Evaluate test function.
    evaled_tensor   = TENSOR_SPACE.evaluate_NNN(eta1, eta2, eta3, proj_tensor)
    evaled_polar_F  = POLAR_SPACE_F.evaluate_NNN(eta1, eta2, eta3, proj_polar_F)
    evaled_polar_F1 = POLAR_SPACE_F1.evaluate_NNN(eta1, eta2, eta3, proj_polar_F1)

    # Evaluate derivative.
    evaled_1_1_tensor   = TENSOR_SPACE.evaluate_DNN(eta1, eta2, eta3, f1_1_ten)
    evaled_1_1_polar_F  = POLAR_SPACE_F.evaluate_DNN(eta1, eta2, eta3, f1_1_pol_F)
    evaled_1_1_polar_F1 = POLAR_SPACE_F1.evaluate_DNN(eta1, eta2, eta3, f1_1_pol_F1)

    evaled_1_2_tensor   = TENSOR_SPACE.evaluate_NDN(eta1, eta2, eta3, f1_2_ten)
    evaled_1_2_polar_F  = POLAR_SPACE_F.evaluate_NDN(eta1, eta2, eta3, f1_2_pol_F)
    evaled_1_2_polar_F1 = POLAR_SPACE_F1.evaluate_NDN(eta1, eta2, eta3, f1_2_pol_F1)

    evaled_1_3_tensor   = TENSOR_SPACE.evaluate_NND(eta1, eta2, eta3, f1_3_ten)
    evaled_1_3_polar_F  = POLAR_SPACE_F.evaluate_NND(eta1, eta2, eta3, f1_3_pol_F)
    evaled_1_3_polar_F1 = POLAR_SPACE_F1.evaluate_NND(eta1, eta2, eta3, f1_3_pol_F1)

    evaled_1_tensor   = [evaled_1_1_tensor  , evaled_1_2_tensor  , evaled_1_3_tensor  ]
    evaled_1_polar_F  = [evaled_1_1_polar_F , evaled_1_2_polar_F , evaled_1_3_polar_F ]
    evaled_1_polar_F1 = [evaled_1_1_polar_F1, evaled_1_2_polar_F1, evaled_1_3_polar_F1]

    # Push to canonical domain. Polar splines should be smooth for the polar_F1. Only for polar_F1.
    # pushed_F1_0_tensor   = DOMAIN_F1.push(evaled_tensor, eta1, eta2, eta3, kind_fun='0_form')
    # pushed_F1_0_polar_F  = DOMAIN_F1.push(evaled_polar_F, eta1, eta2, eta3, kind_fun='0_form')
    pushed_F1_0_polar_F1 = DOMAIN_F1.push(evaled_polar_F1, eta1, eta2, eta3, kind_fun='0_form')

    # pushed_F1_1_1_tensor   = DOMAIN_F1.push(evaled_1_tensor, eta1, eta2, eta3, kind_fun='1_form_1')
    # pushed_F1_1_1_polar_F  = DOMAIN_F1.push(evaled_1_polar_F, eta1, eta2, eta3, kind_fun='1_form_1')
    pushed_F1_1_1_polar_F1 = DOMAIN_F1.push(evaled_1_polar_F1, eta1, eta2, eta3, kind_fun='1_form_1')

    # pushed_F1_1_2_tensor   = DOMAIN_F1.push(evaled_1_tensor, eta1, eta2, eta3, kind_fun='1_form_2')
    # pushed_F1_1_2_polar_F  = DOMAIN_F1.push(evaled_1_polar_F, eta1, eta2, eta3, kind_fun='1_form_2')
    pushed_F1_1_2_polar_F1 = DOMAIN_F1.push(evaled_1_polar_F1, eta1, eta2, eta3, kind_fun='1_form_2')

    # pushed_F1_1_3_tensor   = DOMAIN_F1.push(evaled_1_tensor, eta1, eta2, eta3, kind_fun='1_form_3')
    # pushed_F1_1_3_polar_F  = DOMAIN_F1.push(evaled_1_polar_F, eta1, eta2, eta3, kind_fun='1_form_3')
    pushed_F1_1_3_polar_F1 = DOMAIN_F1.push(evaled_1_polar_F1, eta1, eta2, eta3, kind_fun='1_form_3')

    # pushed_F1_1_tensor   = np.array([pushed_F1_1_1_tensor  , pushed_F1_1_2_tensor  , pushed_F1_1_3_tensor  ])
    # pushed_F1_1_polar_F  = np.array([pushed_F1_1_1_polar_F , pushed_F1_1_2_polar_F , pushed_F1_1_3_polar_F ])
    pushed_F1_1_polar_F1 = np.array([pushed_F1_1_1_polar_F1, pushed_F1_1_2_polar_F1, pushed_F1_1_3_polar_F1])

    # Or push to physical.
    pushed_F_tensor   = DOMAIN_F.push(evaled_tensor, eta1, eta2, eta3, kind_fun='0_form')
    pushed_F_polar_F  = DOMAIN_F.push(evaled_polar_F, eta1, eta2, eta3, kind_fun='0_form')
    pushed_F_polar_F1 = DOMAIN_F.push(evaled_polar_F1, eta1, eta2, eta3, kind_fun='0_form')

    pushed_F_1_1_tensor   = DOMAIN_F.push(evaled_1_tensor, eta1, eta2, eta3, kind_fun='1_form_1')
    pushed_F_1_1_polar_F  = DOMAIN_F.push(evaled_1_polar_F, eta1, eta2, eta3, kind_fun='1_form_1')
    pushed_F_1_1_polar_F1 = DOMAIN_F.push(evaled_1_polar_F1, eta1, eta2, eta3, kind_fun='1_form_1')

    pushed_F_1_2_tensor   = DOMAIN_F.push(evaled_1_tensor, eta1, eta2, eta3, kind_fun='1_form_2')
    pushed_F_1_2_polar_F  = DOMAIN_F.push(evaled_1_polar_F, eta1, eta2, eta3, kind_fun='1_form_2')
    pushed_F_1_2_polar_F1 = DOMAIN_F.push(evaled_1_polar_F1, eta1, eta2, eta3, kind_fun='1_form_2')

    pushed_F_1_3_tensor   = DOMAIN_F.push(evaled_1_tensor, eta1, eta2, eta3, kind_fun='1_form_3')
    pushed_F_1_3_polar_F  = DOMAIN_F.push(evaled_1_polar_F, eta1, eta2, eta3, kind_fun='1_form_3')
    pushed_F_1_3_polar_F1 = DOMAIN_F.push(evaled_1_polar_F1, eta1, eta2, eta3, kind_fun='1_form_3')

    pushed_F_1_tensor   = np.array([pushed_F_1_1_tensor  , pushed_F_1_2_tensor  , pushed_F_1_3_tensor  ])
    pushed_F_1_polar_F  = np.array([pushed_F_1_1_polar_F , pushed_F_1_2_polar_F , pushed_F_1_3_polar_F ])
    pushed_F_1_polar_F1 = np.array([pushed_F_1_1_polar_F1, pushed_F_1_2_polar_F1, pushed_F_1_3_polar_F1])

    # Analytical.
    x = F_x(eta1, eta2, eta3)
    y = F_y(eta1, eta2, eta3)
    z = F_z(eta1, eta2, eta3)
    orig_fun = fun(x, y, z)
    orig_df  = np.array([pfpx(x, y, z), pfpy(x, y, z), np.zeros_like(orig_fun)])
    # orig_df  = np.array([pfpx(eta1, eta2, eta3), pfpy(eta1, eta2, eta3), np.zeros_like(orig_fun)])

    print(f'Is pushed tensor equivalent to the original function? {np.allclose(orig_fun, pushed_F_tensor)}')
    print(f'Is pushed polar_F equivalent to the original function? {np.allclose(orig_fun, pushed_F_polar_F)}')
    print(f'Is pushed polar_F1 equivalent to the original function? {np.allclose(orig_fun, pushed_F_polar_F1)}')
    print(f'Is pushed tensor equivalent to pushed polar_F? {np.allclose(pushed_F_tensor, pushed_F_polar_F)}')
    print(f'Shape of orig_fun          : {orig_fun.shape}')
    print(f'Shape of orig_df           : {orig_df.shape}')
    print(f'Shape of pushed_F_tensor   : {pushed_F_tensor.shape}')
    print(f'Shape of pushed_F_polar_F  : {pushed_F_polar_F.shape}')
    print(f'Shape of pushed_F_polar_F1 : {pushed_F_polar_F1.shape}')
    # print(orig_fun-pushed_tensor)

    if plot:

        metadata = {
            'suptitle': 'Compare 3D Polar Splines'
        }
        plot_comparison(metadata, x, y, z, 
        orig_fun, pushed_F_tensor , pushed_F_polar_F , pushed_F_polar_F1 , 
        orig_df , pushed_F_1_tensor , pushed_F_1_polar_F, pushed_F_1_polar_F1 )

        metadata = {
            'suptitle': 'Compare 3D Polar Splines (difference)'
        }
        plot_comparison(metadata, x, y, z, 
        orig_fun, pushed_F_tensor - orig_fun, pushed_F_polar_F - orig_fun, pushed_F_polar_F1 - orig_fun, 
        orig_df , pushed_F_1_tensor - orig_df, pushed_F_1_polar_F-  orig_df, pushed_F_1_polar_F1 - orig_df)

        plt.show()




def case_02_circle_scaled(Nel, p, spl_kind, nq_el, nq_pr, bc, plot):

    import numpy as np
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test case 02: Scaled circle.')



def case_03_ellipse(Nel, p, spl_kind, nq_el, nq_pr, bc, plot):

    import numpy as np
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test case 03: Ellipse.')



def case_04_ellipse_rotated(Nel, p, spl_kind, nq_el, nq_pr, bc, plot):

    import numpy as np
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test case 04: Rotated ellipse.')



def case_05_circular_shifted(Nel, p, spl_kind, nq_el, nq_pr, bc, plot):

    import numpy as np
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test case 05: Circular with grad-shafranov shift.')



def case_06_spline_3D(Nel, p, spl_kind, nq_el, nq_pr, bc, plot):

    import numpy as np
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test case 06: Generic spline map.')

    # ============================================================
    # Imports.
    # ============================================================

    import os
    import h5py
    import tempfile
    temp_dir = tempfile.TemporaryDirectory(prefix='STRUPHY-')
    print(f'Created temp directory at: {temp_dir.name}')

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
        reader = GVEC_Reader(read_filepath, read_filename, gvec_filepath, gvec_filename)

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
    # Create FEM space and setup projectors.
    # ============================================================

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [spl.Spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]
    # Set projectors.
    [space.set_projectors(nq=nq) for space, nq in zip(spaces_FEM, nq_pr) if not hasattr(space, 'projectors')]

    # 3d tensor-product B-spline space for finite elements
    TENSOR_SPACE = spl.Tensor_spline_space(spaces_FEM, ck=-1)
    # Set projectors.
    if not hasattr(TENSOR_SPACE, 'projectors'):
        TENSOR_SPACE.set_projectors('general') # def set_projectors(self, which='tensor'). Use 'general' for polar splines.
    print('Tensor space and projector set up done.')



    # ============================================================
    # Create splines (using PI_0 projector).
    # ============================================================

    # Calculate spline coefficients using PI_0 projector.
    cx = TENSOR_SPACE.projectors.pi_0(X)
    cy = TENSOR_SPACE.projectors.pi_0(Y)
    cz = TENSOR_SPACE.projectors.pi_0(Z)

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
    EQ_MHD = Equilibrium_mhd_gvec(params, DOMAIN, EQ_MHD_P, TENSOR_SPACE, SOURCE_DOMAIN)
    print('Initialized the `Equilibrium_mhd_gvec` class.')

    temp_dir.cleanup()
    print('Removed temp directory.')



def plot_comparison(metadata, eta1, eta2, eta3, analytical_f, tensor, polar_F, polar_F1, analytical_df, df_tensor, df_polar_F, df_polar_F1):

    import numpy             as np
    import matplotlib.pyplot as plt
    import matplotlib.tri    as tri

    # Plot settings.
    row = 2
    col = 4
    dpi = 100
    width, height = (1920 / dpi, 1200 / dpi)
    fig = plt.figure(figsize=(width,height), dpi=dpi)
    gs  = fig.add_gridspec(row, col, width_ratios=[1,1]*2)
    fig.suptitle(metadata['suptitle'], y=0.98)
    # ax1_f = fig.add_subplot(gs[0, 0], projection='3d')
    # ax2_f = fig.add_subplot(gs[0, 1], projection='3d')
    # ax3_f = fig.add_subplot(gs[1, 0], projection='3d')
    # ax4_f = fig.add_subplot(gs[1, 1], projection='3d')
    ax1_f = fig.add_subplot(gs[0, 0])
    ax2_f = fig.add_subplot(gs[0, 1])
    ax3_f = fig.add_subplot(gs[1, 0])
    ax4_f = fig.add_subplot(gs[1, 1])
    # ax1_df = fig.add_subplot(gs[0, 2], projection='3d')
    # ax2_df = fig.add_subplot(gs[0, 3], projection='3d')
    # ax3_df = fig.add_subplot(gs[1, 2], projection='3d')
    # ax4_df = fig.add_subplot(gs[1, 3], projection='3d')
    ax1_df = fig.add_subplot(gs[0, 2])
    ax2_df = fig.add_subplot(gs[0, 3])
    ax3_df = fig.add_subplot(gs[1, 2])
    ax4_df = fig.add_subplot(gs[1, 3])
    axes_f  = [ax1_f,  ax2_f,  ax3_f,  ax4_f ]
    axes_df = [ax1_df, ax2_df, ax3_df, ax4_df]

    for ax in axes_f:
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        # ax.zaxis.get_major_locator().set_params(integer=True)
        # ax.set_xlabel('$\eta^1$')
        # ax.set_ylabel('$\eta^2$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', adjustable='box')
        # ax.set_aspect('auto', adjustable='box')
        # ax.set_zlabel('$f$')

    for ax in axes_df:
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        # ax.zaxis.get_major_locator().set_params(integer=True)
        # ax.set_xlabel('$\eta^1$')
        # ax.set_ylabel('$\eta^2$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', adjustable='box')
        # ax.set_zlabel('$df$')
        # ax.set_xlim(0, 1.2)

    ax1_f.set_title('Original function $f(x,y)$')
    ax2_f.set_title('$f(x,y)$ on tensor product F')
    ax3_f.set_title('$f(x,y)$ on polar splines F')
    ax4_f.set_title('$f(x,y)$ on polar splines F1')

    ax1_df.set_title('Derivative $\\nabla f(x,y)$')
    ax2_df.set_title('$\\nabla f(x,y)$ on tensor product F')
    ax3_df.set_title('$\\nabla f(x,y)$ on polar splines F')
    ax4_df.set_title('$\\nabla f(x,y)$ on polar splines F1')



    # Use wireframe if the function is dependent only on eta1 and eta2 (2D).
    # img1 = ax1_f.plot_surface(eta1[:,:,0], eta2[:,:,0], analytical_f[:,:,0])
    # img2 = ax2_f.plot_surface(eta1[:,:,0], eta2[:,:,0], tensor[:,:,0])
    # img3 = ax3_f.plot_surface(eta1[:,:,0], eta2[:,:,0], polar_F[:,:,0])
    # img4 = ax4_f.plot_surface(eta1[:,:,0], eta2[:,:,0], polar_F1[:,:,0])
    img1 = ax1_f.contourf(eta1[:,:,0], eta2[:,:,0], analytical_f[:,:,0])
    img2 = ax2_f.contourf(eta1[:,:,0], eta2[:,:,0], tensor[:,:,0])
    img3 = ax3_f.contourf(eta1[:,:,0], eta2[:,:,0], polar_F[:,:,0])
    img4 = ax4_f.contourf(eta1[:,:,0], eta2[:,:,0], polar_F1[:,:,0])

    # Use Scatter plot if the function is dependent on eta3 as well (3D).
    # img1 = ax1_f.scatter(eta1, eta2, eta3, c=analytical_f, cmap=plt.cm.plasma, marker='.')
    # img2 = ax2_f.scatter(eta1, eta2, eta3, c=tensor, cmap=plt.cm.plasma, marker='.')
    # img3 = ax3_f.scatter(eta1, eta2, eta3, c=polar_F, cmap=plt.cm.plasma, marker='.')
    # img4 = ax4_f.scatter(eta1, eta2, eta3, c=polar_F1, cmap=plt.cm.plasma, marker='.')
    cbar1 = fig.colorbar(img1, ax=ax1_f, shrink=0.5, pad=0.1, label='Original function')
    cbar2 = fig.colorbar(img2, ax=ax2_f, shrink=0.5, pad=0.1, label='From tensor product')
    cbar3 = fig.colorbar(img3, ax=ax3_f, shrink=0.5, pad=0.1, label='From F polar spline')
    cbar4 = fig.colorbar(img4, ax=ax4_f, shrink=0.5, pad=0.1, label='From F1 polar spline')



    df_mag = np.sqrt(df_tensor[0]**2 + df_tensor[1]**2 + df_tensor[2]**2)
    print('df_mag.shape', df_mag.shape)
    print('Max |df|:', np.max(df_mag))
    print('Min |df|:', np.min(df_mag))

    # 3D:
    # Flatten and normalize.
    c = (df_mag.ravel() - df_mag.min()) / df_mag.ptp()
    print('c.shape', c.shape)
    # Repeat for each body line and two head lines in a quiver.
    c = np.concatenate((c, np.repeat(c, 2)))
    print('c.shape', c.shape)
    # Colormap.
    c = plt.cm.plasma(c)
    print('c.shape', c.shape)

    # img3 = ax3_df.quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), analytical_df[0].flatten(), analytical_df[1].flatten(), analytical_df[2].flatten(), colors=c, length=1, arrow_length_ratio=.3, cmap=plt.cm.plasma)
    # img3.set_array(df_mag.flatten())
    # img3.set_clim(np.min(df_mag), np.max(df_mag))
    # cbar3 = fig.colorbar(img3, ax=ax3_f, shrink=0.5, pad=0.1, label='Gradient of f: $|\\nabla f|$')

    # 2D:
    # Flatten and normalize.
    c = (df_mag[:,:,0].ravel() - df_mag[:,:,0].min()) / df_mag[:,:,0].ptp()
    print('c.shape', c.shape)
    # Repeat for each body line and two head lines in a quiver.
    c = np.concatenate((c, np.repeat(c, 2)))
    print('c.shape', c.shape)
    # Colormap.
    c = plt.cm.plasma(c)
    print('c.shape', c.shape)

    # img1 = ax1_df.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), analytical_df[0,:,:,0].flatten(), analytical_df[1,:,:,0].flatten(), color=c, cmap=plt.cm.plasma, scale=50)
    # img1.set_array(df_mag[:,:,0].flatten())
    # img1.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
    # # cbar1 = fig.colorbar(img1, ax=ax1_df, shrink=0.5, pad=0.1, label=f'Gradient of f: $|\\nabla f|$ (Scale: {0.01}')

    # img2 = ax2_df.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_tensor[0,:,:,0].flatten(), df_tensor[1,:,:,0].flatten(), color=c, cmap=plt.cm.plasma, scale=50)
    # img2.set_array(df_mag[:,:,0].flatten())
    # img2.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))

    # img3 = ax3_df.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_polar_F[0,:,:,0].flatten(), df_polar_F[1,:,:,0].flatten(), color=c, cmap=plt.cm.plasma, scale=50)
    # img3.set_array(df_mag[:,:,0].flatten())
    # img3.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))

    # img4 = ax4_df.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_polar_F1[0,:,:,0].flatten(), df_polar_F1[1,:,:,0].flatten(), color=c, cmap=plt.cm.plasma, scale=50)
    # img4.set_array(df_mag[:,:,0].flatten())
    # img4.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
    # img1 = ax1_df.plot_surface(eta1[:,:,0], eta2[:,:,0], analytical_df[0,:,:,0])
    # img2 = ax2_df.plot_surface(eta1[:,:,0], eta2[:,:,0], df_tensor[0,:,:,0])
    # img3 = ax3_df.plot_surface(eta1[:,:,0], eta2[:,:,0], df_polar_F[0,:,:,0])
    # img4 = ax4_df.plot_surface(eta1[:,:,0], eta2[:,:,0], df_polar_F1[0,:,:,0])
    # img1 = ax1_df.plot_surface(eta1[:,:,0], eta2[:,:,0], analytical_df[1,:,:,0])
    # img2 = ax2_df.plot_surface(eta1[:,:,0], eta2[:,:,0], df_tensor[1,:,:,0])
    # img3 = ax3_df.plot_surface(eta1[:,:,0], eta2[:,:,0], df_polar_F[1,:,:,0])
    # img4 = ax4_df.plot_surface(eta1[:,:,0], eta2[:,:,0], df_polar_F1[1,:,:,0])
    img1 = ax1_df.contourf(eta1[:,:,0], eta2[:,:,0], analytical_df[0,:,:,0])
    img2 = ax2_df.contourf(eta1[:,:,0], eta2[:,:,0], df_tensor[0,:,:,0])
    img3 = ax3_df.contourf(eta1[:,:,0], eta2[:,:,0], df_polar_F[0,:,:,0])
    img4 = ax4_df.contourf(eta1[:,:,0], eta2[:,:,0], df_polar_F1[0,:,:,0])
    # img1 = ax1_df.contourf(eta1[:,:,0], eta2[:,:,0], analytical_df[1,:,:,0])
    # img2 = ax2_df.contourf(eta1[:,:,0], eta2[:,:,0], df_tensor[1,:,:,0])
    # img3 = ax3_df.contourf(eta1[:,:,0], eta2[:,:,0], df_polar_F[1,:,:,0])
    # img4 = ax4_df.contourf(eta1[:,:,0], eta2[:,:,0], df_polar_F1[1,:,:,0])
    cbar1 = fig.colorbar(img1, ax=ax1_df, shrink=0.5, pad=0.1, label='Original derivative')
    cbar2 = fig.colorbar(img2, ax=ax2_df, shrink=0.5, pad=0.1, label='From tensor product')
    cbar3 = fig.colorbar(img3, ax=ax3_df, shrink=0.5, pad=0.1, label='From F polar spline')
    cbar4 = fig.colorbar(img4, ax=ax4_df, shrink=0.5, pad=0.1, label='From F1 polar spline')

    # TODO: Mark position on mapping pole.



    # ============================================================
    # Show the figure.
    # ============================================================

    print('Before `tight_layout()`   | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.tight_layout()
    print('After `tight_layout()`    | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.subplots_adjust(hspace=fig.subplotpars.hspace * 1.2, wspace=fig.subplotpars.wspace * .9)
    print('After `subplots_adjust()` | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))



if __name__ == "__main__":
    test_polar_splines_3D(plot=True)
