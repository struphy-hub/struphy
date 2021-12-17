import numpy as np
import matplotlib.pyplot as plt
import struphy.geometry.domain_3d as dom
import struphy.feec.spline_space as spl

def test_polar_splines_3D(func_test=None, map_type=None, plot_0form=False, plot_1form=False, plot_2form=False):
    """Test constructing 3D polar splines for a non-axisymmetric GVEC equilibrium via an intermediate axisymmetric torus mapping.

    Even though it says 3D, all test functions are :math:``f(x,y)``, independent of :math:``z``.
    For scalar case, the test function is :math:``f(x,y)``.
    For vector case, the test function is :math:``f(x,y) \hat{i} + f(x,y) \hat{j} + f(x,y) \hat{k}``.

    Parameters
    ----------
    func_test : FuncTest
        An Enum of implemented test functions.
    map_type : FuncTest
        An Enum of implemented mapping domains.
    plot_0form : boolean
        Whether to display plots of spline comparisons of 0-form test function.
    plot_1form : boolean
        Whether to display plots of spline comparisons of 1-form test function.
    plot_2form : boolean
        Whether to display plots of spline comparisons of 2-form test function.

    Notes
    -----
    To execute this on Windows, at the project root, enter `python`.
    Then import this test function `from struphy.tests.xx_test_polar_splines_3D import test_polar_splines_3D as test_polar`.
    Finally call `test_polar(True)`.
    """

    if func_test is None:
        func_test = FuncTest.GAUSSIANCOSINE

    if map_type is None:
        map_type = MapType.CIRCLESCALED

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

    import matplotlib.pyplot as plt



    # ============================================================
    # Configure STRUPHY splines.
    # ============================================================

    # Create splines to STRUPHY, unrelated to splines in GVEC.
    # Nel      = [8, 30, 6]
    # Nel      = [8, 12, 6]
    # Nel      = [16*1, 18*1, 3]
    Nel      = [16*2, 18*2, 3]
    # Nel      = [16*4, 18*4, 3]
    # p        = [2, 2, 1]
    p        = [3, 3, 1]
    # p        = [4, 4, 1]
    nq_el    = [4, 4, 4]  # Element integration, quadrature points per grid cell
    nq_pr    = [4, 4, 4]  # Greville integration, quadrature points per histopolation cell (for projection)
    # nq_pr    = [2, 2, 1]
    bc       = ['f', 'f'] # BC in s-direction
    spl_kind = [False, True, True] # Spline type: True=periodic, False=clamped



    # ============================================================
    # Run test cases.
    # ============================================================

    print('Begin testing 3D polar splines.')

    func, dfdx, dfdy = generate_test_function(func_test, params=None)
    func_3d, curl_3d, div_3d = func_3d_wrapper(func, dfdx, dfdy)
    if map_type == MapType.SPLINE:
        DOMAIN_F, gvec = get_gvec_domain(Nel, p, spl_kind, nq_el, nq_pr, bc)
    else:
        DOMAIN_F = None
    DOMAIN_F = map_generator(map_type, DOMAIN_F)

    case_0form = case_01_circle_identity_0form
    case_0form_args = [Nel, p, spl_kind, nq_el, nq_pr, bc, func   , dfdx   , dfdy  , DOMAIN_F]
    case_1form = case_01_circle_identity_1form
    case_1form_args = [Nel, p, spl_kind, nq_el, nq_pr, bc, func_3d, curl_3d, div_3d, DOMAIN_F]
    case_2form = case_01_circle_identity_2form
    case_2form_args = [Nel, p, spl_kind, nq_el, nq_pr, bc, func_3d, curl_3d, div_3d, DOMAIN_F]

    if plot_0form:
        func_form = FuncForm.ZERO
        plot_data_0form = case_0form(*case_0form_args)
        plot_handles_0form = plot_wrapper(*plot_data_0form, func_form, Nel, p)
        ref_fun_0form = plot_controls(case_0form, case_0form_args, func_test, func_form, plot_handles_0form)
        ref_spl_0form = plot_spl_config(case_0form, case_0form_args, func_test, func_form, plot_handles_0form)
        plt.show()
    if plot_1form:
        func_form = FuncForm.ONE
        plot_data_1form = case_1form(*case_1form_args)
        plot_handles_1form = plot_wrapper(*plot_data_1form, func_form, Nel, p)
        ref_fun_1form = plot_controls(case_1form, case_1form_args, func_test, func_form, plot_handles_1form)
        ref_spl_1form = plot_spl_config(case_1form, case_1form_args, func_test, func_form, plot_handles_1form)
        plt.show()
    if plot_2form:
        func_form = FuncForm.TWO
        plot_data_2form = case_2form(*case_2form_args)
        plot_handles_2form = plot_wrapper(*plot_data_2form, func_form, Nel, p)
        ref_fun_2form = plot_controls(case_2form, case_2form_args, func_test, func_form, plot_handles_2form)
        ref_spl_2form = plot_spl_config(case_2form, case_2form_args, func_test, func_form, plot_handles_2form)
        plt.show()

    print('Done testing 3D polar splines.')



from enum import Enum, unique

@unique
class FuncTest(Enum):
    """Enum for test function used."""
    GAUSSIAN = 1
    GAUSSIANCOSINE = 2
    SIGMOID = 3
    SINEX = 4
    LINEARX = 5

@unique
class MapType(Enum):
    """Enum for F-map functions used."""
    CIRCLEIDENTICAL = 1
    CIRCLESCALED = 2
    CIRCLESHIFTED = 3
    ELLIPSE = 4
    ELLIPSEROTATED = 5
    SOLOVIEV = 6
    SOLOVIEVSQRT = 7
    SPLINE = 8

@unique
class Comparison(Enum):
    """Enum for either plotting original or difference."""
    ORIG = 1
    DIFF = 2

@unique
class PlotTypeLeft(Enum):
    """Enum for chart type on the left hand side (original function)."""
    CONTOUR2D = 1 # contourf, 2D.
    CONTOUR3D = 2 # contourf, 3D.
    SCATTER = 3 # scatter, 3D.
    SURFACE = 4 # plot_surface, 3D.
    WIREFRAME = 5 # wireframe, 3D.

@unique
class PlotTypeRight(Enum):
    """Enum for chart type on the right hand side (first derivative function)."""
    CONTOUR2D = 1 # contourf, 2D.
    CONTOUR3D = 2 # contourf, 3D.
    QUIVER2D = 3 # quiver, 2D.
    QUIVER3D = 4 # quiver, 3D.
    SURFACE = 5 # plot_surface, 3D.

@unique
class FuncForm(Enum):
    """Enum for p-form of the test function itself.
    
    It is implied that its "derivative" is (p+1)-form according to de Rahm's sequence."""
    ZERO = 10
    ONE = 11
    TWO = 12



def func_gaussian(mu_x=.1, sd_x=.1, mu_y=.1, sd_y=.1):
    """A 2D independent Gaussian pdf.

    The steepest slope is at mean = s.d..

    Parameters
    ----------
    mu_x : float
        Mean along x-direction.
    sd_x : float
        Standard deviation along x-direction.
    mu_y : float
        Mean along y-direction.
    sd_y : float
        Standard deviation along y-direction.

    Returns
    -------
    tuple of callables
        A 3-tuple consisting of (1) a Gaussian function and its partial derivatives (2) along x and (3) along y.
    """
    import numpy as np
    func = lambda x, y, z : 1 / (2 * np.pi * sd_x * sd_y) * np.exp(-0.5 * ((x - mu_x)**2 / sd_x**2 + (y - mu_y)**2 / sd_y**2))
    dfdx = lambda x, y, z : 1 / (2 * np.pi * sd_x * sd_y) * np.exp(-0.5 * ((x - mu_x)**2 / sd_x**2 + (y - mu_y)**2 / sd_y**2)) * (mu_x - x) / sd_x**2
    dfdy = lambda x, y, z : 1 / (2 * np.pi * sd_x * sd_y) * np.exp(-0.5 * ((x - mu_x)**2 / sd_x**2 + (y - mu_y)**2 / sd_y**2)) * (mu_y - y) / sd_y**2
    return func, dfdx, dfdy



def func_gaussian_cosine(mu_x=1, sd_x=1, mu_y=1, sd_y=1, om_x=6.2831853, ph_x=0, om_y=6.2831853, ph_y=0):
    """A 2D independent mixed Guassian and Cosine function.

    The steepest slope is when mean = s.d. coincide with (some) zeroes of cosine.
    In particular, we want the first zero of the cosine to coincide with the s.d. of the Gaussian.

    Parameters
    ----------
    mu_x : float
        Mean along x-direction.
    sd_x : float
        Standard deviation along x-direction.
    mu_y : float
        Mean along y-direction.
    sd_y : float
        Standard deviation along y-direction.
    om_x : float
        Angular frequency along x-direction.
    ph_x : float
        Phase shift along x-direction.
    om_y : float
        Angular frequency along y-direction.
    ph_y : float
        Phase shift along y-direction.

    Returns
    -------
    tuple of callables
        A 3-tuple consisting of (1) the mixed Gaussian-Cosine function and its partial derivatives (2) along x and (3) along y.
    """
    import numpy as np
    func = lambda x, y, z : 1 / (2 * np.pi * sd_x * sd_y) * np.exp(-0.5 * ((x - mu_x)**2 / sd_x**2 + (y - mu_y)**2 / sd_y**2)) * np.cos(om_x * x + ph_x) * np.cos(om_y * y + ph_y)
    dfdx = lambda x, y, z : 1 / (2 * np.pi * sd_x * sd_y) * np.exp(-0.5 * ((x - mu_x)**2 / sd_x**2 + (y - mu_y)**2 / sd_y**2)) * np.cos(om_x * x + ph_x) * np.cos(om_y * y + ph_y) * ((mu_x - x) / sd_x**2 - om_x * np.tan(om_x * x + ph_x))
    dfdy = lambda x, y, z : 1 / (2 * np.pi * sd_x * sd_y) * np.exp(-0.5 * ((x - mu_x)**2 / sd_x**2 + (y - mu_y)**2 / sd_y**2)) * np.cos(om_x * x + ph_x) * np.cos(om_y * y + ph_y) * ((mu_y - y) / sd_y**2 - om_y * np.tan(om_y * x + ph_y))
    # Pure Cosine for testing default phase offset ph_x=(mu_x+10)*2*np.pi:
    # func = lambda x, y, z : np.cos(om_x * x + ph_x) * np.cos(om_y * y + ph_y)
    # dfdx = lambda x, y, z : - om_x * np.sin(om_x * x + ph_x) * np.cos(om_y * y + ph_y)
    # dfdy = lambda x, y, z : - om_y * np.cos(om_x * x + ph_x) * np.sin(om_y * y + ph_y)
    return func, dfdx, dfdy



def func_sineX(om_x=6.2831853, ph_x=0):
    """A sine function in x.

    The steepest slope is at the origin (or other zeros).

    Parameters
    ----------
    om_x : float
        Angular frequency along x-direction.
    ph_x : float
        Phase shift along x-direction.

    Returns
    -------
    tuple of callables
        A 3-tuple consisting of (1) the sine function of only x, and its partial derivatives (2) along x and (3) along y.
    """
    import numpy as np
    func = lambda x, y, z : np.sin(om_x * x + ph_x)
    dfdx = lambda x, y, z : om_x * np.cos(om_x * x + ph_x)
    dfdy = lambda x, y, z : np.zeros_like(x)
    return func, dfdx, dfdy



def func_sigmoid(L=1, mu_x=0, k_x=1, mu_y=0, k_y=1):
    """A 2D independent generalized logistic cdf.

    The steepest slope is at mean = 0.

    Parameters
    ----------
    L : float
        Maximum value of the logistic function.
    mu_x : float
        Mean along x-direction.
    k_x : float
        Steepnesss of the slope along x-direction.
    mu_y : float
        Mean along y-direction.
    k_y : float
        Steepnesss of the slope along y-direction.

    Returns
    -------
    tuple of callables
        A 3-tuple consisting of (1) a generalized logistic function and its partial derivatives (2) along x and (3) along y.
    """
    import numpy as np
    func = lambda x, y, z : L / (1 + np.exp(-k_x * (x - mu_x))) / (1 + np.exp(-k_y * (y - mu_y)))
    dfdx = lambda x, y, z : k_x * func(x, y, z) * (1 - 1 / (1 + np.exp(-k_x * (x - mu_x))))
    dfdy = lambda x, y, z : k_y * func(x, y, z) * (1 - 1 / (1 + np.exp(-k_y * (y - mu_y))))
    return func, dfdx, dfdy



def func_linearX(mu_x=.1):
    """A linear function in x: f(x) = x.

    Parameters
    ----------
    mu_x : float
        Mean along x-direction.

    Returns
    -------
    tuple of callables
        A 3-tuple consisting of (1) a linear function in only x, and its partial derivatives (2) along x and (3) along y.
    """
    import numpy as np
    func = lambda x, y, z : x - mu_x
    dfdx = lambda x, y, z : np.ones_like(x)
    dfdy = lambda x, y, z : np.zeros_like(x)
    return func, dfdx, dfdy



def func_3d_wrapper(func, dfdx, dfdy):
    """Create a vector-valued function of only x, y from one of the provided test functions.

    Parameters
    ----------
    func : callable
        One of the provided test functions `func_...`.
    dfdx : callable
        Partial derivative of `func` w.r.t. x-direction.
    dfdy : callable
        Partial derivative of `func` w.r.t. y-direction.

    Returns
    -------
    tuple of callables
        A 3-tuple consisting of (1) a vector-valued function (2) its curl and (3) its divergence.

    Notes
    -----
    .. math::

        \vec{f}(x,y)               & = f(x,y) \hat{i} + f(x,y) \hat{j} + f(x,y) \hat{k} \\
        \nabla \times \vec{f}(x,y) & = 0 \hat{i} + 0 \hat{j} + (\pdv{f(x,y)}{y} - \pdv{f(x,y)}{x}) \hat{k} \\
        \nabla \cdot \vec{f}(x,y)  & = \pdv{f(x,y)}{x} + \pdv{f(x,y)}{y} + 0
    """
    import numpy as np
    # func_3d = [lambda x, y, z : func(x, y, z), lambda x, y, z : func(x, y, z), lambda x, y, z : func(x, y, z)]
    # curl_3d = [lambda x, y, z : np.zeros_like(x), lambda x, y, z : np.zeros_like(x), lambda x, y, z : dfdy(x, y, z) - dfdx(x, y, z)]
    func_3d = lambda x, y, z : np.array([func(x, y, z), func(x, y, z), func(x, y, z)])
    curl_3d = lambda x, y, z : np.array([dfdy(x, y, z), - dfdx(x, y, z), dfdx(x, y, z) - dfdy(x, y, z)])
    div_3d  = lambda x, y, z : dfdx(x, y, z) + dfdy(x, y, z)
    return func_3d, curl_3d, div_3d



def generate_test_function(func_test=FuncTest.GAUSSIANCOSINE, params=None):

    import numpy as np

    # ============================================================
    # Test functions.
    # ============================================================

    # TODO: take care of boundary conditions, shifts, etc. (-> define proper function)
    # The Gaussian should be shifted away from (0,0), 
    # otherwise its derivative at the pole is analytically zero, 
    # and would not be of interest to us,
    # because we want to test C1 continuity.
    # A jump is expected in TENSOR_SPACE.

    # Basic Gaussian implementation tests.
    from scipy.integrate import dblquad
    standard_normal, _, _ = func_gaussian(0,1,0,1)
    print(f'Check maximum of 2D standard normal (== 1/(2pi)?) {standard_normal(0,0,0):.8f} == {1 / 2 / np.pi:.8f}?')
    integration_test = dblquad(lambda x, y : standard_normal(x, y, 0), -np.inf, np.inf, -np.inf, np.inf)
    print(f'Check integrating 2D Gaussian (== 1?): {integration_test}')

    # Test case 1: 2D Gaussian
    if func_test == FuncTest.GAUSSIAN:

        if params is not None:
            mu_x = params['mu_x']
            mu_y = params['mu_y']
            sd_x = params['sd_x']
            sd_y = params['sd_y']
        else: # Default.
            sd_x = 0.1
            sd_y = 0.05
            mu_x = sd_x + 10
            mu_y = sd_y

        func, dfdx, dfdy = func_gaussian(mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y)

    # Test case 2: Mixed Gaussian and cosine.
    elif func_test == FuncTest.GAUSSIANCOSINE:

        if params is not None:
            mu_x = params['mu_x']
            mu_y = params['mu_y']
            sd_x = params['sd_x']
            sd_y = params['sd_y']
            om_x = params['om_x']
            om_y = params['om_y']
            ph_x = params['ph_x']
            ph_y = params['ph_y']
        else: # Default.
            sd_x = 0.1
            sd_y = 0.05
            mu_x = sd_x + 10
            mu_y = sd_y
            v_x = 1
            v_y = 1
            om_x = 2 * np.pi * v_x / (4 * sd_x) # Steepest gradient.
            om_y = 2 * np.pi * v_y / (4 * sd_y) # Steepest gradient.
            # om_x = 2 * np.pi
            # om_y = 2 * np.pi
            ph_x = 2 * np.pi * (mu_x)
            ph_y = 2 * np.pi * (mu_y)

        func, dfdx, dfdy = func_gaussian_cosine(mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y, om_x=om_x, ph_x=ph_x, om_y=om_y, ph_y=ph_y)

    # Test case 3: Sine in x-direction.
    elif func_test == FuncTest.SINEX:

        if params is not None:
            om_x = params['om_x']
            ph_x = params['ph_x']
        else: # Default.
            om_x = 2 * np.pi
            ph_x = 2 * np.pi * 10

        func, dfdx, dfdy = func_sineX(om_x=om_x, ph_x=ph_x)

    # Test case 4: Logistic function.
    elif func_test == FuncTest.SIGMOID:

        if params is not None:
            mu_x = params['mu_x']
            mu_y = params['mu_y']
            k_x = params['k_x']
            k_y = params['k_y']
        else: # Default.
            mu_x = 0.0 + 10
            mu_y = 0.0
            k_x = 100
            k_y = 100

        func, dfdx, dfdy = func_sigmoid(L=1, mu_x=mu_x, k_x=k_x, mu_y=mu_y, k_y=k_y)

    # Test case 5: Linear function in x.
    elif func_test == FuncTest.LINEARX:

        if params is not None:
            mu_x = params['mu_x']
        else: # Default.
            mu_x = 0 + 10

        func, dfdx, dfdy = func_linearX(mu_x=mu_x)

    else:

        raise NotImplementedError(f'Test case {func_test} not implemented.')

    # # Shifted Gaussian.
    # shift_x = 0.02 + 10
    # shift_y = 0.04
    # hw = 0.16
    # func = lambda x, y, z : np.exp(-((x - shift_x)**2 + (y - shift_y)**2) / hw**2)

    # # Gradient to the "test function".
    # dfdx = lambda x, y, z : np.exp(-((x - shift_x)**2 + (y - shift_y)**2) / hw**2) * (- 2 * (x - shift_x) / hw**2)
    # dfdy = lambda x, y, z : np.exp(-((x - shift_x)**2 + (y - shift_y)**2) / hw**2) * (- 2 * (y - shift_y) / hw**2)

    # # Shifted sin/cos.
    # shift_x = 0.5 + 10
    # shift_y = 0.5
    # k_x = 2 * np.pi
    # k_y = 2 * np.pi

    # func = lambda x, y, z : np.sin(k_x * (x - shift_x)) * np.cos(k_y * (y - shift_y))

    # # Gradient to the "test function".
    # dfdx = lambda x, y, z :   k_x * np.cos(k_x * (x - shift_x)) * np.cos(k_y * (y - shift_y))
    # dfdy = lambda x, y, z : - k_y * np.sin(k_x * (x - shift_x)) * np.sin(k_y * (y - shift_y))

    # # Rotated sin/cos.
    # shift_x = 0.02 + 10
    # shift_y = 0.1
    # k_x = 2 * np.pi * 0.23 * 2
    # k_y = 2 * np.pi * 0.13 * 2

    # func = lambda x, y, z : np.sin(k_x * (x - shift_x + y)) * np.cos(k_y * (y - shift_y - x))

    # # Gradient to the "test function".
    # dfdx = lambda x, y, z :   k_x * np.cos(k_x * (x - shift_x + y)) * np.cos(k_y * (y - shift_y - x)) + k_y * np.sin(k_x * (x - shift_x + y)) * np.sin(k_y * (y - shift_y - x))
    # dfdy = lambda x, y, z : - k_y * np.sin(k_x * (x - shift_x + y)) * np.sin(k_y * (y - shift_y - x)) + k_x * np.cos(k_x * (x - shift_x + y)) * np.cos(k_y * (y - shift_y - x))

    return func, dfdx, dfdy



def map_generator(map_type:MapType, DOMAIN_F:dom.Domain=None):

    if map_type == MapType.CIRCLEIDENTICAL: # Unit circle centered at (10,0).

        print('Running test case 01: Identity map == F1 (unit circle).')
        if DOMAIN_F is None: # A circle of radius (a2 - a1) offset by (a1 + R0, a1).
            DOMAIN_F = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 1., 'R0': 10.})

        return DOMAIN_F

    elif map_type == MapType.CIRCLESCALED: # Circle scaled to radius = 2.

        print('Running test case 02: Scaled circle.')
        if DOMAIN_F is None:
            DOMAIN_F = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 2., 'R0': 10.})

        return DOMAIN_F

    elif map_type == MapType.CIRCLESHIFTED: # Unit circle shifted to 10.005 instead of 10.

        print('Running test case 03: Circular with x-axis offset.')
        if DOMAIN_F is None:
            DOMAIN_F = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 1., 'R0': 10.005})

        return DOMAIN_F

    elif map_type == MapType.ELLIPSE: # Ellipse centered at (10,0), with major and minor radii 1 and 0.5 respectively.

        print('Running test case 04: Ellipse.')
        if DOMAIN_F is None:
            DOMAIN_F = dom.Domain('ellipse', {'cx': 10., 'cy': 0., 'cz': 0., 'rx': 1., 'ry': 0.5, 'Lz': 10.})

        return DOMAIN_F

    elif map_type == MapType.ELLIPSEROTATED: # Ellipse centered at (10,0), with major and minor radii 1 and 0.5 respectively, rotated 30 degrees.

        print('Running test case 05: Rotated ellipse.')
        if DOMAIN_F is None:
            DOMAIN_F = dom.Domain('rotated_ellipse', {'cx': 10., 'cy': 0., 'cz': 0., 'rx': 1., 'ry': 0.5, 'Lz': 10., 'theta': 30/360})

        return DOMAIN_F

    elif map_type == MapType.SOLOVIEV: # Unit circle centered at (10,0), with a Grad-Shafranov shift delta of 0.1.

        print('Running test case 06: Soloviev equilibrium (circular with Grad-Shafranov shift).')
        if DOMAIN_F is None:
            DOMAIN_F = dom.Domain('soloviev_approx', {'cx': 10., 'cy': 0., 'cz': 0., 'rx': 1., 'ry': 1., 'Lz': 10., 'delta': 0.1})

        return DOMAIN_F

    elif map_type == MapType.SOLOVIEVSQRT: # Unit circle centered at (10,0), with a Grad-Shafranov shift delta of 0.1.

        print('Running test case 07: Soloviev equilibrium but with square root dependence on eta1, instead of square.')
        if DOMAIN_F is None:
            DOMAIN_F = dom.Domain('soloviev_sqrt', {'cx': 10., 'cy': 0., 'cz': 0., 'rx': 1., 'ry': 1., 'Lz': 10., 'delta': 0.1})

        return DOMAIN_F

    elif map_type == MapType.SPLINE:

        print('Running test case 08: Generic spline map.')
        if DOMAIN_F is None:
            raise ValueError('DOMAIN must not be None for spline map.')

        return DOMAIN_F

    else:

        raise NotImplementedError(f'Map {map} not implemented.')



def case_01_circle_identity_0form(Nel, p, spl_kind, nq_el, nq_pr, bc, func   , dfdx   , dfdy  , DOMAIN_F):

    import numpy as np
    import matplotlib.pyplot as plt
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test: 0-form and its gradient.')



    # ============================================================
    # Define mappings F = F2 \circ F1.
    # ============================================================

    # Map F:
    # DOMAIN_F = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 2., 'R0': 10.})

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



    # Use F for pullback!!!
    def fun_L(eta1, eta2, eta3):
        return DOMAIN_F.pull(func, eta1, eta2, eta3, kind_fun='0_form')

    proj_tensor_0form   = TENSOR_SPACE.projectors.pi_0(fun_L)
    proj_polar_F_0form  = POLAR_SPACE_F.projectors.pi_0(fun_L)
    proj_polar_F1_0form = POLAR_SPACE_F1.projectors.pi_0(fun_L)
    print(f'Shape of proj_tensor_0form   : {proj_tensor_0form.shape}')
    print(f'Shape of proj_polar_F_0form  : {proj_polar_F_0form.shape}')
    print(f'Shape of proj_polar_F1_0form : {proj_polar_F1_0form.shape}')

    # f[form]_[component]_[spline] are coefficients cijk.
    f0_ten    = TENSOR_SPACE.extract_0(proj_tensor_0form)
    f0_pol_F  = POLAR_SPACE_F.extract_0(proj_polar_F_0form)
    f0_pol_F1 = POLAR_SPACE_F1.extract_0(proj_polar_F1_0form)

    # Apply discrete gradient -> 1-form. Are all 3 components continuous when pushed forward?
    proj_tensor_1form   = TENSOR_SPACE.G.dot(proj_tensor_0form) # Or f0_ten. Unused because the shapes are identical.
    proj_polar_F_1form  = POLAR_SPACE_F.G.dot(proj_polar_F_0form)
    proj_polar_F1_1form = POLAR_SPACE_F1.G.dot(proj_polar_F1_0form)

    # f[form]_[component]_[spline] are coefficients cijk.
    f1_1_ten   , f1_2_ten   , f1_3_ten    = TENSOR_SPACE.extract_1(proj_tensor_1form)
    f1_1_pol_F , f1_2_pol_F , f1_3_pol_F  = POLAR_SPACE_F.extract_1(proj_polar_F_1form)
    f1_1_pol_F1, f1_2_pol_F1, f1_3_pol_F1 = POLAR_SPACE_F1.extract_1(proj_polar_F1_1form)

    # TODO: evaluate splines, push forward
    lim_s = 1
    num_s_log = 5
    eta1_range = np.concatenate((np.logspace(-num_s_log, -2, num_s_log-1), np.linspace(1e-1, lim_s, 101-num_s_log)))
    # eta1_range = np.linspace(1e-4, lim_s, 101)
    eta2_range = np.linspace(0, 1, 101)
    eta3_range = np.linspace(0, 1, 3)
    # Evaluate at evaluation points:
    # eta1_range = TENSOR_SPACE.spaces[0].el_b
    # eta2_range = TENSOR_SPACE.spaces[1].el_b
    # eta3_range = TENSOR_SPACE.spaces[2].el_b
    # print(f'eta1_range {eta1_range}')
    # print(f'eta2_range {eta2_range}')
    # print(f'eta3_range {eta3_range}')

    eta1_sparse, eta2_sparse, eta3_sparse = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=True)
    eta1_dense,  eta2_dense,  eta3_dense  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)
    eta1,  eta2,  eta3  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)

    # Evaluate test function.
    evaled_0_tensor   = TENSOR_SPACE.evaluate_NNN(eta1, eta2, eta3, f0_ten)
    evaled_0_polar_F  = POLAR_SPACE_F.evaluate_NNN(eta1, eta2, eta3, f0_pol_F)
    evaled_0_polar_F1 = POLAR_SPACE_F1.evaluate_NNN(eta1, eta2, eta3, f0_pol_F1)

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



    # Analytical.
    x0 = F_x(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    y0 = F_y(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    z0 = F_z(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    x = F_x(eta1, eta2, eta3)
    y = F_y(eta1, eta2, eta3)
    z = F_z(eta1, eta2, eta3)
    orig_func = func(x, y, z)
    orig_grad = np.array([dfdx(x, y, z), dfdy(x, y, z), np.zeros_like(orig_func)])



    # Push to canonical domain.
    # Polar splines should be smooth for the polar_F1.
    # Comparison only valid for polar_F1.
    # pushed_F1_0_tensor   = DOMAIN_F1.push(evaled_0_tensor, eta1, eta2, eta3, kind_fun='0_form')
    # pushed_F1_0_polar_F  = DOMAIN_F1.push(evaled_0_polar_F, eta1, eta2, eta3, kind_fun='0_form')
    pushed_F1_0_polar_F1 = DOMAIN_F1.push(evaled_0_polar_F1, eta1, eta2, eta3, kind_fun='0_form')

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



    # Or push to physical domain.
    pushed_F_0_tensor   = DOMAIN_F.push(evaled_0_tensor, eta1, eta2, eta3, kind_fun='0_form')
    pushed_F_0_polar_F  = DOMAIN_F.push(evaled_0_polar_F, eta1, eta2, eta3, kind_fun='0_form')
    pushed_F_0_polar_F1 = DOMAIN_F.push(evaled_0_polar_F1, eta1, eta2, eta3, kind_fun='0_form')

    print(f'Max |orig_func|          : {np.max(np.abs(orig_func))}')
    print(f'Max |pushed_F_0_tensor|  : {np.max(np.abs(pushed_F_0_tensor))}')
    print(f'Max |pushed_F_0_polar_F| : {np.max(np.abs(pushed_F_0_polar_F))}')
    print(f'Max |pushed_F_0_polar_F1|: {np.max(np.abs(pushed_F_0_polar_F1))}')

    print(f'Min |orig_func|          : {np.min(np.abs(orig_func))}')
    print(f'Min |pushed_F_0_tensor|  : {np.min(np.abs(pushed_F_0_tensor))}')
    print(f'Min |pushed_F_0_polar_F| : {np.min(np.abs(pushed_F_0_polar_F))}')
    print(f'Min |pushed_F_0_polar_F1|: {np.min(np.abs(pushed_F_0_polar_F1))}')

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

    print(f'Max |orig_grad|          : {np.max(np.abs(orig_grad))}')
    print(f'Max |pushed_F_1_tensor|  : {np.max(np.abs(pushed_F_1_tensor))}')
    print(f'Max |pushed_F_1_polar_F| : {np.max(np.abs(pushed_F_1_polar_F))}')
    print(f'Max |pushed_F_1_polar_F1|: {np.max(np.abs(pushed_F_1_polar_F1))}')

    print(f'Min |orig_grad|          : {np.min(np.abs(orig_grad))}')
    print(f'Min |pushed_F_1_tensor|  : {np.min(np.abs(pushed_F_1_tensor))}')
    print(f'Min |pushed_F_1_polar_F| : {np.min(np.abs(pushed_F_1_polar_F))}')
    print(f'Min |pushed_F_1_polar_F1|: {np.min(np.abs(pushed_F_1_polar_F1))}')



    print(f'Is pushed tensor equivalent to the original function? {np.allclose(orig_func, pushed_F_0_tensor)}')
    print(f'Is pushed polar_F equivalent to the original function? {np.allclose(orig_func, pushed_F_0_polar_F)}')
    print(f'Is pushed polar_F1 equivalent to the original function? {np.allclose(orig_func, pushed_F_0_polar_F1)}')
    print(f'Is pushed tensor equivalent to pushed polar_F? {np.allclose(pushed_F_0_tensor, pushed_F_0_polar_F)}')
    print(f'Shape of orig_func           : {orig_func.shape}')
    print(f'Shape of orig_grad           : {orig_grad.shape}')
    print(f'Shape of pushed_F_0_tensor   : {pushed_F_0_tensor.shape}')
    print(f'Shape of pushed_F_0_polar_F  : {pushed_F_0_polar_F.shape}')
    print(f'Shape of pushed_F_0_polar_F1 : {pushed_F_0_polar_F1.shape}')
    # print(orig_func-pushed_tensor)

    return (x, y, z, x0, y0, z0, 
    orig_func, pushed_F_0_tensor, pushed_F_0_polar_F, pushed_F_0_polar_F1, 
    orig_grad, pushed_F_1_tensor, pushed_F_1_polar_F, pushed_F_1_polar_F1)



def case_01_circle_identity_1form(Nel, p, spl_kind, nq_el, nq_pr, bc, func_3d, curl_3d, div_3d, DOMAIN_F):

    import numpy as np
    import matplotlib.pyplot as plt
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test: 1-form and its curl.')



    # ============================================================
    # Define mappings F = F2 \circ F1.
    # ============================================================

    # Map F:
    # DOMAIN_F = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 2., 'R0': 10.})

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



    # Use F for pullback!!!
    def fun_L_1_1(eta1, eta2, eta3):
        return DOMAIN_F.pull(func_3d, eta1, eta2, eta3, kind_fun='1_form_1', flat_eval=False)
    def fun_L_1_2(eta1, eta2, eta3):
        return DOMAIN_F.pull(func_3d, eta1, eta2, eta3, kind_fun='1_form_2', flat_eval=False)
    def fun_L_1_3(eta1, eta2, eta3):
        return DOMAIN_F.pull(func_3d, eta1, eta2, eta3, kind_fun='1_form_3', flat_eval=False)
    fun_L = [fun_L_1_1, fun_L_1_2, fun_L_1_3]

    proj_tensor_1form   = TENSOR_SPACE.projectors.pi_1(fun_L)
    proj_polar_F_1form  = POLAR_SPACE_F.projectors.pi_1(fun_L)
    proj_polar_F1_1form = POLAR_SPACE_F1.projectors.pi_1(fun_L)
    print(f'Shape of proj_tensor_1form   : {proj_tensor_1form.shape}')
    print(f'Shape of proj_polar_F_1form  : {proj_polar_F_1form.shape}')
    print(f'Shape of proj_polar_F1_1form : {proj_polar_F1_1form.shape}')

    # f[form]_[component]_[spline] are coefficients cijk.
    f1_1_ten   , f1_2_ten   , f1_3_ten    = TENSOR_SPACE.extract_1(proj_tensor_1form)
    f1_1_pol_F , f1_2_pol_F , f1_3_pol_F  = POLAR_SPACE_F.extract_1(proj_polar_F_1form)
    f1_1_pol_F1, f1_2_pol_F1, f1_3_pol_F1 = POLAR_SPACE_F1.extract_1(proj_polar_F1_1form)

    # Apply discrete curl -> 2-form. Are all 3 components continuous when pushed forward?
    proj_tensor_2form   = TENSOR_SPACE.C.dot(proj_tensor_1form)
    proj_polar_F_2form  = POLAR_SPACE_F.C.dot(proj_polar_F_1form)
    proj_polar_F1_2form = POLAR_SPACE_F1.C.dot(proj_polar_F1_1form)

    # f[form]_[component]_[spline] are coefficients cijk.
    f2_1_ten   , f2_2_ten   , f2_3_ten    = TENSOR_SPACE.extract_2(proj_tensor_2form)
    f2_1_pol_F , f2_2_pol_F , f2_3_pol_F  = POLAR_SPACE_F.extract_2(proj_polar_F_2form)
    f2_1_pol_F1, f2_2_pol_F1, f2_3_pol_F1 = POLAR_SPACE_F1.extract_2(proj_polar_F1_2form)

    # TODO: evaluate splines, push forward
    lim_s = 1
    num_s_log = 5
    eta1_range = np.concatenate((np.logspace(-num_s_log, -2, num_s_log-1), np.linspace(1e-1, lim_s, 101-num_s_log)))
    # eta1_range = np.linspace(1e-4, lim_s, 101)
    eta2_range = np.linspace(0, 1, 101)
    eta3_range = np.linspace(0, 1, 3)
    # Evaluate at evaluation points:
    # eta1_range = TENSOR_SPACE.spaces[0].el_b
    # eta2_range = TENSOR_SPACE.spaces[1].el_b
    # eta3_range = TENSOR_SPACE.spaces[2].el_b
    # print(f'eta1_range {eta1_range}')
    # print(f'eta2_range {eta2_range}')
    # print(f'eta3_range {eta3_range}')

    eta1_sparse, eta2_sparse, eta3_sparse = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=True)
    eta1_dense,  eta2_dense,  eta3_dense  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)
    eta1,  eta2,  eta3  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)

    # Evaluate test function.
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

    # Evaluate derivative.
    evaled_2_1_tensor   = TENSOR_SPACE.evaluate_NDD(eta1, eta2, eta3, f2_1_ten)
    evaled_2_1_polar_F  = POLAR_SPACE_F.evaluate_NDD(eta1, eta2, eta3, f2_1_pol_F)
    evaled_2_1_polar_F1 = POLAR_SPACE_F1.evaluate_NDD(eta1, eta2, eta3, f2_1_pol_F1)

    evaled_2_2_tensor   = TENSOR_SPACE.evaluate_DND(eta1, eta2, eta3, f2_2_ten)
    evaled_2_2_polar_F  = POLAR_SPACE_F.evaluate_DND(eta1, eta2, eta3, f2_2_pol_F)
    evaled_2_2_polar_F1 = POLAR_SPACE_F1.evaluate_DND(eta1, eta2, eta3, f2_2_pol_F1)

    evaled_2_3_tensor   = TENSOR_SPACE.evaluate_DDN(eta1, eta2, eta3, f2_3_ten)
    evaled_2_3_polar_F  = POLAR_SPACE_F.evaluate_DDN(eta1, eta2, eta3, f2_3_pol_F)
    evaled_2_3_polar_F1 = POLAR_SPACE_F1.evaluate_DDN(eta1, eta2, eta3, f2_3_pol_F1)

    evaled_2_tensor   = [evaled_2_1_tensor  , evaled_2_2_tensor  , evaled_2_3_tensor  ]
    evaled_2_polar_F  = [evaled_2_1_polar_F , evaled_2_2_polar_F , evaled_2_3_polar_F ]
    evaled_2_polar_F1 = [evaled_2_1_polar_F1, evaled_2_2_polar_F1, evaled_2_3_polar_F1]



    # Analytical.
    x0 = F_x(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    y0 = F_y(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    z0 = F_z(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    x = F_x(eta1, eta2, eta3)
    y = F_y(eta1, eta2, eta3)
    z = F_z(eta1, eta2, eta3)
    orig_func = func_3d(x, y, z)
    orig_curl = curl_3d(x, y, z)



    # Push to canonical domain.
    # Polar splines should be smooth for the polar_F1.
    # Comparison only valid for polar_F1.
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

    # pushed_F1_2_1_tensor   = DOMAIN_F1.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_1')
    # pushed_F1_2_1_polar_F  = DOMAIN_F1.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_1')
    pushed_F1_2_1_polar_F1 = DOMAIN_F1.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_1')

    # pushed_F1_2_2_tensor   = DOMAIN_F1.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_2')
    # pushed_F1_2_2_polar_F  = DOMAIN_F1.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_2')
    pushed_F1_2_2_polar_F1 = DOMAIN_F1.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_2')

    # pushed_F1_2_3_tensor   = DOMAIN_F1.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_3')
    # pushed_F1_2_3_polar_F  = DOMAIN_F1.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_3')
    pushed_F1_2_3_polar_F1 = DOMAIN_F1.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_3')

    # pushed_F1_2_tensor   = np.array([pushed_F1_2_1_tensor  , pushed_F1_2_2_tensor  , pushed_F1_2_3_tensor  ])
    # pushed_F1_2_polar_F  = np.array([pushed_F1_2_1_polar_F , pushed_F1_2_2_polar_F , pushed_F1_2_3_polar_F ])
    pushed_F1_2_polar_F1 = np.array([pushed_F1_2_1_polar_F1, pushed_F1_2_2_polar_F1, pushed_F1_2_3_polar_F1])



    # Or push to physical domain.
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

    print(f'Max |orig_func|          : {np.max(np.abs(orig_func))}')
    print(f'Max |pushed_F_1_tensor|  : {np.max(np.abs(pushed_F_1_tensor))}')
    print(f'Max |pushed_F_1_polar_F| : {np.max(np.abs(pushed_F_1_polar_F))}')
    print(f'Max |pushed_F_1_polar_F1|: {np.max(np.abs(pushed_F_1_polar_F1))}')

    print(f'Min |orig_func|          : {np.min(np.abs(orig_func))}')
    print(f'Min |pushed_F_1_tensor|  : {np.min(np.abs(pushed_F_1_tensor))}')
    print(f'Min |pushed_F_1_polar_F| : {np.min(np.abs(pushed_F_1_polar_F))}')
    print(f'Min |pushed_F_1_polar_F1|: {np.min(np.abs(pushed_F_1_polar_F1))}')

    pushed_F_2_1_tensor   = DOMAIN_F.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_1')
    pushed_F_2_1_polar_F  = DOMAIN_F.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_1')
    pushed_F_2_1_polar_F1 = DOMAIN_F.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_1')

    pushed_F_2_2_tensor   = DOMAIN_F.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_2')
    pushed_F_2_2_polar_F  = DOMAIN_F.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_2')
    pushed_F_2_2_polar_F1 = DOMAIN_F.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_2')

    pushed_F_2_3_tensor   = DOMAIN_F.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_3')
    pushed_F_2_3_polar_F  = DOMAIN_F.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_3')
    pushed_F_2_3_polar_F1 = DOMAIN_F.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_3')

    pushed_F_2_tensor   = np.array([pushed_F_2_1_tensor  , pushed_F_2_2_tensor  , pushed_F_2_3_tensor  ])
    pushed_F_2_polar_F  = np.array([pushed_F_2_1_polar_F , pushed_F_2_2_polar_F , pushed_F_2_3_polar_F ])
    pushed_F_2_polar_F1 = np.array([pushed_F_2_1_polar_F1, pushed_F_2_2_polar_F1, pushed_F_2_3_polar_F1])

    print(f'Max |orig_curl|          : {np.max(np.abs(orig_curl))}')
    print(f'Max |pushed_F_2_tensor|  : {np.max(np.abs(pushed_F_2_tensor))}')
    print(f'Max |pushed_F_2_polar_F| : {np.max(np.abs(pushed_F_2_polar_F))}')
    print(f'Max |pushed_F_2_polar_F1|: {np.max(np.abs(pushed_F_2_polar_F1))}')

    print(f'Min |orig_curl|          : {np.min(np.abs(orig_curl))}')
    print(f'Min |pushed_F_2_tensor|  : {np.min(np.abs(pushed_F_2_tensor))}')
    print(f'Min |pushed_F_2_polar_F| : {np.min(np.abs(pushed_F_2_polar_F))}')
    print(f'Min |pushed_F_2_polar_F1|: {np.min(np.abs(pushed_F_2_polar_F1))}')



    print(f'Is pushed tensor equivalent to the original function? {np.allclose(orig_func, pushed_F_1_tensor)}')
    print(f'Is pushed polar_F equivalent to the original function? {np.allclose(orig_func, pushed_F_1_polar_F)}')
    print(f'Is pushed polar_F1 equivalent to the original function? {np.allclose(orig_func, pushed_F_1_polar_F1)}')
    print(f'Is pushed tensor equivalent to pushed polar_F? {np.allclose(pushed_F_1_tensor, pushed_F_1_polar_F)}')
    print(f'Shape of orig_func           : {orig_func.shape}')
    print(f'Shape of orig_curl           : {orig_curl.shape}')
    print(f'Shape of pushed_F_1_tensor   : {pushed_F_1_tensor.shape}')
    print(f'Shape of pushed_F_1_polar_F  : {pushed_F_1_polar_F.shape}')
    print(f'Shape of pushed_F_1_polar_F1 : {pushed_F_1_polar_F1.shape}')
    # print(orig_func-pushed_tensor)

    return (x, y, z, x0, y0, z0, 
    orig_func, pushed_F_1_tensor, pushed_F_1_polar_F, pushed_F_1_polar_F1, 
    orig_curl, pushed_F_2_tensor, pushed_F_2_polar_F, pushed_F_2_polar_F1)



def case_01_circle_identity_2form(Nel, p, spl_kind, nq_el, nq_pr, bc, func_3d, curl_3d, div_3d, DOMAIN_F):

    import numpy as np
    import matplotlib.pyplot as plt
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

    print('Running test: 2-form and its divergence.')



    # ============================================================
    # Define mappings F = F2 \circ F1.
    # ============================================================

    # Map F:
    # DOMAIN_F = dom.Domain('hollow_cyl', {'a1': .0, 'a2': 2., 'R0': 10.})

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



    # Use F for pullback!!!
    def fun_L_2_1(eta1, eta2, eta3):
        return DOMAIN_F.pull(func_3d, eta1, eta2, eta3, kind_fun='2_form_1', flat_eval=False)
    def fun_L_2_2(eta1, eta2, eta3):
        return DOMAIN_F.pull(func_3d, eta1, eta2, eta3, kind_fun='2_form_2', flat_eval=False)
    def fun_L_2_3(eta1, eta2, eta3):
        return DOMAIN_F.pull(func_3d, eta1, eta2, eta3, kind_fun='2_form_3', flat_eval=False)
    fun_L = [fun_L_2_1, fun_L_2_2, fun_L_2_3]

    proj_tensor_2form   = TENSOR_SPACE.projectors.pi_2(fun_L)
    proj_polar_F_2form  = POLAR_SPACE_F.projectors.pi_2(fun_L)
    proj_polar_F1_2form = POLAR_SPACE_F1.projectors.pi_2(fun_L)
    print(f'Shape of proj_tensor_2form   : {proj_tensor_2form.shape}')
    print(f'Shape of proj_polar_F_2form  : {proj_polar_F_2form.shape}')
    print(f'Shape of proj_polar_F1_2form : {proj_polar_F1_2form.shape}')

    # f[form]_[component]_[spline] are coefficients cijk.
    f2_1_ten   , f2_2_ten   , f2_3_ten    = TENSOR_SPACE.extract_2(proj_tensor_2form)
    f2_1_pol_F , f2_2_pol_F , f2_3_pol_F  = POLAR_SPACE_F.extract_2(proj_polar_F_2form)
    f2_1_pol_F1, f2_2_pol_F1, f2_3_pol_F1 = POLAR_SPACE_F1.extract_2(proj_polar_F1_2form)

    # Apply discrete divergence -> 3-form. Are all 3 components continuous when pushed forward?
    proj_tensor_3form   = TENSOR_SPACE.D.dot(proj_tensor_2form)
    proj_polar_F_3form  = POLAR_SPACE_F.D.dot(proj_polar_F_2form)
    proj_polar_F1_3form = POLAR_SPACE_F1.D.dot(proj_polar_F1_2form)

    # f[form]_[component]_[spline] are coefficients cijk.
    f3_ten    = TENSOR_SPACE.extract_3(proj_tensor_3form)
    f3_pol_F  = POLAR_SPACE_F.extract_3(proj_polar_F_3form)
    f3_pol_F1 = POLAR_SPACE_F1.extract_3(proj_polar_F1_3form)

    # TODO: evaluate splines, push forward
    lim_s = 1
    num_s_log = 5
    eta1_range = np.concatenate((np.logspace(-num_s_log, -2, num_s_log-1), np.linspace(1e-1, lim_s, 101-num_s_log)))
    # eta1_range = np.linspace(1e-4, lim_s, 101)
    eta2_range = np.linspace(0, 1, 101)
    eta3_range = np.linspace(0, 1, 3)
    # Evaluate at evaluation points:
    # eta1_range = TENSOR_SPACE.spaces[0].el_b
    # eta2_range = TENSOR_SPACE.spaces[1].el_b
    # eta3_range = TENSOR_SPACE.spaces[2].el_b
    # print(f'eta1_range {eta1_range}')
    # print(f'eta2_range {eta2_range}')
    # print(f'eta3_range {eta3_range}')

    eta1_sparse, eta2_sparse, eta3_sparse = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=True)
    eta1_dense,  eta2_dense,  eta3_dense  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)
    eta1,  eta2,  eta3  = np.meshgrid(eta1_range, eta2_range, eta3_range, indexing='ij', sparse=False)

    # Evaluate test function.
    evaled_2_1_tensor   = TENSOR_SPACE.evaluate_NDD(eta1, eta2, eta3, f2_1_ten)
    evaled_2_1_polar_F  = POLAR_SPACE_F.evaluate_NDD(eta1, eta2, eta3, f2_1_pol_F)
    evaled_2_1_polar_F1 = POLAR_SPACE_F1.evaluate_NDD(eta1, eta2, eta3, f2_1_pol_F1)

    evaled_2_2_tensor   = TENSOR_SPACE.evaluate_DND(eta1, eta2, eta3, f2_2_ten)
    evaled_2_2_polar_F  = POLAR_SPACE_F.evaluate_DND(eta1, eta2, eta3, f2_2_pol_F)
    evaled_2_2_polar_F1 = POLAR_SPACE_F1.evaluate_DND(eta1, eta2, eta3, f2_2_pol_F1)

    evaled_2_3_tensor   = TENSOR_SPACE.evaluate_DDN(eta1, eta2, eta3, f2_3_ten)
    evaled_2_3_polar_F  = POLAR_SPACE_F.evaluate_DDN(eta1, eta2, eta3, f2_3_pol_F)
    evaled_2_3_polar_F1 = POLAR_SPACE_F1.evaluate_DDN(eta1, eta2, eta3, f2_3_pol_F1)

    evaled_2_tensor   = [evaled_2_1_tensor  , evaled_2_2_tensor  , evaled_2_3_tensor  ]
    evaled_2_polar_F  = [evaled_2_1_polar_F , evaled_2_2_polar_F , evaled_2_3_polar_F ]
    evaled_2_polar_F1 = [evaled_2_1_polar_F1, evaled_2_2_polar_F1, evaled_2_3_polar_F1]

    # Evaluate derivative.
    evaled_3_tensor   = TENSOR_SPACE.evaluate_DDD(eta1, eta2, eta3, f3_ten)
    evaled_3_polar_F  = POLAR_SPACE_F.evaluate_DDD(eta1, eta2, eta3, f3_pol_F)
    evaled_3_polar_F1 = POLAR_SPACE_F1.evaluate_DDD(eta1, eta2, eta3, f3_pol_F1)



    # Analytical.
    x0 = F_x(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    y0 = F_y(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    z0 = F_z(np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1)))
    x = F_x(eta1, eta2, eta3)
    y = F_y(eta1, eta2, eta3)
    z = F_z(eta1, eta2, eta3)
    orig_func = func_3d(x, y, z)
    orig_div  = div_3d(x, y, z)



    # Push to canonical domain.
    # Polar splines should be smooth for the polar_F1.
    # Comparison only valid for polar_F1.

    # pushed_F1_2_1_tensor   = DOMAIN_F1.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_1')
    # pushed_F1_2_1_polar_F  = DOMAIN_F1.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_1')
    pushed_F1_2_1_polar_F1 = DOMAIN_F1.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_1')

    # pushed_F1_2_2_tensor   = DOMAIN_F1.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_2')
    # pushed_F1_2_2_polar_F  = DOMAIN_F1.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_2')
    pushed_F1_2_2_polar_F1 = DOMAIN_F1.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_2')

    # pushed_F1_2_3_tensor   = DOMAIN_F1.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_3')
    # pushed_F1_2_3_polar_F  = DOMAIN_F1.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_3')
    pushed_F1_2_3_polar_F1 = DOMAIN_F1.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_3')

    # pushed_F1_2_tensor   = np.array([pushed_F1_2_1_tensor  , pushed_F1_2_2_tensor  , pushed_F1_2_3_tensor  ])
    # pushed_F1_2_polar_F  = np.array([pushed_F1_2_1_polar_F , pushed_F1_2_2_polar_F , pushed_F1_2_3_polar_F ])
    pushed_F1_2_polar_F1 = np.array([pushed_F1_2_1_polar_F1, pushed_F1_2_2_polar_F1, pushed_F1_2_3_polar_F1])

    # pushed_F1_3_tensor   = DOMAIN_F1.push(evaled_3_tensor, eta1, eta2, eta3, kind_fun='3_form')
    # pushed_F1_3_polar_F  = DOMAIN_F1.push(evaled_3_polar_F, eta1, eta2, eta3, kind_fun='3_form')
    pushed_F1_3_polar_F1 = DOMAIN_F1.push(evaled_3_polar_F1, eta1, eta2, eta3, kind_fun='3_form')



    # Or push to physical domain.
    pushed_F_2_1_tensor   = DOMAIN_F.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_1')
    pushed_F_2_1_polar_F  = DOMAIN_F.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_1')
    pushed_F_2_1_polar_F1 = DOMAIN_F.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_1')

    pushed_F_2_2_tensor   = DOMAIN_F.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_2')
    pushed_F_2_2_polar_F  = DOMAIN_F.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_2')
    pushed_F_2_2_polar_F1 = DOMAIN_F.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_2')

    pushed_F_2_3_tensor   = DOMAIN_F.push(evaled_2_tensor, eta1, eta2, eta3, kind_fun='2_form_3')
    pushed_F_2_3_polar_F  = DOMAIN_F.push(evaled_2_polar_F, eta1, eta2, eta3, kind_fun='2_form_3')
    pushed_F_2_3_polar_F1 = DOMAIN_F.push(evaled_2_polar_F1, eta1, eta2, eta3, kind_fun='2_form_3')

    pushed_F_2_tensor   = np.array([pushed_F_2_1_tensor  , pushed_F_2_2_tensor  , pushed_F_2_3_tensor  ])
    pushed_F_2_polar_F  = np.array([pushed_F_2_1_polar_F , pushed_F_2_2_polar_F , pushed_F_2_3_polar_F ])
    pushed_F_2_polar_F1 = np.array([pushed_F_2_1_polar_F1, pushed_F_2_2_polar_F1, pushed_F_2_3_polar_F1])

    print(f'Max |orig_func|          : {np.max(np.abs(orig_func))}')
    print(f'Max |pushed_F_2_tensor|  : {np.max(np.abs(pushed_F_2_tensor))}')
    print(f'Max |pushed_F_2_polar_F| : {np.max(np.abs(pushed_F_2_polar_F))}')
    print(f'Max |pushed_F_2_polar_F1|: {np.max(np.abs(pushed_F_2_polar_F1))}')

    print(f'Min |orig_func|          : {np.min(np.abs(orig_func))}')
    print(f'Min |pushed_F_2_tensor|  : {np.min(np.abs(pushed_F_2_tensor))}')
    print(f'Min |pushed_F_2_polar_F| : {np.min(np.abs(pushed_F_2_polar_F))}')
    print(f'Min |pushed_F_2_polar_F1|: {np.min(np.abs(pushed_F_2_polar_F1))}')

    pushed_F_3_tensor   = DOMAIN_F.push(evaled_3_tensor, eta1, eta2, eta3, kind_fun='3_form')
    pushed_F_3_polar_F  = DOMAIN_F.push(evaled_3_polar_F, eta1, eta2, eta3, kind_fun='3_form')
    pushed_F_3_polar_F1 = DOMAIN_F.push(evaled_3_polar_F1, eta1, eta2, eta3, kind_fun='3_form')

    print(f'Max |orig_div|           : {np.max(np.abs(orig_div))}')
    print(f'Max |pushed_F_3_tensor|  : {np.max(np.abs(pushed_F_3_tensor))}')
    print(f'Max |pushed_F_3_polar_F| : {np.max(np.abs(pushed_F_3_polar_F))}')
    print(f'Max |pushed_F_3_polar_F1|: {np.max(np.abs(pushed_F_3_polar_F1))}')

    print(f'Min |orig_div|           : {np.min(np.abs(orig_div))}')
    print(f'Min |pushed_F_3_tensor|  : {np.min(np.abs(pushed_F_3_tensor))}')
    print(f'Min |pushed_F_3_polar_F| : {np.min(np.abs(pushed_F_3_polar_F))}')
    print(f'Min |pushed_F_3_polar_F1|: {np.min(np.abs(pushed_F_3_polar_F1))}')



    print(f'Is pushed tensor equivalent to the original function? {np.allclose(orig_func, pushed_F_2_tensor)}')
    print(f'Is pushed polar_F equivalent to the original function? {np.allclose(orig_func, pushed_F_2_polar_F)}')
    print(f'Is pushed polar_F1 equivalent to the original function? {np.allclose(orig_func, pushed_F_2_polar_F1)}')
    print(f'Is pushed tensor equivalent to pushed polar_F? {np.allclose(pushed_F_2_tensor, pushed_F_2_polar_F)}')
    print(f'Shape of orig_func           : {orig_func.shape}')
    print(f'Shape of orig_div            : {orig_div.shape}')
    print(f'Shape of pushed_F_2_tensor   : {pushed_F_2_tensor.shape}')
    print(f'Shape of pushed_F_2_polar_F  : {pushed_F_2_polar_F.shape}')
    print(f'Shape of pushed_F_2_polar_F1 : {pushed_F_2_polar_F1.shape}')
    # print(orig_func-pushed_tensor)

    return (x, y, z, x0, y0, z0,
    orig_func, pushed_F_2_tensor, pushed_F_2_polar_F, pushed_F_2_polar_F1, 
    orig_div , pushed_F_3_tensor, pushed_F_3_polar_F, pushed_F_3_polar_F1)



def get_gvec_domain(Nel, p, spl_kind, nq_el, nq_pr, bc):

    import numpy as np
    import struphy.geometry.domain_3d as dom
    import struphy.feec.spline_space as spl

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
    # bounds = {'b1': 0.3, 'e1': 0.8, 'b2': 0.3, 'e2': 0.8, 'b3': 0.3, 'e3': 0.8}
    bounds = {'b1': 0.0, 'e1': 1.0, 'b2': 0.0, 'e2': 1.0, 'b3': 0.0, 'e3': 1.0}
    SOURCE_DOMAIN = dom.Domain('cuboid', params_map=bounds)

    def s(eta1, eta2, eta3):
        return SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'x')

    def u(eta1, eta2, eta3):
        return SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'y')

    def v(eta1, eta2, eta3):
        return SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'z')

    def eta123_to_suv(eta1, eta2, eta3):
        return s(eta1, eta2, eta3), u(eta1, eta2, eta3), v(eta1, eta2, eta3)

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



    return DOMAIN, gvec



def plot_wrapper(x, y, z, x0, y0, z0,
f_exact, f_ten_F, f_pol_F, f_pol_F1, 
df_exact, df_ten_F, df_pol_F, df_pol_F1, 
func_form, Nel, p):

    import copy
    import numpy as np
    import matplotlib.pyplot as plt

    plot_handles = []

    metadata_base = {
        'func_form': func_form,
        'origin': [x0, y0, z0,],
        'suptitle': 'Compare 3D Polar Splines (LHS: $\hat{f}^1_1$, RHS: $[\\nabla \\times \hat{f}^1]_1 = \hat{f}^2_1$) (Original)',
        'comptype': Comparison.ORIG,
        'anacolor': True, # Use analytical solution to scale plots.
        'colscale': 2,    # Factor to scale up/down the max/min of the plot's color code.
        'collevel': 20,   # Number of color levels on a contour plot.
        'unlink': False,  # Set color and z-axis limits individually.
        'lplot': PlotTypeLeft.CONTOUR2D,
        'lxlabel': '$x$', # '$\eta^1$',
        'lylabel': '$y$', # '$\eta^2$',
        # 'lzlabel': '$f(x,y)$',
        # 'lzlabeld': '$f(x,y) - \hat{f}(x,y)$',
        'rplot': PlotTypeRight.CONTOUR2D,
        'rxlabel': '$x$',
        'rylabel': '$y$',
        # 'rzlabel': '$\\nabla f(x,y)$',
        # 'rzlabeld': '$\\nabla f(x,y) - \\nabla \hat{f}(x,y)$',
    }



    if func_form == FuncForm.ZERO:

        # LHS: Original 0-form function.
        # RHS: 1-form gradient, only x component, because y is identical and z is zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.

        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^0$, RHS: $[\hat{\\nabla} \hat{f}^0]_1 = \hat{f}^1_1$) (Original)'
        metadata['comptype'] = Comparison.ORIG
        metadata['lzlabel'] = '$f(x,y)=\hat{f}^0$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \hat{f}^0]_1$'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact   ,  f_ten_F   ,  f_pol_F   ,  f_pol_F1   , 
        df_exact[0], df_ten_F[0], df_pol_F[0], df_pol_F1[0])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^0$, RHS: $[\hat{\\nabla} \hat{f}^0]_1 = \hat{f}^1_1$) (Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$f(x,y)=\hat{f}^0$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \hat{f}^0]_1$'
        metadata['lzlabeld'] = 'Difference from analytical'
        metadata['rzlabeld'] = 'Difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact   ,   f_ten_F -  f_exact    ,   f_pol_F -  f_exact    ,   f_pol_F1 -  f_exact    , 
        df_exact[0], (df_ten_F - df_exact)[0], (df_pol_F - df_exact)[0], (df_pol_F1 - df_exact)[0])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^0$, RHS: $[\hat{\\nabla} \hat{f}^0]_1 = \hat{f}^1_1$) (Absolute Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$f(x,y)=\hat{f}^0$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \hat{f}^0]_1$'
        metadata['lzlabeld'] = 'Absolute difference from analytical'
        metadata['rzlabeld'] = 'Absolute difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact   , np.abs( f_ten_F -  f_exact)   , np.abs( f_pol_F -  f_exact)   , np.abs( f_pol_F1 -  f_exact)   , 
        df_exact[0], np.abs(df_ten_F - df_exact)[0], np.abs(df_pol_F - df_exact)[0], np.abs(df_pol_F1 - df_exact)[0])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^0$, RHS: $[\hat{\\nabla} \hat{f}^0]_1 = \hat{f}^1_1$) (Log Absolute Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$f(x,y)=\hat{f}^0$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \hat{f}^0]_1$'
        metadata['lzlabeld'] = 'Log absolute difference from analytical'
        metadata['rzlabeld'] = 'Log absolute difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact   , np.log(np.abs( f_ten_F -  f_exact))   , np.log(np.abs( f_pol_F -  f_exact))   , np.log(np.abs( f_pol_F1 -  f_exact))   , 
        df_exact[0], np.log(np.abs(df_ten_F - df_exact))[0], np.log(np.abs(df_pol_F - df_exact))[0], np.log(np.abs(df_pol_F1 - df_exact))[0])
        plot_handles.append(handle)



    elif func_form == FuncForm.ONE:

        # LHS: Original 1-form function, only x-component because all directions are identical.
        # RHS: 2-form curl, only z component because x and y are zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.

        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^1_1$, RHS: $[\hat{\\nabla} \\times \hat{f}^1]_1 = \hat{f}^2_1$) (Original)'
        metadata['comptype'] = Comparison.ORIG
        metadata['lzlabel'] = '$\hat{f}^1_1$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \\times \hat{f}^1]_1 = \hat{f}^2_1$'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[0],  f_ten_F[0],  f_pol_F[0],  f_pol_F1[0], 
        df_exact[0], df_ten_F[0], df_pol_F[0], df_pol_F1[0])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^1_1$, RHS: $[\hat{\\nabla} \\times \hat{f}^1]_1 = \hat{f}^2_1$) (Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$\hat{f}^1_1$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \\times \hat{f}^1]_1 = \hat{f}^2_1$'
        metadata['lzlabeld'] = 'Difference from analytical'
        metadata['rzlabeld'] = 'Difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[0], ( f_ten_F -  f_exact)[0], ( f_pol_F -  f_exact)[0], ( f_pol_F1 -  f_exact)[0], 
        df_exact[0], (df_ten_F - df_exact)[0], (df_pol_F - df_exact)[0], (df_pol_F1 - df_exact)[0])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^1_2$, RHS: $[\hat{\\nabla} \\times \hat{f}^1]_2 = \hat{f}^2_2$) (Original)'
        metadata['comptype'] = Comparison.ORIG
        metadata['lzlabel'] = '$\hat{f}^1_2$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \\times \hat{f}^1]_2 = \hat{f}^2_2$'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[1],  f_ten_F[1],  f_pol_F[1],  f_pol_F1[1], 
        df_exact[1], df_ten_F[1], df_pol_F[1], df_pol_F1[1])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^1_2$, RHS: $[\hat{\\nabla} \\times \hat{f}^1]_2 = \hat{f}^2_2$) (Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$\hat{f}^1_2$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \\times \hat{f}^1]_2 = \hat{f}^2_2$'
        metadata['lzlabeld'] = 'Difference from analytical'
        metadata['rzlabeld'] = 'Difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[1], ( f_ten_F -  f_exact)[1], ( f_pol_F -  f_exact)[1], ( f_pol_F1 -  f_exact)[1], 
        df_exact[1], (df_ten_F - df_exact)[1], (df_pol_F - df_exact)[1], (df_pol_F1 - df_exact)[1])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^1_3$, RHS: $[\hat{\\nabla} \\times \hat{f}^1]_3 = \hat{f}^2_3$) (Original)'
        metadata['comptype'] = Comparison.ORIG
        metadata['lzlabel'] = '$\hat{f}^1_3$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \\times \hat{f}^1]_3 = \hat{f}^2_3$'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[2],  f_ten_F[2],  f_pol_F[2],  f_pol_F1[2], 
        df_exact[2], df_ten_F[2], df_pol_F[2], df_pol_F1[2])
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^1_3$, RHS: $[\hat{\\nabla} \\times \hat{f}^1]_3 = \hat{f}^2_3$) (Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$\hat{f}^1_3$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \\times \hat{f}^1]_3 = \hat{f}^2_3$'
        metadata['lzlabeld'] = 'Difference from analytical'
        metadata['rzlabeld'] = 'Difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[2], ( f_ten_F -  f_exact)[2], ( f_pol_F -  f_exact)[2], ( f_pol_F1 -  f_exact)[2], 
        df_exact[2], (df_ten_F - df_exact)[2], (df_pol_F - df_exact)[2], (df_pol_F1 - df_exact)[2])
        plot_handles.append(handle)



    elif func_form == FuncForm.TWO:

        # LHS: Original 2-form function, only x-component because all directions are identical.
        # RHS: 3-form div.
        # Figure 1: Actual value.
        # Figure 2: Difference.

        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^2_1$, RHS: $[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$) (Original)'
        metadata['comptype'] = Comparison.ORIG
        metadata['lzlabel'] = '$\hat{f}^2_1$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[0],  f_ten_F[0],  f_pol_F[0],  f_pol_F1[0], 
        df_exact   , df_ten_F   , df_pol_F   , df_pol_F1   )
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^2_1$, RHS: $[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$) (Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$\hat{f}^2_1$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$'
        metadata['lzlabeld'] = 'Difference from analytical'
        metadata['rzlabeld'] = 'Difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[0], ( f_ten_F -  f_exact)[0], ( f_pol_F -  f_exact)[0], ( f_pol_F1 -  f_exact)[0], 
        df_exact   , (df_ten_F - df_exact)   , (df_pol_F - df_exact)   , (df_pol_F1 - df_exact)   )
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^2_2$, RHS: $[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$) (Original)'
        metadata['comptype'] = Comparison.ORIG
        metadata['lzlabel'] = '$\hat{f}^2_2$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[1],  f_ten_F[1],  f_pol_F[1],  f_pol_F1[1], 
        df_exact   , df_ten_F   , df_pol_F   , df_pol_F1   )
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^2_2$, RHS: $[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$) (Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$\hat{f}^2_2$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$'
        metadata['lzlabeld'] = 'Difference from analytical'
        metadata['rzlabeld'] = 'Difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[1], ( f_ten_F -  f_exact)[1], ( f_pol_F -  f_exact)[1], ( f_pol_F1 -  f_exact)[1], 
        df_exact   , (df_ten_F - df_exact)   , (df_pol_F - df_exact)   , (df_pol_F1 - df_exact)   )
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^2_3$, RHS: $[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$) (Original)'
        metadata['comptype'] = Comparison.ORIG
        metadata['lzlabel'] = '$\hat{f}^2_3$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[2],  f_ten_F[2],  f_pol_F[2],  f_pol_F1[2], 
        df_exact   , df_ten_F   , df_pol_F   , df_pol_F1   )
        plot_handles.append(handle)


        metadata = copy.deepcopy(metadata_base)
        metadata['suptitle'] = 'Compare 3D Polar Splines (LHS: $\hat{f}^2_3$, RHS: $[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$) (Error)'
        metadata['comptype'] = Comparison.DIFF
        metadata['lzlabel'] = '$\hat{f}^2_3$'
        metadata['rzlabel'] = '$[\hat{\\nabla} \cdot \hat{f}^2] = \hat{f}^3$'
        metadata['lzlabeld'] = 'Difference from analytical'
        metadata['rzlabeld'] = 'Difference from analytical'
        handle = plot_comparison(metadata, x, y, z, 
         f_exact[2], ( f_ten_F -  f_exact)[2], ( f_pol_F -  f_exact)[2], ( f_pol_F1 -  f_exact)[2], 
        df_exact   , (df_ten_F - df_exact)   , (df_pol_F - df_exact)   , (df_pol_F1 - df_exact)   )
        plot_handles.append(handle)



    else:

        raise NotImplementedError(f'FuncForm {func_form} not recognized.')



    update_fig_suptitle(plot_handles, Nel, p)



    return plot_handles



def plot_comparison(metadata, eta1, eta2, eta3, 
f_exact, f_ten_F, f_pol_F, f_pol_F1, 
df_exact, df_ten_F, df_pol_F, df_pol_F1):
    """Expects 3D input with useless z-component, effectively 2D."""

    import numpy             as np
    import matplotlib.pyplot as plt
    import matplotlib.tri    as tri

    func_form = metadata['func_form']
    origin = metadata['origin']
    levels = metadata['collevel']
    axlim = 0.01

    if metadata['comptype'] == Comparison.ORIG:
        print('Comparison type: Original values.')
        lmin = np.min([
            np.min(f_exact),
            np.min(f_ten_F),
            np.min(f_pol_F),
            np.min(f_pol_F1),
        ])
        lmax = np.max([
            np.max(f_exact),
            np.max(f_ten_F),
            np.max(f_pol_F),
            np.max(f_pol_F1),
        ])
        rmin = np.min([
            np.min(df_exact),
            np.min(df_ten_F),
            np.min(df_pol_F),
            np.min(df_pol_F1),
        ])
        rmax = np.max([
            np.max(df_exact),
            np.max(df_ten_F),
            np.max(df_pol_F),
            np.max(df_pol_F1),
        ])
        if metadata['anacolor']:
            print('Scale with analytical solution.')
            lmin = np.min(f_exact)
            lmax = np.max(f_exact)
            rmin = np.min(df_exact)
            rmax = np.max(df_exact)
    else:
        print('Comparison type: Errors.')
        lmin = np.min([
            np.min(f_ten_F),
            np.min(f_pol_F),
            np.min(f_pol_F1),
        ])
        lmax = np.max([
            np.max(f_ten_F),
            np.max(f_pol_F),
            np.max(f_pol_F1),
        ])
        rmin = np.min([
            np.min(df_ten_F),
            np.min(df_pol_F),
            np.min(df_pol_F1),
        ])
        rmax = np.max([
            np.max(df_ten_F),
            np.max(df_pol_F),
            np.max(df_pol_F1),
        ])

    # Because np.log(0)!
    if np.isneginf(lmin):
        print('Overwritten infinite lmin.')
        lmin = -10000
    if np.isneginf(rmin):
        print('Overwritten infinite rmin.')
        rmin = -10000
    if np.isposinf(lmax):
        print('Overwritten infinite lmax.')
        lmax = 10000
    if np.isposinf(rmax):
        print('Overwritten infinite rmax.')
        rmax = 10000

    if metadata['colscale'] != 1:
        lrange = lmax - lmin
        rrange = rmax - rmin
        lmid = (lmax + lmin) / 2
        rmid = (rmax + rmin) / 2
        lrange_new = lrange * metadata['colscale']
        rrange_new = rrange * metadata['colscale']
        lmin = lmid - lrange_new / 2
        lmax = lmid + lrange_new / 2
        rmin = rmid - rrange_new / 2
        rmax = rmid + rrange_new / 2

    if metadata['unlink']:
        lmin=None
        lmax=None
        rmin=None
        rmax=None

    print(f'lmin: {lmin}')
    print(f'lmax: {lmax}')
    print(f'rmin: {rmin}')
    print(f'rmax: {rmax}')

    # Plot settings.
    row = 2
    col = 4
    dpi = 100
    width, height = (1920 / dpi, 1200 / dpi)
    fig = plt.figure(figsize=(width,height), dpi=dpi)
    gs  = fig.add_gridspec(row, col, width_ratios=[1,1]*2)
    fig.canvas.manager.set_window_title(metadata['suptitle'] + '\n')
    fig.suptitle(metadata['suptitle'] + '\n', y=0.98)

    l_is_3d = metadata['lplot'] in [PlotTypeLeft.CONTOUR3D, PlotTypeLeft.SCATTER, PlotTypeLeft.SURFACE, PlotTypeLeft.WIREFRAME]
    r_is_3d = metadata['rplot'] in [PlotTypeRight.CONTOUR3D, PlotTypeRight.QUIVER3D, PlotTypeRight.SURFACE]

    if l_is_3d:
        lax1 = fig.add_subplot(gs[0, 0], projection='3d')
        lax2 = fig.add_subplot(gs[0, 1], projection='3d')
        lax3 = fig.add_subplot(gs[1, 0], projection='3d')
        lax4 = fig.add_subplot(gs[1, 1], projection='3d')
    else:
        lax1 = fig.add_subplot(gs[0, 0])
        lax2 = fig.add_subplot(gs[0, 1])
        lax3 = fig.add_subplot(gs[1, 0])
        lax4 = fig.add_subplot(gs[1, 1])
    if r_is_3d:
        print('Plot is 3D')
        rax1 = fig.add_subplot(gs[0, 2], projection='3d')
        rax2 = fig.add_subplot(gs[0, 3], projection='3d')
        rax3 = fig.add_subplot(gs[1, 2], projection='3d')
        rax4 = fig.add_subplot(gs[1, 3], projection='3d')
    else:
        rax1 = fig.add_subplot(gs[0, 2])
        rax2 = fig.add_subplot(gs[0, 3])
        rax3 = fig.add_subplot(gs[1, 2])
        rax4 = fig.add_subplot(gs[1, 3])
    laxes = [lax1, lax2, lax3, lax4]
    raxes = [rax1, rax2, rax3, rax4]

    for idx, ax in enumerate(laxes):
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        # ax.zaxis.get_major_locator().set_params(integer=True)
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel(metadata['lxlabel'])
        ax.set_ylabel(metadata['lylabel'])
        if l_is_3d:
            if not metadata['unlink']:
                ax.set_zlim(lmin, lmax)
            if metadata['comptype'] == Comparison.ORIG:
                ax.set_zlabel(metadata['lzlabel'])
            else:
                if idx == 0:
                    ax.set_zlabel(metadata['lzlabel'])
                else:
                    ax.set_zlabel(metadata['lzlabeld'])
            ax.set_aspect('auto', adjustable='box')
        else:
            ax.set_aspect('equal', adjustable='box')

    for idx, ax in enumerate(raxes):
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        # ax.zaxis.get_major_locator().set_params(integer=True)
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel(metadata['rxlabel'])
        ax.set_ylabel(metadata['rylabel'])
        if r_is_3d:
            if not metadata['unlink']:
                ax.set_zlim(rmin, rmax)
            if metadata['comptype'] == Comparison.ORIG:
                ax.set_zlabel(metadata['rzlabel'])
            else:
                if idx == 0:
                    ax.set_zlabel(metadata['rzlabel'])
                else:
                    ax.set_zlabel(metadata['rzlabeld'])
            ax.set_aspect('auto', adjustable='box')
        else:
            ax.set_aspect('equal', adjustable='box')

    lax1.set_title(metadata['lzlabel'] + ' analytical')
    lax2.set_title(metadata['lzlabel'] + ' pushed, on tensor F')
    lax3.set_title(metadata['lzlabel'] + ' pushed, on polar F')
    lax4.set_title(metadata['lzlabel'] + ' pushed, on polar F1')

    if metadata['rplot'] in [PlotTypeRight.QUIVER2D, PlotTypeRight.QUIVER3D]:
        rax1.set_title('Derivative $\\nabla f(x,y)$')
        rax2.set_title('$\\nabla f(x,y)$ on tensor F')
        rax3.set_title('$\\nabla f(x,y)$ on polar F')
        rax4.set_title('$\\nabla f(x,y)$ on polar F1')

    rax1.set_title(metadata['rzlabel'] + ' analytical')
    rax2.set_title(metadata['rzlabel'] + ' pushed, on tensor F')
    rax3.set_title(metadata['rzlabel'] + ' pushed, on polar F')
    rax4.set_title(metadata['rzlabel'] + ' pushed, on polar F1')



    # ============================================================
    # Four original functions on the left.
    # ============================================================

    if metadata['lplot'] in [PlotTypeLeft.CONTOUR2D, PlotTypeLeft.CONTOUR3D]:
        if metadata['comptype'] == Comparison.ORIG:
            limg1 = lax1.contourf(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
        else:
            limg1 = lax1.contourf(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0])
        limg2 = lax2.contourf(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
        limg3 = lax3.contourf(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
        limg4 = lax4.contourf(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
    elif metadata['lplot'] == PlotTypeLeft.SURFACE:
        limg1 = lax1.plot_surface(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
        limg2 = lax2.plot_surface(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
        limg3 = lax3.plot_surface(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
        limg4 = lax4.plot_surface(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
    elif metadata['lplot'] == PlotTypeLeft.WIREFRAME:
        limg1 = lax1.plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0])
        limg2 = lax2.plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0])
        limg3 = lax3.plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0])
        limg4 = lax4.plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0])
    elif metadata['lplot'] == PlotTypeLeft.SCATTER:
        # Use Scatter plot for 2D data:
        limg1 = lax1.scatter(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0], c=f_exact[:,:,0], cmap=plt.cm.viridis, marker='.')
        limg2 = lax2.scatter(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0], c=f_ten_F[:,:,0], cmap=plt.cm.viridis, marker='.')
        limg3 = lax3.scatter(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0], c=f_pol_F[:,:,0], cmap=plt.cm.viridis, marker='.')
        limg4 = lax4.scatter(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0], c=f_pol_F1[:,:,0], cmap=plt.cm.viridis, marker='.')
        # When the function is dependent on eta3 as well (3D).
        # limg1 = lax1.scatter(eta1, eta2, eta3, c=f_exact, cmap=plt.cm.viridis, marker='.')
        # limg2 = lax2.scatter(eta1, eta2, eta3, c=f_ten_F, cmap=plt.cm.viridis, marker='.')
        # limg3 = lax3.scatter(eta1, eta2, eta3, c=f_pol_F, cmap=plt.cm.viridis, marker='.')
        # limg4 = lax4.scatter(eta1, eta2, eta3, c=f_pol_F1, cmap=plt.cm.viridis, marker='.')



    if metadata['lplot'] != PlotTypeLeft.WIREFRAME:
        if metadata['comptype'] == Comparison.ORIG:
            lcbar1 = fig.colorbar(limg1, ax=lax1, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabel'])
            lcbar2 = fig.colorbar(limg2, ax=lax2, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabel'])
            lcbar3 = fig.colorbar(limg3, ax=lax3, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabel'])
            lcbar4 = fig.colorbar(limg4, ax=lax4, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabel'])
        else:
            lcbar1 = fig.colorbar(limg1, ax=lax1, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabel'])
            lcbar2 = fig.colorbar(limg2, ax=lax2, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabeld'])
            lcbar3 = fig.colorbar(limg3, ax=lax3, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabeld'])
            lcbar4 = fig.colorbar(limg4, ax=lax4, shrink=0.9, pad=0.15, location='bottom', label=metadata['lzlabeld'])
        lcbar1.ax.locator_params(nbins=5)
        lcbar2.ax.locator_params(nbins=5)
        lcbar3.ax.locator_params(nbins=5)
        lcbar4.ax.locator_params(nbins=5)

    limgs = [limg1, limg2, limg3, limg4]
    if metadata['lplot'] != PlotTypeLeft.WIREFRAME:
        lcbars = [lcbar1, lcbar2, lcbar3, lcbar4]
    else:
        lcbars = None



    # ============================================================
    # Four derivatives on the right.
    # ============================================================

    if func_form == FuncForm.ZERO:

        # LHS: Original 0-form function.
        # RHS: 1-form gradient, only x component, because y is identical and z is zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.
        df_mag = np.sqrt(2 * df_ten_F**2)

    elif func_form == FuncForm.ONE:

        # LHS: Original 1-form function, only x-component because all directions are identical.
        # RHS: 2-form curl, only z component because x and y are zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.
        df_mag = np.sqrt(3 * df_ten_F**2)

    elif func_form == FuncForm.TWO:

        # LHS: Original 2-form function, only x-component because all directions are identical.
        # RHS: 3-form div.
        # Figure 1: Actual value.
        # Figure 2: Difference.
        df_mag = np.sqrt(3 * df_ten_F**2)

    else:

        raise NotImplementedError(f'FuncForm {func_form} not recognized.')

    # df_mag = np.sqrt(df_ten_F[0]**2 + df_ten_F[1]**2 + df_ten_F[2]**2)
    print('df_mag.shape', df_mag.shape)
    print('Max |df|:', np.max(df_mag))
    print('Min |df|:', np.min(df_mag))

    # Flatten and normalize.
    if r_is_3d:
        c = (df_mag.ravel() - df_mag.min()) / df_mag.ptp()
    else:
        c = (df_mag[:,:,0].ravel() - df_mag[:,:,0].min()) / df_mag[:,:,0].ptp()
    # print('c.shape', c.shape)
    # Repeat for each body line and two head lines in a quiver.
    c = np.concatenate((c, np.repeat(c, 2)))
    # print('c.shape', c.shape)
    # Colormap.
    c = plt.cm.viridis(c)
    # print('c.shape', c.shape)



    if metadata['rplot'] in [PlotTypeRight.CONTOUR2D, PlotTypeRight.CONTOUR3D]:
        if metadata['comptype'] == Comparison.ORIG:
            rimg1 = rax1.contourf(eta1[:,:,0], eta2[:,:,0], df_exact[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
        else:
            rimg1 = rax1.contourf(eta1[:,:,0], eta2[:,:,0], df_exact[:,:,0])
        rimg2 = rax2.contourf(eta1[:,:,0], eta2[:,:,0], df_ten_F[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
        rimg3 = rax3.contourf(eta1[:,:,0], eta2[:,:,0], df_pol_F[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
        rimg4 = rax4.contourf(eta1[:,:,0], eta2[:,:,0], df_pol_F1[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
    elif metadata['rplot'] == PlotTypeRight.SURFACE:
        rimg1 = rax1.plot_surface(eta1[:,:,0], eta2[:,:,0], df_exact[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
        rimg2 = rax2.plot_surface(eta1[:,:,0], eta2[:,:,0], df_ten_F[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
        rimg3 = rax3.plot_surface(eta1[:,:,0], eta2[:,:,0], df_pol_F[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
        rimg4 = rax4.plot_surface(eta1[:,:,0], eta2[:,:,0], df_pol_F1[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
    elif metadata['rplot'] == PlotTypeRight.QUIVER2D:
        scale = 500
        metadata['rzlabel'] = metadata['rzlabel'] + f' Scale: {1/scale}'
        rimg1 = rax1.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_exact[0,:,:,0].flatten(), df_exact[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimg2 = rax2.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_ten_F[0,:,:,0].flatten(), df_ten_F[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimg3 = rax3.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_pol_F[0,:,:,0].flatten(), df_pol_F[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimg4 = rax4.quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_pol_F1[0,:,:,0].flatten(), df_pol_F1[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimg1.set_array(df_mag[:,:,0].flatten())
        rimg2.set_array(df_mag[:,:,0].flatten())
        rimg3.set_array(df_mag[:,:,0].flatten())
        rimg4.set_array(df_mag[:,:,0].flatten())
        rimg1.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
        rimg2.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
        rimg3.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
        rimg4.set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
    elif metadata['rplot'] == PlotTypeRight.QUIVER3D:
        scale = 500
        metadata['rzlabel'] = metadata['rzlabel'] + f' Scale: {1/scale}'
        rimg1 = rax1.quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_exact[0].flatten(), df_exact[1].flatten(), df_exact[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimg2 = rax2.quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_ten_F[0].flatten(), df_ten_F[1].flatten(), df_ten_F[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimg3 = rax3.quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_pol_F[0].flatten(), df_pol_F[1].flatten(), df_pol_F[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimg4 = rax4.quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_pol_F1[0].flatten(), df_pol_F1[1].flatten(), df_pol_F1[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimg1.set_array(df_mag.flatten())
        rimg2.set_array(df_mag.flatten())
        rimg3.set_array(df_mag.flatten())
        rimg4.set_array(df_mag.flatten())
        rimg1.set_clim(np.min(df_mag), np.max(df_mag))
        rimg2.set_clim(np.min(df_mag), np.max(df_mag))
        rimg3.set_clim(np.min(df_mag), np.max(df_mag))
        rimg4.set_clim(np.min(df_mag), np.max(df_mag))



    if metadata['comptype'] == Comparison.ORIG:
        rcbar1 = fig.colorbar(rimg1, ax=rax1, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabel'])
        rcbar2 = fig.colorbar(rimg2, ax=rax2, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabel'])
        rcbar3 = fig.colorbar(rimg3, ax=rax3, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabel'])
        rcbar4 = fig.colorbar(rimg4, ax=rax4, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabel'])
    else:
        rcbar1 = fig.colorbar(rimg1, ax=rax1, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabel'])
        rcbar2 = fig.colorbar(rimg2, ax=rax2, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabeld'])
        rcbar3 = fig.colorbar(rimg3, ax=rax3, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabeld'])
        rcbar4 = fig.colorbar(rimg4, ax=rax4, shrink=0.9, pad=0.15, location='bottom', label=metadata['rzlabeld'])
    rcbar1.ax.locator_params(nbins=5)
    rcbar2.ax.locator_params(nbins=5)
    rcbar3.ax.locator_params(nbins=5)
    rcbar4.ax.locator_params(nbins=5)

    rimgs = [rimg1, rimg2, rimg3, rimg4]
    rcbars = [rcbar1, rcbar2, rcbar3, rcbar4]



    # Mark position on mapping pole.
    lpoles = []
    for idx, ax in enumerate(laxes):
        ax.set_xlim(origin[0] - axlim, origin[0] + axlim)
        ax.set_ylim(origin[1] - axlim, origin[1] + axlim)
        if l_is_3d:
            pole = ax.scatter(origin[0], origin[1], origin[2], marker='o', edgecolors='red', facecolors='none', label='Pole')
        else:
            pole = ax.scatter(origin[0], origin[1], 100,       marker='o', edgecolors='red', facecolors='none', label='Pole')
        lpoles.append(pole)
    rpoles = []
    for idx, ax in enumerate(raxes):
        ax.set_xlim(origin[0] - axlim, origin[0] + axlim)
        ax.set_ylim(origin[1] - axlim, origin[1] + axlim)
        if r_is_3d:
            pole = ax.scatter(origin[0], origin[1], origin[2], marker='o', edgecolors='red', facecolors='none', label='Pole')
        else:
            pole = ax.scatter(origin[0], origin[1], 100,       marker='o', edgecolors='red', facecolors='none', label='Pole')
        rpoles.append(pole)



    # ============================================================
    # Show the figure.
    # ============================================================

    # print('Before `tight_layout()`   | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.tight_layout()
    # print('After `tight_layout()`    | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.subplots_adjust(hspace=fig.subplotpars.hspace * 1.2, wspace=fig.subplotpars.wspace * .9)
    # print('After `subplots_adjust()` | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))

    return {
        'fig': fig,
        'laxes': laxes,
        'raxes': raxes,
        'limgs': limgs,
        'rimgs': rimgs,
        'lcbars': lcbars,
        'rcbars': rcbars,
        'lpoles': lpoles,
        'rpoles': rpoles,
        'metadata': metadata,
    }



def plot_updater(plot_handles, x, y, z, x0, y0, z0, 
f_exact, f_ten_F, f_pol_F, f_pol_F1, 
df_exact, df_ten_F, df_pol_F, df_pol_F1):

    import numpy as np
    import matplotlib.pyplot as plt

    import copy
    metadata = plot_handles[0]['metadata']
    func_form = metadata['func_form']

    if func_form == FuncForm.ZERO:

        # LHS: Original 0-form function.
        # RHS: 1-form gradient, only x component, because y is identical and z is zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.

        handle = plot_handles[0]
        plot_update(handle, x, y, z, 
         f_exact   ,  f_ten_F   ,  f_pol_F   ,  f_pol_F1   , 
        df_exact[0], df_ten_F[0], df_pol_F[0], df_pol_F1[0])

        handle = plot_handles[1]
        plot_update(handle, x, y, z, 
         f_exact   ,   f_ten_F -  f_exact    ,   f_pol_F -  f_exact    ,   f_pol_F1 -  f_exact    , 
        df_exact[0], (df_ten_F - df_exact)[0], (df_pol_F - df_exact)[0], (df_pol_F1 - df_exact)[0])

        handle = plot_handles[2]
        plot_update(handle, x, y, z, 
         f_exact   , np.abs( f_ten_F -  f_exact)   , np.abs( f_pol_F -  f_exact)   , np.abs( f_pol_F1 -  f_exact)   , 
        df_exact[0], np.abs(df_ten_F - df_exact)[0], np.abs(df_pol_F - df_exact)[0], np.abs(df_pol_F1 - df_exact)[0])

        handle = plot_handles[3]
        plot_update(handle, x, y, z, 
         f_exact   , np.log(np.abs( f_ten_F -  f_exact))   , np.log(np.abs( f_pol_F -  f_exact))   , np.log(np.abs( f_pol_F1 -  f_exact))   , 
        df_exact[0], np.log(np.abs(df_ten_F - df_exact))[0], np.log(np.abs(df_pol_F - df_exact))[0], np.log(np.abs(df_pol_F1 - df_exact))[0])



    elif func_form == FuncForm.ONE:

        # LHS: Original 1-form function, only x-component because all directions are identical.
        # RHS: 2-form curl, only z component because x and y are zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.

        handle = plot_handles[0]
        plot_update(handle, x, y, z, 
         f_exact[0],  f_ten_F[0],  f_pol_F[0],  f_pol_F1[0], 
        df_exact[0], df_ten_F[0], df_pol_F[0], df_pol_F1[0])

        handle = plot_handles[1]
        plot_update(handle, x, y, z, 
         f_exact[0], ( f_ten_F -  f_exact)[0], ( f_pol_F -  f_exact)[0], ( f_pol_F1 -  f_exact)[0], 
        df_exact[0], (df_ten_F - df_exact)[0], (df_pol_F - df_exact)[0], (df_pol_F1 - df_exact)[0])

        handle = plot_handles[2]
        plot_update(handle, x, y, z, 
         f_exact[1],  f_ten_F[1],  f_pol_F[1],  f_pol_F1[1], 
        df_exact[1], df_ten_F[1], df_pol_F[1], df_pol_F1[1])

        handle = plot_handles[3]
        plot_update(handle, x, y, z, 
         f_exact[1], ( f_ten_F -  f_exact)[1], ( f_pol_F -  f_exact)[1], ( f_pol_F1 -  f_exact)[1], 
        df_exact[1], (df_ten_F - df_exact)[1], (df_pol_F - df_exact)[1], (df_pol_F1 - df_exact)[1])

        handle = plot_handles[4]
        plot_update(handle, x, y, z, 
         f_exact[2],  f_ten_F[2],  f_pol_F[2],  f_pol_F1[2], 
        df_exact[2], df_ten_F[2], df_pol_F[2], df_pol_F1[2])

        handle = plot_handles[5]
        plot_update(handle, x, y, z, 
         f_exact[2], ( f_ten_F -  f_exact)[2], ( f_pol_F -  f_exact)[2], ( f_pol_F1 -  f_exact)[2], 
        df_exact[2], (df_ten_F - df_exact)[2], (df_pol_F - df_exact)[2], (df_pol_F1 - df_exact)[2])



    elif func_form == FuncForm.TWO:

        # LHS: Original 2-form function, only x-component because all directions are identical.
        # RHS: 3-form div.
        # Figure 1: Actual value.
        # Figure 2: Difference.

        handle = plot_handles[0]
        plot_update(handle, x, y, z, 
         f_exact[0],  f_ten_F[0],  f_pol_F[0],  f_pol_F1[0], 
        df_exact   , df_ten_F   , df_pol_F   , df_pol_F1   )

        handle = plot_handles[1]
        plot_update(handle, x, y, z, 
         f_exact[0], ( f_ten_F -  f_exact)[0], ( f_pol_F -  f_exact)[0], ( f_pol_F1 -  f_exact)[0], 
        df_exact   , (df_ten_F - df_exact)   , (df_pol_F - df_exact)   , (df_pol_F1 - df_exact)   )

        handle = plot_handles[2]
        plot_update(handle, x, y, z, 
         f_exact[1],  f_ten_F[1],  f_pol_F[1],  f_pol_F1[1], 
        df_exact   , df_ten_F   , df_pol_F   , df_pol_F1   )

        handle = plot_handles[3]
        plot_update(handle, x, y, z, 
         f_exact[1], ( f_ten_F -  f_exact)[1], ( f_pol_F -  f_exact)[1], ( f_pol_F1 -  f_exact)[1], 
        df_exact   , (df_ten_F - df_exact)   , (df_pol_F - df_exact)   , (df_pol_F1 - df_exact)   )

        handle = plot_handles[4]
        plot_update(handle, x, y, z, 
         f_exact[2],  f_ten_F[2],  f_pol_F[2],  f_pol_F1[2], 
        df_exact   , df_ten_F   , df_pol_F   , df_pol_F1   )

        handle = plot_handles[5]
        plot_update(handle, x, y, z, 
         f_exact[2], ( f_ten_F -  f_exact)[2], ( f_pol_F -  f_exact)[2], ( f_pol_F1 -  f_exact)[2], 
        df_exact   , (df_ten_F - df_exact)   , (df_pol_F - df_exact)   , (df_pol_F1 - df_exact)   )



    else:

        raise NotImplementedError(f'FuncForm {func_form} not recognized.')



    return plot_handles



def plot_update(handle, eta1, eta2, eta3, 
f_exact, f_ten_F, f_pol_F, f_pol_F1, 
df_exact, df_ten_F, df_pol_F, df_pol_F1):

    import numpy             as np
    import matplotlib.pyplot as plt
    import matplotlib.tri    as tri

    # `handle` is a dict that looks like this:
    # handle == {
    #     'fig': fig,
    #     'laxes': laxes,
    #     'raxes': raxes,
    #     'limgs': limgs,
    #     'rimgs': rimgs,
    #     'lcbars': lcbars,
    #     'rcbars': rcbars,
    #     'metadata': metadata,
    # }
    fig = handle['fig']
    laxes = handle['laxes']
    raxes = handle['raxes']
    limgs = handle['limgs']
    rimgs = handle['rimgs']
    lcbars = handle['lcbars']
    rcbars = handle['rcbars']
    lpoles = handle['lpoles']
    rpoles = handle['rpoles']
    metadata = handle['metadata']
    func_form = metadata['func_form']
    origin = metadata['origin']
    levels = metadata['collevel']
    axlim = 0.01

    print(f'Updating plot {metadata["suptitle"]}')

    print(f'type(laxes[0]): {type(laxes[0])}')
    print(f'type(limgs[0]): {type(limgs[0])}')

    if metadata['comptype'] == Comparison.ORIG:
        print('Comparison type: Original values.')
        lmin = np.min([
            np.min(f_exact),
            np.min(f_ten_F),
            np.min(f_pol_F),
            np.min(f_pol_F1),
        ])
        lmax = np.max([
            np.max(f_exact),
            np.max(f_ten_F),
            np.max(f_pol_F),
            np.max(f_pol_F1),
        ])
        rmin = np.min([
            np.min(df_exact),
            np.min(df_ten_F),
            np.min(df_pol_F),
            np.min(df_pol_F1),
        ])
        rmax = np.max([
            np.max(df_exact),
            np.max(df_ten_F),
            np.max(df_pol_F),
            np.max(df_pol_F1),
        ])
        if metadata['anacolor']:
            print('Scale with analytical solution.')
            lmin = np.min(f_exact)
            lmax = np.max(f_exact)
            rmin = np.min(df_exact)
            rmax = np.max(df_exact)
    else:
        print('Comparison type: Errors.')
        lmin = np.min([
            np.min(f_ten_F),
            np.min(f_pol_F),
            np.min(f_pol_F1),
        ])
        lmax = np.max([
            np.max(f_ten_F),
            np.max(f_pol_F),
            np.max(f_pol_F1),
        ])
        rmin = np.min([
            np.min(df_ten_F),
            np.min(df_pol_F),
            np.min(df_pol_F1),
        ])
        rmax = np.max([
            np.max(df_ten_F),
            np.max(df_pol_F),
            np.max(df_pol_F1),
        ])

    if metadata['colscale'] != 1:
        lrange = lmax - lmin
        rrange = rmax - rmin
        lmid = (lmax + lmin) / 2
        rmid = (rmax + rmin) / 2
        lrange_new = lrange * metadata['colscale']
        rrange_new = rrange * metadata['colscale']
        lmin = lmid - lrange_new / 2
        lmax = lmid + lrange_new / 2
        rmin = rmid - rrange_new / 2
        rmax = rmid + rrange_new / 2

    if metadata['unlink']:
        lmin=None
        lmax=None
        rmin=None
        rmax=None

    print(f'lmin: {lmin}')
    print(f'lmax: {lmax}')
    print(f'rmin: {rmin}')
    print(f'rmax: {rmax}')

    l_is_3d = metadata['lplot'] in [PlotTypeLeft.CONTOUR3D, PlotTypeLeft.SCATTER, PlotTypeLeft.SURFACE, PlotTypeLeft.WIREFRAME]
    r_is_3d = metadata['rplot'] in [PlotTypeRight.CONTOUR3D, PlotTypeRight.QUIVER3D, PlotTypeRight.SURFACE]



    # ============================================================
    # Four original functions on the left.
    # ============================================================

    if metadata['lplot'] in [PlotTypeLeft.CONTOUR2D, PlotTypeLeft.CONTOUR3D]:
        for img in limgs:
            for c in img.collections:
                c.remove() # Remove only the contours, leave the rest intact.
        if metadata['comptype'] == Comparison.ORIG:
            limgs[0] = laxes[0].contourf(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
            limgs[0].set_clim(lmin, lmax)
        else:
            limgs[0] = laxes[0].contourf(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0])
        limgs[1] = laxes[1].contourf(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
        limgs[2] = laxes[2].contourf(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
        limgs[3] = laxes[3].contourf(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0], levels=np.linspace(lmin,lmax,levels), vmin=lmin, vmax=lmax)
        limgs[1].set_clim(lmin, lmax)
        limgs[2].set_clim(lmin, lmax)
        limgs[3].set_clim(lmin, lmax)
    elif metadata['lplot'] == PlotTypeLeft.SURFACE:
        # for img in limgs:
        #     img.remove()
        for ax, img in zip(laxes, limgs):
            ax.collections.remove(img)
        limgs[0] = laxes[0].plot_surface(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
        limgs[1] = laxes[1].plot_surface(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
        limgs[2] = laxes[2].plot_surface(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
        limgs[3] = laxes[3].plot_surface(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0], vmin=lmin, vmax=lmax, cmap=plt.cm.viridis)
    elif metadata['lplot'] == PlotTypeLeft.WIREFRAME:
        for ax, img in zip(laxes, limgs):
            ax.collections.remove(img)
        # for img in limgs:
        #     img.remove()
        limgs[0] = laxes[0].plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0])
        limgs[1] = laxes[1].plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0])
        limgs[2] = laxes[2].plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0])
        limgs[3] = laxes[3].plot_wireframe(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0])
    elif metadata['lplot'] == PlotTypeLeft.SCATTER:
        for ax, img in zip(laxes, limgs):
            ax.collections.remove(img)
        # limgs[0]._offsets3d = (eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0])
        # Use Scatter plot for 2D data:
        limgs[0] = laxes[0].scatter(eta1[:,:,0], eta2[:,:,0], f_exact[:,:,0], c=f_exact[:,:,0], cmap=plt.cm.viridis, marker='.')
        limgs[1] = laxes[1].scatter(eta1[:,:,0], eta2[:,:,0], f_ten_F[:,:,0], c=f_ten_F[:,:,0], cmap=plt.cm.viridis, marker='.')
        limgs[2] = laxes[2].scatter(eta1[:,:,0], eta2[:,:,0], f_pol_F[:,:,0], c=f_pol_F[:,:,0], cmap=plt.cm.viridis, marker='.')
        limgs[3] = laxes[3].scatter(eta1[:,:,0], eta2[:,:,0], f_pol_F1[:,:,0], c=f_pol_F1[:,:,0], cmap=plt.cm.viridis, marker='.')
        # When the function is dependent on eta3 as well (3D).
        # limgs[0] = laxes[1].scatter(eta1, eta2, eta3, c=f_exact, cmap=plt.cm.viridis, marker='.')
        # limgs[1] = laxes[2].scatter(eta1, eta2, eta3, c=f_ten_F, cmap=plt.cm.viridis, marker='.')
        # limgs[2] = laxes[3].scatter(eta1, eta2, eta3, c=f_pol_F, cmap=plt.cm.viridis, marker='.')
        # limgs[3] = laxes[4].scatter(eta1, eta2, eta3, c=f_pol_F1, cmap=plt.cm.viridis, marker='.')



    # TODO: Bug: Colorbar only updates with scatter plot but not the others (surface, contourf).
    if metadata['lplot'] != PlotTypeLeft.WIREFRAME:
        try:
            lcbars[0].update_normal(limgs[0])
            lcbars[1].update_normal(limgs[1])
            lcbars[2].update_normal(limgs[2])
            lcbars[3].update_normal(limgs[3])
        except Exception as e:
            print(e)



    # ============================================================
    # Four derivatives on the right.
    # ============================================================

    if func_form == FuncForm.ZERO:

        # LHS: Original 0-form function.
        # RHS: 1-form gradient, only x component, because y is identical and z is zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.
        df_mag = np.sqrt(2 * df_ten_F**2)

    elif func_form == FuncForm.ONE:

        # LHS: Original 1-form function, only x-component because all directions are identical.
        # RHS: 2-form curl, only z component because x and y are zero.
        # Figure 1: Actual value.
        # Figure 2: Difference.
        df_mag = np.sqrt(3 * df_ten_F**2)

    elif func_form == FuncForm.TWO:

        # LHS: Original 2-form function, only x-component because all directions are identical.
        # RHS: 3-form div.
        # Figure 1: Actual value.
        # Figure 2: Difference.
        df_mag = np.sqrt(3 * df_ten_F**2)

    else:

        raise NotImplementedError(f'FuncForm {func_form} not recognized.')

    # df_mag = np.sqrt(df_ten_F[0]**2 + df_ten_F[1]**2 + df_ten_F[2]**2)
    print('df_mag.shape', df_mag.shape)
    print('Max |df|:', np.max(df_mag))
    print('Min |df|:', np.min(df_mag))

    # Flatten and normalize.
    if r_is_3d:
        c = (df_mag.ravel() - df_mag.min()) / df_mag.ptp()
    else:
        c = (df_mag[:,:,0].ravel() - df_mag[:,:,0].min()) / df_mag[:,:,0].ptp()
    # print('c.shape', c.shape)
    # Repeat for each body line and two head lines in a quiver.
    c = np.concatenate((c, np.repeat(c, 2)))
    # print('c.shape', c.shape)
    # Colormap.
    c = plt.cm.viridis(c)
    # print('c.shape', c.shape)



    if metadata['rplot'] in [PlotTypeRight.CONTOUR2D, PlotTypeRight.CONTOUR3D]:
        for img in rimgs:
            for c in img.collections:
                c.remove() # Remove only the contours, leave the rest intact.
        if metadata['comptype'] == Comparison.ORIG:
            rimgs[0] = raxes[0].contourf(eta1[:,:,0], eta2[:,:,0], df_exact[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
            rimgs[0].set_clim(rmin, rmax)
        else:
            rimgs[0] = raxes[0].contourf(eta1[:,:,0], eta2[:,:,0], df_exact[:,:,0])
        rimgs[1] = raxes[1].contourf(eta1[:,:,0], eta2[:,:,0], df_ten_F[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
        rimgs[2] = raxes[2].contourf(eta1[:,:,0], eta2[:,:,0], df_pol_F[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
        rimgs[3] = raxes[3].contourf(eta1[:,:,0], eta2[:,:,0], df_pol_F1[:,:,0], levels=np.linspace(rmin,rmax,levels), vmin=rmin, vmax=rmax)
        rimgs[1].set_clim(rmin, rmax)
        rimgs[2].set_clim(rmin, rmax)
        rimgs[3].set_clim(rmin, rmax)
    elif metadata['rplot'] == PlotTypeRight.SURFACE:
        # for img in rimgs:
        #     img.remove()
        for ax, img in zip(raxes, rimgs):
            ax.collections.remove(img)
        rimgs[0] = raxes[0].plot_surface(eta1[:,:,0], eta2[:,:,0], df_exact[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
        rimgs[1] = raxes[1].plot_surface(eta1[:,:,0], eta2[:,:,0], df_ten_F[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
        rimgs[2] = raxes[2].plot_surface(eta1[:,:,0], eta2[:,:,0], df_pol_F[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
        rimgs[3] = raxes[3].plot_surface(eta1[:,:,0], eta2[:,:,0], df_pol_F1[:,:,0], vmin=rmin, vmax=rmax, cmap=plt.cm.viridis)
    elif metadata['rplot'] == PlotTypeRight.QUIVER2D:
        for img in rimgs:
            img.remove()
        scale = 500
        metadata['rzlabel'] = metadata['rzlabel'] + f' Scale: {1/scale}'
        rimgs[0] = raxes[0].quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_exact[0,:,:,0].flatten(), df_exact[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimgs[1] = raxes[1].quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_ten_F[0,:,:,0].flatten(), df_ten_F[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimgs[2] = raxes[2].quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_pol_F[0,:,:,0].flatten(), df_pol_F[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimgs[3] = raxes[3].quiver(eta1[:,:,0].flatten(), eta2[:,:,0].flatten(), df_pol_F1[0,:,:,0].flatten(), df_pol_F1[1,:,:,0].flatten(), color=c, cmap=plt.cm.viridis, scale=scale)
        rimgs[0].set_array(df_mag[:,:,0].flatten())
        rimgs[1].set_array(df_mag[:,:,0].flatten())
        rimgs[2].set_array(df_mag[:,:,0].flatten())
        rimgs[3].set_array(df_mag[:,:,0].flatten())
        rimgs[0].set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
        rimgs[1].set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
        rimgs[2].set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
        rimgs[3].set_clim(np.min(df_mag[:,:,0]), np.max(df_mag[:,:,0]))
    elif metadata['rplot'] == PlotTypeRight.QUIVER3D:
        for img in rimgs:
            img.remove()
        scale = 500
        metadata['rzlabel'] = metadata['rzlabel'] + f' Scale: {1/scale}'
        rimgs[0] = raxes[0].quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_exact[0].flatten(), df_exact[1].flatten(), df_exact[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimgs[1] = raxes[1].quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_ten_F[0].flatten(), df_ten_F[1].flatten(), df_ten_F[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimgs[2] = raxes[2].quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_pol_F[0].flatten(), df_pol_F[1].flatten(), df_pol_F[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimgs[3] = raxes[3].quiver(eta1.flatten(), eta2.flatten(), eta3.flatten(), df_pol_F1[0].flatten(), df_pol_F1[1].flatten(), df_pol_F1[2].flatten(), color=c, cmap=plt.cm.viridis, length=1/scale)
        rimgs[0].set_array(df_mag.flatten())
        rimgs[1].set_array(df_mag.flatten())
        rimgs[2].set_array(df_mag.flatten())
        rimgs[3].set_array(df_mag.flatten())
        rimgs[0].set_clim(np.min(df_mag), np.max(df_mag))
        rimgs[1].set_clim(np.min(df_mag), np.max(df_mag))
        rimgs[2].set_clim(np.min(df_mag), np.max(df_mag))
        rimgs[3].set_clim(np.min(df_mag), np.max(df_mag))



    # TODO: Bug: Colorbar only updates with scatter plot but not the others (surface, contourf).
    try:
        rcbars[0].update_normal(rimgs[0])
        rcbars[1].update_normal(rimgs[1])
        rcbars[2].update_normal(rimgs[2])
        rcbars[3].update_normal(rimgs[3])
    except Exception as e:
        print(e)



    # Mark position on mapping pole.
    for idx, (ax, pole) in enumerate(zip(laxes, lpoles)):
        ax.collections.remove(pole)
        ax.set_xlim(origin[0] - axlim, origin[0] + axlim)
        ax.set_ylim(origin[1] - axlim, origin[1] + axlim)
        if l_is_3d:
            lpoles[idx] = ax.scatter(origin[0], origin[1], origin[2], marker='o', edgecolors='red', facecolors='none', label='Pole')
        else:
            lpoles[idx] = ax.scatter(origin[0], origin[1], 100,       marker='o', edgecolors='red', facecolors='none', label='Pole')
    for idx, (ax, pole) in enumerate(zip(raxes, rpoles)):
        ax.collections.remove(pole)
        ax.set_xlim(origin[0] - axlim, origin[0] + axlim)
        ax.set_ylim(origin[1] - axlim, origin[1] + axlim)
        if r_is_3d:
            rpoles[idx] = ax.scatter(origin[0], origin[1], origin[2], marker='o', edgecolors='red', facecolors='none', label='Pole')
        else:
            rpoles[idx] = ax.scatter(origin[0], origin[1], 100,       marker='o', edgecolors='red', facecolors='none', label='Pole')



    # fig.canvas.draw()
    # fig.canvas.flush_events()

    fig.canvas.draw_idle()



def plot_controls(case, case_args, func_test, func_form, plot_handles):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    references = []



    if func_test == FuncTest.GAUSSIAN:

        # Plot settings.
        row = 3
        col = 2
        dpi = 100
        width, height = (640 / dpi, 320 / dpi)
        fig = plt.figure(figsize=(width,height), dpi=dpi)#, constrained_layout=True)
        gs  = fig.add_gridspec(row, col)#, width_ratios=[1,1]*2)
        fig.canvas.manager.set_window_title('Change Plot Parameters (Gaussian)')
        fig.suptitle('Change Plot Parameters (Gaussian)')

        axis_color = 'lightgoldenrodyellow'

        # Define an axis and draw a slider.
        sd_x = 0.1
        sd_y = 0.05
        mu_x = sd_x
        mu_y = sd_y
        # mu_x_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
        # mu_y_slider_ax = fig.add_axes([0.25, 0.10, 0.65, 0.03], facecolor=axis_color)
        mu_x_slider_ax = fig.add_subplot(gs[0, 0], facecolor=axis_color)
        mu_y_slider_ax = fig.add_subplot(gs[0, 1], facecolor=axis_color)
        sd_x_slider_ax = fig.add_subplot(gs[1, 0], facecolor=axis_color)
        sd_y_slider_ax = fig.add_subplot(gs[1, 1], facecolor=axis_color)
        mu_x_slider = Slider(mu_x_slider_ax, '$\mu_x$', -2.0, 2.0, valinit=mu_x, valfmt='%+.3f')
        mu_y_slider = Slider(mu_y_slider_ax, '$\mu_y$', -2.0, 2.0, valinit=mu_y, valfmt='%+.3f')
        sd_x_slider = Slider(sd_x_slider_ax, '$\sigma_x$', 0.001, 2.0, valinit=sd_x, valfmt='%+.3f')
        sd_y_slider = Slider(sd_y_slider_ax, '$\sigma_y$', 0.001, 2.0, valinit=sd_y, valfmt='%+.3f')
        references.append(mu_x_slider)
        references.append(mu_y_slider)
        references.append(sd_x_slider)
        references.append(sd_y_slider)

        # Define a listener for modifying the line when any slider's value changes.
        def sliders_on_changed(val):
            print(f'Updated {val}')
            func, dfdx, dfdy = generate_test_function(func_test, params={
                'mu_x': mu_x_slider.val + 10,
                'mu_y': mu_y_slider.val,
                'sd_x': sd_x_slider.val,
                'sd_y': sd_y_slider.val,
            })
            func_3d, curl_3d, div_3d = func_3d_wrapper(func, dfdx, dfdy)
            if func_form == FuncForm.ZERO:
                case_args[6] = func
                case_args[7] = dfdx
                case_args[8] = dfdy
            else:
                case_args[6] = func_3d
                case_args[7] = curl_3d
                case_args[8] = div_3d
            plot_data = case(*case_args)
            plot_updater(plot_handles, *plot_data)
            fig.canvas.draw_idle()
        for slider in references:
            slider.on_changed(sliders_on_changed)

        # Add a button for resetting the parameters.
        reset_button_ax = fig.add_subplot(gs[2, 0:2])
        reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        references.append(reset_button)

        def reset_button_on_clicked(mouse_event):
            print('Reset!')
            mu_x_slider.reset()
            mu_y_slider.reset()
            sd_x_slider.reset()
            sd_y_slider.reset()
        reset_button.on_clicked(reset_button_on_clicked)



    elif func_test == FuncTest.GAUSSIANCOSINE:

        # Plot settings.
        row = 5
        col = 2
        dpi = 100
        width, height = (640 / dpi, 480 / dpi)
        fig = plt.figure(figsize=(width,height), dpi=dpi)#, constrained_layout=True)
        gs  = fig.add_gridspec(row, col)#, width_ratios=[1,1]*2)
        fig.canvas.manager.set_window_title('Change Plot Parameters (Gaussian-Cosine)')
        fig.suptitle('Change Plot Parameters (Gaussian-Cosine)')

        axis_color = 'lightgoldenrodyellow'

        # Define an axis and draw a slider.
        sd_x = 0.1
        sd_y = 0.05
        mu_x = sd_x
        mu_y = sd_y
        v_x = 1
        v_y = 1
        om_x = 2 * np.pi * v_x / (4 * sd_x) # Steepest gradient.
        om_y = 2 * np.pi * v_y / (4 * sd_y) # Steepest gradient.
        ph_x = 2 * np.pi * (mu_x)
        ph_y = 2 * np.pi * (mu_y)
        mu_x_slider_ax = fig.add_subplot(gs[0, 0], facecolor=axis_color)
        sd_x_slider_ax = fig.add_subplot(gs[1, 0], facecolor=axis_color)
        om_x_slider_ax = fig.add_subplot(gs[2, 0], facecolor=axis_color)
        ph_x_slider_ax = fig.add_subplot(gs[3, 0], facecolor=axis_color)
        mu_y_slider_ax = fig.add_subplot(gs[0, 1], facecolor=axis_color)
        sd_y_slider_ax = fig.add_subplot(gs[1, 1], facecolor=axis_color)
        om_y_slider_ax = fig.add_subplot(gs[2, 1], facecolor=axis_color)
        ph_y_slider_ax = fig.add_subplot(gs[3, 1], facecolor=axis_color)
        mu_x_slider = Slider(mu_x_slider_ax, '$\mu_x$', -2.0, 2.0, valinit=mu_x, valfmt='%+.3f')
        mu_y_slider = Slider(mu_y_slider_ax, '$\mu_y$', -2.0, 2.0, valinit=mu_y, valfmt='%+.3f')
        sd_x_slider = Slider(sd_x_slider_ax, '$\sigma_x$', 0.001, 2.0, valinit=sd_x, valfmt='%+.3f')
        sd_y_slider = Slider(sd_y_slider_ax, '$\sigma_y$', 0.001, 2.0, valinit=sd_y, valfmt='%+.3f')
        om_x_slider = Slider(om_x_slider_ax, '$\omega_x$', -100.0, 100.0, valinit=om_x, valfmt='%+.3f')
        om_y_slider = Slider(om_y_slider_ax, '$\omega_y$', -100.0, 100.0, valinit=om_y, valfmt='%+.3f')
        ph_x_slider = Slider(ph_x_slider_ax, '$\\varphi_x$', -10.0, 10.0, valinit=ph_x, valfmt='%+.3f')
        ph_y_slider = Slider(ph_y_slider_ax, '$\\varphi_y$', -10.0, 10.0, valinit=ph_y, valfmt='%+.3f')
        references.append(mu_x_slider)
        references.append(mu_y_slider)
        references.append(sd_x_slider)
        references.append(sd_y_slider)
        references.append(om_x_slider)
        references.append(om_y_slider)
        references.append(ph_x_slider)
        references.append(ph_y_slider)

        # Define a listener for modifying the line when any slider's value changes.
        def sliders_on_changed(val):
            print(f'Updated {val}')
            func, dfdx, dfdy = generate_test_function(func_test, params={
                'mu_x': mu_x_slider.val + 10,
                'mu_y': mu_y_slider.val,
                'sd_x': sd_x_slider.val,
                'sd_y': sd_y_slider.val,
                'om_x': om_x_slider.val,
                'om_y': om_y_slider.val,
                'ph_x': ph_x_slider.val + 10 * 2 * np.pi,
                'ph_y': ph_y_slider.val,
            })
            func_3d, curl_3d, div_3d = func_3d_wrapper(func, dfdx, dfdy)
            if func_form == FuncForm.ZERO:
                case_args[6] = func
                case_args[7] = dfdx
                case_args[8] = dfdy
            else:
                case_args[6] = func_3d
                case_args[7] = curl_3d
                case_args[8] = div_3d
            plot_data = case(*case_args)
            plot_updater(plot_handles, *plot_data)
            fig.canvas.draw_idle()
        for slider in references:
            slider.on_changed(sliders_on_changed)

        # Add a button for resetting the parameters.
        reset_button_ax = fig.add_subplot(gs[4, 0:2])
        reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        references.append(reset_button)

        def reset_button_on_clicked(mouse_event):
            print('Reset!')
            mu_x_slider.reset()
            mu_y_slider.reset()
            sd_x_slider.reset()
            sd_y_slider.reset()
            om_x_slider.reset()
            om_y_slider.reset()
            ph_x_slider.reset()
            ph_y_slider.reset()
        reset_button.on_clicked(reset_button_on_clicked)



    elif func_test == FuncTest.SINEX:

        # Plot settings.
        row = 3
        col = 1
        dpi = 100
        width, height = (640 / dpi, 320 / dpi)
        fig = plt.figure(figsize=(width,height), dpi=dpi)#, constrained_layout=True)
        gs  = fig.add_gridspec(row, col)#, width_ratios=[1,1]*2)
        fig.canvas.manager.set_window_title('Change Plot Parameters (Sine(x))')
        fig.suptitle('Change Plot Parameters (Sine(x))')

        axis_color = 'lightgoldenrodyellow'

        # Define an axis and draw a slider.
        om_x = 2 * np.pi
        ph_x = 0
        om_x_slider_ax = fig.add_subplot(gs[0, 0], facecolor=axis_color)
        ph_x_slider_ax = fig.add_subplot(gs[1, 0], facecolor=axis_color)
        om_x_slider = Slider(om_x_slider_ax, '$\omega_x$', -100.0, 100.0, valinit=om_x, valfmt='%+.3f')
        ph_x_slider = Slider(ph_x_slider_ax, '$\\varphi_x$', -10.0, 10.0, valinit=ph_x, valfmt='%+.3f')
        references.append(om_x_slider)
        references.append(ph_x_slider)

        # Define a listener for modifying the line when any slider's value changes.
        def sliders_on_changed(val):
            print(f'Updated {val}')
            func, dfdx, dfdy = generate_test_function(func_test, params={
                'om_x': om_x_slider.val,
                'ph_x': ph_x_slider.val + 10 * 2 * np.pi,
            })
            func_3d, curl_3d, div_3d = func_3d_wrapper(func, dfdx, dfdy)
            if func_form == FuncForm.ZERO:
                case_args[6] = func
                case_args[7] = dfdx
                case_args[8] = dfdy
            else:
                case_args[6] = func_3d
                case_args[7] = curl_3d
                case_args[8] = div_3d
            plot_data = case(*case_args)
            plot_updater(plot_handles, *plot_data)
            fig.canvas.draw_idle()
        for slider in references:
            slider.on_changed(sliders_on_changed)

        # Add a button for resetting the parameters.
        reset_button_ax = fig.add_subplot(gs[2, 0])
        reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        references.append(reset_button)

        def reset_button_on_clicked(mouse_event):
            print('Reset!')
            om_x_slider.reset()
            ph_x_slider.reset()
        reset_button.on_clicked(reset_button_on_clicked)




    elif func_test == FuncTest.SIGMOID:

        # Plot settings.
        row = 3
        col = 2
        dpi = 100
        width, height = (640 / dpi, 320 / dpi)
        fig = plt.figure(figsize=(width,height), dpi=dpi)#, constrained_layout=True)
        gs  = fig.add_gridspec(row, col)#, width_ratios=[1,1]*2)
        fig.canvas.manager.set_window_title('Change Plot Parameters (Sigmoid/Logistic)')
        fig.suptitle('Change Plot Parameters (Sigmoid/Logistic)')

        axis_color = 'lightgoldenrodyellow'

        # Define an axis and draw a slider.
        mu_x = 0.0
        mu_y = 0.0
        k_x = 100
        k_y = 100
        mu_x_slider_ax = fig.add_subplot(gs[0, 0], facecolor=axis_color)
        mu_y_slider_ax = fig.add_subplot(gs[0, 1], facecolor=axis_color)
        sd_x_slider_ax = fig.add_subplot(gs[1, 0], facecolor=axis_color)
        sd_y_slider_ax = fig.add_subplot(gs[1, 1], facecolor=axis_color)
        mu_x_slider = Slider(mu_x_slider_ax, '$\mu_x$', -2.0, 2.0, valinit=mu_x, valfmt='%+.3f')
        mu_y_slider = Slider(mu_y_slider_ax, '$\mu_y$', -2.0, 2.0, valinit=mu_y, valfmt='%+.3f')
        k_x_slider = Slider(sd_x_slider_ax, '$k_x$', 0.001, 1000, valinit=k_x, valfmt='%+.3f')
        k_y_slider = Slider(sd_y_slider_ax, '$k_y$', 0.001, 1000, valinit=k_y, valfmt='%+.3f')
        references.append(mu_x_slider)
        references.append(mu_y_slider)
        references.append(k_x_slider)
        references.append(k_y_slider)

        # Define a listener for modifying the line when any slider's value changes.
        def sliders_on_changed(val):
            print(f'Updated {val}')
            func, dfdx, dfdy = generate_test_function(func_test, params={
                'mu_x': mu_x_slider.val + 10,
                'mu_y': mu_y_slider.val,
                'k_x': k_x_slider.val,
                'k_y': k_y_slider.val,
            })
            func_3d, curl_3d, div_3d = func_3d_wrapper(func, dfdx, dfdy)
            if func_form == FuncForm.ZERO:
                case_args[6] = func
                case_args[7] = dfdx
                case_args[8] = dfdy
            else:
                case_args[6] = func_3d
                case_args[7] = curl_3d
                case_args[8] = div_3d
            plot_data = case(*case_args)
            plot_updater(plot_handles, *plot_data)
            fig.canvas.draw_idle()
        for slider in references:
            slider.on_changed(sliders_on_changed)

        # Add a button for resetting the parameters.
        reset_button_ax = fig.add_subplot(gs[2, 0:2])
        reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        references.append(reset_button)

        def reset_button_on_clicked(mouse_event):
            print('Reset!')
            mu_x_slider.reset()
            mu_y_slider.reset()
            k_x_slider.reset()
            k_y_slider.reset()
        reset_button.on_clicked(reset_button_on_clicked)



    elif func_test == FuncTest.LINEARX:

        # Plot settings.
        row = 3
        col = 2
        dpi = 100
        width, height = (640 / dpi, 320 / dpi)
        fig = plt.figure(figsize=(width,height), dpi=dpi)#, constrained_layout=True)
        gs  = fig.add_gridspec(row, col)#, width_ratios=[1,1]*2)
        fig.canvas.manager.set_window_title('No Configurable Plot Parameters (Linear X)')
        fig.suptitle('No Configurable Plot Parameters (Linear X)')



    else:

        raise NotImplementedError(f'Test case {func_test} not implemented.')



    # print('Before `tight_layout()`   | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.tight_layout()
    # print('After `tight_layout()`    | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.subplots_adjust(hspace=fig.subplotpars.hspace * 0.5, wspace=fig.subplotpars.wspace * 1.2)
    # print('After `subplots_adjust()` | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))

    return references



def plot_spl_config(case, case_args, func_test, func_form, plot_handles):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    references = []

    Nel = case_args[0]
    p = case_args[1]

    # Plot settings.
    row = 4
    col = 2
    dpi = 100
    width, height = (640 / dpi, 320 / dpi)
    fig = plt.figure(figsize=(width,height), dpi=dpi)#, constrained_layout=True)
    gs  = fig.add_gridspec(row, col)#, width_ratios=[1,1]*2)
    fig.canvas.manager.set_window_title('Change Spline Configurations')
    fig.suptitle('Change Spline Configurations')

    axis_color = 'lightgoldenrodyellow'

    Nel1_slider_ax = fig.add_subplot(gs[0, 0], facecolor=axis_color)
    Nel2_slider_ax = fig.add_subplot(gs[1, 0], facecolor=axis_color)
    Nel3_slider_ax = fig.add_subplot(gs[2, 0], facecolor=axis_color)
    deg1_slider_ax = fig.add_subplot(gs[0, 1], facecolor=axis_color)
    deg2_slider_ax = fig.add_subplot(gs[1, 1], facecolor=axis_color)
    deg3_slider_ax = fig.add_subplot(gs[2, 1], facecolor=axis_color)
    Nel1_slider = Slider(Nel1_slider_ax, '$Nel_s$'      , 3, 100, valinit=Nel[0], valfmt='%d', valstep=1)
    Nel2_slider = Slider(Nel2_slider_ax, '$Nel_\\theta$', 3, 100, valinit=Nel[1], valfmt='%d', valstep=3)
    Nel3_slider = Slider(Nel3_slider_ax, '$Nel_\zeta$'  , 3, 100, valinit=Nel[2], valfmt='%d', valstep=1)
    deg1_slider = Slider(deg1_slider_ax, '$p_s$'      , 2, 6, valinit=p[0], valfmt='%d', valstep=1)
    deg2_slider = Slider(deg2_slider_ax, '$p_\\theta$', 2, 6, valinit=p[1], valfmt='%d', valstep=1)
    deg3_slider = Slider(deg3_slider_ax, '$p_\zeta$'  , 2, 6, valinit=p[2], valfmt='%d', valstep=1)
    references.append(Nel1_slider)
    references.append(Nel2_slider)
    references.append(Nel3_slider)
    references.append(deg1_slider)
    references.append(deg2_slider)
    references.append(deg3_slider)

    # Define a listener for modifying the line when any slider's value changes.
    def sliders_on_changed(val):
        print(f'Updated {val}')
        # Nel
        case_args[0][0] = int(Nel1_slider.val)
        case_args[0][1] = int(Nel2_slider.val)
        case_args[0][2] = int(Nel3_slider.val)
        # p
        case_args[1][0] = int(deg1_slider.val)
        case_args[1][1] = int(deg2_slider.val)
        case_args[1][2] = int(deg3_slider.val)
        Nel = case_args[0]
        p = case_args[1]
        # TODO: Update DOMAIN_F in case_args, if MapType.SPLINE is used.
        plot_data = case(*case_args)
        plot_updater(plot_handles, *plot_data)
        update_fig_suptitle(plot_handles, Nel, p)
        fig.canvas.draw_idle()
    for slider in references:
        slider.on_changed(sliders_on_changed)

    # Add a button for resetting the parameters.
    reset_button_ax = fig.add_subplot(gs[3, 0:2])
    reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
    references.append(reset_button)

    def reset_button_on_clicked(mouse_event):
        print('Reset!')
        # Reset all except the reset button itself.
        for slider in references[:-1]:
            slider.reset()
    reset_button.on_clicked(reset_button_on_clicked)



    # print('Before `tight_layout()`   | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.tight_layout()
    # print('After `tight_layout()`    | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))
    fig.subplots_adjust(hspace=fig.subplotpars.hspace * 0.5, wspace=fig.subplotpars.wspace * 1.2)
    # print('After `subplots_adjust()` | hspace: {}; wspace: {}.'.format(fig.subplotpars.hspace, fig.subplotpars.wspace))

    return references



def update_fig_suptitle(plot_handles, Nel, p):
    for handle in plot_handles:
        fig = handle['fig']
        title_x1 = fig._suptitle.get_text()
        title_x2 = title_x1.split('\n')[0]
        title_x3 = title_x2 + f'\nNel: {Nel}   p: {p}'
        print(f'Original suptitle: {title_x1}')
        print(f'First line of suptitle: {title_x2}')
        fig.canvas.manager.set_window_title(title_x3)
        fig.suptitle(title_x3, y=0.98)
        print(f'Updated suptitle: {title_x3}')



if __name__ == "__main__":
    test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.CIRCLESCALED, plot_0form=True, plot_1form=False, plot_2form=False)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.CIRCLESCALED, plot_0form=False, plot_1form=True, plot_2form=False)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.CIRCLEIDENTICAL, plot_0form=False, plot_1form=False, plot_2form=True)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.CIRCLESCALED, plot_0form=False, plot_1form=False, plot_2form=True)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.CIRCLESHIFTED, plot_0form=False, plot_1form=False, plot_2form=True)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.ELLIPSE, plot_0form=False, plot_1form=False, plot_2form=True)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.ELLIPSEROTATED, plot_0form=False, plot_1form=False, plot_2form=True)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.SOLOVIEV, plot_0form=False, plot_1form=False, plot_2form=True)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.SOLOVIEVSQRT, plot_0form=False, plot_1form=False, plot_2form=True)
    # test_polar_splines_3D(func_test=FuncTest.GAUSSIAN, map_type=MapType.SPLINE, plot_0form=False, plot_1form=False, plot_2form=True)
