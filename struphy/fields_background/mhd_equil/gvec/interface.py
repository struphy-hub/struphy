import numpy as np

import os

basedir = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, os.path.join(basedir, '..'))

# Import necessary struphy.modules.
from struphy.geometry.domain_3d import Domain

from gvec_to_python import GVEC, Form, Variable
from gvec_to_python.reader.gvec_reader import GVEC_Reader

import h5py
import tempfile



class AttributeDict(dict):
    """Call dict keys using dot notation.

    Used to imitate a dummy `Equilibrium_mhd_physical` class."""

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value



class GVECtoSTRUPHY:
    """Provide an interface to `gvec_to_python` package, which presents a GVEC MHD equilibrium."""

    def __init__(self, params, DOMAIN, TENSOR_SPACE, SOURCE_DOMAIN=None):

        self.params = params # Only ['mhd_equilibrium']['params_gvec'] key.

        self.temp_dir = tempfile.TemporaryDirectory(prefix='STRUPHY-')
        print(f'Created temp directory at: {self.temp_dir.name}')

        # Check if configuration file has all the necessary input parameters.
        assert params is not None,      'GVEC equilibrium: Input params should not be None.'
        assert 'filepath' in params,    'GVEC equilibrium: Key "filepath" not in params["mhd_equilibrium"]["params_gvec"].'
        assert 'filename' in params,    'GVEC equilibrium: Key "filename" not in params["mhd_equilibrium"]["params_gvec"].'
        assert params['filename'].endswith('.dat') or params['filename'].endswith('.json'), (
                                        'GVEC equilibrium: GVEC input file type not supported. Only .dat and .json.')

        # Geometric parameters.
        self.DOMAIN = DOMAIN # Domain object (spline) defining the GVEC mapping.
        self.SOURCE_DOMAIN = SOURCE_DOMAIN # Slice of the cuboid logical cube.
        if SOURCE_DOMAIN is None:
            # Default (s,u,v) = (eta1, eta2, eta3). LHS is coordinates in GVEC, RHS is STRUPHY.
            # bounds = [0,1,0,1,0,1]
            bounds = {'b1': 0., 'e1': 1., 'b2': 0., 'e2': 1., 'b3': 0., 'e3': 1.}
            self.SOURCE_DOMAIN = Domain('cuboid', params_map=bounds)

        # Create 3D projector. It's not automatic.
        for space in TENSOR_SPACE.spaces:
            if not hasattr(space, 'projectors'):
                space.set_projectors() # def set_projectors(self, nq=6):
        if not hasattr(TENSOR_SPACE, 'projectors'):
            TENSOR_SPACE.set_projectors() # def set_projectors(self, which='tensor'). Use 'general' for polar splines.

        self.TENSOR_SPACE = TENSOR_SPACE
        self.PROJ_3D = TENSOR_SPACE.projectors



        # ============================================================
        # Convert GVEC .dat output to .json.
        # ============================================================

        if params['filename'].endswith('.dat'):

            read_filepath = params['filepath']
            read_filename = params['filename']
            gvec_filepath = self.temp_dir.name
            gvec_filename = params['filename'][:-4] + '.json'
            reader = GVEC_Reader(read_filepath, read_filename, gvec_filepath, gvec_filename, with_spl_coef=True)

        elif params['filename'].endswith('.json'):

            gvec_filepath = params['filepath']
            gvec_filename = params['filename'][:-4] + '.json'



        # ============================================================
        # Load GVEC mapping.
        # ============================================================

        self.gvec = GVEC(gvec_filepath, gvec_filename)



        # ============================================================
        # Imitate behavior of `Equilibrium_mhd_physical`.
        # ============================================================

        # Magnitude of magnetic field "b0_eq" or "b_eq" are not implemented.
        self.MHD = AttributeDict({
            'p_eq': self.p_eq,
            # 'r_eq': MHD.r_eq,
            'r_eq': self.r_eq,
            'b_eq_x': self.b_eq_x,
            'b_eq_y': self.b_eq_y,
            'b_eq_z': self.b_eq_z,
            'j_eq_x': self.j_eq_x,
            'j_eq_y': self.j_eq_y,
            'j_eq_z': self.j_eq_z,
        })

        # Dummy density profile
        self.r1 = 4. # params['r1']
        self.r2 = 3. # params['r2']
        self.ra = 0. # params['ra']



    # ===============================================================
    #                      Map source domain
    # ===============================================================

    def s(self, eta1, eta2, eta3):
        return self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'x')

    def u(self, eta1, eta2, eta3):
        return self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'y')

    def v(self, eta1, eta2, eta3):
        return self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'z')

    def eta123_to_suv(self, eta1, eta2, eta3):
        return self.s(eta1, eta2, eta3), self.u(eta1, eta2, eta3), self.v(eta1, eta2, eta3)

    def source_det(self, eta1, eta2, eta3):
        return self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'det_df')

    def source_df(self, eta1, eta2, eta3):
        df = np.array(
            (
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_11'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_12'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_13')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_21'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_22'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_23')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_31'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_32'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_33')),
            )
        )
        return self.swap_J_axes(df)

    def source_df_inv(self, eta1, eta2, eta3):
        df_inv = np.array(
            (
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_11'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_12'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_13')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_21'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_22'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_23')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_31'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_32'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_inv_33')),
            )
        )
        return self.swap_J_axes(df_inv)

    def source_G(self, eta1, eta2, eta3):
        G = np.array(
            (
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_11'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_12'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_13')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_21'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_22'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_23')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_31'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_32'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_33')),
            )
        )
        return self.swap_J_axes(G)

    def source_G_inv(self, eta1, eta2, eta3):
        G_inv = np.array(
            (
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_11'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_12'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_13')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_21'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_22'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_23')),
                (self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_31'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_32'), self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'g_inv_33')),
            )
        )
        return self.swap_J_axes(G_inv)

    def swap_J_axes(self, J):
        """Swap axes of a batch of Jacobians, such that it is compatible with numpy's batch processing.

        When the inputs are 1D arrays or 3D arrays of meshgrids, the Jacobian dimensions by default will be (3, 3, eta1, eta2, eta3).
        However, all of numpy's matrix operations expect the 3x3 part to be the last two dimensions, i.e. (eta1, eta2, eta3, 3, 3).
        This function will first check if the Jacobian has dimensions > 2 (there is no point swapping axis of a scalar input).
        Then it will check if the 3x3 portion is at the beginning of the `shape` tuple.
        If the conditions are met, it will move the first two axes of 5D Jacobian to the last two, such that it is compatible with numpy's batch processing.

        Parameters
        ----------
        J : numpy.ndarray of shape (3, 3) or (3, 3, ...)
            A batch of Jacobians.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of Jacobians.
        """

        if J.ndim > 2 and J.shape[:2] == (3, 3):
            J = np.moveaxis(J, 0, -1)
            J = np.moveaxis(J, 0, -1)
        return J

    def source_domain_prefactor_matrix_multiplication(self, prefactor, evaled):

        # print(f'prefactor.shape {prefactor.shape}') # (20, 20, 5, 3, 3)
        # print(f'evaled.shape {evaled.shape}')       # (3, 20, 20, 5)

        # If `prefactor` is a batch of tensors in a meshgrid.
        if prefactor.ndim == 5:
            for i in range(prefactor.shape[0]):
                for j in range(prefactor.shape[1]):
                    for k in range(prefactor.shape[2]):
                        evaled[:, i, j, k] = prefactor[i, j, k] @ evaled[:, i, j, k]
        # If `prefactor` is one tensor.
        else:
            evaled = prefactor @ evaled

        return evaled

    def radius(self, s, u, v):
        """Because reasons."""
        x_hollow_cyl = self.gvec.a_minor + (self.gvec.r_major - self.gvec.a_minor) * s * np.cos(2*np.pi*u)
        y_hollow_cyl = self.gvec.a_minor + (self.gvec.r_major - self.gvec.a_minor) * s * np.sin(2*np.pi*u)
        x = (x_hollow_cyl + self.gvec.r_major) * np.cos(2*np.pi*v)
        y = y_hollow_cyl
        z = (x_hollow_cyl + self.gvec.r_major) * np.sin(2*np.pi*v)
        return np.sqrt((np.sqrt(x**2 + z**2) - self.gvec.r_major)**2 + y**2)

    def rho(self, r):
        """Dummy density at given radius."""
        return (1 - self.ra)*(1 - (r/self.gvec.a_minor)**self.r1)**self.r2 + self.ra



    # ===============================================================
    #                       MHD variables
    # ===============================================================

    def p_eq(self, eta1, eta2, eta3=None):
        """Equilibrium bulk pressure (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.PRESSURE, form=Form.PHYSICAL)
        # return self.gvec.P(s, u, v)

    def p0_eq(self, eta1, eta2, eta3=None):
        """Equilibrium bulk pressure (0-form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.PRESSURE, form=Form.ZERO)
        # return self.gvec.P_0(s, u, v)

    def p3_eq(self, eta1, eta2, eta3=None):
        """Equilibrium bulk pressure (3-form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        source_det = self.source_det(eta1, eta2, eta3)
        return source_det * self.gvec.get_variable(s, u, v, variable=Variable.PRESSURE, form=Form.THREE)
        # return self.gvec.P_3(s, u, v)

    def r_eq(self, eta1, eta2, eta3=None):
        """Equilibrium bulk density (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        radius = self.radius(s, u, v)
        return self.rho(radius)

    def r0_eq(self, eta1, eta2, eta3):
        """Equilibrium bulk density (0-form on logical domain)."""
        return self.DOMAIN.pull(self.MHD.r_eq, eta1, eta2, eta3, '0_form')

    def r3_eq(self, eta1, eta2, eta3):
        """Equilibrium bulk density (3-form on logical domain)."""
        return self.DOMAIN.pull(self.MHD.r_eq, eta1, eta2, eta3, '3_form')



    # ===============================================================
    #                       Profiles
    # ===============================================================

    def iota_eq(self, eta1, eta2, eta3=None):
        """Equilibrium iota profile (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.IOTA, form=Form.PHYSICAL)
        # return self.gvec.IOTA(s, u, v)

    def q_eq(self, eta1, eta2, eta3=None):
        """Equilibrium q profile (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return 2 * np.pi / self.gvec.get_variable(s, u, v, variable=Variable.IOTA, form=Form.PHYSICAL)
        # return 2 * np.pi / self.gvec.IOTA(s, u, v)

    def toroidal_flux_eq(self, eta1, eta2, eta3=None):
        """Equilibrium toroidal flux (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.PHI, form=Form.PHYSICAL)
        # return self.gvec.PHI(s, u, v)

    def poloidal_flux_eq(self, eta1, eta2, eta3=None):
        """Equilibrium poloidal flux (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.CHI, form=Form.PHYSICAL)
        # return self.gvec.CHI(s, u, v)

    def dtoroidal_flux_eq(self, eta1, eta2, eta3=None):
        """Derivative of equilibrium toroidal flux along radial direction (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        ds_deta1 = self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_11')
        return ds_deta1 * self.gvec.get_dvariable(s, u, v, variable=Variable.PHI, form=Form.PHYSICAL)
        # return ds_deta1 * self.gvec.dPHI(s, u, v)

    def dpoloidal_flux_eq(self, eta1, eta2, eta3=None):
        """Derivative of equilibrium poloidal flux along radial direction (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        ds_deta1 = self.SOURCE_DOMAIN.evaluate(eta1, eta2, eta3, 'df_11')
        return ds_deta1 * self.gvec.get_dvariable(s, u, v, variable=Variable.CHI, form=Form.PHYSICAL)
        # return ds_deta1 * self.gvec.dCHI(s, u, v)



    # ===============================================================
    #                  3D Variables (w/o projection)
    # ===============================================================

    def a_eq_x(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (physical form on logical domain, x-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_df_inv(eta1, eta2, eta3).T
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.PHYSICAL)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[0]

    def a_eq_y(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (physical form on logical domain, y-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_df_inv(eta1, eta2, eta3).T
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.PHYSICAL)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[1]

    def a_eq_z(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (physical form on logical domain, z-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_df_inv(eta1, eta2, eta3).T
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.PHYSICAL)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[2]

    def a1_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (1-form on logical domain, 1-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.ONE)[0]
        # return self.gvec.A_1(s, u, v)[0]

    def a1_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (1-form on logical domain, 2-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.ONE)[1]
        # return self.gvec.A_1(s, u, v)[1]

    def a1_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (1-form on logical domain, 3-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.ONE)[2]
        # return self.gvec.A_1(s, u, v)[2]

    def a2_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (2-form on logical domain, 1-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_det(eta1, eta2, eta3) * self.source_G_inv(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.TWO)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[0]

    def a2_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (2-form on logical domain, 2-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_det(eta1, eta2, eta3) * self.source_G_inv(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.TWO)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[1]

    def a2_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (2-form on logical domain, 3-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_det(eta1, eta2, eta3) * self.source_G_inv(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.TWO)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[2]

    def av_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (Contravariant (vector field) form on logical domain, 1-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_G_inv(eta1, eta2, eta3) # Only works because DF of SOURCE_DOMAIN is diagonal.
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.CONTRAVARIANT)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[0]

    def av_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (Contravariant (vector field) form on logical domain, 2-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_G_inv(eta1, eta2, eta3) # Only works because DF of SOURCE_DOMAIN is diagonal.
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.CONTRAVARIANT)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[1]

    def av_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic vector potential (Contravariant (vector field) form on logical domain, 3-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_G_inv(eta1, eta2, eta3) # Only works because DF of SOURCE_DOMAIN is diagonal.
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.A, form=Form.CONTRAVARIANT)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[2]

    def b_eq_x(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (physical form on logical domain, x-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_df(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.PHYSICAL)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[0]

    def b_eq_y(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (physical form on logical domain, y-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_df(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.PHYSICAL)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[1]

    def b_eq_z(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (physical form on logical domain, z-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_df(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.PHYSICAL)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[2]

    def b1_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (1-form on logical domain, 1-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_G(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.ONE)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[0]

    def b1_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (1-form on logical domain, 2-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_G(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.ONE)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[1]

    def b1_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (1-form on logical domain, 3-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        prefactor = self.source_G(eta1, eta2, eta3)
        evaled = self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.ONE)
        evaled = self.source_domain_prefactor_matrix_multiplication(prefactor, evaled)
        return evaled[2]

    def b2_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (2-form on logical domain, 1-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        source_det = self.source_det(eta1, eta2, eta3)
        return source_det * self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.TWO)[0]
        # return source_det * self.gvec.B_2(s, u, v)[0]

    def b2_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (2-form on logical domain, 2-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        source_det = self.source_det(eta1, eta2, eta3)
        return source_det * self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.TWO)[1]
        # return source_det * self.gvec.B_2(s, u, v)[1]

    def b2_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (2-form on logical domain, 3-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        source_det = self.source_det(eta1, eta2, eta3)
        return source_det * self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.TWO)[2]
        # return source_det * self.gvec.B_2(s, u, v)[2]

    def bv_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (Contravariant (vector field) form on logical domain, 1-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.CONTRAVARIANT)[0]
        # return self.gvec.B_vec(s, u, v)[0]

    def bv_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (Contravariant (vector field) form on logical domain, 2-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.CONTRAVARIANT)[1]
        # return self.gvec.B_vec(s, u, v)[1]

    def bv_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium magnetic field (Contravariant (vector field) form on logical domain, 3-component)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        return self.gvec.get_variable(s, u, v, variable=Variable.B, form=Form.CONTRAVARIANT)[2]
        # return self.gvec.B_vec(s, u, v)[2]



    # ===============================================================
    #                  3D Variables (w/ projection)
    # ===============================================================

    # TODO: Check if we need to apply SOURCE_DOMAIN prefactor again.
    def j_eq_x(self, eta1, eta2, eta3=None):
        """Equilibrium current (physical form on logical domain, x-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        if hasattr(self.PROJ_3D, 'pi_1'):
            b1_coeff = self.PROJ_3D.pi_1(b1) # Coefficients of projected 1-form.
        else:
            b1_coeff = self.PROJ_3D.PI_1(*b1) # Coefficients of projected 1-form.
            b1_coeff = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten())) # Because discrete Div/Curl takes only 1D array.

        # Take discrete Curl.
        j2 = self.TENSOR_SPACE.C.dot(b1_coeff)
        j2_1, j2_2, j2_3 = self.TENSOR_SPACE.extract_2(j2)

        # Evaluate coefficients of equilibrium current.
        evaled_21 = self.TENSOR_SPACE.evaluate_NDD(eta1, eta2, eta3, j2_1)
        evaled_22 = self.TENSOR_SPACE.evaluate_DND(eta1, eta2, eta3, j2_2)
        evaled_23 = self.TENSOR_SPACE.evaluate_DDN(eta1, eta2, eta3, j2_3)
        evaled    = [evaled_21, evaled_22, evaled_23]

        # Push evaluation to Cartesian.
        pushed_jx = self.DOMAIN.push(evaled, s, u, v, '2_form_1')
        # pushed_jx = self.DOMAIN.push(evaled, eta1, eta2, eta3, '2_form_1')

        return pushed_jx

    def j_eq_y(self, eta1, eta2, eta3=None):
        """Equilibrium current (physical form on logical domain, y-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        if hasattr(self.PROJ_3D, 'pi_1'):
            b1_coeff = self.PROJ_3D.pi_1(b1) # Coefficients of projected 1-form.
        else:
            b1_coeff = self.PROJ_3D.PI_1(*b1) # Coefficients of projected 1-form.
            b1_coeff = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten())) # Because discrete Div/Curl takes only 1D array.

        # Take discrete Curl.
        j2 = self.TENSOR_SPACE.C.dot(b1_coeff)
        j2_1, j2_2, j2_3 = self.TENSOR_SPACE.extract_2(j2)

        # Evaluate coefficients of equilibrium current.
        evaled_21 = self.TENSOR_SPACE.evaluate_NDD(eta1, eta2, eta3, j2_1)
        evaled_22 = self.TENSOR_SPACE.evaluate_DND(eta1, eta2, eta3, j2_2)
        evaled_23 = self.TENSOR_SPACE.evaluate_DDN(eta1, eta2, eta3, j2_3)
        evaled    = [evaled_21, evaled_22, evaled_23]

        # Push evaluation to Cartesian.
        pushed_jy = self.DOMAIN.push(evaled, s, u, v, '2_form_2')
        # pushed_jy = self.DOMAIN.push(evaled, eta1, eta2, eta3, '2_form_2')

        return pushed_jy

    def j_eq_z(self, eta1, eta2, eta3=None):
        """Equilibrium current (physical form on logical domain, z-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        if hasattr(self.PROJ_3D, 'pi_1'):
            b1_coeff = self.PROJ_3D.pi_1(b1) # Coefficients of projected 1-form.
        else:
            b1_coeff = self.PROJ_3D.PI_1(*b1) # Coefficients of projected 1-form.
            b1_coeff = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten())) # Because discrete Div/Curl takes only 1D array.

        # Take discrete Curl.
        j2 = self.TENSOR_SPACE.C.dot(b1_coeff)
        j2_1, j2_2, j2_3 = self.TENSOR_SPACE.extract_2(j2)

        # Evaluate coefficients of equilibrium current.
        evaled_21 = self.TENSOR_SPACE.evaluate_NDD(eta1, eta2, eta3, j2_1)
        evaled_22 = self.TENSOR_SPACE.evaluate_DND(eta1, eta2, eta3, j2_2)
        evaled_23 = self.TENSOR_SPACE.evaluate_DDN(eta1, eta2, eta3, j2_3)
        evaled    = [evaled_21, evaled_22, evaled_23]

        # Push evaluation to Cartesian.
        pushed_jz = self.DOMAIN.push(evaled, s, u, v, '2_form_3')
        # pushed_jz = self.DOMAIN.push(evaled, eta1, eta2, eta3, '2_form_3')

        return pushed_jz

    # TODO: Check if we need to apply SOURCE_DOMAIN prefactor again.
    def j2_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium current (2-form on logical domain, 1-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        if hasattr(self.PROJ_3D, 'pi_1'):
            b1_coeff = self.PROJ_3D.pi_1(b1) # Coefficients of projected 1-form.
        else:
            b1_coeff = self.PROJ_3D.PI_1(*b1) # Coefficients of projected 1-form.
            b1_coeff = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten())) # Because discrete Div/Curl takes only 1D array.

        # Take discrete Curl.
        j2 = self.TENSOR_SPACE.C.dot(b1_coeff)
        j2_1, j2_2, j2_3 = self.TENSOR_SPACE.extract_2(j2)

        # Evaluate coefficients of equilibrium current.
        evaled_21 = self.TENSOR_SPACE.evaluate_NDD(eta1, eta2, eta3, j2_1)

        return evaled_21

    def j2_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium current (2-form on logical domain, 2-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        if hasattr(self.PROJ_3D, 'pi_1'):
            b1_coeff = self.PROJ_3D.pi_1(b1) # Coefficients of projected 1-form.
        else:
            b1_coeff = self.PROJ_3D.PI_1(*b1) # Coefficients of projected 1-form.
            b1_coeff = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten())) # Because discrete Div/Curl takes only 1D array.

        # Take discrete Curl.
        j2 = self.TENSOR_SPACE.C.dot(b1_coeff)
        j2_1, j2_2, j2_3 = self.TENSOR_SPACE.extract_2(j2)

        # Evaluate coefficients of equilibrium current.
        evaled_22 = self.TENSOR_SPACE.evaluate_DND(eta1, eta2, eta3, j2_2)

        return evaled_22

    def j2_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium current (2-form on logical domain, 3-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        if hasattr(self.PROJ_3D, 'pi_1'):
            b1_coeff = self.PROJ_3D.pi_1(b1) # Coefficients of projected 1-form.
        else:
            b1_coeff = self.PROJ_3D.PI_1(*b1) # Coefficients of projected 1-form.
            b1_coeff = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten())) # Because discrete Div/Curl takes only 1D array.

        # Take discrete Curl.
        j2 = self.TENSOR_SPACE.C.dot(b1_coeff)
        j2_1, j2_2, j2_3 = self.TENSOR_SPACE.extract_2(j2)

        # Evaluate coefficients of equilibrium current.
        evaled_23 = self.TENSOR_SPACE.evaluate_DDN(eta1, eta2, eta3, j2_3)

        return evaled_23

    # TODO: Check if we need to apply SOURCE_DOMAIN prefactor again.
    def jv_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium current (Contravariant (vector field) form on logical domain, 1-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        evaled_j_eq_x = self.j_eq_x(eta1, eta2, eta3)
        evaled_j_eq_y = self.j_eq_y(eta1, eta2, eta3)
        evaled_j_eq_z = self.j_eq_z(eta1, eta2, eta3)
        evaled_j_eq   = [evaled_j_eq_x, evaled_j_eq_y, evaled_j_eq_z]

        return self.DOMAIN.pull(evaled_j_eq, eta1, eta2, eta3, 'vector_1')

        raise NotImplementedError('Contravariant form of equilibrium current not implemented.')

    def jv_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium current (Contravariant (vector field) form on logical domain, 2-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        evaled_j_eq_x = self.j_eq_x(eta1, eta2, eta3)
        evaled_j_eq_y = self.j_eq_y(eta1, eta2, eta3)
        evaled_j_eq_z = self.j_eq_z(eta1, eta2, eta3)
        evaled_j_eq   = [evaled_j_eq_x, evaled_j_eq_y, evaled_j_eq_z]

        return self.DOMAIN.pull(evaled_j_eq, eta1, eta2, eta3, 'vector_2')

        raise NotImplementedError('Contravariant form of equilibrium current not implemented.')

    def jv_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium current (Contravariant (vector field) form on logical domain, 3-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        evaled_j_eq_x = self.j_eq_x(eta1, eta2, eta3)
        evaled_j_eq_y = self.j_eq_y(eta1, eta2, eta3)
        evaled_j_eq_z = self.j_eq_z(eta1, eta2, eta3)
        evaled_j_eq   = [evaled_j_eq_x, evaled_j_eq_y, evaled_j_eq_z]

        return self.DOMAIN.pull(evaled_j_eq, eta1, eta2, eta3, 'vector_3')

        raise NotImplementedError('Contravariant form of equilibrium current not implemented.')
