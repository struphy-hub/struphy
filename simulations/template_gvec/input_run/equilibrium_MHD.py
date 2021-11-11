import numpy as np
import scipy as sc

import os
import sys

basedir = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, os.path.join(basedir, '..'))

# Import necessary struphy.modules.
import struphy.geometry.domain_3d as dom
from gvec_to_python import GVEC, Form, Variable



class equilibrium_mhd:

    def __init__(self, tensor_space, domain, source_domain=None, gvec_filepath=None, gvec_filename=None, rho0=1., beta=200.):

        # Geometric parameters.
        self.domain = domain  # Domain object defining the mapping.
        self.source_domain = source_domain
        if source_domain is None: # Default (s,u,v) = (eta1, eta2, eta3)
            bounds = [0,1,0,1,0,1]
            self.source_domain = dom.domain('cuboid slice', params_map=bounds)

        # Create 3D projector. It's not automatic.
        for space in tensor_space.spaces:
            if not hasattr(space, 'projectors'):
                space.set_projectors() # def set_projectors(self, nq=6):
        if not hasattr(tensor_space, 'projectors'):
            tensor_space.set_projectors() # def set_projectors(self, which='tensor', nq=[6, 6]):

        self.spaces = tensor_space.spaces
        self.tensor_space = tensor_space
        self.proj_3d = tensor_space.projectors

        # Density.
        # self.rho0 = rho0

        # Pressure.
        # self.gamma = 5/3
        # self.beta = beta

        # ============================================================
        # Load GVEC mapping.
        # ============================================================

        if gvec_filepath is None:
            gvec_filepath = 'testcases/ellipstell/'
            gvec_filepath = os.path.join(basedir, '..', gvec_filepath)
        if gvec_filename is None:
            gvec_filename = 'GVEC_ellipStell_profile_update_State_0000_00010000.json'
        self.gvec = GVEC(gvec_filepath, gvec_filename)



    # ===============================================================
    #                      Map source domain
    # ===============================================================

    def s(self, eta1, eta2, eta3):
        return self.source_domain.evaluate(eta1, eta2, eta3, 'x')

    def u(self, eta1, eta2, eta3):
        return self.source_domain.evaluate(eta1, eta2, eta3, 'y')

    def v(self, eta1, eta2, eta3):
        return self.source_domain.evaluate(eta1, eta2, eta3, 'z')

    def eta123_to_suv(self, eta1, eta2, eta3):
        return self.s(eta1, eta2, eta3), self.u(eta1, eta2, eta3), self.v(eta1, eta2, eta3)

    def source_det(self, eta1, eta2, eta3):
        return self.source_domain.evaluate(eta1, eta2, eta3, 'det_df')

    def source_df(self, eta1, eta2, eta3):
        df = np.array(
            (
                (self.source_domain.evaluate(eta1, eta2, eta3, 'df_11'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_12'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_13')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'df_21'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_22'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_23')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'df_31'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_32'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_33')),
            )
        )
        return self.swap_J_axes(df)

    def source_df_inv(self, eta1, eta2, eta3):
        df_inv = np.array(
            (
                (self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_11'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_12'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_13')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_21'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_22'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_23')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_31'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_32'), self.source_domain.evaluate(eta1, eta2, eta3, 'df_inv_33')),
            )
        )
        return self.swap_J_axes(df_inv)

    def source_G(self, eta1, eta2, eta3):
        G = np.array(
            (
                (self.source_domain.evaluate(eta1, eta2, eta3, 'g_11'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_12'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_13')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'g_21'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_22'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_23')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'g_31'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_32'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_33')),
            )
        )
        return self.swap_J_axes(G)

    def source_G_inv(self, eta1, eta2, eta3):
        G_inv = np.array(
            (
                (self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_11'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_12'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_13')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_21'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_22'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_23')),
                (self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_31'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_32'), self.source_domain.evaluate(eta1, eta2, eta3, 'g_inv_33')),
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
        ds_deta1 = self.source_domain.evaluate(eta1, eta2, eta3, 'df_11')
        return ds_deta1 * self.gvec.get_dvariable(s, u, v, variable=Variable.PHI, form=Form.PHYSICAL)
        # return ds_deta1 * self.gvec.dPHI(s, u, v)

    def dpoloidal_flux_eq(self, eta1, eta2, eta3=None):
        """Derivative of equilibrium poloidal flux along radial direction (physical form on logical domain)."""
        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)
        ds_deta1 = self.source_domain.evaluate(eta1, eta2, eta3, 'df_11')
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



    # ===============================================================
    #                  3D Variables (w/ projection)
    # ===============================================================

    # TODO: Physical form doesn't work. To be fixed in a later commit.
    def j_eq_x(self, eta1, eta2, eta3=None):
        """Equilibrium current (physical form on logical domain, x-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        b1_coeff = self.proj_3d.PI_1(*b1) # Coefficients of projected 1-form.

        # Because discrete Div/Curl takes only 1D array.
        b1_coeff_concat = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten()))

        # Take discrete Curl.
        j2 = self.tensor_space.C.dot(b1_coeff_concat)
        j2_1, j2_2, j2_3 = self.tensor_space.extract_2(j2)

        # Push to Cartesian.
        pushed_jx = self.domain.push([j2_1, j2_2, j2_3], eta1, eta2, eta3, '2_form_1')

        return pushed_jx

    def j_eq_y(self, eta1, eta2, eta3=None):
        """Equilibrium current (physical form on logical domain, y-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        b1_coeff = self.proj_3d.PI_1(*b1) # Coefficients of projected 1-form.

        # Because discrete Div/Curl takes only 1D array.
        b1_coeff_concat = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten()))

        # Take discrete Curl.
        j2 = self.tensor_space.C.dot(b1_coeff_concat)
        j2_1, j2_2, j2_3 = self.tensor_space.extract_2(j2)

        # Push to Cartesian.
        pushed_jy = self.domain.push([j2_1, j2_2, j2_3], eta1, eta2, eta3, '2_form_2')

        return pushed_jy

    def j_eq_z(self, eta1, eta2, eta3=None):
        """Equilibrium current (physical form on logical domain, z-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        b1_coeff = self.proj_3d.PI_1(*b1) # Coefficients of projected 1-form.

        # Because discrete Div/Curl takes only 1D array.
        b1_coeff_concat = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten()))

        # Take discrete Curl.
        j2 = self.tensor_space.C.dot(b1_coeff_concat)
        j2_1, j2_2, j2_3 = self.tensor_space.extract_2(j2)

        # Push to Cartesian.
        pushed_jz = self.domain.push([j2_1, j2_2, j2_3], eta1, eta2, eta3, '2_form_3')

        return pushed_jz

    # TODO: These are just the coefficients! Where's the evaluation?
    def j2_eq_1(self, eta1, eta2, eta3=None):
        """Equilibrium current (2-form on logical domain, 1-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        b1_coeff = self.proj_3d.PI_1(*b1) # Coefficients of projected 1-form.

        # Because discrete Div/Curl takes only 1D array.
        b1_coeff_concat = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten()))

        # Take discrete Curl.
        j2 = self.tensor_space.C.dot(b1_coeff_concat)
        j2_1, j2_2, j2_3 = self.tensor_space.extract_2(j2)

        return j2_1

    def j2_eq_2(self, eta1, eta2, eta3=None):
        """Equilibrium current (2-form on logical domain, 2-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        b1_coeff = self.proj_3d.PI_1(*b1) # Coefficients of projected 1-form.

        # Because discrete Div/Curl takes only 1D array.
        b1_coeff_concat = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten()))

        # Take discrete Curl.
        j2 = self.tensor_space.C.dot(b1_coeff_concat)
        j2_1, j2_2, j2_3 = self.tensor_space.extract_2(j2)

        return j2_2

    def j2_eq_3(self, eta1, eta2, eta3=None):
        """Equilibrium current (2-form on logical domain, 3-component, curl of equilibrium magnetic field)."""

        s, u, v = self.eta123_to_suv(eta1, eta2, eta3)

        # Supply a tuple of each 1-form B-field components B=(B1,B2,B3).
        b1 = [self.b1_eq_1, self.b1_eq_2, self.b1_eq_3]

        # Do the Pi_1 projection.
        b1_coeff = self.proj_3d.PI_1(*b1) # Coefficients of projected 1-form.

        # Because discrete Div/Curl takes only 1D array.
        b1_coeff_concat = np.concatenate((b1_coeff[0].flatten(), b1_coeff[1].flatten(), b1_coeff[2].flatten()))

        # Take discrete Curl.
        j2 = self.tensor_space.C.dot(b1_coeff_concat)
        j2_1, j2_2, j2_3 = self.tensor_space.extract_2(j2)

        return j2_3
