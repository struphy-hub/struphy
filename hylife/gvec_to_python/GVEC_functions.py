# ============================================================
# Imports.
# ============================================================

import os
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

# import logging
from gvec_to_python.util.logger import logger
# logger = logging.getLogger(__name__)

import numpy as np

from gvec_to_python.base.make_base import make_base
from gvec_to_python.reader.read_json import read_json
from gvec_to_python.hmap.suv_to_xyz import suv_to_xyz as MapFull



from enum import Enum, unique

class Form(Enum):
    # The numbers should start from 1 so as not to evaluate to `False`...
    ZERO  = 10
    ONE   = 11
    TWO   = 12
    THREE = 13
    COVARIANT = 42
    CO        = 42
    CONTRAVARIANT = 1337
    CONTRA        = 1337
    VECTORFIELDS  = 1337
    PHYSICAL = 1999
    REAL     = 1999

@unique # Require Enum values to be unique.
class Profile(Enum):
    PRESSURE = 1
    PHI      = 2
    CHI      = 3
    IOTA     = 4
    SPOS     = 5
    A        = 11
    B        = 10



class GVEC:
    """Primary class of `gvec_to_python` that wraps around everything as a simple interface.

    Attributes
    ----------
    data : dict
        A `dict` containing GVEC output, where its keys and values are documented in `List_of_data_entries.md`.
    X1_coef : numpy.ndarray
        Spline coefficients (per Fourier mode) of GVEC basis X1.
    X2_coef : numpy.ndarray
        Spline coefficients (per Fourier mode) of GVEC basis X2.
    LA_coef : numpy.ndarray
        Spline coefficients (per Fourier mode) of GVEC basis Lambda.
    X1_base : Base
        GVEC basis X1.
    X2_base : Base
        GVEC basis X2.
    LA_base : Base
        GVEC basis Lambda.
    mapfull : suv_to_xyz
        A mapping class that maps from logical to Cartesian.
    mapX : suv_to_xyz
        A mapping class that maps from logical to Cartesian.
    mapY : suv_to_xyz
        A mapping class that maps from logical to Cartesian.
    mapZ : suv_to_xyz
        A mapping class that maps from logical to Cartesian.
    phi_coef : numpy.ndarray
        Values of phi profile at Grevielle points.
    chi_coef : numpy.ndarray
        Values of chi profile at Grevielle points.
    iota_coef : numpy.ndarray
        Values of iota profile at Grevielle points.
    pres_coef : numpy.ndarray
        Values of pressure profile at Grevielle points.
    spos_coef : numpy.ndarray
        Grevielle points.

    Methods
    -------
    f
        Mapping f.
    df
        Jacobian of mapping f.
    df_det
        Determinant of Jacobian.
    df_inv
        Inverse of Jacobian.
    G
        Metric tensor.
    G_inv
        Inverse metric tensor.
    """

    def __init__(self, gvec_data_filepath: str, gvec_data_filename: str):
        """Load GVEC data from a JSON file, initialize GVEC basis and coordinate map, load equilibrium profile.

        Parameters
        ----------
        gvec_data_filepath : str
            The path to the JSON file.
        gvec_data_filename:
            The name of the JSON file.

        """

        # ============================================================
        # Read GVEC JSON output file.
        # ============================================================

        data = read_json(gvec_data_filepath, gvec_data_filename)

        self.data = data



        # ============================================================
        # Init GVEC basis.
        # ============================================================

        X1_coef = np.array(data['X1']['coef'])
        X2_coef = np.array(data['X2']['coef'])
        LA_coef = np.array(data['LA']['coef'])
        X1_base = make_base(data, 'X1')
        X2_base = make_base(data, 'X2')
        LA_base = make_base(data, 'LA')

        self.X1_coef = X1_coef
        self.X2_coef = X2_coef
        self.LA_coef = LA_coef
        self.X1_base = X1_base
        self.X2_base = X2_base
        self.LA_base = LA_base



        # ============================================================
        # Init hmap class.
        # ============================================================

        mapfull = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef)
        mapX    = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef, 'x')
        mapY    = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef, 'y')
        mapZ    = MapFull(X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef, 'z')

        self.mapfull = mapfull
        self.mapX    = mapX
        self.mapY    = mapY
        self.mapZ    = mapZ



        # ============================================================
        # Expose mapping functions.
        # ============================================================

        self.f      = mapfull.f
        self.df     = mapfull.df
        self.df_det = mapfull.df_det
        self.df_det_from_J = mapfull.df_det_from_J
        self.df_inv_from_J = mapfull.df_inv_from_J
        self.df_inv = mapfull.df_inv
        self.G      = mapfull.G
        self.G_inv  = mapfull.G_inv
        self.G_from_J = mapfull.G_from_J
        self.G_inv_from_J = mapfull.G_inv_from_J



        # ============================================================
        # Access profiles from `data`.
        # ============================================================

        phi_coef  = np.array(data['profiles']['phi'])
        chi_coef  = np.array(data['profiles']['chi'])
        iota_coef = np.array(data['profiles']['iota'])
        pres_coef = np.array(data['profiles']['pres'])
        spos_coef = np.array(data['profiles']['spos'])

        self.phi_coef  = phi_coef
        self.chi_coef  = chi_coef
        self.iota_coef = iota_coef
        self.pres_coef = pres_coef
        self.spos_coef = spos_coef

        assert len(phi_coef)  == data['profiles']['nPoints'],      'Phi profile should have the same length as `nPoints`.'
        assert len(chi_coef)  == data['profiles']['nPoints'],      'Chi profile should have the same length as `nPoints`.'
        assert len(iota_coef) == data['profiles']['nPoints'],     'Iota profile should have the same length as `nPoints`.'
        assert len(pres_coef) == data['profiles']['nPoints'], 'Pressure profile should have the same length as `nPoints`.'
        assert len(spos_coef) == data['profiles']['nPoints'],   'Spline profile should have the same length as `nPoints`.'



    def assert_profile_form(self, profile, form):

        if profile in [Profile.PRESSURE, Profile.PHI, Profile.CHI, Profile.IOTA, Profile.SPOS]:
            assert (form == Form.PHYSICAL) or (form == Form.ZERO) or (form == Form.THREE)
        elif profile in [Profile.A, Profile.B]:
            assert (form == Form.PHYSICAL) or (form == Form.ONE) or (form == Form.TWO) or (form == Form.COVARIANT) or (form == Form.CONTRAVARIANT)
        else:
            raise NotImplementedError('Profile {} is not implemented.'.format(profile))

    # TODO: Pay attention to a potential (2*pi)^2 factor in our Jacobian!
    def get_profile(self, s, u, v, profile, form):

        self.assert_profile_form(profile, form)

        if isinstance(s, np.ndarray):

            assert isinstance(s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(u, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(u))
            assert isinstance(v, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(v))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)



        if profile == Profile.PRESSURE:

            # Identical operations for both 0-form and 3-form.
            evaled = self.X1_base.eval_profile(s, self.pres_coef)
            evaled = evaled * np.ones_like(u) * np.ones_like(v)

            if form == Form.THREE:
                det = self.df_det(s, u, v)
                evaled = det * evaled

        elif profile == Profile.PHI:

            # Identical operations for both 0-form and 3-form.
            evaled = self.X1_base.eval_profile(s, self.phi_coef)
            evaled = evaled * np.ones_like(u) * np.ones_like(v)

            if form == Form.THREE:
                det = self.df_det(s, u, v)
                evaled = det * evaled

        elif profile == Profile.CHI:

            # Identical operations for both 0-form and 3-form.
            evaled = self.X1_base.eval_profile(s, self.chi_coef)
            evaled = evaled * np.ones_like(u) * np.ones_like(v)

            if form == Form.THREE:
                det = self.df_det(s, u, v)
                evaled = det * evaled

        elif profile == Profile.IOTA:

            # Identical operations for both 0-form and 3-form.
            evaled = self.X1_base.eval_profile(s, self.iota_coef)
            evaled = evaled * np.ones_like(u) * np.ones_like(v)

            if form == Form.THREE:
                det = self.df_det(s, u, v)
                evaled = det * evaled

        elif profile == Profile.SPOS:

            # Identical operations for both 0-form and 3-form.
            evaled = self.X1_base.eval_profile(s, self.spos_coef)
            evaled = evaled * np.ones_like(u) * np.ones_like(v)

            if form == Form.THREE:
                det = self.df_det(s, u, v)
                evaled = det * evaled

        elif profile == Profile.A:

            J = self.df(s, u, v)
            det = self.df_det_from_J(J)

            # A-field is in 1-form basis (GVEC Eq. 1.24, 1.25).
            # i.e. covariant components with contravariant basis.
            A1 = - self.LA_base.eval_suv(s, u, v, self.LA_coef) * self.X1_base.eval_dprofile(s, self.phi_coef)
            A2 =   self.X1_base.eval_profile(s, self.phi_coef) * 2 * np.pi * np.ones_like(u) * np.ones_like(v)
            A3 = - self.X1_base.eval_profile(s, self.chi_coef) * 2 * np.pi * np.ones_like(u) * np.ones_like(v)
            evaled = np.array([A1, A2, A3])

            if form == Form.CONTRAVARIANT:
                G_inv = self.G_inv_from_J(J)
                # If `G_inv` is a batch of inverse metric tensors in a meshgrid.
                if G_inv.ndim == 5:
                    for i in range(G_inv.shape[0]):
                        for j in range(G_inv.shape[1]):
                            for k in range(G_inv.shape[2]):
                                evaled[:, i, j, k] = G_inv[i, j, k] @ evaled[:, i, j, k]
                # If `G_inv` is one inverse metric tensor.
                else:
                    evaled = G_inv @ evaled

            elif form == Form.TWO:
                # DF^{-T} \hat{v}^1 = \frac{1}{\sqrt{g}} * DF \hat{v}^2
                # \sqrt{g} DF^{-1} DF^{-T} \hat{v}^1 = DF^{-1} * DF \hat{v}^2
                # \sqrt{g} G^{-1} \hat{v}^1 =\hat{v}^2
                G_inv = self.G_inv_from_J(J)
                # If `G_inv` is a batch of inverse metric tensors in a meshgrid.
                if G_inv.ndim == 5:
                    for i in range(G_inv.shape[0]):
                        for j in range(G_inv.shape[1]):
                            for k in range(G_inv.shape[2]):
                                evaled[:, i, j, k] = G_inv[i, j, k] @ evaled[:, i, j, k]
                # If `G_inv` is one inverse metric tensor.
                else:
                    evaled = G_inv @ evaled
                evaled = det * evaled # Multiply with determinant.

            elif form == Form.PHYSICAL:
                J_inv = self.df_inv_from_J(J)
                # If `J_inv` is a batch of inverse metric tensors in a meshgrid.
                if J_inv.ndim == 5:
                    for i in range(J_inv.shape[0]):
                        for j in range(J_inv.shape[1]):
                            for k in range(J_inv.shape[2]):
                                evaled[:, i, j, k] = J_inv[i, j, k].T @ evaled[:, i, j, k]
                # If `J_inv` is one inverse metric tensor.
                else:
                    evaled = J_inv.T @ evaled

        elif profile == Profile.B:

            J = self.df(s, u, v)
            det = self.df_det_from_J(J)

            # B-field is in contravariant form, a.k.a. vector fields (GVEC Eq. 1.26).
            # i.e. contravariant components with covariant basis.
            # Pay attention to a 2*pi factor, because we are calculating du and dv here, not d(theta) d(zeta)!
            dchi_ds = self.X1_base.eval_dprofile(s, self.chi_coef)
            dphi_ds = self.X1_base.eval_dprofile(s, self.phi_coef)
            dlambda_dtheta = self.LA_base.eval_suv_du(s, u, v, self.LA_coef) / (2 * np.pi)
            dlambda_dzeta  = self.LA_base.eval_suv_dv(s, u, v, self.LA_coef) / (2 * np.pi)
            B1 = np.zeros_like(dlambda_dzeta)
            B2 = ( dchi_ds - dlambda_dzeta * dphi_ds ) * 2 * np.pi / det
            B3 = ( (1 + dlambda_dtheta)    * dphi_ds ) * 2 * np.pi / det
            evaled = np.array([B1, B2, B3])

            if form == Form.ONE or form == Form.COVARIANT:
                G = self.G_from_J(J)
                # If `G` is a batch of metric tensors in a meshgrid.
                if G.ndim == 5:
                    for i in range(G.shape[0]):
                        for j in range(G.shape[1]):
                            for k in range(G.shape[2]):
                                evaled[:, i, j, k] = G[i, j, k] @ evaled[:, i, j, k]
                # If `G` is one metric tensor.
                else:
                    evaled = G @ evaled

            elif form == Form.TWO:
                evaled = det * evaled

            elif form == Form.PHYSICAL:
                # If `J` is a batch of metric tensors in a meshgrid.
                if J.ndim == 5:
                    for i in range(J.shape[0]):
                        for j in range(J.shape[1]):
                            for k in range(J.shape[2]):
                                evaled[:, i, j, k] = J[i, j, k] @ evaled[:, i, j, k]
                # If `J` is one metric tensor.
                else:
                    evaled = J @ evaled

        else:

            raise NotImplementedError('Profile {} is not implemented.'.format(profile))

        return evaled



    # ============================================================
    # Set aliases for profiles.
    # ============================================================

    def P(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.PRESSURE, form=Form.PHYSICAL)

    def P_0(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.PRESSURE, form=Form.ZERO)

    def P_3(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.PRESSURE, form=Form.THREE)

    def PHI(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.PHI, form=Form.PHYSICAL)

    def PHI_0(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.PHI, form=Form.ZERO)

    def PHI_3(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.PHI, form=Form.THREE)

    def CHI(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.CHI, form=Form.PHYSICAL)

    def CHI_0(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.CHI, form=Form.ZERO)

    def CHI_3(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.CHI, form=Form.THREE)

    def IOTA(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.IOTA, form=Form.PHYSICAL)

    def IOTA_0(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.IOTA, form=Form.ZERO)

    def IOTA_3(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.IOTA, form=Form.THREE)

    def SPOS(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.SPOS, form=Form.PHYSICAL)

    def SPOS_0(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.SPOS, form=Form.ZERO)

    def SPOS_3(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.SPOS, form=Form.THREE)

    def A(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.A, form=Form.PHYSICAL)

    def A_vec(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.A, form=Form.CONTRAVARIANT)

    def A_1(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.A, form=Form.ONE)

    def A_2(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.A, form=Form.TWO)

    def B(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.B, form=Form.PHYSICAL)

    def B_vec(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.B, form=Form.CONTRAVARIANT)

    def B_1(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.B, form=Form.ONE)

    def B_2(self, s, u, v): # pragma: no cover
        return self.get_profile(s, u, v, profile=Profile.B, form=Form.TWO)
