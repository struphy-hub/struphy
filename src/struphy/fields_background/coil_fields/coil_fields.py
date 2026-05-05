import logging

import cunumpy as xp

from struphy.feec.psydac_derham import Derham
from struphy.fields_background.coil_fields.base import CoilMagneticField, load_csv_data
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")


class RatGUI(CoilMagneticField):
    """Interface to RatGUI."""

    def __init__(self, csv_path=None, num_elements=[16, 16, 16], degree=[3, 3, 3], domain=None, **params):
        logger.info("Hello.")
        self._csv_path = csv_path

        # TODO: load csv data from absolute/relative path
        self._ratgui_csv_data = load_csv_data(csv_path)

        grid = TensorProductGrid(num_elements=num_elements)
        derham_opts = DerhamOptions(degree=degree, bcs=(("free", "free"), ("free", "free"), None))
        derham = Derham(
            grid=grid, options=derham_opts
        )  # Assuming (R=eta1, Z=eta2, phi=eta3) coordinates for csv data (periodic in eta3 only).

        self._interpolate = (
            derham.Pv.solve
        )  # This is a method for spline interpolation of degree degree on the grid num_elements in eta-space.
        self._rhs = derham.Vv.zeros()  # This is the vector where we want to store the csv data. It holds all three B-components and will be passed to the interpolator.

        # Extract B_R, B_Z, B_phi from loaded data
        B_R = self._ratgui_csv_data["B_R"]
        B_Z = self._ratgui_csv_data["B_Z"]
        B_phi = self._ratgui_csv_data["B_phi"]

        # Fill the rhs vector with reshaped data
        self.rhs[0][:] = B_R
        self.rhs[1][:] = B_Z
        self.rhs[2][:] = B_phi

        logger.info(f"{self.rhs =}")
        logger.info(f"{derham.Vvsplines.nbasis =}")
        logger.info(f"{self.rhs[0] =}")
        logger.info(f"{self.rhs[1] =}")
        logger.info(f"{self.rhs[2] =}")
        logger.info(f"{self.rhs[0][:].shape =}")
        logger.info(f"{self.rhs[1][:].shape =}")
        logger.info(f"{self.rhs[2][:].shape =}")
        # We need to choose num_elements and degree such that the csv_data fits into this vector.
        # For a periodic direction, the size of the vector is num_elements, for non-periodic the size is num_elements + degree.

        # TODO: fill ratgui_csv_data into rhs vector

        # create callable FEMfield and fill with FE coeffs obtained from interpolation
        self._bfield_RZphi = derham.create_field("ratgui_field", "H1vec")
        self.bfield_RZphi.vector = self.interpolate(self.rhs)

    @property
    def csv_path(self):
        """Path to csv data."""
        return self._csv_path

    @property
    def ratgui_csv_data(self):
        """Data from RatGUI file."""
        return self._ratgui_csv_data

    @property
    def interpolate(self):
        """Spline interpolation according to :attr:`~struphy.feec.projectors.CommutingProjector.solve` of space H1."""
        return self._interpolate

    @property
    def rhs(self):
        """Point data for interpolation, obtained from ratgui_csv_data."""
        return self._rhs

    @property
    def bfield_RZphi(self):
        """Callable :class:`~struphy.feec.psydac_derham.Derham.Field` obtained from interpolation of rhs."""
        return self._bfield_RZphi

    def b_xyz(self, x, y, z):
        """Cartesian coil magnetic field in physical space. Must return the components as a tuple."""
        # compute (R, Z, phi) corrdinates from (x, y, z), for example:
        R = xp.sqrt(x**2 + y**2)
        Z = z
        phi = -xp.arctan2(y / x)

        return self.bfield_RZphi(R, Z, phi)
