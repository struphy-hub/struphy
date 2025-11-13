import cunumpy as xp

from struphy.feec.psydac_derham import Derham
from struphy.fields_background.coil_fields.base import CoilMagneticField, load_csv_data


class RatGUI(CoilMagneticField):
    """Interface to RatGUI."""

    def __init__(self, csv_path=None, Nel=[16, 16, 16], p=[3, 3, 3], domain=None, **params):
        print("Hello.")
        self._csv_path = csv_path

        # TODO: load csv data from absolute/relative path
        self._ratgui_csv_data = load_csv_data(csv_path)

        derham = Derham(
            Nel=Nel,
            p=p,
            spl_kind=[False, False, True],
        )  # Assuming (R=eta1, Z=eta2, phi=eta3) coordinates for csv data (periodic in eta3 only).
        self._interpolate = derham.P[
            "v"
        ].solve  # This is a method for spline interpolation of degree p on the grid Nel in eta-space.
        self._rhs = derham.Vh[
            "v"
        ].zeros()  # This is the vector where we want to store the csv data. It holds all three B-components and will be passed to the interpolator.

        # Extract B_R, B_Z, B_phi from loaded data
        B_R = self._ratgui_csv_data["B_R"]
        B_Z = self._ratgui_csv_data["B_Z"]
        B_phi = self._ratgui_csv_data["B_phi"]

        # Fill the rhs vector with reshaped data
        self.rhs[0][:] = B_R
        self.rhs[1][:] = B_Z
        self.rhs[2][:] = B_phi

        print(f"{self.rhs =}")
        print(f"{derham.nbasis['v'] =}")
        print(f"{self.rhs[0] =}")
        print(f"{self.rhs[1] =}")
        print(f"{self.rhs[2] =}")
        print(f"{self.rhs[0][:].shape =}")
        print(f"{self.rhs[1][:].shape =}")
        print(f"{self.rhs[2][:].shape =}")
        # We need to choose Nel and p such that the csv_data fits into this vector.
        # For a periodic direction, the size of the vector is Nel, for non-periodic (spl_kind=False) the size is Nel + p.

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
