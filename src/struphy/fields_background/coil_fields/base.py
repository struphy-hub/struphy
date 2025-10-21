from abc import ABCMeta, abstractmethod

from matplotlib import pyplot as plt
from pyevtk.hl import gridToVTK

from struphy.utils.arrays import xp as np


class CoilMagneticField(metaclass=ABCMeta):
    """
    Base class for magnetic vacuum fields derived from current-carrying coils.
    """

    @property
    def domain(self):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        assert hasattr(self, "_domain"), (
            "Domain for CoilMagneticField not set. Only bfield_logical available at this stage. Please do obj.domain = ... to have access to all transformations (1-form, 2-form, etc.)"
        )
        return self._domain

    @domain.setter
    def domain(self, domain):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        self._domain = domain

    @abstractmethod
    def b_xyz(self, x, y, z):
        """Cartesian equilibrium magnetic field in physical space. Must return the components as a tuple."""
        pass

    def absB0(self, *etas, squeeze_out=False):
        """0-form absolute value of equilibrium magnetic field on logical cube [0, 1]^3."""
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        return np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

    def absB3(self, *etas, squeeze_out=False):
        """3-form absolute value of equilibrium magnetic field on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.absB0(*etas, squeeze_out=False),
            *etas,
            kind="0_to_3",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def b1(self, *etas, squeeze_out=False):
        """1-form components of equilibrium magnetic field on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.bv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_1",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def b2(self, *etas, squeeze_out=False):
        """2-form components of equilibrium magnetic field on logical cube [0, 1]^3."""
        return self.domain.transform(
            self.bv(*etas, squeeze_out=False),
            *etas,
            kind="v_to_2",
            a_kwargs={"squeeze_out": False},
            squeeze_out=squeeze_out,
        )

    def bv(self, *etas, squeeze_out=False):
        """Contra-variant components of equilibrium magnetic field on logical cube [0, 1]^3."""
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.b_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind="v", squeeze_out=squeeze_out)

    def b_cart(self, *etas, squeeze_out=False):
        """Cartesian components of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        b_out = self.domain.push(
            self.bv(*etas, squeeze_out=False), *etas, kind="v", a_kwargs={"squeeze_out": False}, squeeze_out=squeeze_out
        )
        return b_out, self.domain(*etas, squeeze_out=squeeze_out)

    def unit_b1(self, *etas, squeeze_out=False):
        """Unit vector components of equilibrium magnetic field (1-form) on logical cube [0, 1]^3."""
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="1", squeeze_out=squeeze_out)

    def unit_b2(self, *etas, squeeze_out=False):
        """Unit vector components of equilibrium magnetic field (2-form) on logical cube [0, 1]^3."""
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="2", squeeze_out=squeeze_out)

    def unit_bv(self, *etas, squeeze_out=False):
        """Unit vector components of  equilibrium magnetic field (contra-variant) on logical cube [0, 1]^3."""
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind="v", squeeze_out=squeeze_out)

    def unit_b_cart(self, *etas, squeeze_out=False):
        """Unit vector Cartesian components of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z)."""
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        absB = self.absB0(*etas, squeeze_out=squeeze_out)
        out = np.array([b[0] / absB, b[1] / absB, b[2] / absB], dtype=float)
        return out, xyz


def load_csv_data(csv_path):
    """
    Load CSV data from the given path.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Dictionary containing 'R', 'Z', 'phi', 'B_R', 'B_Z', 'B_phi'.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_columns = {"R", "Z", "phi", "B_R", "B_Z", "B_phi"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    # Return the data as a dictionary of numpy arrays
    return {
        "R": data["R"].values,
        "Z": data["Z"].values,
        "phi": data["phi"].values,
        "B_R": data["B_R"].values,
        "B_Z": data["B_Z"].values,
        "B_phi": data["B_phi"].values,
    }
