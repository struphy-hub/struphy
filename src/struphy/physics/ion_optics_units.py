"""SI units for electrostatic ion optics.

Ion-optics designs are specified by a length and a voltage, while Struphy
derives its units from ``BaseUnits(x, B, n)``. The magnetic field unit is only
a normalization parameter here: choosing it as

.. math::

    B = \\frac{1}{L}\\sqrt{\\frac{A m_\\mathrm{H} \\Phi}{Z e}}

with the ``"cyclotron"`` velocity scale makes the unit of the electrostatic
potential equal to :math:`\\Phi`. For the bulk ion (charge number :math:`Z`,
mass number :math:`A`) the normalized equations of motion then read
:math:`\\dot{\\mathbf v} = -\\nabla\\phi` (``epsilon = 1``), and the kinetic
energy :math:`\\tfrac12 |\\mathbf v|^2` is measured in units of :math:`Z e \\Phi`.
With the charge unit :math:`\\varepsilon_0 \\Phi L`, Poisson's equation is
:math:`-\\nabla^2 \\phi = \\rho`, and the Child–Langmuir current density of a planar
diode of gap :math:`L` and voltage :math:`\\Phi` is :math:`4\\sqrt 2/9`.
"""

from dataclasses import dataclass

import numpy as np

from struphy.io.options import BaseUnits
from struphy.physics.physics import ConstantsOfNature


@dataclass(frozen=True)
class IonOpticsUnits:
    """Conversion between SI ion-optics quantities and Struphy normalization.

    Parameters
    ----------
    length : float
        Unit of length in m.

    voltage : float
        Unit of electrostatic potential in V.

    mass_number : float
        Mass number of the bulk ion species (must match the model).

    charge_number : int
        Charge number of the bulk ion species (must match the model).

    density : float
        Unit of number density in 1e20/m^3; it only enters the space-charge
        coupling parameters, not single-particle motion.
    """

    length: float
    voltage: float
    mass_number: float = 1.0
    charge_number: int = 1
    density: float = 1.0

    def __post_init__(self):
        if self.length <= 0.0 or self.voltage <= 0.0:
            raise ValueError("The length and voltage units must be positive.")
        if self.mass_number <= 0.0 or self.charge_number <= 0:
            raise ValueError("The bulk ion needs a positive mass number and charge number.")

    @property
    def magnetic_field(self) -> float:
        """Normalization magnetic field in T (no physical field is implied)."""
        con = ConstantsOfNature()
        return np.sqrt(self.mass_number * con.mH * self.voltage / (self.charge_number * con.e)) / self.length

    def base_units(self) -> BaseUnits:
        """Base units to pass to ``IonOpticsElectrostatic(base_units=...)``."""
        return BaseUnits(x=self.length, B=self.magnetic_field, n=self.density)

    @property
    def velocity(self) -> float:
        """Unit of velocity in m/s."""
        con = ConstantsOfNature()
        return np.sqrt(self.charge_number * con.e * self.voltage / (self.mass_number * con.mH))

    @property
    def time(self) -> float:
        """Unit of time in s."""
        return self.length / self.velocity

    @property
    def electric_field(self) -> float:
        """Unit of electric field in V/m."""
        return self.voltage / self.length

    @property
    def charge(self) -> float:
        """Unit of charge in C, :math:`\\varepsilon_0 \\Phi L`.

        With this unit, Poisson's equation reads :math:`-\\nabla^2\\phi = \\rho` in normalized
        units, and marker weights are marker charges (see ``IonOpticsElectrostatic(space_charge=True)``).
        """
        con = ConstantsOfNature()
        return con.eps0 * self.voltage * self.length

    @property
    def current(self) -> float:
        """Unit of current in A (charge unit per time unit)."""
        return self.charge / self.time

    @property
    def current_density(self) -> float:
        """Unit of current density in A/m^2."""
        return self.current / self.length**2

    def potential(self, volts):
        """Normalized potential of a voltage in V."""
        return np.asarray(volts) / self.voltage

    def volts(self, potential):
        """Voltage in V of a normalized potential."""
        return np.asarray(potential) * self.voltage

    def speed(self, kinetic_energy_eV, mass_number: float = None):
        """Normalized speed of an ion with the given kinetic energy in eV."""
        con = ConstantsOfNature()
        mass = (self.mass_number if mass_number is None else mass_number) * con.mH
        return np.sqrt(2.0 * np.asarray(kinetic_energy_eV) * con.e / mass) / self.velocity

    def kinetic_energy_eV(self, speed, mass_number: float = None):
        """Kinetic energy in eV of an ion with the given normalized speed."""
        con = ConstantsOfNature()
        mass = (self.mass_number if mass_number is None else mass_number) * con.mH
        return 0.5 * mass * (np.asarray(speed) * self.velocity) ** 2 / con.e
