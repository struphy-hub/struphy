"""Plasma models for ion extraction: charge of the compensating (unmodelled) plasma species.

In positive-ion extraction (Kalvas 2013, §5.2.1), the ions are traced as particles and
the plasma electrons are thermal, with the Boltzmann density

.. math::

    \\rho_e(\\phi) = -\\rho_{e0} \\exp\\left(\\frac{\\phi - \\phi_P}{T_e}\\right),

which makes Poisson's equation nonlinear. Ions must enter the sheath with at least
the Bohm velocity :math:`v_B = \\sqrt{k T_e / m_i}`. All quantities here are
normalized with :class:`~struphy.physics.ion_optics_units.IonOpticsUnits`: potential
and :math:`T_e` in units of the voltage unit, charge density in units of
``units.charge / units.length**3``.
"""

from dataclasses import dataclass

import numpy as np

from struphy.physics.physics import ConstantsOfNature


@dataclass(frozen=True)
class BoltzmannElectrons:
    """Thermal electrons in Boltzmann equilibrium with the plasma potential.

    Parameters
    ----------
    density : float
        Electron charge density magnitude :math:`\\rho_{e0}` at the plasma potential (normalized).

    temperature : float
        Electron temperature :math:`k T_e / e` in units of the voltage unit.

    plasma_potential : float
        Potential :math:`\\phi_P` of the (quasi-neutral) plasma, in units of the voltage unit.
    """

    density: float
    temperature: float
    plasma_potential: float = 0.0

    def __post_init__(self):
        if self.density <= 0.0 or self.temperature <= 0.0:
            raise ValueError("The electron density and temperature must be positive.")

    def _exponent(self, phi):
        # clip to avoid overflow far above the plasma potential (Newton iterates can overshoot)
        return np.minimum((np.asarray(phi) - self.plasma_potential) / self.temperature, 50.0)

    def charge_density(self, phi):
        """Electron charge density (negative) at potential ``phi``."""
        return -self.density * np.exp(self._exponent(phi))

    def charge_density_derivative(self, phi):
        """Derivative of :meth:`charge_density` with respect to ``phi``."""
        return -self.density / self.temperature * np.exp(self._exponent(phi))

    @classmethod
    def from_si(cls, units, electron_density, electron_temperature_eV, plasma_potential_V=0.0):
        """Build from SI values: density in 1/m^3, temperature in eV, plasma potential in V."""
        con = ConstantsOfNature()
        rho_unit = units.charge / units.length**3
        return cls(
            density=con.e * electron_density / rho_unit,
            temperature=electron_temperature_eV / units.voltage,
            plasma_potential=plasma_potential_V / units.voltage,
        )

    def bohm_speed(self, mass_number: float = 1.0, bulk_mass_number: float = 1.0, bulk_charge_number: int = 1):
        """Normalized Bohm speed of an ion species (``kT_e / m_i``).

        In normalized units, the velocity unit squared is ``Z_b e Phi / (A_b m_H)``, so
        ``v_B² = T_e * (A_b / Z_b) / A``.
        """
        return float(np.sqrt(self.temperature * bulk_mass_number / bulk_charge_number / mass_number))

    def debye_length(self):
        """Normalized electron Debye length, ``sqrt(T_e / rho_e0)``."""
        return float(np.sqrt(self.temperature / self.density))
