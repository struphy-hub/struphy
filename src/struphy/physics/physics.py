import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits

logger = logging.getLogger("struphy")


@dataclass
class ConstantsOfNature:
    e = 1.602176634e-19  # elementary charge (C)
    mH = 1.67262192369e-27  # proton mass (kg)
    mu0 = 1.25663706212e-6  # magnetic constant (N/A^2)
    eps0 = 8.8541878128e-12  # vacuum permittivity (F/m)
    kB = 1.380649e-23  # Boltzmann constant (J/K)
    c = 299792458  # speed of light (m/s)


class Units:
    """
    Colllects base units and derives other units from these. See :ref:`normalization`.
    """

    def __init__(self, base: BaseUnits = None):
        if base is None:
            base = BaseUnits()

        self._x = base.x
        self._B = base.B
        self._n = base.n * 1e20
        self._kBT = base.kBT

    @property
    def x(self):
        return self._x

    @property
    def B(self):
        return self._B

    @property
    def n(self):
        """Unit of particle number density in 1/m^3."""
        return self._n

    @property
    def kBT(self):
        return self._kBT

    @property
    def v(self):
        """Unit of velocity in m/s."""
        if not hasattr(self, "_v"):
            raise AttributeError("Must call Units.derive_units() to get full set of units.")
        return self._v

    @property
    def t(self):
        """Unit of time in s."""
        if not hasattr(self, "_t"):
            raise AttributeError("Must call Units.derive_units() to get full set of units.")
        return self._t

    @property
    def p(self):
        """Unit of pressure in Pa, equal to B^2/mu0 if velocity_scale='alfvén'."""
        if not hasattr(self, "_p"):
            raise AttributeError("Must call Units.derive_units() to get full set of units.")
        return self._p

    @property
    def rho(self):
        """Unit of mass density in kg/m^3."""
        if not hasattr(self, "_rho"):
            raise AttributeError("Must call Units.derive_units() to get full set of units.")
        return self._rho

    @property
    def j(self):
        """Unit of current density in A/m^2."""
        if not hasattr(self, "_j"):
            raise AttributeError("Must call Units.derive_units() to get full set of units.")
        return self._j

    def derive_units(self, velocity_scale: str = "light", A_bulk: int = None, Z_bulk: int = None, verbose=False):
        """Derive the remaining units from the base units, velocity scale and bulk species' A and Z."""

        con = ConstantsOfNature()

        # velocity (m/s)
        if velocity_scale is None:
            self._v = 1.0

        elif velocity_scale == "light":
            self._v = con.c

        elif velocity_scale == "alfvén":
            assert A_bulk is not None, 'Need bulk species to choose velocity scale "alfvén".'
            self._v = self.B / xp.sqrt(self.n * A_bulk * con.mH * con.mu0)

        elif velocity_scale == "cyclotron":
            assert Z_bulk is not None, 'Need bulk species to choose velocity scale "cyclotron".'
            assert A_bulk is not None, 'Need bulk species to choose velocity scale "cyclotron".'
            self._v = Z_bulk * con.e * self.B / (A_bulk * con.mH) * self.x

        elif velocity_scale == "thermal":
            assert A_bulk is not None, 'Need bulk species to choose velocity scale "thermal".'
            assert self.kBT is not None
            self._v = xp.sqrt(self.kBT * 1000 * con.e / (con.mH * A_bulk))

        # time (s)
        self._t = self.x / self.v

        # return if no bulk is present
        if A_bulk is None:
            self._p = None
            self._rho = None
            self._j = None
        else:
            # pressure (Pa), equal to B^2/mu0 if velocity_scale='alfvén'
            self._p = A_bulk * con.mH * self.n * self.v**2

            # mass density (kg/m^3)
            self._rho = A_bulk * con.mH * self.n

            # current density (A/m^2)
            self._j = con.e * self.n * self.v

        # print to screen
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            units_used = (
                " m",
                " T",
                " m⁻³",
                "keV",
                " m/s",
                " s",
                " bar",
                " kg/m³",
                " A/m²",
            )
            logger.info("")
            for (k, v), u in zip(self.__dict__.items(), units_used):
                if v is None:
                    logger.info(f"Unit of {k[1:]} not specified.")
                else:
                    logger.info(
                        f"Unit of {k[1:]}:".ljust(25),
                        "{:4.3e}".format(v) + u,
                    )
