import logging

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.stencil import StencilVector

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")
rank = MPI.COMM_WORLD.Get_rank()


class HasegawaWakatani(StruphyModel):
    r"""Hasegawa-Wakatani equations in 2D.

    :ref:`normalization`:

    .. math::

        \hat u = \hat v_\textnormal{th}\,,\qquad \hat \phi = \hat u\, \hat x \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial n}{\partial t} = C (\phi - n) - [\phi, n] - \kappa\, \partial_y \phi + \nu\, \nabla^{2N} n\,,
        \\[2mm]
        &\frac{\partial \omega}{\partial t} = C (\phi - n) - [\phi, \omega] + \nu\, \nabla^{2N} \omega \,,
        \\[3mm]
        &\Delta \phi = \omega\,,

    where :math:`[\phi, n] = \partial_x \phi \partial_y n - \partial_y \phi \partial_x n`, :math:`C = C(x, y)` and
    :math:`\kappa` and :math:`\nu` are constants (at the moment only :math:`N=1` is available).

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Poisson`
    2. :class:`~struphy.propagators.propagators_fields.HasegawaWakatani`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class Plasma(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="H1")
            self.vorticity = FEECVariable(space="H1")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.poisson = propagators_fields.Poisson()
            self.hw = propagators_fields.HasegawaWakatani()

    ## abstract methods

    def __init__(self):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.plasma = self.Plasma()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.poisson.variables.phi = self.em_fields.phi
        self.propagators.hw.variables.n = self.plasma.density
        self.propagators.hw.variables.omega = self.plasma.vorticity

        # define scalars for update_scalar_quantities

    @property
    def bulk_species(self):
        return self.plasma

    @property
    def velocity_scale(self):
        return "alfvén"

    def update_rho(self):
        omega = self.plasma.vorticity.spline.vector
        self._rho = Propagator.mass_ops.M0.dot(omega, out=self._rho)
        self._rho.update_ghost_regions()
        return self._rho

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._rho: StencilVector = Propagator.derham.V0.zeros()
        self.update_rho()

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("\nINITIAL POISSON SOLVE:")

        self.update_rho()
        self.propagators.poisson(1.0)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("Done.")

    def update_scalar_quantities(self):
        pass

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "hw.Options" in line:
                    new_file += [
                        "model.propagators.hw.options = model.propagators.hw.Options(phi=model.em_fields.phi)\n",
                    ]
                elif "vorticity.add_background" in line:
                    new_file += ["model.plasma.density.add_background(FieldsBackground())\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
