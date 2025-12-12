from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import ModelTypes
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)

rank = MPI.COMM_WORLD.Get_rank()


class TwoFluidQuasiNeutralToy(StruphyModel):
    r"""Linearized, quasi-neutral two-fluid model with zero electron inertia.

    :ref:`normalization`:

    .. math::

        \hat u = \hat v_\textnormal{th}\,,\qquad  e\hat \phi = m \hat v_\textnormal{th}^2\,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{\partial \mathbf u}{\partial t} &= - \nabla \phi + \frac{\mathbf u \times \mathbf B_0}{\varepsilon} + \nu \Delta \mathbf u + \mathbf f\,,
        \\[2mm]
        0 &= \nabla \phi - \frac{\mathbf u_e \times \mathbf B_0}{\varepsilon} + \nu_e \Delta \mathbf u_e + \mathbf f_e \,,
        \\[3mm]
        \nabla & \cdot (\mathbf u - \mathbf u_e) = 0\,,

    where :math:`\mathbf B_0` is a static magnetic field and :math:`\mathbf f, \mathbf f_e` are given forcing terms,
    and with the normalization parameter

    .. math::

        \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.TwoFluidQuasiNeutralFull`

    :ref:`Model info <add_model>`:

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    model_type: ModelTypes = "Fluid"

    ## species

    class EMfields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="L2")
            self.init_variables()

    class Ions(FluidSpecies):
        def __init__(self):
            self.u = FEECVariable(space="Hdiv")
            self.init_variables()

    class Electrons(FluidSpecies):
        def __init__(self):
            self.u = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.qn_full = propagators_fields.TwoFluidQuasiNeutralFull()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMfields()
        self.ions = self.Ions()
        self.electrons = self.Electrons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.qn_full.variables.u = self.ions.u
        self.propagators.qn_full.variables.ue = self.electrons.u
        self.propagators.qn_full.variables.phi = self.em_fields.phi

        # define scalars for update_scalar_quantities

    @property
    def bulk_species(self):
        return self.ions

    @property
    def velocity_scale(self):
        return "thermal"

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        pass

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "BaseUnits()" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
