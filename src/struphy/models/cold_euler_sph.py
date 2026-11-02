from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergySPH
from struphy.models.species import ParticleSpecies
from struphy.models.variables import SPHVariable
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_v_in_force_field import PushVinForceField
from struphy.propagators.push_vxb import PushVxB


class ColdEulerSPH(StruphyModel):
    r"""Pressureless (cold) Euler equations with external forces, discretized with smoothed particle hydrodynamics
    (SPH).

    The models of the SPH family inherit from this model: they call ``super().__init__()`` and extend the hooks of
    :meth:`~struphy.models.base.StruphyModel._setup` (species, propagators, variable links, scalars).
    Formerly named ``PressureLessSPH``.

    Parameters
    ----------
    base_units : BaseUnits, optional
        Reference units used to derive model normalization constants.
    charge_number : int, optional
        Species charge number (default is 1).
    mass_number : float, optional
        Species mass number (default is 1.0).
    epsilon : float, optional
        Scaling parameter of the forces (default is None, computed from the units).
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class Fluid(ParticleSpecies):
        """SPH fluid of the SPH family (named ``fluid`` in all models of the family)."""

        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
        ):
            self.var = SPHVariable()
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        """Propagators of the cold Euler equations: ``push_eta`` (advection), ``push_vxb`` (Lorentz force of the
        equilibrium magnetic field, nothing without one) and ``push_v`` (external force, nothing without force field
        or potential, set e.g. with ``model.propagators.push_v.potential = ...`` in the parameter file)."""

        def __init__(self):
            self.push_eta = PushEta()
            self.push_vxb = PushVxB()
            self.push_v = PushVinForceField()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    fluid: Fluid
    propagators: Propagators

    # names before the SPH family (old run metadata)
    _renamed_species = {"cold_fluid": "fluid"}

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):
        # 0. store input parameters (a model inheriting from ColdEulerSPH stores its own before calling
        # super().__init__)
        self._store_params(locals())

        # 1.-5. species, units, propagators, variable links and scalars, see StruphyModel._setup
        self._setup(base_units)

    def _create_species(self):
        """:meta private:"""
        self.fluid = self.Fluid(**self._particle_species_args())

    def _link_propagator_variables(self):
        """:meta private:"""
        self.propagators.push_eta.variables.var = self.fluid.var
        self.propagators.push_vxb.variables.ions = self.fluid.var
        self.propagators.push_v.variables.var = self.fluid.var

    def _create_scalars(self):
        """:meta private:"""
        return {"en_kin": KineticEnergySPH(self.fluid.var)}

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return None

    def post_allocate(self):
        pass

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "push_v.Options" in line:
                    new_file += ["# external force F = -grad(phi), e.g.:\n"]
                    new_file += ["# model.propagators.push_v.potential = equil.p0\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
        return params_path

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u}
            = \frac{1}{\varepsilon} \left( \mathbf{F} + \mathbf{u} \times \mathbf{B}_0 \right)

        where :math:`\mathbf{F}` is an external force field per unit charge, normalized like an electric field (e.g.
        :math:`\mathbf F = -\nabla \phi` with a given potential :math:`\phi`), :math:`\mathbf{B}_0` the magnetic
        field of the equilibrium (if any) and :math:`\varepsilon` the species parameter ``epsilon``.

        **Conserved quantities** (periodic boundaries or no outflow):

        Mass:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \rho \, \textnormal d \mathbf x = 0

        Kinetic energy, for :math:`\mathbf F = 0` (the Lorentz force does no work):

        .. math::

            \frac{\textnormal d}{\textnormal d t} \frac 12 \int \rho |\mathbf u|^2 \, \textnormal d \mathbf x = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""There is no characteristic velocity: the velocity unit :math:`\hat u` follows from the base units
        (:math:`\hat t = \hat x / \hat u`). The force field and the magnetic field are normalized like an electric
        and a magnetic field,

        .. math::

            \hat{F} = \hat{u} \hat{B}\,, \qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t}\,,

        where :math:`\hat \Omega_\textnormal{c} = q \hat B / m` is the cyclotron frequency of the species.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - SPH kinetic energy: :math:`\frac 12 \int \rho |\mathbf u|^2 \, dV` (``en_kin``)"""

    @classmethod
    def doc_long_description(cls):
        r"""ColdEulerSPH (formerly ``PressureLessSPH``) is a meshfree particle model for a pressureless fluid.
        It is primarily useful as a simple SPH benchmark or as a reduced
        particle transport model in a prescribed force field or potential.

        This is the parent model of the SPH family: :class:`~struphy.models.EulerSPH` adds the pressure force
        (and its child :class:`~struphy.models.ViscousEulerSPH` the viscous force), and
        :class:`~struphy.models.IncompressibleNavierStokesSPH` the incompressibility constraint and the viscous
        force; all inherit the species, propagators and scalars of this model."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a cold Euler SPH model:

        .. code-block:: python

            from struphy.models import ColdEulerSPH

            model = ColdEulerSPH()
            model.fluid.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - meshfree pressureless-flow benchmarks
        - particle transport with external forcing and in a background magnetic field
        - testing SPH particle loading and diagnostics"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - fluids with pressure forces (see :class:`~struphy.models.EulerSPH`)
        - FEEC-based grid discretizations
        - self-consistent electromagnetic plasma dynamics
        - viscous or thermal closures"""
