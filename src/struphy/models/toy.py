from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from struphy.feec.projectors import L2Projector
from struphy.feec.variational_utilities import InternalEnergyEvaluator
from struphy.models.base import StruphyModel
from struphy.models.species import FieldSpecies, FluidSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.propagators import propagators_coupling, propagators_fields, propagators_markers
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()

rank = MPI.COMM_WORLD.Get_rank()


class Maxwell(StruphyModel):
    r"""Maxwell's equations in vacuum.

    :ref:`normalization`:

    .. math::

        \hat E = c \hat B\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \mathbf E}{\partial t} - \nabla\times\mathbf B = 0\,,

        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    """
    
    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")
        
        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field
        
        # define scalars for update_scalar_quantities
        self.add_scalar("electric energy")
        self.add_scalar("magnetic energy")
        self.add_scalar("total energy")

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        en_E = 0.5 * self.mass_ops.M1.dot_inner(
            self.em_fields.e_field.spline.vector, self.em_fields.e_field.spline.vector
        )
        en_B = 0.5 * self.mass_ops.M2.dot_inner(
            self.em_fields.b_field.spline.vector, self.em_fields.b_field.spline.vector
        )

        self.update_scalar("electric energy", en_E)
        self.update_scalar("magnetic energy", en_B)
        self.update_scalar("total energy", en_E + en_B)


class Vlasov(StruphyModel):
    r"""Vlasov equation in static background magnetic field.

    :ref:`normalization`:

    .. math::

        \hat v = \hat \Omega_\textnormal{c} \hat x\,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left(\mathbf{v}\times\mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushVxB`
    2. :class:`~struphy.propagators.propagators_markers.PushEta`
    """
    
    ## species

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_vxb = propagators_markers.PushVxB()
            self.push_eta = propagators_markers.PushEta()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}' ***")
            
        # 1. instantiate all species
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        
        # define scalars for update_scalar_quantities
        self.add_scalar("en_f", compute="from_particles", variable=self.kinetic_ions.var)

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "cyclotron"

    def allocate_helpers(self):
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        particles = self.kinetic_ions.var.particles
        self._tmp[0] = particles.markers_wo_holes[:, 6].dot(
            particles.markers_wo_holes[:, 3] ** 2
            + particles.markers_wo_holes[:, 4] ** 2
            + particles.markers_wo_holes[:, 5] ** 2,
        ) / (2 * particles.Np)

        self.update_scalar("en_f", self._tmp[0])


class GuidingCenter(StruphyModel):
    r"""Guiding-center equation in static background magnetic field.

    :ref:`normalization`:

    .. math::

        \hat v = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[\frac{1}{\epsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

    .. math::

        \mathbf{E}^* = -\epsilon \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \epsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,.

    Moreover,

    .. math::

        \epsilon = \frac{1 }{ \hat \Omega_{\textnormal{c}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterBxEstar`
    2. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterParallel`
    """
    
    ## species
    
    class KineticIons(KineticSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles5D")
            self.init_variables()
        
    ## propagators
    
    class Propagators:
        def __init__(self):
            self.push_bxe = propagators_markers.PushGuidingCenterBxEstar()
            self.push_parallel = propagators_markers.PushGuidingCenterParallel()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}' ***")
            
        # 1. instantiate all species
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()
        
        # 3. assign variables to propagators
        self.propagators.push_bxe.variables.ions = self.kinetic_ions.var
        self.propagators.push_parallel.variables.ions = self.kinetic_ions.var
        
        # define scalars for update_scalar_quantities
        self.add_scalar("en_fv", compute="from_particles", variable=self.kinetic_ions.var)
        self.add_scalar("en_fB", compute="from_particles", variable=self.kinetic_ions.var)
        self.add_scalar("en_tot", compute="from_particles", variable=self.kinetic_ions.var)

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "alfvén"
    
    def allocate_helpers(self):
        self._en_fv = xp.empty(1, dtype=float)
        self._en_fB = xp.empty(1, dtype=float)
        self._en_tot = xp.empty(1, dtype=float)
        self._n_lost_particles = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        particles = self.kinetic_ions.var.particles

        # particles' kinetic energy
        self._en_fv[0] = particles.markers[~particles.holes, 5].dot(
            particles.markers[~particles.holes, 3] ** 2,
        ) / (2.0 * particles.Np)

        particles.save_magnetic_background_energy()
        self._en_tot[0] = (
            particles.markers[~particles.holes, 5].dot(
                particles.markers[~particles.holes, 8],
            )
            / particles.Np
        )

        self._en_fB[0] = self._en_tot[0] - self._en_fv[0]

        self.update_scalar("en_fv", self._en_fv[0])
        self.update_scalar("en_fB", self._en_fB[0])
        self.update_scalar("en_tot", self._en_tot[0])

        self._n_lost_particles[0] = particles.n_lost_markers
        self.derham.comm.Allreduce(
            MPI.IN_PLACE,
            self._n_lost_particles,
            op=MPI.SUM,
        )


class ShearAlfven(StruphyModel):
    r"""ShearAlfven propagator from :class:`~struphy.models.fluid.LinearMHD` with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \hat U =  \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t}
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfven`

    :ref:`Model info <add_model>`:
    """

    ## species
    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self):
            self.velocity = FEECVariable(space="Hdiv")
            self.init_variables()

    class Propagators:
        def __init__(self) -> None:
            self.shear_alf = propagators_fields.ShearAlfven()

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        # project background magnetic field (2-form) and pressure (3-form)
        self._b_eq = self.derham.P["2"](
            [
                self.equil.b2_1,
                self.equil.b2_2,
                self.equil.b2_3,
            ]
        )

        # temporary vectors for scalar quantities
        self._tmp_b1 = self.derham.Vh["2"].zeros()
        self._tmp_b2 = self.derham.Vh["2"].zeros()

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        # Scalar variables to be saved during simulation
        self.add_scalar("en_tot")

        self.add_scalar("en_U", compute="from_field")
        self.add_scalar("en_B", compute="from_field")
        self.add_scalar("en_B_eq", compute="from_field")
        self.add_scalar("en_B_tot", compute="from_field")
        self.add_scalar("en_tot2", summands=["en_U", "en_B", "en_B_eq"])

    def update_scalar_quantities(self):
        # perturbed fields
        en_U = 0.5 * self.mass_ops.M2n.dot_inner(self.mhd.velocity.spline.vector, self.mhd.velocity.spline.vector)
        en_B = 0.5 * self.mass_ops.M2.dot_inner(
            self.em_fields.b_field.spline.vector, self.em_fields.b_field.spline.vector
        )

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_tot", en_U + en_B)

        # background fields
        self.mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        en_B0 = self._b_eq.inner(self._tmp_b1) / 2
        self.update_scalar("en_B_eq", en_B0)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector

        self.mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        en_Btot = self._tmp_b1.inner(self._tmp_b2) / 2

        self.update_scalar("en_B_tot", en_Btot)


class VariationalPressurelessFluid(StruphyModel):
    r"""Pressure-less fluid equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`

    :ref:`Model info <add_model>`:
    """

    ## species

    class Fluid(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.variat_dens = propagators_fields.VariationalDensityEvolve()
            self.variat_mom = propagators_fields.VariationalMomentumAdvection()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.fluid = self.Fluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.fluid.density
        self.propagators.variat_dens.variables.u = self.fluid.velocity
        self.propagators.variat_mom.variables.u = self.fluid.velocity

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        u = self.fluid.velocity.spline.vector
        en_U = 0.5 * self.mass_ops.WMM.massop.dot_inner(u, u)
        self.update_scalar("en_U", en_U)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='pressureless')\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class VariationalBarotropicFluid(StruphyModel):
    r"""Barotropic fluid equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A} \qquad \hat{\mathcal U} = \frac{\hat \rho}{2} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho))}{\partial \rho} = 0 \,.

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho/2`.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`

    :ref:`Model info <add_model>`:
    """

    ## species

    class Fluid(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.variat_dens = propagators_fields.VariationalDensityEvolve()
            self.variat_mom = propagators_fields.VariationalMomentumAdvection()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.fluid = self.Fluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.fluid.density
        self.propagators.variat_dens.variables.u = self.fluid.velocity
        self.propagators.variat_mom.variables.u = self.fluid.velocity

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_tot")

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        rho = self.fluid.density.spline.vector
        u = self.fluid.velocity.spline.vector

        en_U = 0.5 * self.mass_ops.WMM.massop.dot_inner(u, u)
        self.update_scalar("en_U", en_U)

        en_thermo = 0.5 * self.mass_ops.M3.dot_inner(rho, rho)
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo
        self.update_scalar("en_tot", en_tot)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='barotropic')\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class VariationalCompressibleFluid(StruphyModel):
    r"""Fully compressible fluid equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = K\,,\qquad \hat s = \hat \rho C_v \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial s} = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = 0 \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`

    :ref:`Model info <add_model>`:
    """

    ## species

    class Fluid(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.entropy = FEECVariable(space="L2")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.variat_dens = propagators_fields.VariationalDensityEvolve()
            self.variat_mom = propagators_fields.VariationalMomentumAdvection()
            self.variat_ent = propagators_fields.VariationalEntropyEvolve()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.fluid = self.Fluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.fluid.density
        self.propagators.variat_dens.variables.u = self.fluid.velocity
        self.propagators.variat_mom.variables.u = self.fluid.velocity
        self.propagators.variat_ent.variables.s = self.fluid.entropy
        self.propagators.variat_ent.variables.u = self.fluid.velocity

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_tot")

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        projV3 = L2Projector("L2", self._mass_ops)

        def f(e1, e2, e3):
            return 1

        f = xp.vectorize(f)
        self._integrator = projV3(f)

        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self.propagators.variat_ent.options.gamma)

    def update_scalar_quantities(self):
        rho = self.fluid.density.spline.vector
        u = self.fluid.velocity.spline.vector

        en_U = 0.5 * self.mass_ops.WMM.massop.dot_inner(u, u)
        self.update_scalar("en_U", en_U)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo
        self.update_scalar("en_tot", en_tot)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='full',\n"
                    ]
                    new_file += [
                        "                                                                              s=model.fluid.entropy)\n"
                    ]
                elif "variat_ent.Options" in line:
                    new_file += [
                        "model.propagators.variat_ent.options = model.propagators.variat_ent.Options(model='full',\n"
                    ]
                    new_file += [
                        "                                                                            rho=model.fluid.density)\n"
                    ]
                elif "entropy.add_background" in line:
                    new_file += ["model.fluid.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    def update_thermo_energy(self):
        """Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        """
        en_prop = self.propagators.variat_ent

        self._energy_evaluator.sf.vector = self.fluid.entropy.spline.vector
        self._energy_evaluator.rhof.vector = self.fluid.density.spline.vector
        sf_values = self._energy_evaluator.sf.eval_tp_fixed_loc(
            self._energy_evaluator.integration_grid_spans,
            self._energy_evaluator.integration_grid_bd,
            out=self._energy_evaluator._sf_values,
        )
        rhof_values = self._energy_evaluator.rhof.eval_tp_fixed_loc(
            self._energy_evaluator.integration_grid_spans,
            self._energy_evaluator.integration_grid_bd,
            out=self._energy_evaluator._rhof_values,
        )
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term * e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_ds)
        en_thermo = self._integrator.inner(en_prop._linear_form_dl_ds)
        self.update_scalar("en_thermo", en_thermo)
        return en_thermo

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        return xp.power(rho, self.propagators.variat_ent.options.gamma) * xp.exp(s / rho)


class Poisson(StruphyModel):
    r"""Weak discretization of Poisson's equation with diffusion matrix, stabilization
    and time-depedent right-hand side.

    :ref:`normalization`:

    .. math::

        \hat D = \frac{\hat n}{\hat x^2}\,,\qquad \hat \rho = \hat n \,.

    :ref:`Equations <gempic>`: Find :math:`\phi \in H^1` such that

    .. math::

        - \nabla \cdot D_0(\mathbf x) \nabla \phi + n_0(\mathbf x) \phi =  \rho(t, \mathbf x)\,,

    where :math:`n_0, \rho(t):\Omega \to \mathbb R` are real-valued functions, :math:`\rho(t)` parametrized with time :math:`t`,
    and :math:`D_0:\Omega \to \mathbb R^{3\times 3}` is a positive matrix.
    Boundary terms from integration by parts are assumed to vanish.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.TimeDependentSource`
    2. :class:`~struphy.propagators.propagators_fields.ImplicitDiffusion`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.source = FEECVariable(space="H1")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.source = propagators_fields.TimeDependentSource()
            self.poisson = propagators_fields.Poisson()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.source.variables.source = self.em_fields.source
        self.propagators.poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return None

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        pass

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles
        super().allocate_propagators()

        # # use setter to assign source
        # self.propagators.poisson.rho = self.mass_ops.M0.dot(self.em_fields.source.spline.vector)

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nSolving initial Poisson problem...")

        self.propagators.poisson(1.0)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Done.")

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "poisson.Options" in line:
                    new_file += [
                        "model.propagators.poisson.options = model.propagators.poisson.Options(rho=model.em_fields.source)\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class DeterministicParticleDiffusion(StruphyModel):
    r"""Diffusion equation discretized with a deterministic particle method;
    the solution is :math:`L^2`-projected onto :math:`V^0 \subset H^1` to compute the flux.

    :ref:`normalization`:

    .. math::

        \hat D := \frac{\hat x^2}{\hat t } \,.

    :ref:`Equations <gempic>`: Find :math:`u:\mathbb R\times \Omega\to \mathbb R^+` such that

    .. math::

        \frac{\partial u}{\partial t} +  \nabla \cdot\left(\mathbf F(u) u\right) = 0\,, \qquad \mathbf F(u) = -\mathbb D\,\frac{\nabla u}{u}\,,

    where :math:`\mathbb D: \Omega\to \mathbb R^{3\times 3 }` is a positive diffusion matrix.
    At the moment only matrices of the form :math:`D*Id` are implemented, where :math:`D > 0` is a positive diffusion coefficient.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushDeterministicDiffusion`

    :ref:`Model info <add_model>`:
    """

    ## species

    class Hydrogen(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles3D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.det_diff = propagators_markers.PushDeterministicDiffusion()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.hydrogen = self.Hydrogen()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.det_diff.variables.var = self.hydrogen.var

        # define scalars for update_scalar_quantities
        # self.add_scalar("electric energy")
        # self.add_scalar("magnetic energy")
        # self.add_scalar("total energy")

    @property
    def bulk_species(self):
        return self.hydrogen

    @property
    def velocity_scale(self):
        return None

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        pass


class RandomParticleDiffusion(StruphyModel):
    r"""Diffusion equation discretized with a (random) particle method;
    the diffusion is computed through a Wiener process.

    :ref:`normalization`:

    .. math::

        \hat D := \frac{\hat x^2}{\hat t } \,.

    :ref:`Equations <gempic>`: Find :math:`u:\mathbb R\times \Omega\to \mathbb R^+` such that

    .. math::

        \frac{\partial u}{\partial t} -  D \, \Delta u = 0\,,

    where :math:`D > 0` is a positive diffusion coefficient.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushRandomDiffusion`

    :ref:`Model info <add_model>`:
    """

    ## species

    class Hydrogen(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles3D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.rand_diff = propagators_markers.PushRandomDiffusion()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.hydrogen = self.Hydrogen()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.rand_diff.variables.var = self.hydrogen.var

        # define scalars for update_scalar_quantities
        # self.add_scalar("electric energy")
        # self.add_scalar("magnetic energy")
        # self.add_scalar("total energy")

    @property
    def bulk_species(self):
        return self.hydrogen

    @property
    def velocity_scale(self):
        return None

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        pass


class PressureLessSPH(StruphyModel):
    r"""Pressureless fluid discretized with smoothed particle hydrodynamics

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) = - \nabla \phi_0 \,,

    where :math:`\phi_0` is a static external potential.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`

    This is discretized by particles going in straight lines.
    """

    ## species

    class ColdFluid(ParticleSpecies):
        def __init__(self):
            self.var = SPHVariable()
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_eta = propagators_markers.PushEta()
            self.push_v = propagators_markers.PushVinEfield()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.cold_fluid = self.ColdFluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.cold_fluid.var
        self.propagators.push_v.variables.var = self.cold_fluid.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_kin", compute="from_particles", variable=self.cold_fluid.var)

    @property
    def bulk_species(self):
        return self.cold_fluid

    @property
    def velocity_scale(self):
        return None

    # @staticmethod
    # def diagnostics_dct():
    #     dct = {}
    #     dct["projected_density"] = "L2"
    #     return dct

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        particles = self.cold_fluid.var.particles
        valid_parts = particles.markers_wo_holes_and_ghost
        en_kin = valid_parts[:, 6].dot(valid_parts[:, 3] ** 2 + valid_parts[:, 4] ** 2 + valid_parts[:, 5] ** 2) / (
            2.0 * particles.Np
        )

        self.update_scalar("en_kin", en_kin)

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "push_v.Options" in line:
                    new_file += ["phi = equil.p0\n"]
                    new_file += ["model.propagators.push_v.options = model.propagators.push_v.Options(phi=phi)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


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
