import numpy as np
from mpi4py import MPI
from psydac.linalg.block import BlockVector
from psydac.linalg.stencil import StencilVector

from struphy.models.base import StruphyModel
from struphy.models.species import FieldSpecies, FluidSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.polar.basic import PolarVector
from struphy.propagators import propagators_coupling, propagators_fields, propagators_markers

rank = MPI.COMM_WORLD.Get_rank()


class LinearMHD(StruphyModel):
    r"""Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \hat U = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,,
        \\[2mm]
        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        = (\nabla \times \tilde{\mathbf{B}})\times \mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times \tilde{\mathbf{B}} \,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}})
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfven`
    2. :class:`~struphy.propagators.propagators_fields.Magnetosonic`
    """
    
    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.shear_alf = propagators_fields.ShearAlfven()
            self.mag_sonic = propagators_fields.Magnetosonic()

    ## abstract methods

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

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure
        
        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_p")
        self.add_scalar("en_B")
        self.add_scalar("en_p_eq")
        self.add_scalar("en_B_eq")
        self.add_scalar("en_B_tot")
        self.add_scalar("en_tot")

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        self._ones = self.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_b1: BlockVector = self.derham.Vh["2"].zeros()  # TODO: replace derham.Vh dict by class
        self._tmp_b2: BlockVector = self.derham.Vh["2"].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        en_U = 0.5 * self.mass_ops.M2n.dot_inner(
            self.mhd.velocity.spline.vector,
            self.mhd.velocity.spline.vector,
        )
        en_B = 0.5 * self.mass_ops.M2.dot_inner(
            self.em_fields.b_field.spline.vector,
            self.em_fields.b_field.spline.vector,
        )
        en_p = self.mhd.pressure.spline.vector.inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p)
        self.update_scalar("en_tot", en_U + en_B + en_p)

        # background fields
        self.mass_ops.M2.dot(self.projected_equil.b2, apply_bc=False, out=self._tmp_b1)

        en_B0 = self.projected_equil.b2.inner(self._tmp_b1) / 2
        en_p0 = self.projected_equil.p3.inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_B_eq", en_B0)
        self.update_scalar("en_p_eq", en_p0)

        # total magnetic field
        self.projected_equil.b2.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector

        self.mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.inner(self._tmp_b2) / 2

        self.update_scalar("en_B_tot", en_Btot)

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "mag_sonic.Options" in line:
                    new_file += ["model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options(b_field=model.em_fields.b_field)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class LinearExtendedMHDuniform(StruphyModel):
    r"""Linear extended MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).
    For uniform background conditions only.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,,
        \\[2mm]
        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 \,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + \frac{5}{3}\,p_{0}\nabla\cdot \tilde{\mathbf{U}}=0\,,
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 - \frac{1}{\varepsilon} \frac{\nabla\times \tilde{\mathbf{B}}}{\rho_0}\times \mathbf{B}_0 \right)
        = 0\,.

    where

    .. math::

        \varepsilon = \frac{1}{\hat \Omega_{\textnormal{c}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfvenB1`
    2. :class:`~struphy.propagators.propagators_fields.Hall`
    3. :class:`~struphy.propagators.propagators_fields.MagnetosonicUniform`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hcurl")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.shear_alf = propagators_fields.ShearAlfvenB1()
            self.hall = propagators_fields.Hall()
            self.mag_sonic = propagators_fields.MagnetosonicUniform()

    ## abstract methods

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

        self.propagators.hall.variables.b = self.em_fields.b_field

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_p")
        self.add_scalar("en_B")
        self.add_scalar("en_p_eq")
        self.add_scalar("en_B_eq")
        self.add_scalar("en_B_tot")
        self.add_scalar("en_tot")
        self.add_scalar("helicity")

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        self._b_eq = self.projected_equil.b1
        self._a_eq = self.projected_equil.a1
        self._p_eq = self.projected_equil.p3

        self._ones = self.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_b1: BlockVector = self.derham.Vh["1"].zeros()  # TODO: replace derham.Vh dict by class
        self._tmp_b2: BlockVector = self.derham.Vh["1"].zeros()

        # adjust coupling parameters
        epsilon = self.mhd.equation_params.epsilon

        if abs(epsilon - 1) < 1e-6:
            self.mhd.equation_params.epsilon = 1.0

    def update_scalar_quantities(self):
        # perturbed fields
        u = self.mhd.velocity.spline.vector
        p = self.mhd.pressure.spline.vector
        b = self.em_fields.b_field.spline.vector

        en_U = 0.5 * self.mass_ops.M2n.dot_inner(u, u)
        b1 = self.mass_ops.M1.dot(b, out=self._tmp_b1)
        en_B = 0.5 * b.inner(b1)
        helicity = 2.0 * self._a_eq.inner(b1)
        en_p_i = p.inner(self._ones) / (5.0 / 3.0 - 1.0)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p_i)
        self.update_scalar("helicity", helicity)
        self.update_scalar("en_tot", en_U + en_B + en_p_i)

        # background fields
        b1 = self.mass_ops.M1.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        en_B0 = self._b_eq.inner(b1) / 2.0
        en_p0 = self._p_eq.inner(self._ones) / (5.0 / 3.0 - 1.0)

        self.update_scalar("en_B_eq", en_B0)
        self.update_scalar("en_p_eq", en_p0)

        # total magnetic field
        b1 = self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += b

        b2 = self.mass_ops.M1.dot(b1, apply_bc=False, out=self._tmp_b2)
        en_Btot = b1.inner(b2) / 2.0

        self.update_scalar("en_B_tot", en_Btot)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "hall.Options" in line:
                    new_file += [
                        "model.propagators.hall.options = model.propagators.hall.Options(epsilon_from=model.mhd)\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class ColdPlasma(StruphyModel):
    r"""Cold plasma model.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = c \hat B \,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{1}{n_0} &\frac{\partial \mathbf j}{\partial t} = \frac{1}{\varepsilon} \mathbf E + \frac{1}{\varepsilon n_0} \mathbf j \times \mathbf B_0\,,
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,
        \\[2mm]
        -&\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon} \mathbf j \,,

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background and

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,, \qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_fields.OhmCold`
    3. :class:`~struphy.propagators.propagators_fields.JxBCold`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class Electrons(FluidSpecies):
        def __init__(self):
            self.current = FEECVariable(space="Hcurl")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()
            self.ohm = propagators_fields.OhmCold()
            self.jxb = propagators_fields.JxBCold()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.electrons = self.Electrons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        self.propagators.ohm.variables.j = self.electrons.current
        self.propagators.ohm.variables.e = self.em_fields.e_field

        self.propagators.jxb.variables.j = self.electrons.current

        # define scalars for update_scalar_quantities
        self.add_scalar("electric energy")
        self.add_scalar("magnetic energy")
        self.add_scalar("kinetic energy")
        self.add_scalar("total energy")

    @property
    def bulk_species(self):
        return self.electrons

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        self._alpha = self.electrons.equation_params.alpha

    def update_scalar_quantities(self):
        e = self.em_fields.e_field.spline.vector
        b = self.em_fields.b_field.spline.vector
        j = self.electrons.current.spline.vector

        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        en_B = 0.5 * self.mass_ops.M2.dot_inner(b, b)
        en_J = 0.5 * self._alpha**2 * self.mass_ops.M1ninv.dot_inner(j, j)

        self.update_scalar("electric energy", en_E)
        self.update_scalar("magnetic energy", en_B)
        self.update_scalar("kinetic energy", en_J)
        self.update_scalar("total energy", en_E + en_B + en_J)


class ViscoresistiveMHD(StruphyModel):
    r"""Full (non-linear) visco-resistive MHD equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = \frac{\hat{\mathbf B}^2}{\hat \rho \mu_0 (\gamma-1)} \,,\qquad \hat s = \hat \rho\ \textrm{ln}\left(\frac{\hat{\mathbf B}^2}{\mu_0 (\gamma -1) \hat{\rho}}\right) \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial s} + \mathbf B \times \nabla \times \mathbf B - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \mathbf u \right) = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = \frac{1}{T}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2\right) \,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B = 0 \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`,
    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalMagFieldEvolve`
    5. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    6. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"rho3": "L2", "s3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["mhd_rho3", "mhd_uv"],
            propagators_fields.VariationalMomentumAdvection: ["mhd_uv"],
            propagators_fields.VariationalEntropyEvolve: ["mhd_s3", "mhd_uv"],
            propagators_fields.VariationalMagFieldEvolve: ["b2", "mhd_uv"],
            propagators_fields.VariationalViscosity: ["mhd_s3", "mhd_uv"],
            propagators_fields.VariationalResistivity: ["mhd_s3", "b2"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import H1vecMassMatrix_density, InternalEnergyEvaluator
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_entropy = params["fluid"]["mhd"]["options"]["VariationalEntropyEvolve"]["lin_solver"]
        nonlin_solver_entropy = params["fluid"]["mhd"]["options"]["VariationalEntropyEvolve"]["nonlin_solver"]
        lin_solver_magfield = params["em_fields"]["options"]["VariationalMagFieldEvolve"]["lin_solver"]
        nonlin_solver_magfield = params["em_fields"]["options"]["VariationalMagFieldEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["nonlin_solver"]
        lin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["lin_solver"]
        nonlin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["nonlin_solver"]
        if "linearize_current" in params["fluid"]["mhd"]["options"]["VariationalResistivity"].keys():
            self._linearize_current = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["linearize_current"]
        else:
            self._linearize_current = False
        self._gamma = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        self._alpha = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["alpha"]
        self._eta = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta"]
        self._eta_a = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta_a"]
        model = "full"

        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "s": self.pointer["mhd_s3"],
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
            "energy_evaluator": self._energy_evaluator,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        self._kwargs[propagators_fields.VariationalEntropyEvolve] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_entropy,
            "nonlin_solver": nonlin_solver_entropy,
            "energy_evaluator": self._energy_evaluator,
        }

        self._kwargs[propagators_fields.VariationalMagFieldEvolve] = {
            "model": model,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_magfield,
            "nonlin_solver": nonlin_solver_magfield,
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mu": self._mu,
            "mu_a": self._mu_a,
            "alpha": self._alpha,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
            "energy_evaluator": self._energy_evaluator,
        }

        self._kwargs[propagators_fields.VariationalResistivity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_a": self._eta_a,
            "lin_solver": lin_solver_resistivity,
            "nonlin_solver": nonlin_solver_resistivity,
            "linearize_current": self._linearize_current,
            "energy_evaluator": self._energy_evaluator,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag")
        self.add_scalar("en_tot")
        self.add_scalar("dens_tot")
        self.add_scalar("entr_tot")
        self.add_scalar("tot_div_B")

        # temporary vectors for scalar quantities
        self._tmp_div_B = self.derham.Vh_pol["3"].zeros()
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        def f(e1, e2, e3):
            return 1

        f = np.vectorize(f)
        self._integrator = projV3(f, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["mhd_uv"], self.pointer["mhd_uv"])
        self.update_scalar("en_U", en_U)

        en_mag = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])
        self.update_scalar("en_mag", en_mag)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar("en_tot", en_tot)

        dens_tot = self._ones.inner(self.pointer["mhd_rho3"])
        self.update_scalar("dens_tot", dens_tot)
        entr_tot = self._ones.inner(self.pointer["mhd_s3"])
        self.update_scalar("entr_tot", entr_tot)

        div_B = self.derham.div.dot(self.pointer["b2"], out=self._tmp_div_B)
        L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        self.update_scalar("tot_div_B", L2_div_B)

    def update_thermo_energy(self):
        """Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        """
        en_prop = self._propagators[0]
        self._energy_evaluator.sf.vector = self.pointer["mhd_s3"]
        self._energy_evaluator.rhof.vector = self.pointer["mhd_rho3"]
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
        e = self._energy_evaluator.ener
        ener_values = en_prop._proj_rho2_metric_term * e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_drho)
        en_thermo = self._integrator.inner(en_prop._linear_form_dl_drho)
        self.update_scalar("en_thermo", en_thermo)
        return en_thermo


class ViscousFluid(StruphyModel):
    r"""Full (non-linear) viscous Navier-Stokes equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = \frac{\hat{\mathbf B}^2}{\hat \rho \mu_0 (\gamma-1)} \,,\qquad \hat s = \hat \rho\ \textrm{ln}\left(\frac{\hat{\mathbf B}^2}{\mu_0 (\gamma -1) \hat{\rho}}\right) \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial s} - \nabla \cdot \left((\mu +\mu_a(\mathbf x)) \nabla \mathbf u\right) = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = \frac{1}{T}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 \right) \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.
    and :math:`\mu_a(\mathbf x)` is an artificial viscosity coefficient.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["fluid"]["fluid"] = {"rho3": "L2", "s3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "fluid"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["fluid_rho3", "fluid_uv"],
            propagators_fields.VariationalMomentumAdvection: ["fluid_uv"],
            propagators_fields.VariationalEntropyEvolve: ["fluid_s3", "fluid_uv"],
            propagators_fields.VariationalViscosity: ["fluid_s3", "fluid_uv"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        from struphy.feec.variational_utilities import H1vecMassMatrix_density, InternalEnergyEvaluator

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_entropy = params["fluid"]["fluid"]["options"]["VariationalEntropyEvolve"]["lin_solver"]
        nonlin_solver_entropy = params["fluid"]["fluid"]["options"]["VariationalEntropyEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["fluid"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["fluid"]["options"]["VariationalViscosity"]["nonlin_solver"]

        self._gamma = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["fluid"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["fluid"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        model = "full"

        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "s": self.pointer["fluid_s3"],
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
            "energy_evaluator": self._energy_evaluator,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        self._kwargs[propagators_fields.VariationalEntropyEvolve] = {
            "model": model,
            "rho": self.pointer["fluid_rho3"],
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_entropy,
            "nonlin_solver": nonlin_solver_entropy,
            "energy_evaluator": self._energy_evaluator,
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": model,
            "gamma": self._gamma,
            "rho": self.pointer["fluid_rho3"],
            "mu": self._mu,
            "mu_a": self._mu_a,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
            "energy_evaluator": self._energy_evaluator,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_tot")
        self.add_scalar("dens_tot")
        self.add_scalar("entr_tot")

        # temporary vectors for scalar quantities
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        def f(e1, e2, e3):
            return 1

        f = np.vectorize(f)
        self._integrator = projV3(f, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["fluid_uv"], self.pointer["fluid_uv"])
        self.update_scalar("en_U", en_U)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo
        self.update_scalar("en_tot", en_tot)

        dens_tot = self._ones.inner(self.pointer["fluid_rho3"])
        self.update_scalar("dens_tot", dens_tot)
        entr_tot = self._ones.inner(self.pointer["fluid_s3"])
        self.update_scalar("entr_tot", entr_tot)

    def update_thermo_energy(self):
        """Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        """
        en_prop = self._propagators[0]
        self._energy_evaluator.sf.vector = self.pointer["fluid_s3"]
        self._energy_evaluator.rhof.vector = self.pointer["fluid_rho3"]
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
        e = self._energy_evaluator.ener
        ener_values = en_prop._proj_rho2_metric_term * e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_drho)
        en_thermo = self._integrator.inner(en_prop._linear_form_dl_drho)
        self.update_scalar("en_thermo", en_thermo)
        return en_thermo


class ViscoresistiveMHD_with_p(StruphyModel):
    r"""Full (non-linear) visco-resistive MHD equations, with the pressure variable discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \frac{1}{\gamma -1} \nabla p + \mathbf B \times \nabla \times \mathbf B - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \mathbf u \right) = 0 \,,
        \\[4mm]
        &\partial_t p + u \cdot \nabla p + \gamma p \nabla \cdot u = \frac{1}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2\right) \,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalPBEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    5. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"rho3": "L2", "p3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["mhd_rho3", "mhd_uv"],
            propagators_fields.VariationalMomentumAdvection: ["mhd_uv"],
            propagators_fields.VariationalPBEvolve: ["mhd_p3", "b2", "mhd_uv"],
            propagators_fields.VariationalViscosity: ["mhd_p3", "mhd_uv"],
            propagators_fields.VariationalResistivity: ["mhd_p3", "b2"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import H1vecMassMatrix_density
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalPBEvolve"]["lin_solver"]
        nonlin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalPBEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["nonlin_solver"]
        lin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["lin_solver"]
        nonlin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["nonlin_solver"]
        if "linearize_current" in params["fluid"]["mhd"]["options"]["VariationalResistivity"].keys():
            self._linearize_current = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["linearize_current"]
        else:
            self._linearize_current = False
        self._gamma = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        self._alpha = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["alpha"]
        self._eta = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta"]
        self._eta_a = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta_a"]
        model = "full_p"

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        self._kwargs[propagators_fields.VariationalPBEvolve] = {
            "model": model,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_magfield,
            "nonlin_solver": nonlin_solver_magfield,
            "gamma": self._gamma,
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mu": self._mu,
            "mu_a": self._mu_a,
            "alpha": self._alpha,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
        }

        self._kwargs[propagators_fields.VariationalResistivity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_a": self._eta_a,
            "lin_solver": lin_solver_resistivity,
            "nonlin_solver": nonlin_solver_resistivity,
            "linearize_current": self._linearize_current,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag")
        self.add_scalar("en_tot")
        self.add_scalar("dens_tot")
        self.add_scalar("tot_div_B")

        # temporary vectors for scalar quantities
        self._tmp_div_B = self.derham.Vh_pol["3"].zeros()
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        self._integrator = projV3(self.domain.jacobian_det, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["mhd_uv"], self.pointer["mhd_uv"])
        self.update_scalar("en_U", en_U)

        en_mag = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])
        self.update_scalar("en_mag", en_mag)

        en_thermo = self.mass_ops.M3.dot_inner(self.pointer["mhd_p3"], self._integrator) / (self._gamma - 1.0)
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar("en_tot", en_tot)

        dens_tot = self._ones.inner(self.pointer["mhd_rho3"])
        self.update_scalar("dens_tot", dens_tot)

        div_B = self.derham.div.dot(self.pointer["b2"], out=self._tmp_div_B)
        L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        self.update_scalar("tot_div_B", L2_div_B)

    @staticmethod
    def diagnostics_dct():
        dct = {}

        dct["div_u"] = "L2"
        dct["u2"] = "Hdiv"
        return dct

    __diagnostics__ = diagnostics_dct()


class ViscoresistiveLinearMHD(StruphyModel):
    r"""Linear visco-resistive MHD equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \tilde{\rho} + \nabla \cdot ( \rho_0 \tilde{\mathbf u} ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho_0 \tilde{\mathbf u}) + \frac{1}{\gamma -1} \nabla \tilde{p} + \mathbf B_0 \times \nabla \times \tilde{\mathbf B} + \tilde{\mathbf B} \times \nabla \times \mathbf B_0 - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \tilde{\mathbf u} \right) = 0 \,,
        \\[4mm]
        &\partial_t \tilde{p} + \tilde{\mathbf u} \cdot \nabla p_0 + \gamma p_0 \nabla \cdot \tilde{\mathbf u} = \frac{1}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \tilde{\mathbf u}|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \tilde{\mathbf B}|^2\right) \,,
        \\[4mm]
        &\partial_t \tilde{\mathbf B} + \nabla \times ( \mathbf B_0 \times \tilde{\mathbf u} ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \tilde{\mathbf B} = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalPBEvolve`
    3. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    4. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"rho3": "L2", "p3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["mhd_rho3", "mhd_uv"],
            propagators_fields.VariationalPBEvolve: ["mhd_p3", "b2", "mhd_uv"],
            propagators_fields.VariationalViscosity: ["mhd_p3", "mhd_uv"],
            propagators_fields.VariationalResistivity: ["mhd_p3", "b2"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import H1vecMassMatrix_density
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalPBEvolve"]["lin_solver"]
        nonlin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalPBEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["nonlin_solver"]
        lin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["lin_solver"]
        nonlin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["nonlin_solver"]
        if "linearize_current" in params["fluid"]["mhd"]["options"]["VariationalResistivity"].keys():
            self._linearize_current = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["linearize_current"]
        else:
            self._linearize_current = False
        self._gamma = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        self._alpha = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["alpha"]
        self._eta = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta"]
        self._eta_a = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta_a"]
        model = "linear"

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalPBEvolve] = {
            "model": model,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_magfield,
            "nonlin_solver": nonlin_solver_magfield,
            "gamma": self._gamma,
            "div_u": self.pointer["div_u"],
            "u2": self.pointer["u2"],
            "bt2": self.pointer["bt2"],
            "pt3": self.pointer["pt3"],
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": "linear_p",
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mu": self._mu,
            "mu_a": self._mu_a,
            "alpha": self._alpha,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
        }

        self._kwargs[propagators_fields.VariationalResistivity] = {
            "model": "linear_p",
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_a": self._eta_a,
            "lin_solver": lin_solver_resistivity,
            "nonlin_solver": nonlin_solver_resistivity,
            "linearize_current": self._linearize_current,
            "pt3": self.pointer["pt3"],
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag_1")
        self.add_scalar("en_mag_2")
        self.add_scalar("en_tot")

        # self.add_scalar("dens_tot")
        # self.add_scalar("tot_div_B")

        self.add_scalar("en_tot_l1")
        self.add_scalar("en_thermo_l1")
        self.add_scalar("en_mag_l1")

        # temporary vectors for scalar quantities
        self._tmp_div_B = self.derham.Vh_pol["3"].zeros()
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        self._integrator = projV3(self.domain.jacobian_det, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["mhd_uv"], self.pointer["mhd_uv"])
        self.update_scalar("en_U", en_U)

        en_mag1 = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])
        self.update_scalar("en_mag_1", en_mag1)

        en_mag2 = self.mass_ops.M2.dot_inner(self.pointer["bt2"], self.projected_equil.b2)
        self.update_scalar("en_mag_2", en_mag2)

        en_thermo = self.mass_ops.M3.dot_inner(self.pointer["pt3"], self._integrator) / (self._gamma - 1.0)
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo + en_mag1 + en_mag2
        self.update_scalar("en_tot", en_tot)

        # dens_tot = self._ones.inner(self.pointer["mhd_rho3"])
        # self.update_scalar("dens_tot", dens_tot)

        # div_B = self.derham.div.dot(self.pointer["b2"], out=self._tmp_div_B)
        # L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        # self.update_scalar("tot_div_B", L2_div_B)

        en_thermo_l1 = self.mass_ops.M3.dot_inner(self.pointer["mhd_p3"], self._integrator) / (self._gamma - 1.0)
        self.update_scalar("en_thermo_l1", en_thermo_l1)

        en_mag_l1 = self.mass_ops.M2.dot_inner(self.pointer["b2"], self.projected_equil.b2)
        self.update_scalar("en_mag_l1", en_mag_l1)

        en_tot_l1 = en_thermo_l1 + en_mag_l1
        self.update_scalar("en_tot_l1", en_tot_l1)

    @staticmethod
    def diagnostics_dct():
        dct = {}
        dct["bt2"] = "Hdiv"
        dct["pt3"] = "L2"
        dct["div_u"] = "L2"
        dct["u2"] = "Hdiv"
        return dct

    __diagnostics__ = diagnostics_dct()


class ViscoresistiveDeltafMHD(StruphyModel):
    r""":math:`\delta f` visco-resistive MHD equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \tilde{\rho} + \nabla \cdot ( (\tilde{\rho}+\rho_0) \tilde{\mathbf u} ) = 0 \,,
        \\[4mm]
        &\partial_t ((\tilde{\rho}+\rho_0) \tilde{\mathbf u}) + \nabla \cdot ((\tilde{\rho}+\rho_0) \tilde{\mathbf u} \otimes \tilde{\mathbf u}) + \frac{1}{\gamma -1} \nabla \tilde{p} + \mathbf B_0 \times \nabla \times \tilde{\mathbf B} + \tilde{\mathbf B} \times \nabla \times \mathbf B_0 +  \tilde{\mathbf B} \times \nabla \times \tilde{\mathbf B} - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \tilde{\mathbf u} \right) = 0 \,,
        \\[4mm]
        &\partial_t \tilde{p} + \tilde{\mathbf u} \cdot \nabla (\tilde{p} + p_0) + \gamma (\tilde{p} + p_0) \nabla \cdot \tilde{\mathbf u} = \frac{1}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \tilde{\mathbf u}|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \tilde{\mathbf B}|^2\right) \,,
        \\[4mm]
        &\partial_t \tilde{\mathbf B} + \nabla \times ( (\tilde{\mathbf B} + \mathbf B_0) \times \tilde{\mathbf u} ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \tilde{\mathbf B} = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalPBEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    5. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"rho3": "L2", "p3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["mhd_rho3", "mhd_uv"],
            propagators_fields.VariationalMomentumAdvection: ["mhd_uv"],
            propagators_fields.VariationalPBEvolve: ["mhd_p3", "b2", "mhd_uv"],
            propagators_fields.VariationalViscosity: ["mhd_p3", "mhd_uv"],
            propagators_fields.VariationalResistivity: ["mhd_p3", "b2"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import H1vecMassMatrix_density
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalPBEvolve"]["lin_solver"]
        nonlin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalPBEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["nonlin_solver"]
        lin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["lin_solver"]
        nonlin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["nonlin_solver"]
        if "linearize_current" in params["fluid"]["mhd"]["options"]["VariationalResistivity"].keys():
            self._linearize_current = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["linearize_current"]
        else:
            self._linearize_current = False
        self._gamma = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        self._alpha = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["alpha"]
        self._eta = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta"]
        self._eta_a = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta_a"]
        model = "deltaf"

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        self._kwargs[propagators_fields.VariationalPBEvolve] = {
            "model": model,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_magfield,
            "nonlin_solver": nonlin_solver_magfield,
            "gamma": self._gamma,
            "bt2": self.pointer["bt2"],
            "pt3": self.pointer["pt3"],
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": "full_p",
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mu": self._mu,
            "mu_a": self._mu_a,
            "alpha": self._alpha,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
        }

        self._kwargs[propagators_fields.VariationalResistivity] = {
            "model": "delta_p",
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_a": self._eta_a,
            "lin_solver": lin_solver_resistivity,
            "nonlin_solver": nonlin_solver_resistivity,
            "linearize_current": self._linearize_current,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag_1")
        self.add_scalar("en_mag_2")
        self.add_scalar("en_tot")

        # self.add_scalar("dens_tot")
        # self.add_scalar("tot_div_B")

        self.add_scalar("en_tot_l1")
        self.add_scalar("en_thermo_l1")
        self.add_scalar("en_mag_l1")

        # temporary vectors for scalar quantities
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        self._integrator = projV3(self.domain.jacobian_det, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["mhd_uv"], self.pointer["mhd_uv"])
        self.update_scalar("en_U", en_U)

        en_mag1 = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])
        self.update_scalar("en_mag_1", en_mag1)

        en_mag2 = self.mass_ops.M2.dot_inner(self.pointer["bt2"], self.projected_equil.b2)
        self.update_scalar("en_mag_2", en_mag2)

        en_thermo = self.mass_ops.M3.dot_inner(self.pointer["pt3"], self._integrator) / (self._gamma - 1.0)
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo + en_mag1 + en_mag2
        self.update_scalar("en_tot", en_tot)

        # dens_tot = self._ones.inner(self.pointer["mhd_rho3"])
        # self.update_scalar("dens_tot", dens_tot)

        # div_B = self.derham.div.dot(self.pointer["b2"], out=self._tmp_div_B)
        # L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        # self.update_scalar("tot_div_B", L2_div_B)

        en_thermo_l1 = self.mass_ops.M3.dot_inner(self.pointer["mhd_p3"], self._integrator) / (self._gamma - 1.0)
        self.update_scalar("en_thermo_l1", en_thermo_l1)

        en_mag_l1 = self.mass_ops.M2.dot_inner(self.pointer["b2"], self.projected_equil.b2)
        self.update_scalar("en_mag_l1", en_mag_l1)

        en_tot_l1 = en_thermo_l1 + en_mag_l1
        self.update_scalar("en_tot_l1", en_tot_l1)

    @staticmethod
    def diagnostics_dct():
        dct = {}
        dct["bt2"] = "Hdiv"
        dct["pt3"] = "L2"
        dct["div_u"] = "L2"
        dct["u2"] = "Hdiv"
        return dct

    __diagnostics__ = diagnostics_dct()


class ViscoresistiveMHD_with_q(StruphyModel):
    r"""Full (non-linear) visco-resistive MHD equations, with the q variable (square root of the pressure) discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \frac{2q}{\gamma -1} \nabla q + \mathbf B \times \nabla \times \mathbf B - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \mathbf u \right) = 0 \,,
        \\[4mm]
        &\partial_t q + \cdot(\nabla q \mathbf u) + (\gamma/2 -1) q \nabla \cdot u = \frac{2 q}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2\right) \,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalQBEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    5. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"rho3": "L2", "q3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["mhd_rho3", "mhd_uv"],
            propagators_fields.VariationalMomentumAdvection: ["mhd_uv"],
            propagators_fields.VariationalQBEvolve: ["mhd_q3", "b2", "mhd_uv"],
            propagators_fields.VariationalViscosity: ["mhd_q3", "mhd_uv"],
            propagators_fields.VariationalResistivity: ["mhd_q3", "b2"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import H1vecMassMatrix_density
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalQBEvolve"]["lin_solver"]
        nonlin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalQBEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["nonlin_solver"]
        lin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["lin_solver"]
        nonlin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["nonlin_solver"]
        if "linearize_current" in params["fluid"]["mhd"]["options"]["VariationalResistivity"].keys():
            self._linearize_current = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["linearize_current"]
        else:
            self._linearize_current = False
        self._gamma = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        self._alpha = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["alpha"]
        self._eta = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta"]
        self._eta_a = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta_a"]
        model = "full_q"

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        self._kwargs[propagators_fields.VariationalQBEvolve] = {
            "model": model,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_magfield,
            "nonlin_solver": nonlin_solver_magfield,
            "gamma": self._gamma,
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mu": self._mu,
            "mu_a": self._mu_a,
            "alpha": self._alpha,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
        }

        self._kwargs[propagators_fields.VariationalResistivity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_a": self._eta_a,
            "lin_solver": lin_solver_resistivity,
            "nonlin_solver": nonlin_solver_resistivity,
            "linearize_current": self._linearize_current,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag")
        self.add_scalar("en_tot")
        self.add_scalar("dens_tot")
        self.add_scalar("tot_div_B")

        # temporary vectors for scalar quantities
        self._tmp_div_B = self.derham.Vh_pol["3"].zeros()
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        self._integrator = projV3(self.domain.jacobian_det, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["mhd_uv"], self.pointer["mhd_uv"])
        self.update_scalar("en_U", en_U)

        en_mag = 0.5 * self._mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])
        self.update_scalar("en_mag", en_mag)

        en_thermo = 1 / (self._gamma - 1) * self._mass_ops.M3.dot_inner(self.pointer["mhd_q3"], self.pointer["mhd_q3"])
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar("en_tot", en_tot)

        dens_tot = self._ones.inner(self.pointer["mhd_rho3"])
        self.update_scalar("dens_tot", dens_tot)

        div_B = self.derham.div.dot(self.pointer["b2"], out=self._tmp_div_B)
        L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        self.update_scalar("tot_div_B", L2_div_B)

    @staticmethod
    def diagnostics_dct():
        dct = {}

        dct["div_u"] = "L2"
        dct["u2"] = "Hdiv"
        return dct

    __diagnostics__ = diagnostics_dct()


class ViscoresistiveLinearMHD_with_q(StruphyModel):
    r"""Linear visco-resistive MHD equations, with the q variable (square root of the pressure), discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \tilde{\rho} + \nabla \cdot ( \rho_0 \tilde{\mathbf u} ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho_0 \tilde{\mathbf u}) + \frac{2 q_0}{\gamma -1} \nabla \tilde{q} +  \frac{2 \tilde{q}}{\gamma -1} \nabla q_0 + \mathbf B_0 \times \nabla \times \tilde{\mathbf B} + \tilde{\mathbf B} \times \nabla \times \mathbf B_0 - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \tilde{\mathbf u} \right) = 0 \,,
        \\[4mm]
        &\partial_t \tilde{q} + \cdot(\nabla q_0 \mathbf u) + (\gamma/2 -1) q_0 \nabla \cdot u = 0 \,,
        \\[4mm]
        &\partial_t \tilde{\mathbf B} + \nabla \times ( \mathbf B_0 \times \tilde{\mathbf u} ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \tilde{\mathbf B} = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalQBEvolve`
    3. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    4. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"rho3": "L2", "q3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["mhd_rho3", "mhd_uv"],
            propagators_fields.VariationalQBEvolve: ["mhd_q3", "b2", "mhd_uv"],
            propagators_fields.VariationalViscosity: ["mhd_q3", "mhd_uv"],
            propagators_fields.VariationalResistivity: ["mhd_q3", "b2"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import H1vecMassMatrix_density
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalQBEvolve"]["lin_solver"]
        nonlin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalQBEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["nonlin_solver"]
        lin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["lin_solver"]
        nonlin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["nonlin_solver"]
        if "linearize_current" in params["fluid"]["mhd"]["options"]["VariationalResistivity"].keys():
            self._linearize_current = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["linearize_current"]
        else:
            self._linearize_current = False
        self._gamma = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        self._alpha = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["alpha"]
        self._eta = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta"]
        self._eta_a = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta_a"]
        model = "linear_q"

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalQBEvolve] = {
            "model": model,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_magfield,
            "nonlin_solver": nonlin_solver_magfield,
            "gamma": self._gamma,
            "div_u": self.pointer["div_u"],
            "u2": self.pointer["u2"],
            "bt2": self.pointer["bt2"],
            "qt3": self.pointer["qt3"],
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mu": self._mu,
            "mu_a": self._mu_a,
            "alpha": self._alpha,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
            "pt3": self.pointer["qt3"],
        }

        self._kwargs[propagators_fields.VariationalResistivity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_a": self._eta_a,
            "lin_solver": lin_solver_resistivity,
            "nonlin_solver": nonlin_solver_resistivity,
            "linearize_current": self._linearize_current,
            "pt3": self.pointer["qt3"],
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        # self.add_scalar("en_thermo_1")
        # self.add_scalar("en_thermo_2")
        # self.add_scalar("en_mag_1")
        # self.add_scalar("en_mag_2")
        self.add_scalar("en_tot")

        # self.add_scalar("dens_tot")
        # self.add_scalar("tot_div_B")

        # self.add_scalar("en_tot_l1")
        # self.add_scalar("en_thermo_l1")
        # self.add_scalar("en_mag_l1")

        # temporary vectors for scalar quantities
        self._tmp_div_B = self.derham.Vh_pol["3"].zeros()
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        self._integrator = projV3(self.domain.jacobian_det, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["mhd_uv"], self.pointer["mhd_uv"])
        self.update_scalar("en_U", en_U)

        en_mag1 = self._mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])
        # self.update_scalar("en_mag_1", en_mag1)

        en_mag2 = self._mass_ops.M2.dot_inner(self.pointer["bt2"], self.projected_equil.b2)
        # self.update_scalar("en_mag_2", en_mag2)

        en_th_1 = 1 / (self._gamma - 1) * self._mass_ops.M3.dot_inner(self.pointer["mhd_q3"], self.pointer["mhd_q3"])
        # self.update_scalar("en_thermo_1", en_th_1)

        en_th_2 = 2 / (self._gamma - 1) * self._mass_ops.M3.dot_inner(self.pointer["qt3"], self.projected_equil.q3)
        # self.update_scalar("en_thermo_2", en_th_2)

        en_tot = en_U + en_th_1 + en_th_2 + en_mag1 + en_mag2
        self.update_scalar("en_tot", en_tot)

        # dens_tot = self._ones.dot(self.pointer["mhd_rho3"])
        # self.update_scalar("dens_tot", dens_tot)

        # div_B = self.derham.div.dot(self.pointer["b2"], out=self._tmp_div_B)
        # L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        # self.update_scalar("tot_div_B", L2_div_B)

        # en_thermo_l1 = self._integrator.dot(self.mass_ops.M3.dot(self.pointer["mhd_p3"])) / (self._gamma - 1.0)
        # self.update_scalar("en_thermo_l1", en_thermo_l1)

        # wb2 = self._mass_ops.M2.dot(self.pointer["b2"], out=self._tmp_wb2)
        # en_mag_l1 = wb2.dot(self.projected_equil.b2)
        # self.update_scalar("en_mag_l1", en_mag_l1)

        # en_tot_l1 = en_thermo_l1 + en_mag_l1
        # self.update_scalar("en_tot_l1", en_tot_l1)

    @staticmethod
    def diagnostics_dct():
        dct = {}
        dct["bt2"] = "Hdiv"
        dct["qt3"] = "L2"
        dct["div_u"] = "L2"
        dct["u2"] = "Hdiv"
        return dct

    __diagnostics__ = diagnostics_dct()


class ViscoresistiveDeltafMHD_with_q(StruphyModel):
    r"""Linear visco-resistive MHD equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \tilde{\rho} + \nabla \cdot ( \rho_0 \tilde{\mathbf u} ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho_0 \tilde{\mathbf u}) + \frac{2 q_0}{\gamma -1} \nabla \tilde{q} +  \frac{2 \tilde{q}}{\gamma -1} \nabla q_0 + \frac{2 \tilde{q}}{\gamma -1} \nabla \tilde{q} + \mathbf B_0 \times \nabla \times \tilde{\mathbf B} + \tilde{\mathbf B} \times \nabla \times \mathbf B_0 - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \tilde{\mathbf u} \right) = 0 \,,
        \\[4mm]
        &\partial_t \tilde{q} + \cdot(\nabla (q_0 + \tilde{q}) \mathbf u) + (\gamma/2 -1) (q_0 + \tilde{q}) \nabla \cdot u = 0 \,,
        \\[4mm]
        &\partial_t \tilde{\mathbf B} + \nabla \times ( \mathbf (B_0 + \tilde{\mathbf B}) \times \tilde{\mathbf u} ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \tilde{\mathbf B} = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalQBEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    5. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"rho3": "L2", "q3": "L2", "uv": "H1vec"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.VariationalDensityEvolve: ["mhd_rho3", "mhd_uv"],
            propagators_fields.VariationalMomentumAdvection: ["mhd_uv"],
            propagators_fields.VariationalQBEvolve: ["mhd_q3", "b2", "mhd_uv"],
            propagators_fields.VariationalViscosity: ["mhd_q3", "mhd_uv"],
            propagators_fields.VariationalResistivity: ["mhd_q3", "b2"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import H1vecMassMatrix_density
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["mhd"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalQBEvolve"]["lin_solver"]
        nonlin_solver_magfield = params["fluid"]["mhd"]["options"]["VariationalQBEvolve"]["nonlin_solver"]
        lin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["lin_solver"]
        nonlin_solver_viscosity = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["nonlin_solver"]
        lin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["lin_solver"]
        nonlin_solver_resistivity = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["nonlin_solver"]
        if "linearize_current" in params["fluid"]["mhd"]["options"]["VariationalResistivity"].keys():
            self._linearize_current = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["linearize_current"]
        else:
            self._linearize_current = False
        self._gamma = params["fluid"]["mhd"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        self._mu = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu"]
        self._mu_a = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["mu_a"]
        self._alpha = params["fluid"]["mhd"]["options"]["VariationalViscosity"]["physics"]["alpha"]
        self._eta = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta"]
        self._eta_a = params["fluid"]["mhd"]["options"]["VariationalResistivity"]["physics"]["eta_a"]
        model = "deltaf_q"

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": model,
            "gamma": self._gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        self._kwargs[propagators_fields.VariationalQBEvolve] = {
            "model": model,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_magfield,
            "nonlin_solver": nonlin_solver_magfield,
            "gamma": self._gamma,
            "div_u": self.pointer["div_u"],
            "u2": self.pointer["u2"],
            "bt2": self.pointer["bt2"],
            "qt3": self.pointer["qt3"],
        }

        self._kwargs[propagators_fields.VariationalViscosity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "mu": self._mu,
            "mu_a": self._mu_a,
            "alpha": self._alpha,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_viscosity,
            "nonlin_solver": nonlin_solver_viscosity,
            "pt3": self.pointer["qt3"],
        }

        self._kwargs[propagators_fields.VariationalResistivity] = {
            "model": model,
            "rho": self.pointer["mhd_rho3"],
            "gamma": self._gamma,
            "eta": self._eta,
            "eta_a": self._eta_a,
            "lin_solver": lin_solver_resistivity,
            "nonlin_solver": nonlin_solver_resistivity,
            "linearize_current": self._linearize_current,
            "pt3": self.pointer["qt3"],
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo_1")
        self.add_scalar("en_thermo_2")
        self.add_scalar("en_mag_1")
        self.add_scalar("en_mag_2")
        self.add_scalar("en_tot")

        # self.add_scalar("dens_tot")
        # self.add_scalar("tot_div_B")

        # self.add_scalar("en_tot_l1")
        # self.add_scalar("en_thermo_l1")
        # self.add_scalar("en_mag_l1")

        # temporary vectors for scalar quantities
        self._tmp_div_B = self.derham.Vh_pol["3"].zeros()
        tmp_dof = self.derham.Vh_pol["3"].zeros()
        projV3 = L2Projector("L2", self.mass_ops)

        self._integrator = projV3(self.domain.jacobian_det, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_scalar_quantities(self):
        # Update mass matrix
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["mhd_uv"], self.pointer["mhd_uv"])
        self.update_scalar("en_U", en_U)

        en_mag1 = 0.5 * self._mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])
        self.update_scalar("en_mag_1", en_mag1)

        en_mag2 = 0.5 * self._mass_ops.M2.dot_inner(self.pointer["bt2"], self.projected_equil.b2)
        self.update_scalar("en_mag_2", en_mag2)

        en_th_1 = 1 / (self._gamma - 1) * self._mass_ops.M3.dot_inner(self.pointer["mhd_q3"], self.pointer["mhd_q3"])
        self.update_scalar("en_thermo_1", en_th_1)

        en_th_2 = 2 / (self._gamma - 1) * self._mass_ops.M3.dot_inner(self.pointer["qt3"], self.projected_equil.q3)
        self.update_scalar("en_thermo_2", en_th_2)

        en_tot = en_U + en_th_1 + en_th_2 + en_mag1 + en_mag2
        self.update_scalar("en_tot", en_tot)

        # dens_tot = self._ones.dot(self.pointer["mhd_rho3"])
        # self.update_scalar("dens_tot", dens_tot)

        # div_B = self.derham.div.dot(self.pointer["b2"], out=self._tmp_div_B)
        # L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        # self.update_scalar("tot_div_B", L2_div_B)

        # en_thermo_l1 = self._integrator.dot(self.mass_ops.M3.dot(self.pointer["mhd_p3"])) / (self._gamma - 1.0)
        # self.update_scalar("en_thermo_l1", en_thermo_l1)

        # wb2 = self._mass_ops.M2.dot(self.pointer["b2"], out=self._tmp_wb2)
        # en_mag_l1 = wb2.dot(self.projected_equil.b2)
        # self.update_scalar("en_mag_l1", en_mag_l1)

        # en_tot_l1 = en_thermo_l1 + en_mag_l1
        # self.update_scalar("en_tot_l1", en_tot_l1)

    @staticmethod
    def diagnostics_dct():
        dct = {}
        dct["bt2"] = "Hdiv"
        dct["qt3"] = "L2"
        dct["div_u"] = "L2"
        dct["u2"] = "Hdiv"
        return dct

    __diagnostics__ = diagnostics_dct()


class EulerSPH(StruphyModel):
    r"""Euler equations discretized with smoothed particle hydrodynamics (SPH).

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{th} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
        \partial_t \rho + \nabla \cdot (\rho \mathbf u) &= 0\,,
        \\[2mm]
        \rho(\partial_t \mathbf u + \mathbf u \cdot \nabla \mathbf u) &= - \nabla \left(\rho^2 \frac{\partial \mathcal U(\rho, S)}{\partial \rho} \right)\,,
        \\[2mm]
        \partial_t S + \mathbf u \cdot \nabla S &= 0\,,
        \end{align}

    where :math:`S` denotes the entropy per unit mass.
    The internal energy per unit mass can be defined in two ways:

    .. math::

        \mathrm{"isothermal:"}\qquad &\mathcal U(\rho, S) = \kappa(S) \log \rho\,.

        \mathrm{"polytropic:"}\qquad &\mathcal U(\rho, S) = \kappa(S) \frac{\rho^{\gamma - 1}}{\gamma - 1}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVxB`
    3. :class:`~struphy.propagators.propagators_markers.PushVinSPHpressure`
    """

    ## species

    class EulerFluid(ParticleSpecies):
        def __init__(self):
            self.var = SPHVariable()
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, with_B0: bool = True):
            self.push_eta = propagators_markers.PushEta()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            self.push_sph_p = propagators_markers.PushVinSPHpressure()

    ## abstract methods

    def __init__(self, with_B0: bool = True):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        self.with_B0 = with_B0

        # 1. instantiate all species
        self.euler_fluid = self.EulerFluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.euler_fluid.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.euler_fluid.var
        self.propagators.push_sph_p.variables.fluid = self.euler_fluid.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_kin", compute="from_sph", variable=self.euler_fluid.var)

    @property
    def bulk_species(self):
        return self.euler_fluid

    @property
    def velocity_scale(self):
        return "thermal"

    def allocate_helpers(self):
        pass

    # @staticmethod
    # def diagnostics_dct():
    #     dct = {}
    #     dct["projected_density"] = "L2"
    #     return dct

    def update_scalar_quantities(self):
        particles = self.euler_fluid.var.particles
        valid_markers = particles.markers_wo_holes_and_ghost
        en_kin = valid_markers[:, 6].dot(
            valid_markers[:, 3] ** 2 + valid_markers[:, 4] ** 2 + valid_markers[:, 5] ** 2
        ) / (2.0 * particles.Np)
        self.update_scalar("en_kin", en_kin)

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "push_vxb.Options" in line:
                    new_file += ["if model.with_B0:\n"]
                    new_file += ["    " + line]
                elif "set_save_data" in line:
                    new_file += ["\nkd_plot = KernelDensityPlot()\n"]
                    new_file += ["model.euler_fluid.set_save_data(kernel_density_plots=(kd_plot,))\n"]
                elif "base_units = BaseUnits" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


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
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

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

    def allocate_helpers(self):
        self._rho: StencilVector = self.derham.Vh["0"].zeros()
        self.update_rho()

    def update_scalar_quantities(self):
        pass

    def update_rho(self):
        omega = self.plasma.vorticity.spline.vector
        self._rho = self.mass_ops.M0.dot(omega, out=self._rho)
        self._rho.update_ghost_regions()
        return self._rho

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        # initialize fields and particles
        super().allocate_propagators()

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nINITIAL POISSON SOLVE:")

        self.update_rho()
        self.propagators.poisson(1.0)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Done.")

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
                        "model.propagators.hw.options = model.propagators.hw.Options(phi=model.em_fields.phi)\n"
                    ]
                elif "vorticity.add_background" in line:
                    new_file += ["model.plasma.density.add_background(FieldsBackground())\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
