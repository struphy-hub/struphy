
import numpy as np
from dataclasses import dataclass
from mpi4py import MPI

from struphy.models.base import StruphyModel
from struphy.propagators.base import Propagator
from struphy.propagators import propagators_coupling, propagators_fields, propagators_markers
from struphy.models.species import KineticSpecies, FluidSpecies, FieldSpecies
from struphy.models.variables import Variable, FEECVariable, PICVariable, SPHVariable

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

    :ref:`Model info <add_model>`:
    """
    ## species
    
    @dataclass
    class EMFields(FieldSpecies):
        e_field: FEECVariable = FEECVariable(name="e_field", space="Hcurl")
        b_field: FEECVariable = FEECVariable(name="b_field", space="Hdiv")
    
    ## propagators
    
    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()

    ## abstract methods

    def __init__(self):        
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")
        
        # 1. instantiate all species, variables
        self.em_fields = self.EMFields()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()
        
        # 3. assign variables to propagators
        self.propagators.maxwell.set_variables(
            e = self.em_fields.e_field,
            b = self.em_fields.b_field,
            )
        
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
        en_E = 0.5 * self.mass_ops.M1.dot_inner(self.em_fields.e_field.spline.vector, self.em_fields.e_field.spline.vector)
        en_B = 0.5 * self.mass_ops.M2.dot_inner(self.em_fields.b_field.spline.vector, self.em_fields.b_field.spline.vector)

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

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["kinetic"]["ions"] = "Particles6D"
        return dct

    @staticmethod
    def bulk_species():
        return "ions"

    @staticmethod
    def velocity_scale():
        return "cyclotron"

    @staticmethod
    def propagators_dct():
        return {
            propagators_markers.PushVxB: ["ions"],
            propagators_markers.PushEta: ["ions"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        # prelim
        ions_params = self.kinetic["ions"]["params"]

        # project magnetic background
        self._b_eq = self.derham.P["2"](
            [
                self.equil.b2_1,
                self.equil.b2_2,
                self.equil.b2_3,
            ]
        )

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushVxB] = {
            "algo": ions_params["options"]["PushVxB"]["algo"],
            "kappa": 1.0,
            "b2": self._b_eq,
            "b2_add": None,
        }

        self._kwargs[propagators_markers.PushEta] = {"algo": ions_params["options"]["PushEta"]["algo"]}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_f", compute="from_particles", species="ions")

        # MPI operations needed for scalar variables
        self._tmp = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        self._tmp[0] = self.pointer["ions"].markers_wo_holes[:, 6].dot(
            self.pointer["ions"].markers_wo_holes[:, 3] ** 2
            + self.pointer["ions"].markers_wo_holes[:, 4] ** 2
            + self.pointer["ions"].markers_wo_holes[:, 5] ** 2,
        ) / (2 * self.pointer["ions"].Np)

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

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["kinetic"]["ions"] = "Particles5D"
        return dct

    @staticmethod
    def bulk_species():
        return "ions"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_markers.PushGuidingCenterBxEstar: ["ions"],
            propagators_markers.PushGuidingCenterParallel: ["ions"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        # prelim
        ions_params = self.kinetic["ions"]["params"]
        epsilon = self.equation_params["ions"]["epsilon"]

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushGuidingCenterBxEstar] = {
            "epsilon": epsilon,
            "algo": ions_params["options"]["PushGuidingCenterBxEstar"]["algo"],
        }

        self._kwargs[propagators_markers.PushGuidingCenterParallel] = {
            "epsilon": epsilon,
            "algo": ions_params["options"]["PushGuidingCenterParallel"]["algo"],
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_fv", compute="from_particles", species="ions")
        self.add_scalar("en_fB", compute="from_particles", species="ions")
        self.add_scalar("en_tot", compute="from_particles", species="ions")
        self.add_scalar("n_lost_particles", compute="from_particles", species="ions")

        # MPI operations needed for scalar variables
        self._en_fv = np.empty(1, dtype=float)
        self._en_fB = np.empty(1, dtype=float)
        self._en_tot = np.empty(1, dtype=float)
        self._n_lost_particles = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # particles' kinetic energy

        self._en_fv[0] = self.pointer["ions"].markers[~self.pointer["ions"].holes, 5].dot(
            self.pointer["ions"].markers[~self.pointer["ions"].holes, 3] ** 2,
        ) / (2.0 * self.pointer["ions"].Np)

        self.pointer["ions"].save_magnetic_background_energy()
        self._en_tot[0] = (
            self.pointer["ions"]
            .markers[~self.pointer["ions"].holes, 5]
            .dot(
                self.pointer["ions"].markers[~self.pointer["ions"].holes, 8],
            )
            / self.pointer["ions"].Np
        )

        self._en_fB[0] = self._en_tot[0] - self._en_fv[0]

        self._n_lost_particles[0] = self.pointer["ions"].n_lost_markers

        self.update_scalar("en_fv", self._en_fv[0])
        self.update_scalar("en_fB", self._en_fB[0])
        self.update_scalar("en_tot", self._en_tot[0])
        self.update_scalar("n_lost_particles", self._n_lost_particles[0])


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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["b2"] = "Hdiv"
        dct["fluid"]["mhd"] = {"u2": "Hdiv"}
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {propagators_fields.ShearAlfven: ["mhd_u2", "b2"]}

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        from struphy.polar.basic import PolarVector

        # extract necessary parameters
        alfven_solver = params["fluid"]["mhd"]["options"]["ShearAlfven"]["solver"]
        alfven_algo = params["fluid"]["mhd"]["options"]["ShearAlfven"]["algo"]

        # project background magnetic field (2-form) and pressure (3-form)
        self._b_eq = self.derham.P["2"](
            [
                self.equil.b2_1,
                self.equil.b2_2,
                self.equil.b2_3,
            ]
        )

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.ShearAlfven] = {
            "u_space": "Hdiv",
            "solver": alfven_solver,
            "algo": alfven_algo,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        # self.add_scalar('en_U')
        # self.add_scalar('en_B')
        # self.add_scalar('en_B_eq')
        # self.add_scalar('en_B_tot')
        self.add_scalar("en_tot")

        self.add_scalar("en_U", compute="from_field")
        self.add_scalar("en_B", compute="from_field")
        self.add_scalar("en_B_eq", compute="from_field")
        self.add_scalar("en_B_tot", compute="from_field")
        self.add_scalar("en_tot2", summands=["en_U", "en_B", "en_B_eq"])

        # temporary vectors for scalar quantities
        self._tmp_b1 = self.derham.Vh["2"].zeros()
        self._tmp_b2 = self.derham.Vh["2"].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        en_U = 0.5 * self.mass_ops.M2n.dot_inner(self.pointer["mhd_u2"], self.pointer["mhd_u2"])
        en_B = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b2"], self.pointer["b2"])

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_tot", en_U + en_B)

        # background fields
        self.mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        en_B0 = self._b_eq.inner(self._tmp_b1) / 2
        self.update_scalar("en_B_eq", en_B0)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer["b2"]

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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["fluid"]["fluid"] = {"rho3": "L2", "uv": "H1vec"}
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
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.mass import WeightedMassOperator
        from struphy.feec.variational_utilities import H1vecMassMatrix_density

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        # Initialize mass matrix
        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]

        gamma = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": "pressureless",
            "gamma": gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")

    def update_scalar_quantities(self):
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["fluid_uv"], self.pointer["fluid_uv"])
        self.update_scalar("en_U", en_U)


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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}
        dct["fluid"]["fluid"] = {"rho3": "L2", "uv": "H1vec"}
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
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        from struphy.feec.variational_utilities import H1vecMassMatrix_density

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        # Initialize mass matrix
        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]

        gamma = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {
            "model": "barotropic",
            "gamma": gamma,
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_density,
            "nonlin_solver": nonlin_solver_density,
        }

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {
            "mass_ops": self.WMM,
            "lin_solver": lin_solver_momentum,
            "nonlin_solver": nonlin_solver_momentum,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_tot")

    def update_scalar_quantities(self):
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["fluid_uv"], self.pointer["fluid_uv"])
        self.update_scalar("en_U", en_U)

        en_thermo = 0.5 * self.mass_ops.M3.dot_inner(self.pointer["fluid_rho3"], self.pointer["fluid_rho3"])
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo
        self.update_scalar("en_tot", en_tot)


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

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        # Initialize mass matrix
        self.WMM = H1vecMassMatrix_density(self.derham, self.mass_ops, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["lin_solver"]
        nonlin_solver_momentum = params["fluid"]["fluid"]["options"]["VariationalMomentumAdvection"]["nonlin_solver"]
        lin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["lin_solver"]
        nonlin_solver_density = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["nonlin_solver"]
        lin_solver_entropy = params["fluid"]["fluid"]["options"]["VariationalEntropyEvolve"]["lin_solver"]
        nonlin_solver_entropy = params["fluid"]["fluid"]["options"]["VariationalEntropyEvolve"]["nonlin_solver"]

        self._gamma = params["fluid"]["fluid"]["options"]["VariationalDensityEvolve"]["physics"]["gamma"]
        model = "full"

        from struphy.feec.variational_utilities import InternalEnergyEvaluator

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

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_tot")

        # temporary vectors for scalar quantities
        projV3 = L2Projector("L2", self._mass_ops)

        def f(e1, e2, e3):
            return 1

        f = np.vectorize(f)
        self._integrator = projV3(f)

    def update_scalar_quantities(self):
        en_U = 0.5 * self.WMM.massop.dot_inner(self.pointer["fluid_uv"], self.pointer["fluid_uv"])
        self.update_scalar("en_U", en_U)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo
        self.update_scalar("en_tot", en_tot)

    def update_thermo_energy(self):
        """Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        """
        en_prop = self._propagators[2]

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
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term * e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_ds)
        en_thermo = self._integrator.inner(en_prop._linear_form_dl_ds)
        self.update_scalar("en_thermo", en_thermo)
        return en_thermo

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        return np.power(rho, self._gamma) * np.exp(s / rho)


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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["phi"] = "H1"
        dct["em_fields"]["source"] = "H1"
        return dct

    @staticmethod
    def bulk_species():
        return None

    @staticmethod
    def velocity_scale():
        return None

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.TimeDependentSource: ["source"],
            propagators_fields.ImplicitDiffusion: ["phi"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        super().__init__(params, comm=comm, clone_config=clone_config)

        # extract necessary parameters
        model_params = params["em_fields"]["options"]["ImplicitDiffusion"]["model"]
        solver_params = params["em_fields"]["options"]["ImplicitDiffusion"]["solver"]
        omega = params["em_fields"]["options"]["TimeDependentSource"]["omega"]
        hfun = params["em_fields"]["options"]["TimeDependentSource"]["hfun"]

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.TimeDependentSource] = {
            "omega": omega,
            "hfun": hfun,
        }

        self._kwargs[propagators_fields.ImplicitDiffusion] = {
            "sigma_1": model_params["sigma_1"],
            "stab_mat": model_params["stab_mat"],
            "diffusion_mat": model_params["diffusion_mat"],
            "rho": self.pointer["source"],
            "solver": solver_params,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

    def update_scalar_quantities(self):
        pass


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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["kinetic"]["species1"] = "Particles3D"
        return dct

    @staticmethod
    def bulk_species():
        return "species1"

    @staticmethod
    def velocity_scale():
        return None

    @staticmethod
    def propagators_dct():
        return {propagators_markers.PushDeterministicDiffusion: ["species1"]}

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        super().__init__(params, comm=comm, clone_config=clone_config)

        # prelim
        params = self.kinetic["species1"]["params"]
        algo = params["options"]["PushDeterministicDiffusion"]["algo"]
        diffusion_coefficient = params["options"]["PushDeterministicDiffusion"]["diffusion_coefficient"]

        # # project magnetic background
        # self._b_eq = self.derham.P['2']([self.equil.b2_1,
        #                                  self.equil.b2_2,
        #                                  self.equil.b2_3])

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushDeterministicDiffusion] = {
            "algo": algo,
            "bc_type": params["markers"]["bc"],
            "diffusion_coefficient": diffusion_coefficient,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_f")

        # MPI operations needed for scalar variables
        self._tmp = np.empty(1, dtype=float)

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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["kinetic"]["species1"] = "Particles3D"
        return dct

    @staticmethod
    def bulk_species():
        return "species1"

    @staticmethod
    def velocity_scale():
        return None

    @staticmethod
    def propagators_dct():
        return {propagators_markers.PushRandomDiffusion: ["species1"]}

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        super().__init__(params, comm=comm, clone_config=clone_config)

        # prelim
        species1_params = self.kinetic["species1"]["params"]
        algo = species1_params["options"]["PushRandomDiffusion"]["algo"]
        diffusion_coefficient = species1_params["options"]["PushRandomDiffusion"]["diffusion_coefficient"]

        # # project magnetic background
        # self._b_eq = self.derham.P['2']([self.equil.b2_1,
        #                                  self.equil.b2_2,
        #                                  self.equil.b2_3])

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushRandomDiffusion] = {
            "algo": algo,
            "bc_type": species1_params["markers"]["bc"],
            "diffusion_coefficient": diffusion_coefficient,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_f")

        # MPI operations needed for scalar variables
        self._tmp = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        pass


class PressureLessSPH(StruphyModel):
    r"""Pressureless fluid discretized with smoothed particle hydrodynamics

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`

    This is discretized by particles going in straight lines.
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["kinetic"]["p_fluid"] = "ParticlesSPH"
        return dct

    @staticmethod
    def bulk_species():
        return "p_fluid"

    @staticmethod
    def velocity_scale():
        return None

    @staticmethod
    def diagnostics_dct():
        dct = {}
        dct["projected_density"] = "L2"
        return dct

    @staticmethod
    def propagators_dct():
        return {propagators_markers.PushEta: ["p_fluid"]}

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, clone_config=None):
        super().__init__(params, comm=comm, clone_config=clone_config)

        # prelim
        p_fluid_params = self.kinetic["p_fluid"]["params"]
        algo_eta = params["kinetic"]["p_fluid"]["options"]["PushEta"]["algo"]

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushEta] = {
            "algo": algo_eta,
            "density_field": self.pointer["projected_density"],
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_kin", compute="from_particles", species="p_fluid")

    def update_scalar_quantities(self):
        en_kin = self.pointer["p_fluid"].markers_wo_holes_and_ghost[:, 6].dot(
            self.pointer["p_fluid"].markers_wo_holes_and_ghost[:, 3] ** 2
            + self.pointer["p_fluid"].markers_wo_holes_and_ghost[:, 4] ** 2
            + self.pointer["p_fluid"].markers_wo_holes_and_ghost[:, 5] ** 2
        ) / (2.0 * self.pointer["p_fluid"].Np)

        self.update_scalar("en_kin", en_kin)


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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["potential"] = "L2"
        dct["fluid"]["ions"] = {
            "u": "Hdiv",
        }
        dct["fluid"]["electrons"] = {
            "u": "Hdiv",
        }
        return dct

    @staticmethod
    def bulk_species():
        return "ions"

    @staticmethod
    def velocity_scale():
        return "thermal"

    @staticmethod
    def propagators_dct():
        return {propagators_fields.TwoFluidQuasiNeutralFull: ["ions_u", "electrons_u", "potential"]}

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    # add special options
    @classmethod
    def options(cls):
        dct = super().options()
        cls.add_option(
            species=["fluid", "electrons"],
            option=propagators_fields.TwoFluidQuasiNeutralFull,
            dct=dct,
        )
        return dct

    def __init__(self, params, comm, clone_config=None):
        super().__init__(params, comm=comm, clone_config=clone_config)

        # get species paramaters
        electrons_params = params["fluid"]["electrons"]

        # Get coupling strength
        if electrons_params["options"]["TwoFluidQuasiNeutralFull"]["override_eq_params"]:
            self._epsilon = electrons_params["options"]["TwoFluidQuasiNeutralFull"]["eps_norm"]
            print(
                f"\n!!! Override equation parameters: {self._epsilon = }.",
            )
        else:
            self._epsilon = self.equation_params["electrons"]["epsilon"]

        # extract necessary parameters
        stokes_solver = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["solver"]
        stokes_nu = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["nu"]
        stokes_nu_e = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["nu_e"]
        stokes_a = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["a"]
        stokes_R0 = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["R0"]
        stokes_B0 = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["B0"]
        stokes_Bp = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["Bp"]
        stokes_alpha = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["alpha"]
        stokes_beta = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["beta"]
        stokes_sigma = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["stab_sigma"]
        stokes_variant = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["variant"]
        stokes_method_to_solve = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["method_to_solve"]
        stokes_preconditioner = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["preconditioner"]
        stokes_spectralanalysis = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"][
            "spectralanalysis"
        ]
        stokes_lifting = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["lifting"]
        stokes_dimension = params["fluid"]["electrons"]["options"]["TwoFluidQuasiNeutralFull"]["dimension"]
        stokes_1D_dt = params["time"]["dt"]

        # Check MPI size to ensure only one MPI process
        if comm is not None and stokes_variant == "Uzawa":
            if comm.Get_rank() == 0:
                print(f"Error: TwoFluidQuasiNeutralToy only runs with one MPI process.")
            return  # Early return to stop execution for multiple MPI processes

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.TwoFluidQuasiNeutralFull] = {
            "solver": stokes_solver,
            "nu": stokes_nu,
            "nu_e": stokes_nu_e,
            "eps_norm": self._epsilon,
            "a": stokes_a,
            "R0": stokes_R0,
            "B0": stokes_B0,
            "Bp": stokes_Bp,
            "alpha": stokes_alpha,
            "beta": stokes_beta,
            "stab_sigma": stokes_sigma,
            "variant": stokes_variant,
            "method_to_solve": stokes_method_to_solve,
            "preconditioner": stokes_preconditioner,
            "spectralanalysis": stokes_spectralanalysis,
            "dimension": stokes_dimension,
            "D1_dt": stokes_1D_dt,
            "lifting": stokes_lifting,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

    def update_scalar_quantities(self):
        pass
