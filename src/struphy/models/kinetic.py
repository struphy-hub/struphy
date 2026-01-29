import cunumpy as xp
import numpy as np
import matplotlib.pyplot as plt
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.projectors import L2Projector
from struphy.kinetic_background.base import KineticBackground
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.models.base import StruphyModel
from struphy.models.species import FieldSpecies, FluidSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.utils.pyccel import Pyccelkernel


from scimba_torch.flows.deep_flows import DiscreteFlowSpace
from scimba_torch.flows.flow_trainer import FlowTrainer, NaturalGradientFlowTrainer
from scimba_torch.neural_nets.coordinates_based_nets.mlp import GenericMLP
from scimba_torch.neural_nets.coordinates_based_nets.features import PeriodicMLP
from scimba_torch.numerical_solvers.collocation_projector import (
    CollocationProjector,
    NaturalGradientProjector,
)

from scimba_torch.integration.mesh_based_quadrature import RectangleMethod

from scimba_torch.neural_nets.structure_preserving_nets.sympnet import SympNet
from scimba_torch.approximation_space.nn_space import NNxSpace
from scimba_torch.neural_nets.coordinates_based_nets.features import PeriodicMLP
from scimba_torch.domain.mesh_based_domain.cuboid import Cuboid

from scimba_torch.integration.monte_carlo import DomainSampler, TensorizedSampler
from scimba_torch.integration.monte_carlo_parameters import (
    UniformParametricSampler,
    UniformVelocitySampler,
)
import torch

rank = MPI.COMM_WORLD.Get_rank()


class VlasovAmpereOneSpecies_neural(StruphyModel):

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, with_B0: bool = True):
            self.push_eta = propagators_markers.PushEta()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            self.coupling_va = propagators_coupling.VlasovAmpere()

            self.coupling_current = propagators_coupling.ConstantCurrent()

    ## abstract methods

    def __init__(
        self,
        with_B0: bool = False,
        Nt_Psi=0,
        layers_Psi=[10] * 8,
        epochs_Ad_Psi=300,
        epochs_NG_Psi=1000,
        tol_Psi=[None, None],
        Nt_f0=0,
        layers_f0=[20] * 3,
        epochs_Ad_f0=300,
        epochs_NG_f0=1000,
        tol_f0=[None, None],
        plot_distribution_at_each_learning=False,
    ):
        if rank == 0:
            print(
                f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':"
            )

        self.with_B0 = with_B0

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var

        self.propagators.coupling_current.variables.e = self.em_fields.e_field
        self.propagators.coupling_current.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar(
            "en_f", compute="from_particles", variable=self.kinetic_ions.var
        )
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

        # Training parameters
        self.n = 1
        self.Nt_train = Nt_Psi
        self.layer_Psi = layers_Psi
        self.epochs_Ad_Psi = epochs_Ad_Psi
        self.epochs_NG_Psi = epochs_NG_Psi
        self.tol_Psi = tol_Psi
        self.Nt_f0 = Nt_f0
        self.layers_f0 = layers_f0
        self.epochs_Ad_f0 = epochs_Ad_f0
        self.epochs_NG_f0 = epochs_NG_f0
        self.tol_f0 = tol_f0
        self.space_list = []
        self.x_before = []
        self.f0_remap = Nt_Psi > 0
        if Nt_Psi != 0:
            if Nt_f0 % Nt_Psi != 0:
                raise ValueError(f"Nt_f0 = {Nt_f0} must be a multiple of Nt_Psi")
            else:
                self.max_nb_Psi_networks = Nt_f0 // Nt_Psi

        self.plot_distribution_at_each_learning = plot_distribution_at_each_learning
        self.non_periodic_positions = np.empty(0)
        self.n_iter_since_last_training = 0
        self.num_p = 0
        self.x_test = np.empty(0)
        self.electric_energy = []
        self.t = []
        self.time = 0

    #  self.original_f0 = self.kinetic_ions.var.particles.f0

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        particles = self.kinetic_ions.var.particles
        alpha = self.kinetic_ions.equation_params.alpha
        self._tmp[0] = (
            alpha**2
            / (2 * particles.Np)
            * xp.dot(
                particles.markers_wo_holes[:, 3] ** 2
                + particles.markers_wo_holes[:, 4] ** 2
                + particles.markers_wo_holes[:, 5] ** 2,
                particles.markers_wo_holes[:, 6],
            )
        )
        self.update_scalar("en_f", self._tmp[0])

        # en_tot = en_w + en_e
        self.update_scalar("en_tot", en_E + self._tmp[0])

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles

        super().allocate_propagators()

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        self.initial_poisson.options.rho = charge_accum
        self.initial_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        self.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Done.")

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += [
                        "model.initial_poisson.options = model.initial_poisson.Options()\n"
                    ]
                elif "push_vxb.Options" in line:
                    new_file += ["if model.with_B0:\n"]
                    new_file += ["    " + line]
                elif "set_save_data" in line:
                    new_file += [
                        "\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"
                    ]
                    new_file += [
                        "model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    # def compute_bulk_current(self):
    #     from scipy import integrate

    #     x1 = np.linspace(0, 1, 32)
    #     x2 = np.linspace(0, 1, 1)
    #     x3 = np.linspace(0, 1, 1)

    #     x, y, z = np.meshgrid(x1, x2, x3, indexing="ij")

    #     def integrand(vz, vy, vx):
    #         return self.kinetic_ions.var.particles.f0(x, y, z, vx, vy, vz)

    #     result, _ = integrate.tplquad(
    #         integrand,
    #         -10,
    #         10,  # limites pour vx
    #         -10,
    #         10,  # limites pour vy
    #         -10,
    #         10,  # limites pour vz
    #     )
    #     return result

    def compute_backward_flow(
        self,
        x_non_periodic,
        period=1.0,
    ):
        use_natural_gradient = True

        def apply_at_each_step(outputs):
            x = outputs[:, :3] % period

            return torch.cat((x, outputs[:, 3:]), dim=1)

        x0 = self.x_before[0]
        x0_tensor = torch.tensor(x0)
        print(f"x0.shape = {x0_tensor.shape}")
        # pos0 = torch.tensor(x0[0])
        # vel0 = torch.tensor(x0[1])
        pos0 = x0_tensor[0]
        vel0 = x0_tensor[1]

        y_target = torch.cat((pos0, vel0), dim=1)  # (Np, 6)
        x_non_periodic_tensor = torch.tensor(x_non_periodic)

        x_test_tensor = torch.tensor(self.x_test)
        # posT = torch.tensor(x_non_periodic[0])
        # velT = torch.tensor(x_non_periodic[1])
        posT = x_non_periodic_tensor[0]
        velT = x_non_periodic_tensor[1]

        posT_test = x_test_tensor

        x_input = torch.cat((posT, velT), dim=1)  # (Np, 6)
        x_input_test = torch.cat((posT_test, velT), dim=1)
        # sanity checks (leave them during debugging)
        assert x_input.shape == y_target.shape
        assert x_input.shape[1] == 6

        data = x_input, y_target

        space = DiscreteFlowSpace(
            6,
            0,
            flow_type=SympNet,
            rollout=1,
            layer_sizes=self.layer_Psi,
            apply_at_each_step=apply_at_each_step,
            activation_type="silu",
            periodic=True,
            periods=torch.tensor([period, period, period], dtype=torch.float64),
        )

        trainer = FlowTrainer(space, data)
        trainer.solve(
            max_epochs=self.epochs_Ad_Psi,
            verbose=True,
            batch_size=1000,
        )

        if use_natural_gradient:
            trainer_ng = NaturalGradientFlowTrainer(space, data)
            trainer_ng.solve(
                max_epochs=self.epochs_NG_Psi,
                verbose=True,
                batch_size=1000,
            )

        return space

    def update_bulk_current(self):

        # def n0u0(x, y, z):
        #     from scipy import integrate

        #     def integrand(vz, vy, vx):
        #         return self.kinetic_ions.var.particles.f0(x, y, z, vx, vy, vz)

        #     result, error = integrate.tplquad(
        #         integrand,
        #         -10,
        #         10,  # limites pour vx
        #         -10,
        #         10,  # limites pour vy
        #         -10,
        #         10,  # limites pour vz
        #     )

        # self.bulk_current = n0u0

        pass

    def update_f_bulk(self):

        def new_f_bulk(x1, x2, x3, v1, v2, v3):
            x1_flat = np.ravel(x1)
            x2_flat = np.ravel(x2)
            x3_flat = np.ravel(x3)
            v1_flat = np.ravel(v1)
            v2_flat = np.ravel(v2)
            v3_flat = np.ravel(v3)
            original_shape = x1.shape
            z = torch.tensor(
                np.stack(
                    [x1_flat, x2_flat, x3_flat, v1_flat, v2_flat, v3_flat], axis=1
                ),
                dtype=torch.float64,
            )
            mu = torch.empty(0)
            with torch.no_grad():

                for space in reversed(self.space_list):
                    z = space.inference(z, mu, 1).squeeze(0)
            z_np = z.detach().cpu().numpy()

            f_bulk_vals = self.original_f0(
                z_np[:, 0], z_np[:, 1], z_np[:, 2], z_np[:, 3], z_np[:, 4], z_np[:, 5]
            )

            return f_bulk_vals.reshape(original_shape)

        # Mettre à jour f0
        #    particles.f0 = particles._original_f0
        self.kinetic_ions.var.particles.f0 = new_f_bulk
        self.kinetic_ions.var.particles.f0.coords = None
        self.propagators.coupling_current.update_current()
        self.kinetic_ions.var.particles.update_weights()

    def project_density(self):

        pass

    def plot_f_bulk(self):

        vmin = -8
        vmax = 8

        e1 = np.linspace(0.0, 1.0, 256)
        v1 = np.linspace(vmin, vmax, 256)
        E1, V1 = np.meshgrid(e1, v1, indexing="ij")

        E1_flat = E1.flatten()
        V1_flat = V1.flatten()

        X_eval = torch.tensor(
            np.stack(
                [
                    E1_flat,
                    np.zeros_like(E1_flat),
                    np.zeros_like(E1_flat),
                    V1_flat,
                    np.zeros_like(E1_flat),
                    np.zeros_like(E1_flat),
                ],
                axis=1,
            ),
            dtype=torch.float64,
        )

        # Appliquer les transformations
        X_transformed = X_eval.clone()
        # mu = torch.empty(0)

        # for space in reversed(self.space_list):
        #     X_transformed = space.inference(X_transformed, mu, 1).squeeze(0)
        X_np = X_transformed.detach().cpu().numpy()

        # Évaluer f0 transformé
        f_bulk_vals = self.kinetic_ions.var.particles.f0(
            X_np[:, 0],
            X_np[:, 1],
            X_np[:, 2],
            X_np[:, 3],
            X_np[:, 4],
            X_np[:, 5],
        ).reshape(E1.shape)

        # Binning des particules
        # f_e1v1, df_e1v1 = particles.binning(
        #     components=components, bin_edges=[bin_edges_e, bin_edges_v]
        # )

        # Visualisation
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))

        # Plot f0 along the curve B(...)
        im0 = axes.pcolormesh(E1, V1, f_bulk_vals, shading="auto", cmap="turbo")
        axes.set_xlabel(r"$\eta_1$")
        axes.set_ylabel(r"$v_x$")
        axes.set_title(r"$f^0(\Psi(\eta_1, 0, 0,v_x, 0, 0))$")
        fig.colorbar(im0, ax=axes)

        plt.tight_layout()
        plt.show()

    def plot_f(self):

        vmin = -8
        vmax = 8

        e1 = np.linspace(0.0, 1.0, 256)
        v1 = np.linspace(vmin, vmax, 256)
        E1, V1 = np.meshgrid(e1, v1, indexing="ij")

        E1_flat = E1.flatten()
        V1_flat = V1.flatten()

        X_eval = torch.tensor(
            np.stack(
                [
                    E1_flat,
                    np.zeros_like(E1_flat),
                    np.zeros_like(E1_flat),
                    V1_flat,
                    np.zeros_like(E1_flat),
                    np.zeros_like(E1_flat),
                ],
                axis=1,
            ),
            dtype=torch.float64,
        )

        # Appliquer les transformations
        X_transformed = X_eval.clone()
        mu = torch.empty(0)

        for space in reversed(self.space_list):
            X_transformed = space.inference(X_transformed, mu, 1).squeeze(0)
        X_np = X_transformed.detach().cpu().numpy()

        # Évaluer f0 transformé
        f_bulk_vals = self.kinetic_ions.var.particles.f_init(
            X_np[:, 0],
            X_np[:, 1],
            X_np[:, 2],
            X_np[:, 3],
            X_np[:, 4],
            X_np[:, 5],
        ).reshape(E1.shape)

        # Binning des particules
        # f_e1v1, df_e1v1 = particles.binning(
        #     components=components, bin_edges=[bin_edges_e, bin_edges_v]
        # )

        # Visualisation
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))

        # Plot f0 along the curve B(...)
        im0 = axes.pcolormesh(E1, V1, f_bulk_vals, shading="auto", cmap="turbo")
        axes.set_xlabel(r"$\eta_1$")
        axes.set_ylabel(r"$v_x$")
        axes.set_title(r"$f(\Psi(\eta_1, 0, 0,v_x, 0, 0))$")
        fig.colorbar(im0, ax=axes)

        plt.tight_layout()
        plt.show()

    def plot_electric_energy(self):

        fig, axes = plt.subplots(1, 1, figsize=(12, 5))

        # Plot f0 along the curve B(...)
        axes.plot(self.t, self.electric_energy)
        axes.grid()
        axes.set_xlabel(r"$t$")
        axes.set_ylabel(r"$E$")
        axes.set_yscale("log")
        axes.set_title("Electric energy")

        plt.tight_layout()
        plt.show()

    def update_training_particles(self, dt):

        r1 = self.domain.params["r1"]
        r2 = self.domain.params["r2"]
        r3 = self.domain.params["r3"]
        self.non_periodic_positions[:, 0] = (
            self.non_periodic_positions[:, 0] * r1
            + (dt) * self.kinetic_ions.var.particles.velocities[:, 0]
        ) / r1
        self.non_periodic_positions[:, 1] = (
            self.non_periodic_positions[:, 1] * r2
            + (dt) * self.kinetic_ions.var.particles.velocities[:, 1]
        ) / r2

        self.non_periodic_positions[:, 2] = (
            self.non_periodic_positions[:, 2] * r1
            + (dt) * self.kinetic_ions.var.particles.velocities[:, 2]
        ) / r1

    def compute_particle_current(self):
        self.derham
        # Récupérer les données des particules
        positions = self.kinetic_ions.var.particles.positions  # Shape: (n_particles, 3)
        velocities = (
            self.kinetic_ions.var.particles.velocities
        )  # Shape: (n_particles, 3)
        weights = self.kinetic_ions.var.particles.weights  # Shape: (n_particles,)

        # Grille spatiale
        x_grid = np.linspace(0, 1, 100)  # Ajuster la résolution
        dx = x_grid[1] - x_grid[0]

        # Initialiser les courants
        jx = np.zeros_like(x_grid)
        jy = np.zeros_like(x_grid)
        jz = np.zeros_like(x_grid)

        # Déposer le courant sur la grille (méthode NGP - Nearest Grid Point)
        for k in range(len(weights)):
            # Trouver l'indice de grille le plus proche
            i = int(np.round(positions[k, 0] / dx))
            i = i % len(x_grid)  # Périodicité

            # Ajouter la contribution de la particule
            jx[i] += weights[k] * velocities[k, 0]
            jy[i] += weights[k] * velocities[k, 1]
            jz[i] += weights[k] * velocities[k, 2]

        # Normaliser par le volume de cellule
        #  jx /= dx
        #  jy /= dx
        #  jz /= dx
        Np = self.kinetic_ions.var.particles.Np
        jx /= Np
        jy /= Np
        jz /= Np
        # Utilisation pour le plot
        return x_grid, jx, jy, jz

    def plot_current(self):
        x_grid, jx_part, jy_part, jz_part = self.compute_particle_current()

        # Calculer aussi le courant de fond
        jx_bg = np.array(
            [
                self.propagators.coupling_current.compute_bulk_current_x()(x, 0.5, 0.5)
                for x in x_grid
            ]
        )
        jy_bg = np.array(
            [
                self.propagators.coupling_current.compute_bulk_current_y()(x, 0.5, 0.5)
                for x in x_grid
            ]
        )
        jz_bg = np.array(
            [
                self.propagators.coupling_current.compute_bulk_current_z()(x, 0.5, 0.5)
                for x in x_grid
            ]
        )
        self.propagators.coupling_current.compute_bulk_current_x
        # Courant total
        jx_total = jx_part + jx_bg
        jy_total = jy_part + jy_bg
        jz_total = jz_part + jz_bg

        # Plot
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(x_grid, jx_part, label="Particules")
        axes[0].plot(x_grid, jx_bg, label="Fond")
        axes[0].plot(x_grid, jx_total, "k--", label="Total")
        axes[0].set_ylabel("$j_x$")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(x_grid, jy_part, label="Particules")
        axes[1].plot(x_grid, jy_bg, label="Fond")
        axes[1].plot(x_grid, jy_total, "k--", label="Total")
        axes[1].set_ylabel("$j_y$")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(x_grid, jz_part, label="Particules")
        axes[2].plot(x_grid, jz_bg, label="Fond")
        axes[2].plot(x_grid, jz_total, "k--", label="Total")
        axes[2].set_ylabel("$j_z$")
        axes[2].set_xlabel("x")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def integrate(self, dt, split_algo):

        # x_test_list = [0, 0.2, 0.4, 0.6, 0.6, 1, 1.1]
        # current_test = np.zeros(len(x_test_list))

        # nvx, nvy, nvz = 16, 16, 16
        # vx = np.linspace(-6, 6, nvx)
        # vy = np.linspace(-6, 6, nvy)
        # vz = np.linspace(-6, 6, nvz)
        # dv = (vx[1] - vx[0]) * (vy[1] - vy[0]) * (vz[1] - vz[0])
        # jx = self.propagators.coupling_current.compute_bulk_current_x()

        # for x in x_test_list:

        #     print(f"jx(x ={x}): {jx(x,0.5,0.5)}")

        #   assert 1 == 0

        print(f"n = {self.n}")
        # particles = self.kinetic_ions.var.particles
        pos_before = self.kinetic_ions.var.particles.positions.copy()
        vel_before = self.kinetic_ions.var.particles.velocities.copy()
        if self.n == 1:

            self.original_f0 = self.kinetic_ions.var.particles.f0
            self.non_periodic_positions = (
                self.kinetic_ions.var.particles.positions.copy()
            )
            self.x_test = self.non_periodic_positions.copy()
            self.plot_f_bulk()
            self.plot_f()

        if self.n_iter_since_last_training == 0:
            self.non_periodic_positions = (
                self.kinetic_ions.var.particles.positions.copy()
            )

        self.x_before.append([pos_before, vel_before])

        self.propagators.push_eta(dt / 2)

        # self.non_periodic_positions = (
        #     self.non_periodic_positions * r1
        #     + (dt / 2) * self.kinetic_ions.var.particles.velocities
        # ) / r1
        self.update_training_particles(dt / 2)

        self.propagators.coupling_current(dt)

        self.propagators.coupling_va(dt)

        self.propagators.push_eta(dt / 2)
        # self.non_periodic_positions = (
        #     self.non_periodic_positions * r1
        #     + (dt / 2) * self.kinetic_ions.var.particles.velocities
        # ) / r1
        self.update_training_particles(dt / 2)
        self.time += dt

        e = self.em_fields.e_field.spline.vector
        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e).copy()
        self.t.append(self.time)
        self.electric_energy.append(en_E)
        print(f"Nt_train = {self.Nt_train}")
        self.n_iter_since_last_training += 1
        if self.n % self.Nt_train == 0:

            x_non_periodic = [
                self.non_periodic_positions.copy(),
                self.kinetic_ions.var.particles.velocities.copy(),
            ]

            space = self.compute_backward_flow(
                x_non_periodic,
                period=1,
            )
            #   self.store_train_times_Psi.append(self.t)
            self.space_list.append(space)

            self.update_f_bulk()
            # self.kinetic_ions.var.particles.non_periodic_positions = (
            #     self.kinetic_ions.var.particles.positions.copy()
            # )
            #  self.non_periodic_positions %= 1.0
            # assert np.allclose(
            #     self.non_periodic_positions[:, 0],
            #     self.kinetic_ions.var.particles.positions[:, 0],
            # )
            self.x_before.clear()
            assert self.x_before == []
            self.n_iter_since_last_training = 0
            # Plot
            if self.plot_distribution_at_each_learning:
                self.plot_f_bulk()
                self.plot_f()
                self.plot_electric_energy()
            #     self.plot_current()
            print(f"space_list.shape = {len(self.space_list)}")
            # if self.f0_remap and len(self.space_list) == self.max_nb_Psi_networks:
            #     print(f"Remapping f0")
            #     rhs = self.particles.f_bulk
            #     f0_torch = self.project_density(
            #         rhs,
            #         self.particles.boxsize,
            #         self.particles.vmax,
            #         epochs_Adam=epochs_Ad_f0,
            #         epochs_NG=epochs_NG_f0,
            #         layers_size=layers_f0,
            #         tol=tol_f0,
            #         store_losses=self.store_losses_f0,
            #         store_epochs=self.store_epochs_f0,
            #     )
            #     self.store_train_times_f0.append(self.t)

            #     def f0_np(x, v):

            #         x_flat = np.ravel(x)
            #         v_flat = np.ravel(v)

            #         z = torch.tensor(
            #             np.stack([x_flat, v_flat], axis=1), dtype=torch.float64
            #         )
            #         with torch.no_grad():
            #             f0_vals = f0_torch(z)

            #         f0_vals_np = np.abs(f0_vals.detach().cpu().numpy().reshape(x.shape))

            #         return f0_vals_np

            #     self.particles.f_bulk = f0_np
            #     self.particles.f0 = f0_np
            #     self.space_list = []
            #     particles.update_weights()

        self.n += 1


class VlasovAmpereOneSpecies(StruphyModel):
    r"""Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \\[2mm]
        -&\frac{\partial \mathbf{E}}{\partial t} =
        \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v}\,,

    with the normalization parameter

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons.
    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon}  \int_\Omega \int_{\mathbb{R}^3} \psi\, (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)\,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \nabla \times \mathbf B_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}\,,

    where :math:`\mathbf B_0` is the static equilibirum magnetic field.

    Notes
    -----

    * The :ref:`control_var` for Ampère's law is optional; in case it is enabled via the parameter file, the following system is solved:
    Find :math:`(\mathbf E, f) \in H(\textnormal{curl}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x =
            \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl}) \,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,.
        \end{align}


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`
    3. :class:`~struphy.propagators.propagators_markers.PushVxB`
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, with_B0: bool = True):
            self.push_eta = propagators_markers.PushEta()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            self.coupling_va = propagators_coupling.VlasovAmpere()

    ## abstract methods

    def __init__(self, with_B0: bool = True):
        if rank == 0:
            print(
                f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':"
            )

        self.with_B0 = with_B0

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar(
            "en_f", compute="from_particles", variable=self.kinetic_ions.var
        )
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        particles = self.kinetic_ions.var.particles
        alpha = self.kinetic_ions.equation_params.alpha
        self._tmp[0] = (
            alpha**2
            / (2 * particles.Np)
            * xp.dot(
                particles.markers_wo_holes[:, 3] ** 2
                + particles.markers_wo_holes[:, 4] ** 2
                + particles.markers_wo_holes[:, 5] ** 2,
                particles.markers_wo_holes[:, 6],
            )
        )
        self.update_scalar("en_f", self._tmp[0])

        # en_tot = en_w + en_e
        self.update_scalar("en_tot", en_E + self._tmp[0])

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles
        super().allocate_propagators()

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        self.initial_poisson.options.rho = charge_accum
        self.initial_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        self.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Done.")

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += [
                        "model.initial_poisson.options = model.initial_poisson.Options()\n"
                    ]
                elif "push_vxb.Options" in line:
                    new_file += ["if model.with_B0:\n"]
                    new_file += ["    " + line]
                elif "set_save_data" in line:
                    new_file += [
                        "\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"
                    ]
                    new_file += [
                        "model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class VlasovMaxwellOneSpecies(StruphyModel):
    r"""Vlasov-Maxwell equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right)
        \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \\[2mm]
        -&\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf B =
        \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3}  \mathbf{v} f \, \text{d}^3 \mathbf{v}\,,
        \\[2mm]
        &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,

    with the normalization parameters

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons.
    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \psi\, (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)\,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \nabla \times \mathbf B_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}\,,

    where :math:`\mathbf B_0` is the static equilibirum magnetic field.

    Notes
    -----

    * The :ref:`control_var` for Ampère's law is optional; in case it is enabled via the parameter file, the following system is solved:
    Find :math:`(\mathbf E, \tilde{\mathbf B}, f) \in H(\textnormal{curl}) \times H(\textnormal{div}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \times \mathbf{F} \cdot \tilde{\mathbf B}\,\textrm d \mathbf x =
            \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl}) \,,
            \\[2mm]
            &\frac{\partial \tilde{\mathbf B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon}\Big[ \mathbf{E} + \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf B}) \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \end{align}

    where :math:`\tilde{\mathbf B} = \mathbf B - \mathbf B_0` denotes the magnetic perturbation.


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_markers.PushEta`
    3. :class:`~struphy.propagators.propagators_markers.PushVxB`
    4. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()
            self.push_eta = propagators_markers.PushEta()
            self.push_vxb = propagators_markers.PushVxB()
            self.coupling_va = propagators_coupling.VlasovAmpere()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(
                f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':"
            )

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_B")
        self.add_scalar(
            "en_f", compute="from_particles", variable=self.kinetic_ions.var
        )
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        b = self.em_fields.b_field.spline.vector

        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        en_B = 0.5 * self.mass_ops.M2.dot_inner(b, b)
        self.update_scalar("en_B", en_B)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        particles = self.kinetic_ions.var.particles
        alpha = self.kinetic_ions.equation_params.alpha
        self._tmp[0] = (
            alpha**2
            / (2 * particles.Np)
            * xp.dot(
                particles.markers_wo_holes[:, 3] ** 2
                + particles.markers_wo_holes[:, 4] ** 2
                + particles.markers_wo_holes[:, 5] ** 2,
                particles.markers_wo_holes[:, 6],
            )
        )
        self.update_scalar("en_f", self._tmp[0])

        # en_tot = en_w + en_e
        self.update_scalar("en_tot", en_E + self._tmp[0])

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles
        super().allocate_propagators()

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        self.initial_poisson.options.rho = charge_accum
        self.initial_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        self.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Done.")

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += [
                        "model.initial_poisson.options = model.initial_poisson.Options()\n"
                    ]
                elif "push_vxb.Options" in line:
                    new_file += [
                        "model.propagators.push_vxb.options = model.propagators.push_vxb.Options(b2_var=model.em_fields.b_field)\n",
                    ]
                elif "set_save_data" in line:
                    new_file += [
                        "\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"
                    ]
                    new_file += [
                        "model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class LinearVlasovAmpereOneSpecies(StruphyModel):
    r"""Linearized Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
            & \frac{\partial \tilde{\mathbf E}}{\partial t} = - \frac{\alpha^2}{\varepsilon} \int_{\mathbb R^3} \mathbf{v} \tilde f\, \textrm d^3 \mathbf v \,,
            \\[2mm]
            & \frac{\partial \tilde f}{\partial t} + \mathbf{v} \cdot \, \nabla \tilde f + \frac{1}{\varepsilon} \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial \tilde f}{\partial \mathbf{v}} = \frac{1}{v_{\text{th}}^2 \varepsilon} \, \tilde{\mathbf E} \cdot \mathbf{v} f_0 \,,
        \end{align}

    with the normalization parameter

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

    .. math::

        f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3}
        \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right) \,,

    and the background electric field has to verify the following compatibility condition between with background density

    .. math::

        \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{1}{v_{\text{th}}^2 \varepsilon} \mathbf{E}_0 \,.

    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon}  \int_\Omega \int_{\mathbb{R}^3} \psi\, \tilde f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \tilde{\mathbf{E}}(t=0) &= -\nabla \phi(t=0) \,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVinEfield`
    3. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="DeltaFParticles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            with_B0: bool = True,
            with_E0: bool = True,
        ):
            self.push_eta = propagators_markers.PushEta()
            if with_E0:
                self.push_vinE = propagators_markers.PushVinEfield()
            self.coupling_Eweights = propagators_coupling.EfieldWeights()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()

    ## abstract methods

    def __init__(
        self,
        with_B0: bool = True,
        with_E0: bool = True,
    ):
        if rank == 0:
            print(
                f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':"
            )

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0, with_E0=with_E0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_E0:
            self.propagators.push_vinE.variables.var = self.kinetic_ions.var
        self.propagators.coupling_Eweights.variables.e = self.em_fields.e_field
        self.propagators.coupling_Eweights.variables.ions = self.kinetic_ions.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar(
            "en_w", compute="from_particles", variable=self.kinetic_ions.var
        )
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        particles = self.kinetic_ions.var.particles

        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        # evaluate f0
        if not hasattr(self, "_f0"):
            backgrounds = self.kinetic_ions.var.backgrounds
            if isinstance(backgrounds, list):
                self._f0 = backgrounds[0]
            else:
                self._f0 = backgrounds
            self._f0_values = xp.zeros(
                self.kinetic_ions.var.particles.markers.shape[0],
                dtype=float,
            )
            assert isinstance(self._f0, Maxwellian3D)

        self._f0_values[particles.valid_mks] = self._f0(*particles.phasespace_coords.T)

        # alpha^2 * v_th^2 / (2*N) * sum_p s_0 * w_p^2 / f_{0,p}
        alpha = self.kinetic_ions.equation_params.alpha
        vth = self._f0.maxw_params["vth1"][0]

        self._tmp[0] = (
            alpha**2
            * vth**2
            / (2 * particles.Np)
            * xp.dot(
                particles.weights**2,  # w_p^2
                particles.sampling_density
                / self._f0_values[particles.valid_mks],  # s_{0,p} / f_{0,p}
            )
        )

        self.update_scalar("en_w", self._tmp[0])
        self.update_scalar("en_tot", self._tmp[0] + en_E)

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles
        super().allocate_propagators()

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        self.initial_poisson.options.rho = charge_accum
        self.initial_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        self.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Done.")

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "maxwellian_1 + maxwellian_2" in line:
                    new_file += ["background = maxwellian_1\n"]
                elif "maxwellian_1pt =" in line:
                    new_file += [
                        "maxwellian_1pt = maxwellians.Maxwellian3D(n=(0.0, perturbation))\n"
                    ]
                elif "set_save_data" in line:
                    new_file += [
                        "\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"
                    ]
                    new_file += [
                        "model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class LinearVlasovMaxwellOneSpecies(LinearVlasovAmpereOneSpecies):
    r"""Linearized Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
            & \frac{\partial \tilde{\mathbf E}}{\partial t} = \nabla \times \tilde{\mathbf B} - \frac{\alpha^2}{\varepsilon} \int_{\mathbb R^3}\mathbf{v} \tilde f\, \textrm d^3 \mathbf v \,,
            \\[2mm]
            & \frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times \tilde{\mathbf E} \,,
            \\[2mm]
            & \frac{\partial \tilde f}{\partial t} + \mathbf{v} \cdot \, \nabla \tilde f + \frac{1}{\varepsilon} \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial \tilde f}{\partial \mathbf{v}} = \frac{1}{v_{\text{th}}^2 \varepsilon} \, \tilde{\mathbf E} \cdot \mathbf{v} f_0 \,,
        \end{align}

    with the normalization parameter

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

    .. math::

        f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3}
        \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right) \,,

    and the background electric field has to verify the following compatibility condition between with background density

    .. math::

        \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{1}{v_{\text{th}}^2 \varepsilon} \mathbf{E}_0 \,.

    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \psi\, \tilde f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \tilde{\mathbf{E}(t=0)} &= -\nabla \phi(t=0) \,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVinEfield`
    3. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`
    5. :class:`~struphy.propagators.propagators_fields.Maxwell`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="DeltaFParticles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            with_B0: bool = True,
            with_E0: bool = True,
        ):
            self.push_eta = propagators_markers.PushEta()
            if with_E0:
                self.push_vinE = propagators_markers.PushVinEfield()
            self.coupling_Eweights = propagators_coupling.EfieldWeights()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            self.maxwell = propagators_fields.Maxwell()

    ## abstract methods

    def __init__(
        self,
        with_B0: bool = True,
        with_E0: bool = True,
    ):
        if rank == 0:
            print(
                f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':"
            )

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0, with_E0=with_E0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_E0:
            self.propagators.push_vinE.variables.var = self.kinetic_ions.var
        self.propagators.coupling_Eweights.variables.e = self.em_fields.e_field
        self.propagators.coupling_Eweights.variables.ions = self.kinetic_ions.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_B")
        self.add_scalar(
            "en_w", compute="from_particles", variable=self.kinetic_ions.var
        )
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    def update_scalar_quantities(self):
        super().update_scalar_quantities()

        # 0.5 * b^T * M_2 * b
        b = self.em_fields.b_field.spline.vector

        en_B = 0.5 * self._mass_ops.M2.dot_inner(b, b)
        self.update_scalar(
            "en_tot", self.scalar_quantities["en_tot"]["value"][0] + en_B
        )


class DriftKineticElectrostaticAdiabatic(StruphyModel):
    r"""Drift-kinetic equation for one ion species in static background magnetic field,
    coupled to quasi-neutrality equation with adiabatic electrons.

    :ref:`normalization`:

    .. math::

       \hat v = \hat v_\textrm{i} = \sqrt{\frac{k_B \hat T_\textrm{i}}{m_\textrm{i}}}\,,\qquad  \hat E = \hat v_\textrm{i}\hat B\,,\qquad \hat \phi = \hat E \hat x \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[\frac{1}{\varepsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.
        \\[2mm]
        - &\nabla_\perp \cdot \left( \frac{n_0}{|B_0|^2} \nabla_\perp \phi \right) + \frac{1}{\varepsilon} n_0 \left(1 + \frac{1}{Z \varepsilon} \frac{1}{T_{0}} \phi \right) = \frac 1 \varepsilon \int f B^*_\parallel \,\textnormal d v_\parallel \textnormal d \mu \,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

    .. math::
        \mathbf{E}^* = - \nabla \phi - \varepsilon \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \varepsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,,

    and with the normalization parameters

    .. math::

        \varepsilon := \frac{1}{\hat \Omega_\textrm{c} \hat t}\,,\qquad \hat \Omega_\textrm{c} = \frac{q_\textrm{i} \hat B}{m_\textrm{i}} \,.

    Notes
    -----

    * The :ref:`control_var` in the Poisson equation is optional; in case it is enabled via the parameter file, the following Poisson equation is solved:
    Find :math:`\phi \in H^1` such that

    .. math::

        \int \frac{n_0}{|B_0|^2} \nabla_\perp \psi \cdot \nabla_\perp \phi\,\textrm d \mathbf x + \frac{1}{Z\varepsilon^2} \int  \frac{n_0}{T_{0}} \psi \phi \,\textrm d \mathbf x  = \frac 1 \varepsilon \int \int \psi \, (f - f_0) B^*_\parallel \,\textrm d \mathbf x\,\textnormal d v_\parallel \textnormal d \mu \qquad \forall \ \psi \in H^1\,.


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ImplicitDiffusion`
    2. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterBxEstar`
    3. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterParallel`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles5D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.gc_poisson = propagators_fields.ImplicitDiffusion()
            self.push_gc_bxe = propagators_markers.PushGuidingCenterBxEstar()
            self.push_gc_para = propagators_markers.PushGuidingCenterParallel()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(
                f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':"
            )

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.gc_poisson.variables.phi = self.em_fields.phi
        self.propagators.push_gc_bxe.variables.ions = self.kinetic_ions.var
        self.propagators.push_gc_para.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_phi")
        self.add_scalar(
            "en_particles", compute="from_particles", variable=self.kinetic_ions.var
        )
        self.add_scalar("en_tot")

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "thermal"

    def allocate_helpers(self):
        self._tmp3 = xp.empty(1, dtype=float)
        self._e_field = self.derham.Vh["1"].zeros()

        assert (
            self.kinetic_ions.charge_number > 0
        ), "Model written only for positive ions."

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles
        super().allocate_propagators()

        # Poisson right-hand side
        particles = self.kinetic_ions.var.particles
        Z = self.kinetic_ions.charge_number
        epsilon = self.kinetic_ions.equation_params.epsilon

        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels_gc.gc_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        rho = charge_accum

        # get neutralizing background density
        if not particles.control_variate:
            l2_proj = L2Projector("H1", self.mass_ops)
            f0e = Z * particles.f0
            assert isinstance(f0e, KineticBackground)
            rho_eh = FEECVariable(space="H1")
            rho_eh.allocate(derham=self.derham, domain=self.domain)
            rho_eh.spline.vector = l2_proj.get_dofs(f0e.n)
            rho = [rho]
            rho += [rho_eh]

        self.propagators.gc_poisson.options.sigma_1 = 1.0 / epsilon**2 / Z
        self.propagators.gc_poisson.options.sigma_2 = 0.0
        self.propagators.gc_poisson.options.sigma_3 = 1.0 / epsilon
        self.propagators.gc_poisson.options.stab_mat = "M0ad"
        self.propagators.gc_poisson.options.diffusion_mat = "M1perp"
        self.propagators.gc_poisson.options.rho = rho
        self.propagators.gc_poisson.allocate()

    def update_scalar_quantities(self):
        phi = self.em_fields.phi.spline.vector
        particles = self.kinetic_ions.var.particles
        epsilon = self.kinetic_ions.equation_params.epsilon

        # energy from polarization
        e1 = self.derham.grad.dot(-phi, out=self._e_field)
        en_phi1 = 0.5 * self.mass_ops.M1gyro.dot_inner(e1, e1)

        # energy from adiabatic electrons
        en_phi = 0.5 / epsilon**2 * self.mass_ops.M0ad.dot_inner(phi, phi)

        # for Landau damping test
        # en_phi = 0.

        # mu_p * |B0(eta_p)|
        particles.save_magnetic_background_energy()

        # 1/N sum_p (w_p v_p^2/2 + mu_p |B0|_p)
        self._tmp3[0] = (
            1
            / particles.Np
            * xp.sum(
                particles.weights * particles.velocities[:, 0] ** 2 / 2.0
                + particles.markers_wo_holes_and_ghost[:, 8],
            )
        )

        self.update_scalar("en_phi", en_phi + en_phi1)
        self.update_scalar("en_particles", self._tmp3[0])
        self.update_scalar("en_tot", en_phi + en_phi1 + self._tmp3[0])

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "BaseUnits(" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
                elif "push_gc_bxe.Options" in line:
                    new_file += [
                        "model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(phi=model.em_fields.phi)\n",
                    ]
                elif "push_gc_para.Options" in line:
                    new_file += [
                        "model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(phi=model.em_fields.phi)\n",
                    ]
                elif "set_save_data" in line:
                    new_file += [
                        "\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"
                    ]
                    new_file += [
                        "model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
