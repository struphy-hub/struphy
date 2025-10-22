import copy

import cunumpy as xp

from struphy.fields_background import equils
from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB
from struphy.fields_background.projected_equils import ProjectedFluidEquilibriumWithB
from struphy.geometry.base import Domain
from struphy.geometry.utilities import TransformedPformComponent
from struphy.initial.base import Perturbation
from struphy.kinetic_background import maxwellians
from struphy.kinetic_background.base import Maxwellian, SumKineticBackground
from struphy.pic import utilities_kernels
from struphy.pic.base import Particles


class Particles6D(Particles):
    """
    A class for initializing particles in models that use the full 6D phase space.

    The numpy marker array is as follows:

    ===== ============== ======================= ======= ====== ====== ==========
    index  | 0 | 1 | 2 | | 3 | 4 | 5           |  6       7       8    >=9
    ===== ============== ======================= ======= ====== ====== ==========
    value position (eta)    velocities           weight   s0     w0    buffer
    ===== ============== ======================= ======= ====== ====== ==========
    """

    @classmethod
    def default_background(cls):
        return maxwellians.Maxwellian3D()

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["type"] = "full_f"

        if "background" not in kwargs:
            kwargs["background"] = self.default_background()
        elif kwargs["background"] is None:
            kwargs["background"] = self.default_background()

        # default number of diagnostics and auxiliary columns
        self._n_cols_diagnostics = kwargs.pop("n_cols_diagn", 0)
        self._n_cols_aux = kwargs.pop("n_cols_aux", 5)

        super().__init__(**kwargs)

        # call projected mhd equilibrium in case of CanonicalMaxwellian
        if isinstance(kwargs["background"], maxwellians.CanonicalMaxwellian):
            assert isinstance(self.equil, FluidEquilibriumWithB), (
                "CanonicalMaxwellian needs background with magnetic field."
            )
            self._absB0_h = self.projected_equil.absB0
            self._b2_h = self.projected_equil.b2
            self._derham = self.projected_equil.derham
            self._epsilon = self.equation_params["epsilon"]

    @property
    def vdim(self):
        """Dimension of the velocity space."""
        return 3

    @property
    def n_cols_diagnostics(self):
        """Number of the diagnostics columns."""
        return self._n_cols_diagnostics

    @property
    def n_cols_aux(self):
        """Number of the auxiliary columns."""
        return self._n_cols_aux

    @property
    def coords(self):
        """Coordinates of the Particles6D, :math:`(v_1, v_2, v_3)`."""
        return "cartesian"

    def svol(self, eta1, eta2, eta3, *v):
        """Sampling density function as volume form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        # load sampling density svol (normalized to 1 in logical space)
        maxw_params = {
            "n": (1.0, None),
            "u1": (self.loading_params.moments[0], None),
            "u2": (self.loading_params.moments[1], None),
            "u3": (self.loading_params.moments[2], None),
            "vth1": (self.loading_params.moments[3], None),
            "vth2": (self.loading_params.moments[4], None),
            "vth3": (self.loading_params.moments[5], None),
        }

        fun = maxwellians.Maxwellian3D(**maxw_params)

        if self.spatial == "uniform":
            return fun(eta1, eta2, eta3, *v)

        elif self.spatial == "disc":
            return fun(eta1, eta2, eta3, *v) * 2 * eta1

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.',
            )

    def s0(self, eta1, eta2, eta3, *v, flat_eval=False, remove_holes=True):
        """Sampling density function as 0 form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        flat_eval : bool
            If true, perform flat (marker) evaluation (etas must be same size 1D).

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """
        assert self.domain, f"self.domain must be set to call the sampling density 0-form."

        return self.domain.transform(
            self.svol(eta1, eta2, eta3, *v),
            eta1,
            eta2,
            eta3,
            flat_eval=flat_eval,
            kind="3_to_0",
            remove_outside=remove_holes,
        )

    def save_constants_of_motion(self):
        """
        Calculate each markers' guiding center constants of motions
        and assign them into diagnostics columns of marker array:

        ================= ============== ======= ============ ============= ==============
        diagnostics index | 0 | 1 | 2 |  |  3  | |    4     | |     5     | |     6      |
        ================= ============== ======= ============ ============= ==============
              value       guiding_center energy  magn. moment can. momentum para. velocity
        ================= ============== ======= ============ ============= ==============

        Only equilibrium magnetic field is considered.
        """

        assert isinstance(self.equil, FluidEquilibriumWithB), "Constants of motion need background with magnetic field."

        # idx and slice
        idx_gc_r = self.first_diagnostics_idx
        slice_gc = slice(self.first_diagnostics_idx, self.first_diagnostics_idx + 3)
        idx_energy = self.first_diagnostics_idx + 3
        idx_can_momentum = self.first_diagnostics_idx + 5

        # save cartesian positions
        self.markers[~self.holes, slice_gc] = self.domain(
            self.positions,
            change_out_order=True,
        )

        # eval guiding center phase space
        utilities_kernels.eval_guiding_center_from_6d(
            self.markers,
            self._derham.args_derham,
            self.domain.args_domain,
            self.first_diagnostics_idx,
            self._epsilon,
            self._b2_h[0]._data,
            self._b2_h[1]._data,
            self._b2_h[2]._data,
            self._absB0_h._data,
        )

        # apply domain inverse map to get logical guiding center positions
        # TODO: currently only possible with the geometry where its inverse map is defined.
        assert hasattr(self.domain, "inverse_map")

        self.markers[~self.holes, slice_gc] = self.domain.inverse_map(
            *self.markers[~self.holes, slice_gc].T,
            change_out_order=True,
        )

        # eval energy
        self.markers[~self.holes, idx_energy] = (
            self.markers[~self.holes, 3] ** 2 + self.markers[~self.holes, 4] ** 2 + self.markers[~self.holes, 5] ** 2
        ) / (2)

        # eval psi at etas
        a1 = self.equil.domain.params["a1"]
        R0 = self.equil.params["R0"]
        B0 = self.equil.params["B0"]

        r = self.markers[~self.holes, idx_gc_r] * (1 - a1) + a1
        self.markers[~self.holes, idx_can_momentum] = self.equil.psi_r(r)

        # send particles to the guiding center positions
        self.markers[~self.holes, self.first_pusher_idx : self.first_pusher_idx + 3] = self.markers[
            ~self.holes, slice_gc
        ]
        if self.mpi_comm is not None:
            self.mpi_sort_markers(alpha=1)

        utilities_kernels.eval_canonical_toroidal_moment_6d(
            self.markers,
            self._derham.args_derham,
            self.first_diagnostics_idx,
            self._epsilon,
            B0,
            R0,
            self._absB0_h._data,
        )

        # send back and clear buffer
        if self.mpi_comm is not None:
            self.mpi_sort_markers()
        self.markers[~self.holes, self.first_pusher_idx : self.first_pusher_idx + 3] = 0


class DeltaFParticles6D(Particles6D):
    """
    A class for kinetic species in full 6D phase space that solve for delta_f = f - f0.
    """

    @classmethod
    def default_background(cls):
        return maxwellians.Maxwellian3D()

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["type"] = "delta_f"
        if "weights_params" in kwargs:
            kwargs["weights_params"].control_variate = False
        super().__init__(**kwargs)

    def _set_initial_condition(self):
        # bp_copy = copy.deepcopy(self.bckgr_params)
        # pp_copy = copy.deepcopy(self.pert_params)

        # # Prepare delta-f perturbation parameters
        # if pp_copy is not None:
        #     for fi in bp_copy:
        #         # Set background to zero (if "use_background_n" in perturbation params is set to false or not in keys)
        #         if fi in pp_copy:
        #             if "use_background_n" in pp_copy[fi]:
        #                 if not pp_copy[fi]["use_background_n"]:
        #                     bp_copy[fi]["n"] = 0.0
        #             else:
        #                 bp_copy[fi]["n"] = 0.0
        #         else:
        #             bp_copy[fi]["n"] = 0.0
        self.set_n_to_zero(self.initial_condition)

        super()._set_initial_condition()

    def set_n_to_zero(self, background: Maxwellian | SumKineticBackground):
        if isinstance(background, Maxwellian):
            background.maxw_params["n"] = (0.0, background.maxw_params["n"][1])
        else:
            assert isinstance(background, SumKineticBackground)
            self.set_n_to_zero(background._f1)
            self.set_n_to_zero(background._f2)


class Particles5D(Particles):
    """
    A class for initializing particles in guiding-center, drift-kinetic or gyro-kinetic models that use the 5D phase space.

    The numpy marker array is as follows:

    ===== ============== ========== ====== ======= ====== ====== ==========
    index  | 0 | 1 | 2 |     3        4       5      6      7       >=8
    ===== ============== ========== ====== ======= ====== ====== ==========
    value position (eta) v_parallel v_perp  weight   s0     w0   buffer
    ===== ============== ========== ====== ======= ====== ====== ==========

    Parameters
    ----------
    name : str
        Name of particle species.

    Np : int
        Number of particles.

    bc : list
        Either 'remove', 'reflect', 'periodic' or 'refill' in each direction.

    loading : str
        Drawing of markers; either 'pseudo_random', 'sobol_standard',
        'sobol_antithetic', 'external' or 'restart'.

    **kwargs : dict
        Parameters for markers, see :class:`~struphy.pic.base.Particles`.
    """

    @classmethod
    def default_background(cls):
        return maxwellians.GyroMaxwellian2D()

    def __init__(
        self,
        projected_equil: ProjectedFluidEquilibriumWithB,
        **kwargs,
    ):
        assert projected_equil is not None, "Particles5D needs a projected MHD equilibrium."

        kwargs["type"] = "full_f"

        # if "bckgr_params" not in kwargs:
        #     kwargs["bckgr_params"] = self.default_bckgr_params()

        # default number of diagnostics and auxiliary columns
        self._n_cols_diagnostics = kwargs.pop("n_cols_diagn", 3)
        self._n_cols_aux = kwargs.pop("n_cols_aux", 12)

        super().__init__(
            projected_equil=projected_equil,
            **kwargs,
        )

        # magnetic background
        if self.equil is not None:
            assert isinstance(self.equil, FluidEquilibriumWithB), "Particles5D needs background with magnetic field."
        self._magn_bckgr = self.equil

        self._absB0_h = self.projected_equil.absB0
        self._unit_b1_h = self.projected_equil.unit_b1
        self._derham = self.projected_equil.derham

        self._tmp0 = self.derham.Vh["0"].zeros()
        self._tmp2 = self.derham.Vh["2"].zeros()

    @property
    def vdim(self):
        """Dimension of the velocity space."""
        return 2

    @property
    def n_cols_diagnostics(self):
        """Number of the diagnostics columns."""
        return self._n_cols_diagnostics

    @property
    def n_cols_aux(self):
        """Number of the auxiliary columns."""
        return self._n_cols_aux

    @property
    def magn_bckgr(self):
        """Fluid equilibrium with B."""
        return self._magn_bckgr

    @property
    def absB0_h(self):
        """Discrete 0-form coefficients of |B_0|."""
        return self._absB0_h

    @property
    def unit_b1_h(self):
        """Discrete 1-form coefficients of B/|B|."""
        return self._unit_b1_h

    @property
    def epsilon(self):
        """One of equation params, epsilon"""
        return self._epsilon

    @property
    def coords(self):
        r"""Coordinates of the Particles5D, :math:`(v_\parallel, \mu)`."""
        return "vpara_mu"

    @property
    def derham(self):
        """Discrete Deram complex."""
        return self._derham

    def svol(self, eta1, eta2, eta3, *v):
        """
        Sampling density function as volume-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        # load sampling density svol (normalized to 1 in logical space)
        maxw_params = {
            "n": 1.0,
            "u_para": self.loading_params.moments[0],
            "u_perp": self.loading_params.moments[1],
            "vth_para": self.loading_params.moments[2],
            "vth_perp": self.loading_params.moments[3],
        }

        self._svol = maxwellians.GyroMaxwellian2D(
            n=(1.0, None),
            u_para=(self.loading_params.moments[0], None),
            u_perp=(self.loading_params.moments[1], None),
            vth_para=(self.loading_params.moments[2], None),
            vth_perp=(self.loading_params.moments[3], None),
            volume_form=True,
            equil=self._magn_bckgr,
        )

        if self.spatial == "uniform":
            out = self._svol(eta1, eta2, eta3, *v)

        elif self.spatial == "disc":
            out = 2 * eta1 * self._svol(eta1, eta2, eta3, *v)

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.',
            )

        return out

    def s3(self, eta1, eta2, eta3, *v):
        """
        Sampling density function as 3-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The 3-form sampling density.
        -------
        """

        return self.svol(eta1, eta2, eta3, *v) / self._svol.velocity_jacobian_det(eta1, eta2, eta3, *v)

    def s0(self, eta1, eta2, eta3, *v, flat_eval=False, remove_holes=True):
        """
        Sampling density function as 0-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_parallel, v_perp : array_like
            Velocity evaluation points.

        flat_eval : bool
            If true, perform flat (marker) evaluation (etas must be same size 1D).

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """

        return self.domain.transform(
            self.s3(eta1, eta2, eta3, *v),
            eta1,
            eta2,
            eta3,
            flat_eval=flat_eval,
            kind="3_to_0",
            remove_outside=remove_holes,
        )

    def draw_markers(self, sort: bool = True, verbose: bool = True):
        super().draw_markers(sort=sort, verbose=verbose)

        utilities_kernels.eval_magnetic_moment_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self._absB0_h._data,
        )

    def save_constants_of_motion(self):
        """
        Calculate each markers' energy and canonical toroidal momentum
        and assign them into diagnostics columns of marker array:

        ================= ======= ============ =============
        diagnostics index |  0  | |    1     | |     2     |
        ================= ======= ============ =============
              value       energy  magn. moment can. momentum
        ================= ======= ============ =============

        Only equilibrium magnetic field is considered.
        """

        assert isinstance(self.equil, FluidEquilibriumWithB), "Constants of motion need background with magnetic field."

        # idx and slice
        idx_can_momentum = self.first_diagnostics_idx + 2

        utilities_kernels.eval_energy_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self.absB0_h._data,
        )

        # eval psi at etas
        a1 = self.equil.domain.params["a1"]
        R0 = self.equil.params["R0"]
        B0 = self.equil.params["B0"]

        r = self.markers[~self.holes, 0] * (1 - a1) + a1
        self.markers[~self.holes, idx_can_momentum] = self.equil.psi_r(r)

        self._epsilon = self.equation_params["epsilon"]

        utilities_kernels.eval_canonical_toroidal_moment_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self.epsilon,
            B0,
            R0,
            self.absB0_h._data,
        )

    def save_magnetic_energy(self, PBb):
        r"""
        Calculate magnetic field energy at each particles' position and assign it into markers[:,self.first_diagnostics_idx].

        Parameters
        ----------

        b2 : BlockVector
            Finite element coefficients of the time-dependent magnetic field.
        """

        E0T = self.derham.extraction_ops["0"].transpose()
        PBbt = E0T.dot(PBb, out=self._tmp0)
        PBbt.update_ghost_regions()

        utilities_kernels.eval_magnetic_energy_PBb(
            self.markers,
            self.derham.args_derham,
            self.domain.args_domain,
            self.first_diagnostics_idx,
            self.absB0_h._data,
            PBbt._data,
        )

    def save_magnetic_background_energy(self):
        r"""
        Evaluate :math:`mu_p |B_0(\boldsymbol \eta_p)|` for each marker.
        The result is stored at markers[:, self.first_diagnostics_idx,].
        """

        utilities_kernels.eval_magnetic_background_energy(
            self.markers,
            self.derham.args_derham,
            self.domain.args_domain,
            self.first_diagnostics_idx,
            self.absB0_h._data,
        )

    def save_magnetic_moment(self):
        r"""
        Calculate magnetic moment of each particles and assign it into markers[:,self.first_diagnostics_idx,+1].
        """

        utilities_kernels.eval_magnetic_moment_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self.absB0_h._data,
        )


class Particles3D(Particles):
    """
    A class for initializing particles in 3D configuration space.

    The numpy marker array is as follows:

    ===== ============== ====== ====== ====== ======
    index  | 0 | 1 | 2 |   3       4     5      >=6
    ===== ============== ====== ====== ====== ======
    value position (eta) weight   s0     w0   buffer
    ===== ============== ====== ====== ====== ======

    Parameters
    ----------
    name : str
        Name of particle species.

    Np : int
        Number of particles.

    bc : list
        Either 'remove', 'reflect', 'periodic' or 'refill' in each direction.

    loading : str
        Drawing of markers; either 'pseudo_random', 'sobol_standard',
        'sobol_antithetic', 'external' or 'restart'.

    **kwargs : dict
        Parameters for markers, see :class:`~struphy.pic.base.Particles`.
    """

    @classmethod
    def default_background(cls):
        return maxwellians.ColdPlasma()

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["type"] = "full_f"

        if "background" not in kwargs:
            kwargs["background"] = self.default_background()
        elif kwargs["background"] is None:
            kwargs["background"] = self.default_background()

        # default number of diagnostics and auxiliary columns
        self._n_cols_diagnostics = kwargs.pop("n_cols_diagn", 0)
        self._n_cols_aux = kwargs.pop("n_cols_aux", 5)

        super().__init__(**kwargs)

    @property
    def vdim(self):
        """Dimension of the velocity space."""
        return 0

    @property
    def n_cols_diagnostics(self):
        """Number of the diagnostics columns."""
        return self._n_cols_diagnostics

    @property
    def n_cols_aux(self):
        """Number of the auxiliary columns."""
        return self._n_cols_aux

    @property
    def coords(self):
        """Coordinates of the Particles3D."""
        return "cartesian"

    def svol(self, eta1, eta2, eta3):
        """Sampling density function as volume form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """

        if self.spatial == "uniform":
            return 1.0 + 0.0 * eta1

        elif self.spatial == "disc":
            return 2.0 * eta1

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.',
            )

    def s0(self, eta1, eta2, eta3, flat_eval=False, remove_holes=True):
        """Sampling density function as 0 form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        flat_eval : bool
            If true, perform flat (marker) evaluation (etas must be same size 1D).

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """
        return self.domain.transform(
            self.svol(eta1, eta2, eta3),
            eta1,
            eta2,
            eta3,
            flat_eval=flat_eval,
            kind="3_to_0",
            remove_outside=remove_holes,
        )


class ParticlesSPH(Particles):
    """
    A class for initializing particles in SPH models.

    The numpy marker array is as follows:

    ===== ============== ======================= ======= ====== ====== ==========
    index  | 0 | 1 | 2 | | 3 | 4 | 5           |  6       7       8    >=9
    ===== ============== ======================= ======= ====== ====== ==========
    value position (eta)    velocities           weight   s0     w0    buffer
    ===== ============== ======================= ======= ====== ====== ==========

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers, see :class:`~struphy.pic.base.Particles`.
    """

    @classmethod
    def default_background(cls):
        return equils.ConstantVelocity()

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["type"] = "sph"

        if "background" not in kwargs:
            bckgr = self.default_background()
            bckgr.domain = kwargs["domain"]
            kwargs["background"] = bckgr
        elif kwargs["background"] is None:
            bckgr = self.default_background()
            bckgr.domain = kwargs["domain"]
            kwargs["background"] = bckgr

        if "boxes_per_dim" not in kwargs:
            kwargs["boxes_per_dim"] = (1, 1, 1)
        else:
            if kwargs["boxes_per_dim"] is None:
                kwargs["boxes_per_dim"] = (1, 1, 1)

        # TODO: maybe this needs a fix
        # else:
        #     if "communicate" not in kwargs["sorting_params"] or not kwargs["sorting_params"]["communicate"]:
        #         print("Enforcing communication of boxes in sph")
        #         kwargs["sorting_params"]["communicate"] = True

        # default number of diagnostics and auxiliary columns
        self._n_cols_diagnostics = kwargs.pop("n_cols_diagn", 0)
        self._n_cols_aux = kwargs.pop("n_cols_aux", 24)

        clone_config = kwargs.get("clone_config", None)
        assert clone_config is None, "SPH can only be launched with --nclones 1"

        super().__init__(**kwargs)

    @property
    def vdim(self):
        """Dimension of the velocity space."""
        return 3

    @property
    def n_cols_diagnostics(self):
        """Number of the diagnostics columns."""
        return self._n_cols_diagnostics

    @property
    def n_cols_aux(self):
        """Number of the auxiliary columns."""
        return self._n_cols_aux

    @property
    def coords(self):
        """Coordinates of the Particles6D, :math:`(v_1, v_2, v_3)`."""
        return "cartesian"

    def svol(self, eta1, eta2, eta3, *v):
        """Sampling density function as volume form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """

        if self.spatial == "uniform":
            return 0 * eta1 + 1.0

        elif self.spatial == "disc":
            return 2 * eta1

        else:
            raise NotImplementedError(f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.')

    def s0(self, eta1, eta2, eta3, *v, flat_eval=False, remove_holes=True):
        """Sampling density function as 0 form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        flat_eval : bool
            If true, perform flat (marker) evaluation (etas must be same size 1D).

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """
        return self.domain.transform(
            self.svol(eta1, eta2, eta3, *v),
            eta1,
            eta2,
            eta3,
            flat_eval=flat_eval,
            kind="3_to_0",
            remove_outside=remove_holes,
        )
