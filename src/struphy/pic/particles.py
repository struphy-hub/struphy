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
    Particles in the full 6D phase space :math:`(\\boldsymbol \\eta, \\mathbf v) \\in [0, 1]^3 \\times \\mathbb R^3`,
    as used e.g. in full-orbit (Vlasov) kinetic models.

    Each marker carries a logical (curvilinear) position :math:`\\boldsymbol \\eta_p` together with a velocity
    :math:`\\mathbf v_p` expressed in the *Cartesian* velocity space attached to that position
    (i.e. velocities are not transformed by the curvilinear map, unlike positions).

    See :class:`~struphy.pic.base.Particles` for the structure of the numpy marker array and the meaning of its columns.
    """

    # Class properties
    vdim = 3
    """Dimension of the (Cartesian) velocity space, here 3."""
    default_background = maxwellians.Maxwellian3D()
    """Default sampling background is a 3D Cartesian Maxwellian."""
    default_n_cols = {"diagnostics": 0, "aux": 5}
    """Default number of buffer columns reserved for diagnostics and auxiliary (pusher/free) use."""

    def __post_init__(self):
        """If the background is a :class:`~struphy.kinetic_background.maxwellians.CanonicalMaxwellian`,
        set up the discrete magnetic field (needed to evaluate canonical invariants) from the projected equilibrium."""
        if isinstance(self.background, maxwellians.CanonicalMaxwellian):
            assert isinstance(self.projected_equil, ProjectedFluidEquilibriumWithB), (
                "CanonicalMaxwellian needs background with magnetic field."
            )
            self._absB0_h = self.projected_equil.absB0
            self._b2_h = self.projected_equil.b2
            self._derham = self.projected_equil.derham

    def svol(self, eta1, eta2, eta3, vx, vy, vz):
        """Sampling density function as volume form, used to draw markers via inverse transform/rejection
        sampling and to compute their initial weights (see :meth:`~struphy.pic.base.Particles.draw_markers`).

        This is a :class:`~struphy.kinetic_background.maxwellians.Maxwellian3D` in the Cartesian velocities
        ``vx, vy, vz``, parametrized by the mean velocities and thermal velocities in :attr:`loading_params`,
        with density normalized to 1 (i.e. uniform in ``eta1, eta2, eta3``), further multiplied by the
        Jacobian factor ``2 * eta1`` if :attr:`spatial` is ``"disc"`` (to sample uniformly in physical
        space on a disc, where ``eta1`` plays the role of a normalized radius).

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        vx, vy, vz : array_like
            Cartesian velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        if not hasattr(self, "_svol"):
            # load sampling density svol (normalized to 1 in logical space)
            self._svol = maxwellians.Maxwellian3D(
                n=(1.0, None),
                u1=(self.loading_params.moments[0], None),
                u2=(self.loading_params.moments[1], None),
                u3=(self.loading_params.moments[2], None),
                vth1=(self.loading_params.moments[3], None),
                vth2=(self.loading_params.moments[4], None),
                vth3=(self.loading_params.moments[5], None),
            )

        if self.spatial == "uniform":
            return self._svol(eta1, eta2, eta3, vx, vy, vz)

        elif self.spatial == "disc":
            return self._svol(eta1, eta2, eta3, vx, vy, vz) * 2 * eta1

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.',
            )

    def s0(self, eta1, eta2, eta3, vx, vy, vz, flat_eval=False, remove_holes=True):
        """Sampling density function as 0 form, i.e. :meth:`svol` pushed forward to a pointwise
        (non-volume-form) density by dividing out the metric Jacobian determinant via
        :meth:`~struphy.geometry.base.Domain.transform`. This is the quantity stored in each
        marker's ``s0`` column (see the class docstring) and used to compute initial weights
        ``w0 = f_init / s0 / Np``.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        vx, vy, vz : array_like
            Cartesian velocity evaluation points.

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
        assert self.domain, "self.domain must be set to call the sampling density 0-form."

        return self.domain.transform(
            self.svol(eta1, eta2, eta3, vx, vy, vz),
            eta1,
            eta2,
            eta3,
            flat_eval=flat_eval,
            kind="3_to_0",
            remove_outside=remove_holes,
        )

    def save_constants_of_motion(self):
        """
        Calculate each marker's guiding-center constants of motion (only the equilibrium
        magnetic field is considered) and assign them into the diagnostics columns of the marker array:

        * ``0:3``: guiding-center position (logical :math:`\\boldsymbol \\eta`)
        * ``3``: energy
        * ``4``: magnetic moment
        * ``5``: canonical toroidal momentum
        * ``6``: parallel velocity
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
            self.equation_params.epsilon,
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
            ~self.holes,
            slice_gc,
        ]
        if self.mpi_comm is not None:
            self.mpi_sort_markers(alpha=1)

        utilities_kernels.eval_canonical_toroidal_moment_6d(
            self.markers,
            self._derham.args_derham,
            self.first_diagnostics_idx,
            self.equation_params.epsilon,
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

    See :class:`~struphy.pic.particles.Particles6D` for more information.
    """

    def __post_init__(self):
        """Force the control-variate weight update off, since delta-f weights already evolve
        the perturbation directly (there is no separate background contribution to subtract)."""
        self.weights_params.control_variate = False

    def _set_initial_condition(self):
        """Zero out the density of the (unperturbed) background before setting the initial
        condition, so that only the perturbation :math:`\\delta f` is initialized on the markers."""
        self.set_n_to_zero(self.initial_condition)
        super()._set_initial_condition()

    def set_n_to_zero(self, background: Maxwellian | SumKineticBackground):
        """Recursively set the density moment ``n`` of ``background`` (and, if it is a
        :class:`~struphy.kinetic_background.base.SumKineticBackground`, of both its summands) to zero,
        keeping any perturbation attached to it.

        Parameters
        ----------
        background : Maxwellian | SumKineticBackground
            The kinetic background whose density is to be zeroed.
        """
        if isinstance(background, Maxwellian):
            background.params["n"] = (0.0, background.params["n"][1])
        else:
            assert isinstance(background, SumKineticBackground)
            self.set_n_to_zero(background._f1)
            self.set_n_to_zero(background._f2)


class Particles5D(Particles):
    """
    Particles in the 5D guiding-center, drift-kinetic or gyro-kinetic phase space
    :math:`(\\boldsymbol \\eta, v_\\parallel, \mu) \\in [0, 1]^3 \\times \\mathbb R \\times \\mathbb R_{\\geq 0}`.

    Each marker carries a logical (curvilinear) position :math:`\\boldsymbol \\eta_p` together with the
    velocity coordinates

    .. math::

        v_{\\parallel, p} = \\mathbf v_p \\cdot \\mathbf b_0(\\boldsymbol \\eta_p) \\,, \\qquad
        \\mu_p = \\frac{1}{2 |\\mathbf B_0|} m |\\mathbf v_p|^2 - v_{\\parallel, p}^2 \\,,

    defined with respect to the equilibrium magnetic field :math:`\\mathbf B_0` and its unit vector
    :math:`\\mathbf b_0 = \\mathbf B_0 / |\\mathbf B_0|` (unlike :class:`Particles6D`, velocities are thus
    not Cartesian but expressed in a field-aligned basis that itself depends on :math:`\\boldsymbol \\eta_p`).

    By default, two diagnostics columns are reserved (``default_n_cols["diagnostics"] = 2``), holding
    each marker's perpendicular energy and canonical toroidal momentum
    (see :meth:`save_constants_of_motion`).

    See :class:`~struphy.pic.base.Particles` for the structure of the numpy marker array and the meaning of its columns.
    """

    # Class properties
    vdim = 2
    """Dimension of the velocity space, here 2 (:math:`v_\\parallel, \\mu`)."""
    default_background = maxwellians.GyroMaxwellian2D()
    """Default sampling background is a gyrotropic Maxwellian in :math:`(v_\\parallel, \\mu)`."""
    default_n_cols = {"diagnostics": 2, "aux": 12}
    """Default number of buffer columns is 2 diagnostics (perpendicular energy, canonical toroidal
    momentum, see :meth:`save_constants_of_motion`) and 12 auxiliary columns."""

    def __post_init__(self):
        """Retrieve the discrete equilibrium magnetic-field quantities (:math:`|B_0|`, unit 1-form
        :math:`\\mathbf b_0`, Derham complex) needed to project marker velocities onto
        :math:`v_\\parallel, \\mu` and to evaluate diagnostics, and allocate the temporary
        FE coefficient vectors used for that."""
        assert self.projected_equil is not None, "Particles5D needs a projected MHD equilibrium."

        # magnetic background
        if self.projected_equil is not None:
            assert isinstance(self.projected_equil, ProjectedFluidEquilibriumWithB), (
                "Particles5D needs background with magnetic field."
            )

        self._absB0_h = self.projected_equil.absB0
        self._unit_b1_h = self.projected_equil.unit_b1
        self._derham = self.projected_equil.derham

        self._tmp0 = self.derham.V0.zeros()
        self._tmp2 = self.derham.V2.zeros()

    @property
    def magn_bckgr(self):
        """Equilibrium fluid background carrying the magnetic field :math:`\\mathbf B_0` with respect to
        which :math:`v_\\parallel, \\mu` are defined."""
        return self.equil

    @property
    def absB0_h(self):
        """Discrete 0-form coefficients of :math:`|B_0|`."""
        return self._absB0_h

    @property
    def unit_b1_h(self):
        """Discrete 1-form coefficients of the equilibrium field-aligned unit vector :math:`\\mathbf b_0 = \\mathbf B_0/|B_0|`."""
        return self._unit_b1_h

    @property
    def epsilon(self):
        """Normalization parameter :math:`\\epsilon` (from :attr:`equation_params`) entering the
        guiding-center equations of motion, e.g. the canonical toroidal momentum evaluation."""
        return self._epsilon

    @property
    def derham(self):
        """Discrete Derham complex of the projected equilibrium."""
        return self._derham

    def svol(self, eta1, eta2, eta3, v_para, mu):
        """
        Sampling density function as volume form, used to draw markers via inverse transform/rejection
        sampling and to compute their initial weights (see :meth:`~struphy.pic.base.Particles.draw_markers`).

        This is a :class:`~struphy.kinetic_background.maxwellians.GyroMaxwellian2D` in
        :math:`(v_\\parallel, \\mu)`, parametrized by the mean/thermal parallel velocity
        and by the equilibrium magnetic field in :attr:`loading_params` . It is normalized to
        1 in logical space (i.e. uniform in ``eta1, eta2, eta3``) and already includes the polar-coordinate
        Jacobian factor :math:`|\\mathbf B_0|` (``volume_form=True``), further multiplied by ``2 * eta1`` if
        :attr:`spatial` is ``"disc"``.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, mu : array_like
            Parallel velocity and magnetic moment evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        if not hasattr(self, "_svol"):
            # load sampling density svol (normalized to 1 in logical space)
            self._svol = maxwellians.GyroMaxwellian2D(
                n=(1.0, None),
                u_para=(self.loading_params.moments[0], None),
                u_perp=(0.0, None),
                vth_para=(self.loading_params.moments[2], None),
                vth_perp=(self.loading_params.moments[3], None),
                volume_form=True,
                equil=self.magn_bckgr,
                B0=self.loading_params.B0,
            )

        if self.spatial == "uniform":
            out = self._svol(eta1, eta2, eta3, v_para, mu)

        elif self.spatial == "disc":
            out = 2 * eta1 * self._svol(eta1, eta2, eta3, v_para, mu)

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.',
            )

        return out

    def s3(self, eta1, eta2, eta3, v_para, mu):
        """
        Sampling density function as 3-form, i.e. :meth:`svol` with the velocity-space
        (:math:`|v_\\perp|`) Jacobian factor divided back out via
        :meth:`~struphy.kinetic_background.maxwellians.GyroMaxwellian2D.velocity_jacobian_det`,
        leaving a density that is a volume form in :math:`\\boldsymbol \\eta` only.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, mu : array_like
            Parallel velocity and magnetic moment evaluation points.

        Returns
        -------
        out : array-like
            The 3-form sampling density.
        -------
        """

        return self.svol(eta1, eta2, eta3, v_para, mu) / self._svol.velocity_jacobian_det(
            eta1, eta2, eta3, v_para, mu
        )

    def s0(self, eta1, eta2, eta3, v_para, mu, flat_eval=False, remove_holes=True):
        """
        Sampling density function as 0-form, i.e. :meth:`s3` pushed forward to a pointwise density by
        dividing out the spatial metric Jacobian determinant via
        :meth:`~struphy.geometry.base.Domain.transform`. This is the quantity stored in each marker's
        ``s0`` column and used to compute initial weights ``w0 = f_init / s0 / Np``.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, mu : array_like
            Parallel velocity and magnetic moment evaluation points.

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
            self.s3(eta1, eta2, eta3, v_para, mu),
            eta1,
            eta2,
            eta3,
            flat_eval=flat_eval,
            kind="3_to_0",
            remove_outside=remove_holes,
        )

    def save_constants_of_motion(self):
        """
        Calculate each marker's guiding-center energy and canonical toroidal momentum (only the
        equilibrium magnetic field is considered) and assign them into the diagnostics columns of
        the marker array:

        * ``first_diagnostics_idx + 0``: energy
        * ``first_diagnostics_idx + 1``: magnetic moment (set once in :meth:`draw_markers`, unchanged here
          since it is an adiabatic invariant)
        * ``first_diagnostics_idx + 2``: canonical toroidal momentum
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

        utilities_kernels.eval_canonical_toroidal_moment_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self.equation_params.epsilon,
            B0,
            R0,
            self.absB0_h._data,
        )

    def save_magnetic_energy(self, PBb):
        r"""
        Calculate the (time-dependent) magnetic field energy at each marker's position and assign it
        into the energy diagnostics column (``self.first_diagnostics_idx``).

        Parameters
        ----------
        PBb : BlockVector
            Finite element coefficients of the time-dependent magnetic field, projected onto V0.
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
        Evaluate the equilibrium magnetic-moment energy :math:`\mu_p |B_0(\boldsymbol \eta_p)|` for each marker.
        The result is stored in the energy diagnostics column (``self.first_diagnostics_idx``).
        """

        utilities_kernels.eval_magnetic_background_energy(
            self.markers,
            self.derham.args_derham,
            self.domain.args_domain,
            self.first_diagnostics_idx,
            self.absB0_h._data,
        )
        

class Particles5Dvperp(Particles):
    """
    Particles in the 5D guiding-center, drift-kinetic or gyro-kinetic phase space
    :math:`(\\boldsymbol \\eta, v_\\parallel, v_\\perp) \\in [0, 1]^3 \\times \\mathbb R \\times \\mathbb R_{\\geq 0}`.

    Each marker carries a logical (curvilinear) position :math:`\\boldsymbol \\eta_p` together with the
    parallel and perpendicular velocity coordinates

    .. math::

        v_{\\parallel, p} = \\mathbf v_p \\cdot \\mathbf b_0(\\boldsymbol \\eta_p) \\,, \\qquad
        v_{\\perp, p} = \\left| \\mathbf v_p - v_{\\parallel, p} \\, \\mathbf b_0(\\boldsymbol \\eta_p) \\right| \\,,

    defined with respect to the equilibrium magnetic field :math:`\\mathbf B_0` and its unit vector
    :math:`\\mathbf b_0 = \\mathbf B_0 / |\\mathbf B_0|` (unlike :class:`Particles6D`, velocities are thus
    not Cartesian but expressed in a field-aligned basis that itself depends on :math:`\\boldsymbol \\eta_p`).

    By default, three diagnostics columns are reserved (``default_n_cols["diagnostics"] = 3``), holding
    each marker's guiding-center energy, magnetic moment and canonical toroidal momentum
    (see :meth:`save_constants_of_motion`).

    See :class:`~struphy.pic.base.Particles` for the structure of the numpy marker array and the meaning of its columns.
    """

    # Class properties
    vdim = 2
    """Dimension of the velocity space, here 2 (:math:`v_\\parallel, v_\\perp`)."""
    default_background = maxwellians.GyroMaxwellian2Dvperp()
    """Default sampling background is a gyrotropic Maxwellian in :math:`(v_\\parallel, v_\\perp)`."""
    default_n_cols = {"diagnostics": 3, "aux": 12}
    """Default number of buffer columns is 3 diagnostics (energy, magnetic moment, canonical toroidal
    momentum, see :meth:`save_constants_of_motion`) and 12 auxiliary columns."""

    def __post_init__(self):
        """Retrieve the discrete equilibrium magnetic-field quantities (:math:`|B_0|`, unit 1-form
        :math:`\\mathbf b_0`, Derham complex) needed to project marker velocities onto
        :math:`v_\\parallel, v_\\perp` and to evaluate diagnostics, and allocate the temporary
        FE coefficient vectors used for that."""
        assert self.projected_equil is not None, "Particles5Dvperp needs a projected MHD equilibrium."

        # magnetic background
        if self.projected_equil is not None:
            assert isinstance(self.projected_equil, ProjectedFluidEquilibriumWithB), (
                "Particles5Dvperp needs background with magnetic field."
            )

        self._absB0_h = self.projected_equil.absB0
        self._unit_b1_h = self.projected_equil.unit_b1
        self._derham = self.projected_equil.derham

        self._tmp0 = self.derham.V0.zeros()
        self._tmp2 = self.derham.V2.zeros()

    @property
    def magn_bckgr(self):
        """Equilibrium fluid background carrying the magnetic field :math:`\\mathbf B_0` with respect to
        which :math:`v_\\parallel, v_\\perp` are defined."""
        return self.equil

    @property
    def absB0_h(self):
        """Discrete 0-form coefficients of :math:`|B_0|`."""
        return self._absB0_h

    @property
    def unit_b1_h(self):
        """Discrete 1-form coefficients of the equilibrium field-aligned unit vector :math:`\\mathbf b_0 = \\mathbf B_0/|B_0|`."""
        return self._unit_b1_h

    @property
    def epsilon(self):
        """Normalization parameter :math:`\\epsilon` (from :attr:`equation_params`) entering the
        guiding-center equations of motion, e.g. the canonical toroidal momentum evaluation."""
        return self._epsilon

    @property
    def derham(self):
        """Discrete Derham complex of the projected equilibrium."""
        return self._derham

    def svol(self, eta1, eta2, eta3, v_para, v_perp):
        """
        Sampling density function as volume form, used to draw markers via inverse transform/rejection
        sampling and to compute their initial weights (see :meth:`~struphy.pic.base.Particles.draw_markers`).

        This is a :class:`~struphy.kinetic_background.maxwellians.GyroMaxwellian2D` in
        :math:`(v_\\parallel, v_\\perp)`, parametrized by the mean/thermal parallel and perpendicular velocities
        in :attr:`loading_params` and by the equilibrium magnetic field :attr:`magn_bckgr`. It is normalized to
        1 in logical space (i.e. uniform in ``eta1, eta2, eta3``) and already includes the polar-coordinate
        Jacobian factor :math:`|v_\\perp|` (``volume_form=True``), further multiplied by ``2 * eta1`` if
        :attr:`spatial` is ``"disc"``.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, v_perp : array_like
            Parallel and perpendicular velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        if not hasattr(self, "_svol"):
            # load sampling density svol (normalized to 1 in logical space)
            self._svol = maxwellians.GyroMaxwellian2Dvperp(
                n=(1.0, None),
                u_para=(self.loading_params.moments[0], None),
                u_perp=(self.loading_params.moments[1], None),
                vth_para=(self.loading_params.moments[2], None),
                vth_perp=(self.loading_params.moments[3], None),
                volume_form=True,
                equil=self.magn_bckgr,
            )

        if self.spatial == "uniform":
            out = self._svol(eta1, eta2, eta3, v_para, v_perp)

        elif self.spatial == "disc":
            out = 2 * eta1 * self._svol(eta1, eta2, eta3, v_para, v_perp)

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.',
            )

        return out

    def s3(self, eta1, eta2, eta3, v_para, v_perp):
        """
        Sampling density function as 3-form, i.e. :meth:`svol` with the velocity-space
        (:math:`|v_\\perp|`) Jacobian factor divided back out via
        :meth:`~struphy.kinetic_background.maxwellians.GyroMaxwellian2D.velocity_jacobian_det`,
        leaving a density that is a volume form in :math:`\\boldsymbol \\eta` only.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, v_perp : array_like
            Parallel and perpendicular velocity evaluation points.

        Returns
        -------
        out : array-like
            The 3-form sampling density.
        -------
        """

        return self.svol(eta1, eta2, eta3, v_para, v_perp) / self._svol.velocity_jacobian_det(
            eta1, eta2, eta3, v_para, v_perp
        )

    def s0(self, eta1, eta2, eta3, v_para, v_perp, flat_eval=False, remove_holes=True):
        """
        Sampling density function as 0-form, i.e. :meth:`s3` pushed forward to a pointwise density by
        dividing out the spatial metric Jacobian determinant via
        :meth:`~struphy.geometry.base.Domain.transform`. This is the quantity stored in each marker's
        ``s0`` column and used to compute initial weights ``w0 = f_init / s0 / Np``.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, v_perp : array_like
            Parallel and perpendicular velocity evaluation points.

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
            self.s3(eta1, eta2, eta3, v_para, v_perp),
            eta1,
            eta2,
            eta3,
            flat_eval=flat_eval,
            kind="3_to_0",
            remove_outside=remove_holes,
        )

    def draw_markers(self, sort: bool = True):
        super().draw_markers(sort=sort)

        # magnetic moment is an adiabatic invariant: evaluate once at draw time (diagnostics column 1)
        utilities_kernels.eval_magnetic_moment_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self._absB0_h._data,
        )

    def save_constants_of_motion(self):
        """
        Calculate each marker's guiding-center energy and canonical toroidal momentum (only the
        equilibrium magnetic field is considered) and assign them into the diagnostics columns of
        the marker array:

        * ``first_diagnostics_idx + 0``: energy
        * ``first_diagnostics_idx + 1``: magnetic moment (set once in :meth:`draw_markers`, unchanged here
          since it is an adiabatic invariant)
        * ``first_diagnostics_idx + 2``: canonical toroidal momentum
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

        utilities_kernels.eval_canonical_toroidal_moment_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self.equation_params.epsilon,
            B0,
            R0,
            self.absB0_h._data,
        )

    def save_magnetic_energy(self, PBb):
        r"""
        Calculate the (time-dependent) magnetic field energy at each marker's position and assign it
        into the energy diagnostics column (``self.first_diagnostics_idx``).

        Parameters
        ----------
        PBb : BlockVector
            Finite element coefficients of the time-dependent magnetic field, projected onto V0.
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
        Evaluate the equilibrium magnetic-moment energy :math:`\mu_p |B_0(\boldsymbol \eta_p)|` for each marker.
        The result is stored in the energy diagnostics column (``self.first_diagnostics_idx``).
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
        Calculate the magnetic moment of each marker and assign it into the magnetic-moment
        diagnostics column (``self.first_diagnostics_idx + 1``).
        """

        utilities_kernels.eval_magnetic_moment_5d(
            self.markers,
            self.derham.args_derham,
            self.first_diagnostics_idx,
            self.absB0_h._data,
        )


class Particles3D(Particles):
    """
    Particles in pure 3D configuration space :math:`\\boldsymbol \\eta \\in [0, 1]^3`, with no velocity
    space attached (``vdim = 0``) — each marker only carries a logical (curvilinear) position, used e.g.
    to represent a (massless) tracer or cold-plasma fluid density.

    See :class:`~struphy.pic.base.Particles` for the structure of the numpy marker array and the meaning of its columns.

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

    # Class properties
    vdim = 0
    """Dimension of the velocity space, here 0 (no velocity coordinates)."""
    default_background = maxwellians.ColdPlasma()
    """Default sampling background is a cold-plasma (velocity-independent) density."""
    default_n_cols = {"diagnostics": 0, "aux": 5}
    """Default number of buffer columns reserved for diagnostics and auxiliary (pusher/free) use."""

    def __post_init__(self):
        """No additional setup is required for this class."""

    def svol(self, eta1, eta2, eta3):
        """Sampling density function as volume form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

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
        """Sampling density function as 0 form, i.e. :meth:`svol` pushed forward to a pointwise
        (non-volume-form) density by dividing out the metric Jacobian determinant via
        :meth:`~struphy.geometry.base.Domain.transform`.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

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
    Particles for Smoothed Particle Hydrodynamics (SPH) models. The particle distribution itself lives
    in pure 3D configuration space :math:`\\boldsymbol \\eta \\in [0, 1]^3`, exactly as for :class:`Particles3D`
    (:meth:`svol` and :meth:`s0` depend only on :math:`\\boldsymbol \\eta_p`).

    Each marker additionally carries a Cartesian velocity :math:`\\mathbf v_p` in its marker-array columns,
    but this is a per-particle *helper* quantity (e.g. the SPH velocity-field sample used by pushers and
    kernel-based reconstructions) rather than a coordinate of a sampled phase-space density.

    See :class:`~struphy.pic.base.Particles` for the structure of the numpy marker array and the meaning of its columns.

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers, see :class:`~struphy.pic.base.Particles`.
    """

    # Class properties
    vdim = 3
    """Dimension of the per-marker Cartesian velocity attribute, here 3 (not a sampled coordinate, see class docstring)."""
    default_background = equils.ConstantVelocity()
    """Default background is a spatially constant velocity field."""
    default_n_cols = {"diagnostics": 0, "aux": 24}
    """Default number of buffer columns reserved for diagnostics and auxiliary (pusher/free) use."""

    def __post_init__(self):
        """Attach the domain to the background (needed to evaluate it at marker positions).
        SPH does not support clone-based (tile-copied) MPI parallelization."""
        assert self.clone_config is None, "SPH can only be launched with --nclones 1"
        self.background.domain = self.domain

    def svol(self, eta1, eta2, eta3, *v):
        """Sampling density function as volume form, used to draw markers via inverse transform/rejection
        sampling and to compute their initial weights (see :meth:`~struphy.pic.base.Particles.draw_markers`).

        This density is purely spatial: uniform (normalized to 1) if :attr:`spatial` is ``"uniform"``, or
        multiplied by the Jacobian factor ``2 * eta1`` if :attr:`spatial` is ``"disc"``.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Accepted for a call signature compatible with generic phase-space evaluation
            (e.g. via :attr:`~struphy.pic.base.Particles.phasespace_coords`), but unused: the marker
            velocities are helper quantities, not coordinates of the sampled density.

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
        """Sampling density function as 0 form, i.e. :meth:`svol` pushed forward to a pointwise
        (non-volume-form) density by dividing out the metric Jacobian determinant via
        :meth:`~struphy.geometry.base.Domain.transform`.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Accepted for a call signature compatible with generic phase-space evaluation, but unused
            (see :meth:`svol`).

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
