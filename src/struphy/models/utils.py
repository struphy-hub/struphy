import inspect
import logging

from feectools.linalg.block import BlockVector

import struphy.models as models
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel, apply_splitting_order
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, KineticEnergyPIC, LostMarkersPIC
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.utils import VelocityMass

logger = logging.getLogger("struphy")


# renamed models (old name -> new name), so that old run metadata and parameter files still find their model
RENAMED_MODELS = {
    "PressureLessSPH": "ColdEulerSPH",
}


# TODO: These utils should be classmethod of the model baseclass!
def get_model_by_name(model_name: str) -> type[StruphyModel]:
    if model_name in RENAMED_MODELS:
        logger.warning(f"The model {model_name} has been renamed to {RENAMED_MODELS[model_name]}.")
        model_name = RENAMED_MODELS[model_name]
    try:
        model_class: StruphyModel = getattr(models, model_name)
        if not issubclass(model_class, StruphyModel):
            raise TypeError(f"{model_name} is not a StruphyModel subclass.")
        else:
            return model_class
    except AttributeError:
        raise ModuleNotFoundError(f"{model_name} not found in models.")


def get_models(model_type: LiteralOptions.ModelTypes | None = None) -> list[type[StruphyModel]]:
    model_classes = []

    for model in StruphyModel:
        if model_type is None or model_type == model.model_type():
            model_classes.append(model)

    return model_classes


def get_model_names(model_type: LiteralOptions.ModelTypes | None = None) -> list[str]:
    model_classes = get_models(model_type=model_type)

    return [model.name() for model in model_classes]


def get_model_type_message(model_type: LiteralOptions.ModelTypes) -> str:
    model_names = get_model_names(model_type=model_type)

    # fluid message
    models_message = f"{model_type} models:\n"
    models_message += "-------------\n"
    models_message += "\n".join(model_names)

    return models_message


def generate_models_message() -> str:
    fluid_message = get_model_type_message(model_type="Fluid")
    kinetic_message = get_model_type_message(model_type="Kinetic")
    hybrid_message = get_model_type_message(model_type="Hybrid")

    # model message
    model_message = "run one of the following models:\n"
    model_message += "\n" + fluid_message
    model_message += "\n\n" + kinetic_message
    model_message += "\n\n" + hybrid_message

    return model_message


# ------------------------------------------------------------------------------------------------------------------
# Utilities shared by the models of a family (or of several families), called explicitly from the hooks of the
# models (no mixin classes). Short code stays in the models.
# ------------------------------------------------------------------------------------------------------------------

## LinearEuler family (with p and with q = sqrt(p))


def ones_l2():
    """The vector of ones in the tensor-product :math:`L^2` space, for volume integrals of 3-forms from their
    tensor-product coefficients (e.g. ``Propagator.projected_equil.tp.p3.inner(ones_l2())``; the sum of the
    tensor-product coefficients of a 3-form is its integral over the logical cube). Allocated, call in
    ``post_allocate``."""
    ones = Propagator.projected_equil.tp.p3.space.zeros()
    ones[:] = 1.0
    return ones


## magnetic field


class MagneticEnergies:
    """Energies of the equilibrium magnetic field, :math:`\\frac 12 \\int |\\mathbf B_0|^2`, and of the total magnetic
    field, :math:`\\frac 12 \\int |\\mathbf B_0 + \\tilde{\\mathbf B}|^2`, in the space of the perturbation (2-form, or
    1-form with ``b_space="Hcurl"``).

    Parameters
    ----------
    b_field : FEECVariable
        Magnetic field perturbation.
    """

    def __init__(self, b_field: FEECVariable):
        self._b_field = b_field

    def allocate(self):
        """Background field, mass matrix and buffers (call in ``post_allocate`` of the model)."""
        if self._b_field.space == "Hcurl":
            self._b_eq = Propagator.projected_equil.b1
            self._M_b = Propagator.mass_ops.M1
        else:
            self._b_eq = Propagator.projected_equil.b2
            self._M_b = Propagator.mass_ops.M2

        self._tmp_b1: BlockVector = self._b_eq.space.zeros()
        self._tmp_b2: BlockVector = self._b_eq.space.zeros()

    def en_B_eq(self):
        """Energy of the equilibrium magnetic field."""
        self._M_b.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        return self._b_eq.inner(self._tmp_b1) / 2

    def en_B_tot(self):
        """Energy of the total magnetic field."""
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self._b_field.spline.vector

        self._M_b.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        return self._tmp_b1.inner(self._tmp_b2) / 2


def add_magnetic_scalars(scalars: dict, b_field: FEECVariable, energies: MagneticEnergies):
    """Add the magnetic energies to the scalars of the parent model: ``en_B``, ``en_B_eq``, ``en_B_tot``, and replace
    ``en_tot`` by the one including the magnetic energy (kept last).

    Parameters
    ----------
    scalars : dict
        Scalars of the parent model, modified in place.
    b_field : FEECVariable
        Magnetic field perturbation.
    energies : MagneticEnergies
        Energies of the equilibrium and the total magnetic field.
    """
    en_tot = scalars.pop("en_tot")
    en_B = BilinearEnergyFEEC(b_field)
    scalars["en_B"] = en_B
    scalars["en_B_eq"] = FunctionScalarFEEC(energies.en_B_eq)
    scalars["en_B_tot"] = FunctionScalarFEEC(energies.en_B_tot)
    scalars["en_tot"] = en_tot + en_B


def magnetic_discretization_doc(model_cls) -> str:
    """Body of ``doc_discretization`` of the linear MHD models with the flag ``b_space``: the propagators of the model
    (with ``b_space="Hdiv"``), followed by the propagators that differ with ``b_space="Hcurl"`` (the curl-curl
    shear-Alfvén step, and the 1-form couplings and pushers of the hybrid models).

    Parameters
    ----------
    model_cls : type
        The model class (its ``Propagators`` take the argument ``b_space``).
    """
    doc = model_cls.propagators_doc()
    hdiv = model_cls.propagators_in_splitting_order()
    hcurl = model_cls.Propagators(b_space="Hcurl")
    apply_splitting_order(hcurl)
    for n, (name, prop) in enumerate(vars(hcurl).items(), start=1):
        prop_cls = type(prop)
        if prop_cls is type(hdiv[name]):
            continue
        doc += f'**{n}. {name}** with ``b_space="Hcurl"`` (:class:`~{prop_cls.__module__}.{prop_cls.__name__}`):\n\n'
        doc += f"{prop_cls.__doc__}\n\n"
    return doc


## nonlinear (perturbative) terms of the models with q = sqrt(p)

GAMMA = 5 / 3


class PerturbativeEnergies:
    """Energies of the perturbation of the full (nonlinear) system in the perturbative models: the kinetic energy with
    the total density, :math:`\\frac 12 \\int (\\rho_0 + \\tilde \\rho) |\\tilde{\\mathbf U}|^2`, the internal energy
    :math:`\\int (2 q_0 \\tilde q + \\tilde q^2) / (\\gamma - 1)` and their sum, plus the change of the magnetic energy
    :math:`\\frac 12 \\int (|\\mathbf B_0 + \\tilde{\\mathbf B}|^2 - |\\mathbf B_0|^2)` if ``magnetic_energies`` is given.

    Parameters
    ----------
    mhd : FluidSpecies
        The fluid species ``mhd`` of the model (``density``, ``velocity``, ``sqrt_p``).
    magnetic_energies : MagneticEnergies, optional
        Magnetic energies of the model with a magnetic field (default: None, no magnetic field).
    """

    def __init__(self, mhd, magnetic_energies: MagneticEnergies = None):
        self._mhd = mhd
        self._magnetic_energies = magnetic_energies

    def allocate(self):
        """Mass matrix of the total density and buffers (call in ``post_allocate`` of the model)."""
        self._kinetic_mass = VelocityMass(
            Propagator.derham,
            Propagator.domain,
            Propagator.mass_ops,
            Propagator.projected_equil,
            rho_field=self._mhd.density,
        )
        self._tmp_u: BlockVector = self._mhd.velocity.spline.vector.space.zeros()
        # linear form of q0 in V3, for int 2 q0 q
        self._M3_q0 = Propagator.mass_ops.M3.dot(Propagator.projected_equil.q3)

    def en_U(self):
        """Kinetic energy with the total density."""
        self._kinetic_mass.update()
        u = self._mhd.velocity.spline.vector
        return u.inner(self._kinetic_mass.M_rho.dot(u, out=self._tmp_u)) / 2

    def en_internal(self):
        """Perturbation of the internal energy."""
        q = self._mhd.sqrt_p.spline.vector
        q2 = q.inner(Propagator.mass_ops.M3.dot(q))
        return (2 * q.inner(self._M3_q0) + q2) / (GAMMA - 1)

    def en_tot(self):
        """Total energy of the perturbation."""
        en_tot = self.en_U() + self.en_internal()
        if self._magnetic_energies is not None:
            en_tot += self._magnetic_energies.en_B_tot() - self._magnetic_energies.en_B_eq()
        return en_tot


def add_perturbative_scalars(scalars: dict, mhd, energies: PerturbativeEnergies):
    """Replace the energies of the linear model by those of the perturbation of the full system: ``en_U`` (total
    density), ``en_q2``, ``en_internal`` and ``en_tot`` (kept last).

    Parameters
    ----------
    scalars : dict
        Scalars of the linear model, modified in place.
    mhd : FluidSpecies
        The fluid species ``mhd`` of the model.
    energies : PerturbativeEnergies
        Energies of the perturbation.
    """
    scalars.pop("en_tot")
    scalars["en_U"] = FunctionScalarFEEC(energies.en_U)
    scalars["en_q2"] = BilinearEnergyFEEC(mhd.sqrt_p, bilinear_form_name="M3", normalization=2 / (GAMMA - 1))
    scalars["en_internal"] = FunctionScalarFEEC(energies.en_internal)
    scalars["en_tot"] = FunctionScalarFEEC(energies.en_tot)


## hybrid MHD models with energetic ions (LinearMHD* and PerturbativeMHDWithQ* hybrids)


def add_energetic_ion_scalars(scalars: dict, mhd, energetic_ions, extra: dict = None, kinetic_name: str = "en_f"):
    """Add the scalars of the energetic ions to those of the fluid model: their kinetic energy ``en_f`` (normalized
    with the mass ratio :math:`A_\\textnormal{h} / A_\\textnormal{b}`), further energies ``extra`` (e.g. the magnetic
    moment energy of guiding centers), the total energy ``en_tot`` (the fluid one plus all of them, kept last) and the
    number of lost markers ``n_lost_particles``.

    Parameters
    ----------
    scalars : dict
        Scalars of the fluid model (with ``en_tot``), modified in place.
    mhd : FluidSpecies
        The fluid species ``mhd`` of the model (for the bulk mass number).
    energetic_ions : ParticleSpecies
        The energetic ions (species with the PIC variable ``var``).
    extra : dict, optional
        Further energies of the energetic ions (name -> scalar), added after ``en_f`` and to ``en_tot``.
    kinetic_name : str
        Name of the kinetic energy scalar (default ``"en_f"``).
    """
    en_tot = scalars.pop("en_tot")
    Ab = mhd.mass_number
    Ah = energetic_ions.var.species.mass_number
    energies = {kinetic_name: KineticEnergyPIC(energetic_ions.var, normalization=Ah / Ab), **(extra or {})}
    for name, energy in energies.items():
        scalars[name] = energy
        en_tot = en_tot + energy
    scalars["en_tot"] = en_tot
    scalars["n_lost_particles"] = LostMarkersPIC(energetic_ions.var)


def check_pressure_coupling_perp(model: StruphyModel):
    r"""Warn if the propagators ``push_eta_pc`` and ``pc6d`` of a pressure-coupling hybrid model use different
    ``use_perp_model``: the energy exchange is exact only if the transport :math:`(\mathbf v + \tilde{\mathbf U}_\perp)
    \cdot \nabla f_\textnormal{h}` and the pressure coupling use the same :math:`\tilde{\mathbf U}_\perp`.

    Parameters
    ----------
    model : StruphyModel
        Model with the propagators ``push_eta_pc`` (PushEtaPC) and ``pc6d`` (PressureCoupling6D).
    """
    perp_eta = model.propagators.push_eta_pc.options.use_perp_model
    perp_pc = model.propagators.pc6d.options.use_perp_model
    if perp_eta != perp_pc:
        logger.warning(
            f"{type(model).__name__}: use_perp_model differs between push_eta_pc ({perp_eta}) and pc6d ({perp_pc}); "
            "transport and pressure coupling then use different U_perp and the total energy is not conserved.",
        )


def add_binning_plot(params_path: str, binning_plot: str = "BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))"):
    """Replace the ``saving_params`` of a default parameter file by ``SavingParameters`` with one binning plot of the
    energetic ions (``binning_plot``, Python code).

    Parameters
    ----------
    params_path : str
        Path of the parameter file (modified in place).
    binning_plot : str
        Python code of the :class:`~struphy.io.options.BinningPlot`.
    """
    new_file = []
    with open(params_path, "r") as f:
        for line in f:
            if "saving_params = " in line:
                new_file += [f"\nbinplot = {binning_plot}\n"]
                new_file += ["saving_params = SavingParameters(binning_plots=(binplot,))\n\n"]
            else:
                new_file += [line]

    with open(params_path, "w") as f:
        for line in new_file:
            f.write(line)


# algorithms of the guiding-center pushers in the default parameter files of the drift-kinetic hybrids: the
# energy-conserving discrete gradient methods of the reference discretization, second order (implicit mid-point) for
# the drift perpendicular to b0, first order (Itoh-Abe) with Newton iterations for the parallel motion
GC_PUSH_ALGOS = {"push_bxe": "discrete_gradient_2nd_order", "push_parallel": "discrete_gradient_1st_order_newton"}


def set_guiding_center_push_algos(params_path: str, algos: dict = GC_PUSH_ALGOS):
    """Set the algorithms of the guiding-center pushers in a default parameter file, ``Options()`` ->
    ``Options(algo=...)``.

    Parameters
    ----------
    params_path : str
        Path of the parameter file (modified in place).
    algos : dict
        Propagator name -> ``algo`` (default :data:`GC_PUSH_ALGOS`).
    """
    with open(params_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for prop, algo in algos.items():
            if f"model.propagators.{prop}.Options()" in line:
                lines[i] = line.replace(
                    f"model.propagators.{prop}.Options()", f'model.propagators.{prop}.Options(algo="{algo}")'
                )
    with open(params_path, "w") as f:
        f.writelines(lines)


def energetic_ion_mu_energy(mhd, energetic_ions, PB, b_field: FEECVariable) -> float:
    """Magnetic-moment energy of the guiding centers, :math:`\\frac{A_\\textnormal{h}}{A_\\textnormal{b}} \\sum_p w_p
    \\mu_p |\\mathbf B|(\\boldsymbol \\eta_p)`, with the parallel component of the magnetic field perturbation
    ``PB b_field`` (local to this process, valid markers only; the weights carry the :math:`1/N_p`).

    Parameters
    ----------
    mhd : FluidSpecies
        The fluid species ``mhd`` of the model (for the bulk mass number).
    energetic_ions : ParticleSpecies
        The energetic ions (species with the ``Particles5D`` variable ``var``).
    PB : LinearOperator
        Parallel projection of the magnetic field perturbation to a 0-form (``basis_ops.PB`` or ``PB1form``).
    b_field : FEECVariable
        The magnetic field perturbation.
    """
    Ab = mhd.mass_number
    Ah = energetic_ions.var.species.mass_number
    particles = energetic_ions.var.particles
    particles.save_magnetic_energy(PB.dot(b_field.spline.vector))
    return particles.markers[5, particles.valid_mks].dot(particles.markers[8, particles.valid_mks]) * Ah / Ab


if __name__ == "__main__":
    fluid_models = get_models(model_type="Fluid")
    kinetic_models = get_models(model_type="Kinetic")
    hybrid_models = get_models(model_type="Hybrid")

    logger.info(f"Fluid models: {[mod.__name__ for mod in fluid_models]}")
    logger.info(f"Kinetic_models: {[mod.__name__ for mod in kinetic_models]}")
    logger.info(f"Hybrid models: {[mod.__name__ for mod in hybrid_models]}")

    model_message = generate_models_message()

    logger.info(model_message)
