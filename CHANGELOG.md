## Version 2.3.1


### Core changes

* Addition of `Particles3D` class. !517

* Addition of `Constant` background function. !517

* New psydac v0.1.13: removed `igakit` from `pyproject.toml` (will not be installed anymore). !517

* New kinetic_background base class `CanonicalMaxwellian` and its test were added. !512

* Parallelized the DOFs evaluation for the local projection operators. !509

* Added new base class `KineticBackground` with the magic methods `__add__`, `__mul__`, `__rmul__` and `__sub__`.
The Maxwellians now inherit this base class, class `Maxwellian(KineticBackground)`. In the parameter file, this enables to specify multiple backgrounds of the same type by appending _1, _2 etc. to the type name. !510

* Modification at Particles5D marker array indexing. New property (abstractmethod) `bufferindex` is added at `Particles` class. !507

* Added dispersion relation `FluidSlabITG`, Tutorial 11 on dispersion relations, `BraginskiiEquilibrium` in new folder and `self.braginskii_equil` to `StruphyModel`. !506

* Several changes to streamline `PolarSplines`. !501

* Implementation of local commuting projection operators bsed on quasi-inter/histopolation. Added the `CommutingProjectorLocal` class to projectors.py !494

* Implemented Particle refilling: So far, it only transfers the particle to the opposite poloidal angle θ\thetaθ but now it also considers the toroidal angle ϕ\phiϕ in order to put the particle at the same magnetic flux surface (`particle_refilling`). !499

* Add a new `SchurSolver` to for the variational propagators. !498

* Changed the abstract methods for `LogicalMHDequilibrium` from `b2`, `j2` to covariant `b1` and `j1`. !495

* Added the new classes `DESCunit` and `DESCequilibrium` to interface to the [DESC equilibirum code](https://github.com/PlasmaControl/DESC). !495

* Add the entropy interfaces `s0` and `s3` to `MHDequilibrium`. !497

* Re-factored `ImplicitDiffusion` propagator; the right-hand side can now be a list of `StencilVector` or (`Auccumulator`, `Particles`)-tuple. !496

* New propagator `propagators_fields.AdiabaticPhi`. !479

* `WeightedMassOperators` has the new attribute `selected_weight` which can be set to change the object from which the weights are taken (usually `eq_mhd` or `eq_braginskii`, but more in the future). !479

* New method `MHDequilibrium.curl_unit_b_dot_b0` for the curvature as a 0-form. !479

* Renaming of kinetic backgrounds: `Maxwellian6D` -> `Maxwellian3D` and `Maxwellian5D` -> `GyroMaxwellian2D` !479

* New key `kBT` in the parameter `units`; this is used only if the velocity scale is set to `thermal`, which is a new option. !479

* `StruphyModel.update_distribution_functions()` now always calculates full-f and delta-f binnings from the weights `w0` and `w`, repsectively. The function `Particles.binning()` has been adapted accordingly. !479

* Renamed `Particles.f_backgr` -> `Particles.f0`. !479


### Model specific changes 

* Addition of `DeterministicParticleDiffusion` and `RandomParticleDiffusion` models, along with corresponding propagators. !517

* Add viscous and resistive terms to fluid and MHD models resulting in the two new models `ViscousFluid` and `ViscoresistiveMHD`. !513

* Implemented implicit forms in the transport operators of `Variational` models. !503

* New model `LinearVlasovAmpereOneSpecies` based Hamiltonian delta-f approach. !504

* Several updates to `Variational` models to run itpa test cases. !501

* Implemented control variate calculation for 5D hybrid model `LinearMHDDriftkineticCC`. !489

* The model `DriftKineticElectrostaticAdiabatic` has been added. !479


### Documentation, tutorials, testing, etc.

* Add a `VariationalMHD` test to Tutorial 04. !511



