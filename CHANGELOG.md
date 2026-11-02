# Changelog


## Struphy 3.6.0 - 2026-10-31

* [PyPI](https://pypi.org/project/struphy/3.6.0)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.6.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.5.0...v3.6.0)


### Headlines

* **Linear MHD with the magnetic field in H(curl)**: `LinearMHD` can use the curl-curl shear-Alfvén step of `ShearAlfven`, with the magnetic field in H(curl) and a Hiptmair-Xu preconditioner, or its exactly energy conserving projected variant; since #835 this is selected with `LinearMHD(b_space="Hcurl", projected=...)`. The PR adds the mixed mass matrix `WeightedMassOperators.M21J`, the perturbation `perturbations.Gaussian` and the examples `LinearMHD/slab_pulse_2d` (compared against an exact Fourier solution) and `LinearMHD/lowmode_tae`, and fixes the pressure update of `Magnetosonic` (see Bug fixes). https://github.com/struphy-hub/struphy/pull/813
* **Implicit Vlasov-Maxwell step with quasi-neutral limit**: the new propagator `VlasovMaxwellCoupling` combines Maxwell's equations and the velocity update of the markers in one Crank-Nicolson step, which conserves the energy exactly; with `lam=0` the displacement current is dropped (quasi-neutral limit). The new model `VlasovMaxwellQuasiNeutral` uses it, normalized with the cyclotron velocity scale and the equation parameters `lam` and `kappa`. https://github.com/struphy-hub/struphy/pull/821
* **Drift-kinetic Maxwell model**: the new propagator `DriftKineticMaxwellCoupling` is the drift-kinetic counterpart of `VlasovMaxwellCoupling`. It advances Maxwell's equations, the parallel acceleration of the guiding centers and, optionally, the E x B current in one energy conserving Crank-Nicolson step, and its polarization term makes the quasi-neutral limit regular (inertial Alfvén wave at `lam=0`). It is used by the new model `DriftKineticMaxwell` with the example `inertial_alfven_wave`; `PushGuidingCenterBxEstar` and `PushGuidingCenterParallel` accept an electric field `e_field` instead of `phi`. https://github.com/struphy-hub/struphy/pull/823
* **A model is defined by its equations**: new conventions in the developer guide and the agent guides. Equations that are already implemented get a new discretization as a model flag or propagator option; a model that adds terms to an existing one inherits from it, without flags that switch terms on or off. Model docstrings list the conserved quantities, propagator docstrings have three sections (continuous, semi-discrete, time discretization), and mass matrices are written `M^alpha_w`. Following these rules, `ShearAlfvenCurlCurl(withProj)` and `LinearMHDCurlCurl(withProj)` are merged into `ShearAlfven` and `LinearMHD` with the flags `b_space` and `projected`. https://github.com/struphy-hub/struphy/pull/835
* **Model families via inheritance**: a model whose equations extend those of another model now inherits from it. Species, propagators, variable links and scalars are created in hooks (`_create_species`, `Propagators`, `_create_scalars`, ...) that a child model extends with `super()`; the new `Propagators.splitting_order` places inherited and new propagators in the time splitting, and `Propagators.strang_groups` integrates groups of them symmetrically. This release contains four families, shown as trees in the model documentation:
  - **Vlasov**: `Vlasov` → `VlasovAmpereOneSpecies` → `VlasovMaxwellOneSpecies`; `Vlasov` → `VlasovPoissonAdiabatic` → `VlasovPoissonAdiabaticPauli`; `Vlasov` → `VlasovMaxwellQuasiNeutral`, `VlasovPauli` (new); `Vlasov` → `LinearVlasovAmpereOneSpecies` → `LinearVlasovMaxwellOneSpecies`;
  - **Guiding-center**: `GuidingCenter` → `DriftKineticElectrostaticAdiabatic`, `DriftKineticMaxwell`;
  - **LinearEuler**: `LinearEuler` (new) → `LinearMHD` → `LinearMHDVlasovCC`, `LinearMHDVlasovPC`, `LinearMHDDriftkineticCC`;
  - **Maxwell and SPH**: `Maxwell` → `ColdPlasma` → `ColdPlasmaVlasov`; `ColdEulerSPH` (was `PressureLessSPH`) → `EulerSPH` (new) → `ViscousEulerSPH`; `ColdEulerSPH` → `IncompressibleNavierStokesSPH`.

  Flags that switched terms on or off are removed and the scalar names are unified (`en_f`, `en_E`, `en_B`, `en_mu`, `en_tot`, ...); old run metadata is still read through `_renamed_propagators`, `_renamed_params` and `_renamed_species` (see API changes). The `new-model` skill asks whether a new model belongs to an existing family. https://github.com/struphy-hub/struphy/pull/839 (continued in https://github.com/struphy-hub/struphy/pull/840, https://github.com/struphy-hub/struphy/pull/850, https://github.com/struphy-hub/struphy/pull/851 and https://github.com/struphy-hub/struphy/pull/859)
* **Model regression tests**: the generic model tests (now in `src/struphy/models/tests/regression/`) compare the default run of every model against stored baselines on 1 and 2 MPI ranks (scalar time series, norms of all FE coefficients, marker moments; `rtol=1e-6`). A PR that intentionally changes results regenerates them with `pytest -m models --update-baselines`. On the way, MPI aborts in `Particles` now exit with a non-zero code (before, CI stayed green after an abort), and the default `dt` of `ViscousEulerSPH` is reduced to the stable value 0.001. https://github.com/struphy-hub/struphy/pull/855

### API changes

* The default `precond` of `ShearAlfvenPropagator.Options` and `ShearAlfvenCurrentCoupling5D.Options` is now `"StiffnessPreconditioner"` (the latter gets the option `exb_correction`). `CurrentCoupling5DCurlb.Options` and `CurrentCoupling5DDensity.Options` get `matrix_free` (default `True`), which applies the particle matrices without assembling them (new `MarkerOperator`); the default `ep_scale` of the 5D coupling options is `None`, i.e. the mass ratio `A_h/A_b`. `CanonicalMaxwellian2D` gets the argument `shifted` (default `True`). Together, a step of `LinearMHDDriftkineticCC` is about 2x faster, and a TAE example driven by energetic ions is added. https://github.com/struphy-hub/struphy/pull/809
* New perturbation `perturbations.Gaussian`, and `Magnetosonic(b_field=...)` accepts a magnetic field in H(curl) (with `u_space="Hdiv"`). https://github.com/struphy-hub/struphy/pull/813
* New model argument `divdiv_alpha` of `ViscoResistiveMHD` (now `VariationalMHD`) and `VariationalCompressibleFluid` for a div-div regularization of the kinetic energy, which penalizes compressions; for `divdiv_alpha > 0` the new scalar `en_divdiv` is part of `en_tot`. https://github.com/struphy-hub/struphy/pull/816
* New option `matrix_free` in `VlasovAmpereCoupling.Options`, `EfieldWeightsCoupling.Options`, `CurrentCoupling6DDensity.Options`, `CurrentCoupling6DCurrent.Options` (default `False`) and `PressureCoupling6D.Options` (default `True`). The default solver of `CurrentCoupling6DDensity.Options` changes from `"pcg"` to `"pbicgstab"`, since the operator is not symmetric. https://github.com/struphy-hub/struphy/pull/817
* The variational MHD models are renamed without aliases: `ViscoResistiveMHD(_with_p, _with_q)` → `VariationalMHD(_with_p, _with_q)`, `ViscoResistiveDeltafMHD(_with_q)` → `VariationalPerturbativeMHD(_with_q)`, and `ViscoResistiveLinearMHD(_with_q)` → `VariationalPerturbativeMHD(_with_q)(linear=True)`. The option values `"deltaf"`/`"deltaf_q"` of the variational propagators become `"perturbative"`/`"perturbative_q"`, and `VariationalQBEvolve.Options` gets `implicit_weights`. https://github.com/struphy-hub/struphy/pull/818
* The default `precond` of `MaxwellWeakAmpere.Options` is now `"StiffnessPreconditioner"`; the docstrings of `MaxwellWeakAmpere` and `VlasovAmpereCoupling` describe their discretization. https://github.com/struphy-hub/struphy/pull/819
* New solver choice `"pcg_sr"` wherever `"pcg"` is accepted (single-reduction PCG with the same iterates and one global reduction per iteration instead of two), from the feectools update of the GPU stack. https://github.com/struphy-hub/struphy/pull/782
* New `Output.num_clones`; runs with domain clones are post-processed serially. The PR adds the `Rwave_resonance` examples of `LinearMHDVlasovCC` and `LinearMHDVlasovPC`. https://github.com/struphy-hub/struphy/pull/831
* New options `schur_precond` (`"StiffnessPreconditioner"` is a Kronecker preconditioner for H1vec velocities) and `exb_correction` in `VariationalPBEvolve.Options`. `VariationalPerturbativeMHD_with_q` gets the flag `split_sound` (default `True`), which splits the step into the new propagators `VariationalAlfvenEvolve` and `VariationalSoundEvolve`. https://github.com/struphy-hub/struphy/pull/829
* The models `ShearAlfvenCurlCurl`, `ShearAlfvenCurlCurlwithProj`, `LinearMHDCurlCurl` and `LinearMHDCurlCurlwithProj` are removed: use `ShearAlfven(b_space="Hcurl", projected=...)` and `LinearMHD(b_space="Hcurl", projected=...)`. The propagator `ShearAlfvenCurlCurlwithProjPropagator` becomes `ShearAlfvenCurlCurlPropagator(projected=True)`, whose `precond` default is now `"auto"`; generated parameter files write all non-default model arguments. https://github.com/struphy-hub/struphy/pull/835
* The linear variational models `VariationalPerturbativeMHD(_with_q)` accept `u_space="Hdiv"`, which reuses the FEEC operators and the Kronecker Schur preconditioner of `LinearMHD`. `VariationalLinearAlfvenEvolve.Options` gets `relative_tol`, `schur_precond` and `exb_correction`, `VariationalLinearSoundEvolve.Options` gets `relative_tol`. https://github.com/struphy-hub/struphy/pull/836
* New option `weights` (`"midpoint"`, the new default, or `"frozen"`) in `VariationalLinearAlfvenEvolve.Options` and `VariationalLinearSoundEvolve.Options`; with midpoint weights each Crank-Nicolson substep is the implicit midpoint rule. The default `solver` of both is now `None`. https://github.com/struphy-hub/struphy/pull/838
* Vlasov family: the model flags `with_B0`, `with_E0` and `with_poisson` are removed (`VlasovPoissonAdiabaticPauli(with_poisson=False)` → `VlasovPauli`), and the scalars are renamed to `en_f`, `en_E`, `en_B`, `en_phi`, `en_tot` (e.g. `kinetic_energy` → `en_f`, `en_particles` → `en_f`, `en_w` → `en_f`). Post-processing scripts that read the old keys need updating. https://github.com/struphy-hub/struphy/pull/839
* Guiding-center family: the propagators are renamed `push_gc_bxe` → `push_bxe`, `push_gc_para` → `push_parallel` (`DriftKineticElectrostaticAdiabatic`) and `push_para` → `push_parallel` (`DriftKineticMaxwell`), and parameter files need the same rename; the scalars of `GuidingCenter` are `en_f`, `en_mu` (were `en_fv`, `en_fB`). https://github.com/struphy-hub/struphy/pull/840
* `LinearMHD` always splits the magnetosonic step into `sound`, `bg_pressure` and `bg_current`, integrated as one symmetric (Strang) block, so the flag `split_sound` and the propagator `mag_sonic` are removed and the default results change. https://github.com/struphy-hub/struphy/pull/850
* `LinearMHDVlasovCC`, `LinearMHDVlasovPC` and `LinearMHDDriftkineticCC` use the split magnetosonic step of `LinearMHD` (results change), `turn_off` is removed, the shear-Alfvén propagators are renamed to `shear_alf`, and `en_tot` contains the thermal energy `en_thermal`. https://github.com/struphy-hub/struphy/pull/851
* `PressureLessSPH` is renamed `ColdEulerSPH`; the flags `with_B0`, `with_p` and `with_viscosity` of the SPH models are removed (`with_p=False` → `EulerSPH` options with `kappa=0.0`), and the species are renamed `fluid` (SPH), `electrons` and `hot_electrons` (`ColdPlasmaVlasov`, with constructor arguments `charge_number`, `mass_number`, `alpha`, `epsilon`). The scalars of `Maxwell` and `ColdPlasma` are renamed to `en_E`, `en_B`, `en_J`, `en_tot`; old outputs are still post-processed through `_renamed_species`. https://github.com/struphy-hub/struphy/pull/859

### Physics models

* New models `LinearMHDCurlCurl` and `LinearMHDCurlCurlwithProj`, linear ideal MHD with the magnetic field in H(curl); since #835 they are `LinearMHD(b_space="Hcurl", projected=...)`. https://github.com/struphy-hub/struphy/pull/813
* `VariationalPerturbativeMHD` and `VariationalPerturbativeMHD_with_q` (perturbative and, with `linear=True`, linear variational MHD) are documented and verified, with a corrected energy diagnostic and a second-order time step of `VariationalQBEvolve`; the other variational MHD models are renamed `VariationalMHD`, `VariationalMHD_with_p` and `VariationalMHD_with_q`. https://github.com/struphy-hub/struphy/pull/818
* New model `VlasovMaxwellQuasiNeutral` with the new propagator `VlasovMaxwellCoupling`. https://github.com/struphy-hub/struphy/pull/821
* New model `DriftKineticMaxwell` with the new propagator `DriftKineticMaxwellCoupling`. https://github.com/struphy-hub/struphy/pull/823
* New propagators `VariationalAlfvenEvolve` (shear-Alfvén and fast waves) and `VariationalSoundEvolve` (sound waves), each energy conserving, for the split step of the q model; in the low-mode TAE a step takes 1.9 s instead of 4.2 s. https://github.com/struphy-hub/struphy/pull/829
* New propagators `VariationalLinearAlfvenEvolve` and `VariationalLinearSoundEvolve`, selected with `crank_nicolson=True` in the linear variational MHD models: one linear solve per substep instead of a Picard/Newton iteration, with exact energy conservation. https://github.com/struphy-hub/struphy/pull/833
* New propagators `SoundWaveLinear` (energy conserving sound wave step), `BackgroundCurrent` and `BackgroundPressureGradient` (exact in time), which split the magnetosonic step of `LinearMHD`. https://github.com/struphy-hub/struphy/pull/837
* New model `VlasovPauli`: classical Pauli particles with the mirror force and without Poisson solve, with the `tokamak_orbits` example. https://github.com/struphy-hub/struphy/pull/839
* New model `LinearEuler`, the linearized compressible Euler equations around a static equilibrium (propagators `sound` and `bg_pressure`), the parent of `LinearMHD`. https://github.com/struphy-hub/struphy/pull/850
* New model `EulerSPH` (`ColdEulerSPH` plus the pressure force), the parent of `ViscousEulerSPH`. https://github.com/struphy-hub/struphy/pull/859

### Bug fixes

* Fix the pressure update of `Magnetosonic`: the Crank-Nicolson factor 1/2 multiplied only one of the two terms, so `2/3 p0 div u` was applied twice and the effective adiabatic index was 7/3 instead of 5/3. This changes the results of `LinearMHD`, `LinearMHDVlasovCC`, `LinearMHDVlasovPC` and `LinearMHDDriftkineticCC` at finite beta (sound speed smaller by sqrt(5/7)). https://github.com/struphy-hub/struphy/pull/813

### Internals

* Remove the `@auto_convert_docstring` decorator, which replaced RST docstrings with HTML and broke their rendering in the Sphinx docs (e.g. of the mass matrices). https://github.com/struphy-hub/struphy/pull/812
* Fix the outdated mock in `models/tests/test_scalars.py` and add the module to a CI shard, where it was missing. https://github.com/struphy-hub/struphy/pull/811
* Variational MHD: the propagators sharing the density-weighted mass matrix `WMMnew` refresh their preconditioners when it is re-assembled (new `MassMatrixPreconditioner.refresh()`), fixes in `FFTSolver`, and `ViscoResistiveMHD` examples. https://github.com/struphy-hub/struphy/pull/815
* The Newton/Picard iterations of the variational propagators stop at a residual relative to the state, which matters for the small states of the linear and perturbative models; fix of `en_thermo` and low-mode TAE examples for the linear variational models. https://github.com/struphy-hub/struphy/pull/824
* Fix a hang of the Pauli verification test with C-compiled kernels on ranks without test markers. https://github.com/struphy-hub/struphy/pull/834
* GPU stack, part 7: chunked marker loading, a row-based weight update and leaner MPI sorting of the markers, and a static field in `PushVxB`. https://github.com/struphy-hub/struphy/pull/788
* `PushGuidingCenterParallel` and `DriftKineticMaxwellCoupling` use the total magnetic field `B0 + B~` in `B*`, so guiding centers stream along the perturbed field lines (magnetic flutter). https://github.com/struphy-hub/struphy/pull/828
* Fix the PyPI downloads badge in the README. https://github.com/struphy-hub/struphy/pull/843
* `VariationalPerturbativeMHD(linear=True, crank_nicolson=True, u_space="Hdiv")` supports viscosity and resistivity, and the dissipated energy heats the auxiliary pressure, so that the total energy is conserved to solver tolerance. https://github.com/struphy-hub/struphy/pull/844
* Shorter model verification workflows in CI; the clones workflow now runs only the tests marked `clones`, which really use clones. https://github.com/struphy-hub/struphy/pull/847
* The domain classes (e.g. `domains.Cuboid`) are visible to Pylance and other type checkers. https://github.com/struphy-hub/struphy/pull/845
* The `VlasovAmpereOneSpecies` examples with pseudo-random marker loading are seeded, so their results are reproducible. https://github.com/struphy-hub/struphy/pull/846
* `Simulation.run()` no longer frees the clone communicators, so a model with `num_clones > 1` can still be used after the run. https://github.com/struphy-hub/struphy/pull/848
* CI: PR examples run only on 2 ranks (all examples), serial examples run in the scheduled workflows. https://github.com/struphy-hub/struphy/pull/858
* CI: faster PR unit tests (testmon no longer reset by image rebuilds, OpenMP checks only when relevant, faster feec tests). https://github.com/struphy-hub/struphy/pull/860


## Struphy 3.5.0 - 2026-10-21

* [PyPI](https://pypi.org/project/struphy/3.5.0)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.5.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.4.0...v3.5.0)


### Headlines

* **Struphy on the feectools CUDA stack**: the `feectools` submodule moves to the feectools CUDA stack, which runs on the CuPy backend with MPI on device buffers, one GPU per MPI rank, and CUDA kernels for the stencil `dot`, `transpose`, `inner` and `axpy`. Field solves of a whole model can now stay on the GPU, also on several ranks. https://github.com/struphy-hub/struphy/pull/680
* **Geometry evaluation on the GPU for all analytic mappings**: CUDA versions of `f`, `df`, the metric, `pull`/`push`/`tran` and the four geometry entry kernels for all analytic mappings, so `Domain.__call__`, `jacobian_det` and the transforms run on the device. This lifts the Cuboid-only restriction of the CUDA kernels; before, particle runs on the CuPy backend failed at setup. https://github.com/struphy-hub/struphy/pull/681
* **Spline mappings on the GPU**: `IGAPolarCylinder`, `IGAPolarTorus`, `Tokamak`, GVEC and DESC domains can be created on the CuPy backend (the one-time setup runs on the host), and the spline mappings are evaluated on the device, so every mapping now runs on the GPU. https://github.com/struphy-hub/struphy/pull/683
* **MHD equilibria on the CuPy backend**: every equilibrium in `fields_background/equils.py` can be created and evaluated on the CuPy backend. SciPy-, gvec- and DESC-based setups and evaluations run on the host through the new helpers `setup_on_host`, `host_call` and `evaluate_on_host`; the time loop uses the projected equilibrium as before. https://github.com/struphy-hub/struphy/pull/714
* **Polar splines on the CuPy backend**: polar extraction, derivatives, commuting projectors (`schur`, `lu`, `ilu`) and mass preconditioners work on the GPU with device work arrays, cached device triangular solves and CUDA-aware MPI for the polar reductions. https://github.com/struphy-hub/struphy/pull/720
* **Kronecker stiffness preconditioner**: new `StiffnessPreconditioner` for operators of the form `d^T M d + sigma M`, built on the new Kronecker machinery of feectools (factors on groups of axes, `KroneckerSumSolver` by fast diagonalization). `MassMatrixPreconditioner` and `MassMatrixDiagonalPreconditioner` share one refactored implementation, and `MassMatrixPreconditioner` gets the option `weight_reduction="midpoint" | "average"`. https://github.com/struphy-hub/struphy/pull/731
* **CurlCurl model and curl-curl preconditioners**: new model `CurlCurl` for `(C^T M2 C + sigma M1) e = j`, with the new `HiptmairXuPreconditioner` and a geometric multigrid with Hiptmair smoother (`MultiGridOptions(hiptmair=True)`). With the Jacobi smoother, the multigrid is the only option whose iteration count is robust in `sigma` on the torus. https://github.com/struphy-hub/struphy/pull/744
* **Struct-of-arrays marker buffer**: `Particles.markers` is stored with one column per marker, shape `(n_cols, n_markers_max)` and indexed `markers[j, ip]`, so every marker quantity is one contiguous block on the CPU and on the GPU. https://github.com/struphy-hub/struphy/pull/774
* **Guidelines for AI coding agents**: coding guidelines for changing Struphy (`AGENTS.md` in the repository, plus guides for adding models and propagators), and usage guidelines shipped with the package. The new command `struphy agents` prints the usage guide, and `struphy agents --install` installs the `setup-simulation` and `postprocess-simulation` skills into a user's project. https://github.com/struphy-hub/struphy/pull/797

### API changes

* `ImplicitDiffusion.Options` and `PoissonSolve.Options` accept `precond="MultiGrid"` and a new field `multigrid: MultiGridOptions` for the new geometric multigrid preconditioner (MPI, 3D semi-coarsening, operators rebuilt on the coarse grids). The old `multigrid_solver` is removed. https://github.com/struphy-hub/struphy/pull/91
* `Compiler()` and `Compiler.compile()` take a new argument `debug` (default `False`), and `struphy compile --debug` calls pyccel with its `--debug` option. https://github.com/struphy-hub/struphy/pull/677
* `MultiGridOptions` is exported from `struphy`. `Output.evaluate()` drops the time dimension when `t` is a single number, and the `MultiGridSolver` wrapper is removed in favour of `inverse(..., pc=...)`. https://github.com/struphy-hub/struphy/pull/682
* MPI is now handled by [maybempi](https://github.com/max-models/maybempi): the environment variable that overrides the serial/MPI decision is `MAYBEMPI=0`/`1` (`STRUPHY_MPI` is no longer read), and the serial stand-in behaves like MPI on one rank instead of returning `None`. https://github.com/struphy-hub/struphy/pull/701
* The commands `struphy format` and `struphy lint` are removed; use `ruff check --fix` and `ruff format` directly. The options `--linters`, `--iterations` and `-y` of `struphy build-init-files` are removed. https://github.com/struphy-hub/struphy/pull/730
* The `precond` option of `ImplicitDiffusion.Options`, `PoissonSolve.Options` and `PoissonAdiabaticGyrokinetic.Options` now reaches the solver (it was ignored before) and defaults to `None`; the new option `precond_params` passes preconditioner settings, and `"StiffnessPreconditioner"` is a new choice. The default `stab_mat` of `PoissonSolve.Options` changes from `"Id"` to `"M0"`, and the default `precond` of `ShearAlfvenCurrentCoupling5D.Options`, `VariationalResistivity.Options` and `VariationalViscosity.Options` changes from `"MassMatrixDiagonalPreconditioner"` to `"MassMatrixPreconditioner"`. https://github.com/struphy-hub/struphy/pull/739
* New option `stab_average` in `ImplicitDiffusion.Options`, `PoissonSolve.Options` and `PoissonAdiabaticGyrokinetic.Options` sets the directions averaged in the stabilization (e.g. `"eta3"`). With it, `StiffnessPreconditioner` works for the adiabatic gyrokinetic Poisson operator, which is now symmetric. https://github.com/struphy-hub/struphy/pull/740
* The propagator `PoissonAdiabaticGyrokinetic` is renamed to `GyrokineticPoissonSolve` (module `struphy.propagators.gyrokinetic_poisson_solve`), without an alias. Its `Options` get a `multigrid` field, and `precond="MultiGrid"` now works together with `stab_average`. https://github.com/struphy-hub/struphy/pull/742
* `CurlCurlSolve` now solves `(C^T M2 C + sigma M1) e = j` (the mass term had a negative sign, which made the operator indefinite). Its `precond` option, which was ignored, now selects `None` (default), `"MassMatrixPreconditioner"`, `"StiffnessPreconditioner"`, `"HiptmairXu"` or `"MultiGrid"`, with new options `precond_params` and `multigrid`. `FEECVariable` gets the argument `projection="commuting" | "L2"`, `MultiGridOptions` the field `hiptmair`, and `TimeDependentSource` accepts variables in every FEEC space. https://github.com/struphy-hub/struphy/pull/744
* Polar splines use 2D x 1D Kronecker solves for the commuting projectors and mass matrices, so building a polar `Derham` takes 0.1 s instead of minutes. The new option `DerhamOptions.polar_poloidal_solver` (`"schur"` default, `"lu"`, `"ilu"`) selects the poloidal solver. https://github.com/struphy-hub/struphy/pull/747
* New `Simulation.compile_cuda_kernels()`, called by `Simulation.run()` after `allocate()`, compiles all CUDA kernels of the time loop before the first step and fails fast if one has no CUDA version; `Simulation.kernels()` lists them. https://github.com/struphy-hub/struphy/pull/711
* New option `enforce_compatibility` in `PoissonSolve.Options` (default `True`), `ImplicitDiffusion.Options` and `GyrokineticPoissonSolve.Options` (default `False`). It removes the net charge from the right-hand side of singular (e.g. periodic) solves; before, the constant mode of `phi` grew to `(net charge) / stab_eps`, which cost `E = -grad phi` most of its digits in PIC runs. https://github.com/struphy-hub/struphy/pull/737
* New `Output.standard_snapshots()` (first, middle and last saved time step), `Output.field_form()` and `Output.field_basis()`; the new argument `vtk_steps` of `Output.pproc()` writes VTK files only for selected time steps. The PR also adds the ITPA TAE benchmark examples for `LinearMHD` and `ShearAlfven`. https://github.com/struphy-hub/struphy/pull/398
* `ShearAlfvenPropagator.Options` accepts `precond="MultiGrid"` with a new field `multigrid` (experimental, not the default). The PR also adds physics descriptions to all examples and corrects the growth rates in the two-stream and Weibel post-processing, which were used as rates of the energies instead of the amplitudes. https://github.com/struphy-hub/struphy/pull/778
* The marker buffer `Particles.markers` is transposed to shape `(n_cols, n_markers_max)` and indexed `markers[j, ip]` (was `markers[ip, j]`); `Particles.n_rows` is renamed to `n_markers_max`. Scripts that access the markers directly have to be adapted. https://github.com/struphy-hub/struphy/pull/774
* `ShearAlfvenPropagator.Options` accepts `precond="StiffnessPreconditioner"`, a Kronecker preconditioner of the Schur operator, with the new option `exb_correction` (default `True`) for the auxiliary space of the E x B flows. https://github.com/struphy-hub/struphy/pull/781
* New command `struphy agents` (`--path`, `--install [DIR]`) for the agent usage guide and skills. https://github.com/struphy-hub/struphy/pull/797
* `struphy.api.models` now also exports `CurlCurl`, `ToyDrift` and `IncompressibleNavierStokesSPH`. https://github.com/struphy-hub/struphy/pull/798
* `GyrokineticPoissonSolve.Options` accepts `which_geometry=None` (Boltzmann response for all modes, no flux-surface average), and `ImplicitDiffusion.Options` accepts `stab_mat="M0ad_withT"` and `diffusion_mat=None`. https://github.com/struphy-hub/struphy/pull/800

### Physics models

* New model `GyrokineticPoisson` for the gyrokinetic Poisson equation with adiabatic electrons, built like `Poisson`, with profiling cases on a hollow cylinder and a hollow torus compared against an exact solution. https://github.com/struphy-hub/struphy/pull/742
* New model `CurlCurl` (`TimeDependentSource` followed by `CurlCurlSolve`, like `Poisson`). Its source uses the `L2` projection, so that the gradient part of the projection error is not amplified by `1/sigma`. https://github.com/struphy-hub/struphy/pull/744
* New models `ShearAlfvenCurlCurl` and `ShearAlfvenCurlCurlwithProj` (propagators `ShearAlfvenCurlCurlPropagator` and `ShearAlfvenCurlCurlwithProjPropagator`): the shear Alfven equations with the magnetic field in H(curl), so that the implicit step is a curl-curl solve. `ShearAlfvenCurlCurl` uses the Hiptmair-Xu preconditioner and has a small numerical damping; `ShearAlfvenCurlCurlwithProj` conserves the energy exactly. https://github.com/struphy-hub/struphy/pull/780
* New model `VlasovPoissonAdiabatic` (full-orbit ions, adiabatic electrons), the 6D counterpart of `DriftKineticElectrostaticAdiabatic`, with the new propagator `AdiabaticPoissonSolve` and `slab_drift_wave` examples for both models. The PR also fixes the full-f Poisson background. https://github.com/struphy-hub/struphy/pull/800 (renamed from `VlasovPoissonElectrostaticAdiabatic` in https://github.com/struphy-hub/struphy/pull/804)
* New model `VlasovPoissonAdiabaticPauli`: guiding-center ions computed with classical Pauli particles (Xiao and Qin, Comput. Phys. Commun. 265, 107981 (2021)), i.e. full-orbit markers with a magnetic moment (`PauliParticles6D`) pushed with time steps larger than the cyclotron period. It reuses the full-orbit pushers with the magnetic-moment force, adds the guiding-center polarization term to the quasi-neutrality equation (`AdiabaticPoissonSolve.Options(polarization=True)`), and comes with `tokamak_orbits` and `slab_drift_wave` examples. https://github.com/struphy-hub/struphy/pull/805

### Bug fixes

* Fix stale ghost regions after writes to a `PolarVector`: the `tp` setter and `set_tp_rings_to_zero` did not mark the ghost regions out of sync, so a following stencil `dot` read old ghost values. https://github.com/struphy-hub/struphy/pull/771
* Polar splines: the commuting projector no longer applies the boundary operator with `apply_bc=False` (wrong projections with Dirichlet BCs), the polar BiCGStab uses a relative tolerance, and the polar operators apply without allocations. https://github.com/struphy-hub/struphy/pull/746
* Fix the total energy at t=0 in the Vlasov-Ampere/Maxwell models: the ghost regions of the initial electric field were not updated (wrong first push near periodic boundaries), and the control-variate weights were reset to full-f after the initial Poisson solve.738 https://github.com/struphy-hub/struphy/pull/
* Fix the setup of sorting boxes on the CuPy backend, which failed in `SortingBoxes._set_boxes` and in the `ParticlesSPH` setup; the box sorting kernels move to kernel folders with an explicit host fallback on CuPy. https://github.com/struphy-hub/struphy/pull/736

### Internals

* CUDA strategy, part 6: `Particles` can be created on the CuPy backend, with `cuda_args_markers` for the CUDA kernels. https://github.com/struphy-hub/struphy/pull/651
* `struphy compile -j N` failed at random because a kernel dependency imported as a function (`from x_kernels import func`) was not detected; it now is. https://github.com/struphy-hub/struphy/pull/661
* CUDA strategy, part 7: `Derham` can be created on the CuPy backend; the data describing the spline spaces stays on the host, and only coefficients and stencil matrices live on the device. https://github.com/struphy-hub/struphy/pull/652
* CUDA strategy, part 8: the kernel argument classes are passed to CUDA kernels as C structs defined once in a shared header, instead of long flat signatures. https://github.com/struphy-hub/struphy/pull/653
* `LinOpWithTransp` is removed; struphy operators subclass the feectools `LinearOperator` and use its generic `toarray()`/`tosparse()`. https://github.com/struphy-hub/struphy/pull/662
* CUDA strategy, part 9: the pusher and evaluation kernels move to one folder per kernel (`pic/pushing/kernels/<name>/`), so each can get its CUDA version next to it. Code that imported e.g. `struphy.pic.pushing.pusher_kernels` has to import the kernel from its folder. https://github.com/struphy-hub/struphy/pull/663
* CUDA strategy, part 10: device helpers for B-splines, the Cuboid Jacobian, 3x3 linear algebra and the kinetic boundary conditions. https://github.com/struphy-hub/struphy/pull/665
* CUDA strategy, part 11: the first CUDA kernel, `push_eta_stage`, for explicit Runge-Kutta position pushes. https://github.com/struphy-hub/struphy/pull/666
* CUDA strategy, part 12: CUDA versions of `push_vxb_analytic` and `push_vxb_implicit`, and reflection and spline evaluation on the device, so the `Vlasov` model runs on the GPU for Cuboid mappings. https://github.com/struphy-hub/struphy/pull/667
* CUDA strategy, part 13: struphy's own kernel dispatch and argument classes are replaced by their [cunumpy](https://github.com/max-models/cunumpy) equivalents. https://github.com/struphy-hub/struphy/pull/668
* CUDA strategy, part 14: the accumulation kernels move to kernel folders, and the first Vlasov-Ampere CUDA kernels (`push_v_with_efield`, `push_weights_with_efield_lin_va`, `charge_density_0form`) are added, with CPU emulation tests of every CUDA kernel. https://github.com/struphy-hub/struphy/pull/670
* CUDA strategy, part 15: each kernel folder declares its kernel with cunumpy's `Kernel.from_folder`, and call sites import kernels instead of looking them up in catalogs. https://github.com/struphy-hub/struphy/pull/671
* CUDA strategy, part 16: one pyccel and one CUDA argument class per argument type, chosen by the owner at creation. https://github.com/struphy-hub/struphy/pull/679
* Refactor of the weight evaluation in `WeightedMassOperators.create_weighted_mass`, and 2D mass-matrix tests that use less than 1 GB of memory. https://github.com/struphy-hub/struphy/pull/684
* Move to cunumpy 0.6.1. https://github.com/struphy-hub/struphy/pull/685
* Refactor of `WeightedMassOperator.__init__`, which also fixes an `AttributeError` for symmetric operators with spline weights. `toarray()`/`tosparse()` now include the extraction and boundary operators; the matrix-only versions are `to_array_mat_only()`/`to_sparse_mat_only()`. https://github.com/struphy-hub/struphy/pull/702
* Update ruff to 0.16.10 with the previous lint rule set. https://github.com/struphy-hub/struphy/pull/732
* feectools inner products stay on the device on the CuPy backend, so a CG iteration copies one scalar to the host instead of two or more. https://github.com/struphy-hub/struphy/pull/716
* Fix two host/device array mix-ups in `SplineFunction.__init__` and `SortingBoxes._set_boxes` that stopped particle runs on the CuPy backend. https://github.com/struphy-hub/struphy/pull/709
* Fuse the pusher preamble into one CUDA kernel, about 2x faster on the GPU. https://github.com/struphy-hub/struphy/pull/708
* Marker exchange between MPI ranks on the device with CUDA-aware MPI (staged through host memory otherwise). https://github.com/struphy-hub/struphy/pull/712
* CUDA version of the `linear_vlasov_ampere` accumulation (`LinearVlasovAmpereOneSpecies`). https://github.com/struphy-hub/struphy/pull/705
* CUDA version of the `vlasov_maxwell` accumulation (`VlasovAmpereOneSpecies`, `VlasovMaxwellOneSpecies`). https://github.com/struphy-hub/struphy/pull/713
* Hand-written CUDA port of the guiding-center kernel `bstar_parallel_3form`; CUDA kernels are written by hand, not generated from the pyccel source. https://github.com/struphy-hub/struphy/pull/718
* New argument classes `SplineArguments`/`CudaSplineArguments` for the spline evaluation kernels of `SplineFunction`. https://github.com/struphy-hub/struphy/pull/721
* Field evaluation, scalars, binning and saving of the diagnostics run on the device on the CuPy backend, with one host copy per dataset and output step. https://github.com/struphy-hub/struphy/pull/724
* The FEEC assembly kernels move to one folder per kernel (`feec/kernels/<name>/`), with a host fallback on CuPy. https://github.com/struphy-hub/struphy/pull/725
* All pyccel kernels declare the arguments they write to (`outputs`), so the CuPy fallback copies back only those. https://github.com/struphy-hub/struphy/pull/723
* The CUDA device helpers live in C++ namespaces named after the pyccel modules they port. https://github.com/struphy-hub/struphy/pull/722
* End-to-end GPU test of `VlasovAmpereOneSpecies` on 1 and 2 ranks with a transfer budget for the time loop. It also fixes the ghost regions of the mid-point field in `VlasovAmpereCoupling` on every backend. https://github.com/struphy-hub/struphy/pull/734
* New benchmark script `profiling/benchmarks/vlasov_benchmark.py` for `VlasovAmpereOneSpecies` and `VlasovMaxwellOneSpecies` on the NumPy and CuPy backends. https://github.com/struphy-hub/struphy/pull/735
* Polar spline factorizations are shared between the solvers of one `Derham` (e.g. 88 sparse LU calls reduced to 9). https://github.com/struphy-hub/struphy/pull/760
* The GitLab CI at MPCDF now runs only the GPU tests; all CPU tests, linting and docs run on GitHub Actions. https://github.com/struphy-hub/struphy/pull/710
* Polar splines in `StiffnessPreconditioner` (grad, curl) and in `stab_average` (`PolarWeightedAverageProjection`), so `GyrokineticPoisson` runs with polar splines. It also fixes simulations on domains with a pole, which crashed on the renamed option `polar_ck`. https://github.com/struphy-hub/struphy/pull/758
* Profiling examples comparing polar and tensor-product splines for `Poisson`, `CurlCurl` and `GyrokineticPoisson` on a cylinder and a torus; fixes the post-processing of polar fields. https://github.com/struphy-hub/struphy/pull/759
* Particles on domains with a pole: markers are pushed through the axis, a new axis boundary map handles markers that cross it, and `PushVxB` and the guiding-center propagators work with polar splines. It also fixes the reflecting boundary for `Particles5D`, which overwrote the weights. https://github.com/struphy-hub/struphy/pull/761
* Profiling examples with long-time particle equilibria for `Vlasov` and `GuidingCenter` (cuboid, and cylinder and torus without hole). https://github.com/struphy-hub/struphy/pull/762
* Fix the OpenMP pragmas so that all kernels compile with `struphy compile --openmp`, and compile the feectools kernels with OpenMP too. https://github.com/struphy-hub/struphy/pull/757
* The profiling cases move from `profiling/examples/` to `examples/<Model>/<case>/` with `params_`, `pproc_` and `submit_` files; the diocotron example moves to `examples/ToyDrift/`. https://github.com/struphy-hub/struphy/pull/768
* The time-dependent examples save their standard figures at the first, middle and last saved snapshot, and run small with `--test-run`. https://github.com/struphy-hub/struphy/pull/773
* Pushers and accumulations process only the marker slots that can hold markers (`Particles.n_markers_used`), not the trailing holes of the buffer. https://github.com/struphy-hub/struphy/pull/775
* Faster setup of the Hiptmair-Xu preconditioner (7.6 s to 3.0 s per rank in the TAE example), with an exact stiffness diagonal and the new hook `Propagator.prepare_time_step(dt)`, which builds preconditioners before the time loop. https://github.com/struphy-hub/struphy/pull/796
* `DriftKineticElectrostaticAdiabatic` solves the Poisson equation between its two guiding-center pushers, so that with Strang splitting both pushes use a current potential. https://github.com/struphy-hub/struphy/pull/799
* The docs preview of PRs skips compiling Struphy and executing the notebooks; the quickstart gets its 2D figure. https://github.com/struphy-hub/struphy/pull/802
* The repository `struphy-parameter-files` is renamed to [struphy-examples](https://github.com/struphy-hub/struphy-examples), which is synced on every release; README and docs link it and the website. https://github.com/struphy-hub/struphy/pull/801
* Verification of the TAE frequency of `ShearAlfven`, `ShearAlfvenCurlCurl` and `ShearAlfvenCurlCurlwithProj` with a new low-mode TAE example; the curl-curl models use free boundary conditions for the magnetic field, which removes most of their energy loss. https://github.com/struphy-hub/struphy/pull/803


## Struphy 3.4.0 - 2026-10-01

* [PyPI](https://pypi.org/project/struphy/3.4.0)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.4.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.3.0...v3.4.0)


### Headlines

* **New lazy, xarray-based post-processing API**: `Output` replaces `PostProcessor`, `PlottingData`, `sim.pproc()` and `sim.load_plotting_data()`. `Simulation.run()` now returns an `Output`, and `Output("path/to/run")` or `open_output()` opens a finished run; post-processed fields, binned distributions, SPH densities and orbits are materialized on demand and returned as `xarray` objects with named dimensions, coordinates and units. https://github.com/struphy-hub/struphy/pull/370
* **Parallel kernel compilation**: `struphy compile -j N` (and `Compiler(jobs=...)`) compiles the pyccel kernels in parallel while preserving their dependency order; kernel dependencies are now detected by static AST parsing instead of importing the kernels. https://github.com/struphy-hub/struphy/pull/382
* **More efficient MPI sorting of markers**: new `Pusher` keywords `pushes_eta` (required) and `local_eval_only` call `mpi_sort_markers` only when markers have actually moved, cutting the number of sorts by 2x to 10x depending on the model with unchanged results, and giving good strong scaling for e.g. `VlasovAmpereOneSpecies` and `ColdPlasmaVlasov`. https://github.com/struphy-hub/struphy/pull/415
* **One folder per mapping**: each of the 14 domains now lives in its own folder under `struphy/geometry/domains/`, together with its own `<name>_kernels.py` for the mapping and Jacobian kernels; this prepares the translation to other backends such as CUDA. Imports like `from struphy.geometry.domains import Cuboid` still work. https://github.com/struphy-hub/struphy/pull/418 (merged via https://github.com/struphy-hub/struphy/pull/417)

### API changes

* `Output` and `open_output` replace `PostProcessor` and `PlottingData` in `struphy.__all__`. On `Simulation`, `load_plotting_data()` and `plotting_data` are removed, and `Simulation.output` and `Simulation.from_output(path_out)` are new. `Domain` gains `outer_boundary_mesh()`, and the `Variable` classes gain `to_dict()`. https://github.com/struphy-hub/struphy/pull/370
* `Compiler.__init__()` and `Compiler.compile()` take a new `jobs` argument (default 1, serial) for parallel kernel compilation. https://github.com/struphy-hub/struphy/pull/382
* New option `HasegawaWakataniStep.Options.coupling` (default `1.0`) sets the coupling coefficient `C` for `c_fun="const"`, which was previously hard-coded to zero. https://github.com/struphy-hub/struphy/pull/388
* The default of `parallel` in `Output.pproc()` and `Output.evaluate()` changed from `False` to `None`: post-processing now runs on all ranks when the MPI job has as many ranks as the saved run, and serially on rank 0 otherwise. https://github.com/struphy-hub/struphy/pull/410
* New option `fast` (default `False`) in `VariationalViscosity.Options` and `VariationalResistivity.Options`, replacing the leftover `fast` key of the old dict-based solver parameters. https://github.com/struphy-hub/struphy/pull/454
* The misspelled field `LoadingParameters.dir_exrernal` is renamed to `dir_external`, as documented. https://github.com/struphy-hub/struphy/pull/479
* `Simulation.spawn_sister()` takes a new `comm` argument, so a sister simulation runs on the same communicator as its parent instead of `MPI.COMM_WORLD`. https://github.com/struphy-hub/struphy/pull/556
* The `moment_factors` setters of `Maxwellian3D`, `GyroMaxwellian2D`, `GyroMaxwellian2Dvperp` and `CanonicalMaxwellian2D` now take a dict that is merged into the existing factors, e.g. `f0.moment_factors = {"n": 3.0}`; before, every assignment failed with a `TypeError`. https://github.com/struphy-hub/struphy/pull/627
* `CanonicalMaxwellian2D.eval_psic()` takes a new keyword `use_cbufs` (default `True`) to bypass the result buffers. https://github.com/struphy-hub/struphy/pull/637

### Physics models

* None.

### Bug fixes

* Fix `create_vtk` in post-processing crashing on `log10(0)` when no time steps were saved. https://github.com/struphy-hub/struphy/pull/381
* Fix stale values in nested scalar sums (`a + b + c`) in `Scalars.update()`, which caused apparent drifts in derived quantities such as the total energy. https://github.com/struphy-hub/struphy/pull/385
* `KineticEnergyPIC` and `KineticEnergySPH` no longer divide by `N_p` twice, which made the kinetic energy of every PIC and SPH model `N_p` times too small. https://github.com/struphy-hub/struphy/pull/387
* Fix `GyroMaxwellian2D.velocity_jacobian_det` for a callable `B0`, which now returns one value per marker. https://github.com/struphy-hub/struphy/pull/394
* Zero-initialise `norm_b_prod` in the pusher kernel `push_gc_cc_J2_stage_H1vec`; its diagonal held uninitialised memory, corrupting `CurrentCoupling5DGradB` with `u_space="H1vec"`. https://github.com/struphy-hub/struphy/pull/459
* Fix `PushRandomDiffusion` adding the full Wiener increment at every Runge-Kutta stage, which made the effective diffusion coefficient 16 D with `rk4`. Each stage now adds its `b[i]` fraction, and the default scheme is `forward_euler` (Euler-Maruyama). https://github.com/struphy-hub/struphy/pull/458
* Fix the neutralising background in full-f `ImplicitDiffusion`/`PoissonSolve` being multiplied by the mass matrix `M0` twice, which gave a large spurious initial potential in `VlasovAmpereOneSpecies`, `ToyDrift` and `DriftKineticElectrostaticAdiabatic`. https://github.com/struphy-hub/struphy/pull/460
* Fix the missing charge source in the initial Poisson solve of `LinearVlasovMaxwellOneSpecies`, which left the initial electric field at zero. https://github.com/struphy-hub/struphy/pull/457
* Fix `dot_inner_tp_rings` in `PolarLinearOperator` writing the first tensor-product ring on every rank, which silently corrupted coefficients when `eta1` is split across MPI ranks. https://github.com/struphy-hub/struphy/pull/462
* Fix the missing barotropic pressure term in `VariationalDensityEvolve` with `model="barotropic"`, which made `VariationalBarotropicFluid` behave like a pressureless fluid. https://github.com/struphy-hub/struphy/pull/463
* Fix crashes of `VariationalViscosity` and `VariationalResistivity` for any non-zero `mu`/`eta`, caused by dict-style access to the solver dataclass, a wrong argument to `eval_3form` and an in-place update on a `FEECVariable`. https://github.com/struphy-hub/struphy/pull/454
* Fix the missing `/poc` in the eta1-derivatives of the `HollowCylinder` Jacobian, which was wrong for any `poc != 1`. https://github.com/struphy-hub/struphy/pull/455
* Fix the missing factor `s` in the eta1-derivatives of the `PoweredEllipticCylinder` Jacobian, a 100% error for the default `s = 0.5`. https://github.com/struphy-hub/struphy/pull/464
* `Magnetosonic.allocate` no longer re-allocates a `b_field` passed in by the model, which dropped the initial conditions of the shared magnetic field in the linear MHD models. https://github.com/struphy-hub/struphy/pull/461
* Fix the missing toroidal-field term in `AxisymmMHDequilibrium.gradB_xyz`; the gradient of |B| for `EQDSKequilibrium` was off by about 4%. https://github.com/struphy-hub/struphy/pull/465
* `EQDSKequilibrium` now rescales the boundary flux `psi1` to Struphy units like `psi0`, fixing runs with non-default `base_units`. https://github.com/struphy-hub/struphy/pull/466
* Fix a copy-paste error in the `G[0,2]` metric entry of the `hybrid_weight` kernel, wrong for any mapping with `df[2,2] != 0` such as `Cuboid` or `Colella`. https://github.com/struphy-hub/struphy/pull/468
* Fix an `AttributeError` (undefined `_extracted_q2`) in `VariationalQBEvolve` with `linearize=True`, which made every linearized run fail on its first step. https://github.com/struphy-hub/struphy/pull/469
* Post-processing now tracks the saved markers, `f` and `n_sph` per kinetic species, fixing a `KeyError` when only some species save particle data. https://github.com/struphy-hub/struphy/pull/470
* Fix corner neighbour ranks left as `None` in `Particles._get_neighbouring_proc` for mixed periodic/non-periodic boundaries, which broke SPH ghost-box communication. https://github.com/struphy-hub/struphy/pull/471
* `ButcherTableau.a_stage` now raises `NotImplementedError` for the `"3/8 rule"` instead of silently dropping coefficients and degrading particle pushes to first order. https://github.com/struphy-hub/struphy/pull/472
* Fix `StencilMatrixFreeMassOperator.transpose` passing coefficient spaces instead of FEM spaces, which crashed `.T` and `transposed=True` for matrix-free weighted mass operators. https://github.com/struphy-hub/struphy/pull/473
* Fix the argument order in `AverageOperator.transpose`, which made `.T` always crash. https://github.com/struphy-hub/struphy/pull/478
* `KineticEnergyPIC` no longer counts the magnetic moment `mu` as a velocity for `Particles5D`, removing a spurious energy term in all drift-kinetic models. https://github.com/struphy-hub/struphy/pull/474
* Add the missing equilibrium gradient `grad_PBeq` in the `H1vec` branch of the accumulation kernel `cc_lin_mhd_5d_gradB`, so both `u_space` choices of `CurrentCoupling5DGradB` give the same physics. https://github.com/struphy-hub/struphy/pull/476
* The SPH linear smoothing-kernel gradients now vanish at zero separation, so particles no longer push on themselves. https://github.com/struphy-hub/struphy/pull/475
* Fix `loading="external"` for particles, which failed with a `TypeError` in `Particles.draw_markers`; the parameter field is renamed to `dir_external`. https://github.com/struphy-hub/struphy/pull/479
* Fix `BasisProjectionOperatorLocal.update_weights` for transposed operators, which kept applying the construction-time weights (e.g. in `BracketOperator`). https://github.com/struphy-hub/struphy/pull/480
* The Uzawa path of `TwoFluidQuasiNeutralFull` now includes `M2/dt` in its `A11` block, so it solves the time step instead of the steady problem. https://github.com/struphy-hub/struphy/pull/483
* `BasisProjectionOperatorLocal` now treats `None` weights as zero, fixing crashes of `BracketOperator` with `derham.with_local_projectors`. https://github.com/struphy-hub/struphy/pull/481
* ... and 67 more bug fixes, see the [closed PRs](https://github.com/struphy-hub/struphy/pulls?q=is%3Apr+is%3Aclosed).

### Internals

* `test_pproc` gets a `show_plot` safeguard so the test doesn't open plot windows. https://github.com/struphy-hub/struphy/pull/372
* Update the managed dependency bounds in `pyproject.toml`. https://github.com/struphy-hub/struphy/pull/374
* Lazy imports: `import struphy` and the public API objects, models and propagators are resolved on first access, and expensive imports are moved into the operations that need them, reducing Python startup time. https://github.com/struphy-hub/struphy/pull/383
* `M2Bn` now uses the equilibrium field components `eq_mhd.b2_*` instead of the curl of a projected vector potential, so the Hall operator keeps uniform background fields in periodic domains. https://github.com/struphy-hub/struphy/pull/384
* `LinearMHD` and `LinearExtendedMHDuniform` get a new scalar `en_thermal` (quadratic compressional energy) that replaces the linear pressure diagnostic `en_p` in `en_tot`, removing the spurious drift of the total energy. https://github.com/struphy-hub/struphy/pull/386
* Replace Python lists by scalars in `surface_kernel_3d_mat` of `mass_kernels.py`, removing the kernels' only dependency on gFTL and with it the need for `cmake >= 3.28` in `struphy compile`. https://github.com/struphy-hub/struphy/pull/389
* New features for quickly plotting the kinetic initial condition with the show-distribution-function utilities. https://github.com/struphy-hub/struphy/pull/390
* New `ProjectorNoBC` class that wraps projectors to prevent applying Dirichlet boundary conditions. https://github.com/struphy-hub/struphy/pull/395
* Rename the post-processing dimensions `e1`, `e2`, `e3` to `eta1`, `eta2`, `eta3`, and pass slices directly to `evaluate()` instead of `isel()`. https://github.com/struphy-hub/struphy/pull/406
* `Output` integrates the optional [plasma-plots](https://struphy-hub.github.io/plasma-plots) package: when it is installed, `out.plot`, `out.analysis` and the `.plasma` accessor on every product are available, e.g. `out.plot.energies()`. Install it with `pip install "struphy[pproc]"`. https://github.com/struphy-hub/struphy/pull/408
* Kinetic boundary conditions are applied per marker inside the pusher kernels by the new kernel `apply_kinetic_bc_marker`, speeding up position updates 2x-4x. https://github.com/struphy-hub/struphy/pull/412
* Particle kernel setup is refactored into explicit `KernelSetup` objects (`struphy/pic/pushing/kernel_setup.py`). https://github.com/struphy-hub/struphy/pull/413
* Bump the `feectools` dependency to 0.1.11, which was missed in #410. https://github.com/struphy-hub/struphy/pull/453
* CUDA strategy, part 1: proof of concept for running Struphy kernels on NVIDIA GPUs via CuPy next to the pyccel kernels (`CudaKernel` in `struphy/utils/kernel_backends.py`). https://github.com/struphy-hub/struphy/pull/643
* CUDA strategy, part 2: `CudaKernel.from_file()` loads CUDA kernels from `<name>_cuda.cu` files next to the pyccel module. https://github.com/struphy-hub/struphy/pull/644
* CUDA strategy, part 3: `KernelCatalog` discovers kernels from the file layout, and a missing CUDA kernel raises a clear error on the GPU backend instead of silently falling back to the CPU. https://github.com/struphy-hub/struphy/pull/645
* CUDA strategy, part 4: `Pusher` accepts a `Kernel` (pyccel/CUDA pair) and chooses the kernel for the active backend once at setup; CPU behaviour is unchanged. https://github.com/struphy-hub/struphy/pull/646
* CUDA strategy, part 5: `Domain.cuda_args_domain` builds the CUDA kernel arguments of a domain on the device, and `Domain` deep-copies correctly on the CuPy backend. https://github.com/struphy-hub/struphy/pull/649


## Struphy 3.3.0 - 2026-09-11

* [PyPI](https://pypi.org/project/struphy/3.3.0)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.3.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.2.0...v3.3.0)


### Headlines

* **Enable automated scaling test workflow**: running a profiling case across multiple MPI ranks, emitting per-run JSON metadata, packaging results into a publishable folder structure, and optionally uploading them to the struphy-hub/profiling-data repository. https://github.com/struphy-hub/struphy/pull/266
* **New tutorials and verification tests for SPH**: https://github.com/struphy-hub/struphy/pull/261
    - tutorials/tutorial_viscous_euler_sph.ipynb
    - tutorials/tutorial_velocity_diffusion_sph.ipynb
    - tutorials/tutorial_hagen_poiseuille_sph.ipynb
    - tutorials/tutorial_dam_break_sph.ipynb
* **New guiding-center 5D phase-space conventions**: switching from `(v_\parallel, v_\perp)` to `(v_\parallel, \mu)` as the standard coordinates in `Particles5D`. Use consistent `mu_idx` handling across particle kernels and PIC utilities. Introduced new `CanonicalMaxwellian2D` with caching capabilities. https://github.com/struphy-hub/struphy/pull/275
* **New surface (boundary) integral operators in the FEEC layer**: enable assembly/application of boundary mass operators needed for non-homogeneous derivative boundary conditions. https://github.com/struphy-hub/struphy/pull/303
* **Enable parallel post processing on each rank**. The default is still serial post-processing, which means that `parallel_pproc=True` needs to be passed for parallel work. https://github.com/struphy-hub/struphy/pull/309
* **Added memory estimates in struphy and feectools**: Adds per-rank memory footprint reporting for particles and key linear-operator types, including a dry_run construction mode for StencilMatrix to estimate allocation size without actually allocating. https://github.com/struphy-hub/struphy/pull/363


### API changes

* `Variables` have been entirely removed from `Propagator`'s `Options` classes. Such "Options"-variables were not updated by the `Propagator`, but nevertheless needed in the sub-step. From now on, such variables have to be passed to the  `Propagator` constructor, which is done internally in the model file. The user cannot pass these variables anymore in the launch file through `Options()`. https://github.com/struphy-hub/struphy/pull/262
* Added a switch in `EnvironmentOptions` that lets you turn off writing of restart checkpoint data, this is convenient for benchmarking. https://github.com/struphy-hub/struphy/pull/345

### Physics models

* New propagator `PoissonAdiabaticGyrokinetic` for the electric potential in gyrokinetic simulations with adiabatic electrons. https://github.com/struphy-hub/struphy/pull/260
* New examples for model `DriftKineticElectrostaticAdiabatic`: ITG in cylinderical geometry and cyclone base case (still issues due to noise). https://github.com/struphy-hub/struphy/pull/260
* New model for TAE energetic particle simulations: The new API now supports TAE energetic particle simulations with the model `LinearMHDDriftkineticCC`. https://github.com/struphy-hub/struphy/pull/275
* New model `IncompressibleNavierStokesSPH`: Incompressible Navier-Stokes equations solved using SPH with the Chorin projection scheme. The Poisson equation is solved on a grid with pressure in H1. This is a PIC scheme, not meshless anymore. https://github.com/struphy-hub/struphy/pull/284
* Three new verification tests for ``IncompressibleNavierStokesSPH`` (Chorin projection). https://github.com/struphy-hub/struphy/pull/308
* Added `Poisson` solver case to the profiling examples. https://github.com/struphy-hub/struphy/pull/302

### Bug fixes

* Fix bug in `mpi_sort_markers`: add missing `self.update_ghost_particles()` to get the correct number of valid_mks. https://github.com/struphy-hub/struphy/pull/261
* Fix bug in matrix-matrix multiplication during weights assembling in `WeightedMassOperators.create_weighted_mass()`. https://github.com/struphy-hub/struphy/pull/267
* Fix incorrect accumulation kernels for `basis_u == 1` and `basis_u == v` in drift-kinetic-MHD hybrid scheme. https://github.com/struphy-hub/struphy/pull/352 

### User news and internals

* For PIC, moved the factor `1/Np` into the definition of the weights. https://github.com/struphy-hub/struphy/pull/261
* New files for sph kernels: `eval_kernels_sph.py` and `pusher_kernels_sph.py`. https://github.com/struphy-hub/struphy/pull/261
* Added a new `Compiler` class wrapping the existing `struphy_compile` CLI logic for API use. https://github.com/struphy-hub/struphy/pull/281
* Added a `to_json()` method to the Simulation class. https://github.com/struphy-hub/struphy/pull/282
* Added workflow to build a docs preview for existing PRs. https://github.com/struphy-hub/struphy/pull/283
* Added a new `to_json()` method to the `Simulation` class. https://github.com/struphy-hub/struphy/pull/282
* Added a new workflow to build a docs preview for existing PRs. https://github.com/struphy-hub/struphy/pull/283
* New accumulation kernel `div_u_weak_1form` for accumulating the divergence of the velocity field; new class `ParticlesToGrid` for passing the accumulation parameters to a propagator in the model init; and new logic in `ImplicitDiffusion` for passing accumulation and filter parameters, respectively. https://github.com/struphy-hub/struphy/pull/284
* Renamed propagator `PushVinEfield` -> `PushVinForceField`. https://github.com/struphy-hub/struphy/pull/308
* Use config.json instead of files in postprocessing. Instead of exporting all the classes as pickled binary objects, export the main information about the simulation as a json file called `config.json`. https://github.com/struphy-hub/struphy/pull/314
* Nightly automatic check of dependency bounds. https://github.com/struphy-hub/struphy/pull/318 
* Enforce that feectools submodule should be pointing to the last commit. https://github.com/struphy-hub/struphy/pull/320
* Only allocate the full eval grid on rank 0. Grids are now rank local, `_create_eval_grids` returns global grids plus per-rank slices from `derham.domain_array`, so each rank holds only `grids_log_loc`. The physical grid is built on rank 0 only. https://github.com/struphy-hub/struphy/pull/322
* Added `setup/modules.json` for loading modules on known HPC systems. https://github.com/struphy-hub/struphy/pull/326
* Added setters to all properties of the `Simulation` class. https://github.com/struphy-hub/struphy/pull/336
* Added more profiling regions. https://github.com/struphy-hub/struphy/pull/344
* Fixed the `xp.all(axis=1)` bottleneck (10x speedup). https://github.com/struphy-hub/struphy/pull/347
* Column major fix of apply markers bc. Added a pre-allocated `self._eta_bc_buf = xp.zeros((self.n_rows, 3))` buffer (1.3x speedup). https://github.com/struphy-hub/struphy/pull/348
* `_sendrecv_get_destinations` now checks neighbor ranks first, and only falls back to checking the rest if some markers remain unmatched (up to 1.5x speedup). https://github.com/struphy-hub/struphy/pull/350 
* Updated profiling setup (`scoper-profiler`set to 0.5.0). Big improvements for GPU profiling and also line-by-line profiling for the regions we already defined using the tool without having to use `line_profiler` directly. https://github.com/struphy-hub/struphy/pull/354 and https://github.com/struphy-hub/struphy/pull/357
* Enable setting log file with environment variable. https://github.com/struphy-hub/struphy/pull/358
* Improve docstring helpers. Added `_html`, `_markdown`, `_latex` methods to the model/domain/equilibria/perturbation baseclasses. The `info()`, `pde()`, ... methods remain the same, they just use the `_html` methods and display the code. https://github.com/struphy-hub/struphy/pull/361
* Split up docstrings for the `Domain`, `Perturbation`, and `FluidEquilibria` classes, mirroring the pattern already used by `StruphyModel`. https://github.com/struphy-hub/struphy/pull/362




## Struphy 3.2.0 - 2026-06-09

* [PyPI](https://pypi.org/project/struphy/3.2.0)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.2.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.1.0...v3.2.0)


### Headlines

* New Quickstart, Userguide, Tutorials and Developer's guide in the documentation: https://github.com/struphy-hub/struphy/pull/252
* Each `Propagator` now sits in his own .py file: https://github.com/struphy-hub/struphy/pull/236
* New propagator `CurlCurlSolve()` for curl-curl problems: https://github.com/struphy-hub/struphy/pull/245
* New classmethods to inspect the model docstring (in a jupyter notebook for example): https://github.com/struphy-hub/struphy/pull/229

### API changes

* Change in the signature of `ParticleSpecies.set_markers()`, which is used in parameter files featuring particles. The new classes `SortingParameters` and `SavingParameters` replace the methods `set_sorting_boxes` and `set_save_data`. Instances of the new classes are passed to `set_markers()`: https://github.com/struphy-hub/struphy/pull/247

### User news

* Clean-up logging levels for simulation output: https://github.com/struphy-hub/struphy/pull/247
* New plotting functionality for kinetic backgrounds: https://github.com/struphy-hub/struphy/pull/239
* Addition of a matrix-free averaging operator for distributed FEEC data: https://github.com/struphy-hub/struphy/pull/246
* New default for `boxes_per_dim` is `tuple = (1, 1, 1)`: https://github.com/struphy-hub/struphy/pull/247

### Bug fixes

* Adaptation to general equilibria and new tests of the gyrokinetic Poisson solve: https://github.com/struphy-hub/struphy/pull/238




## Struphy 3.1.0 - 2026-04-24

* [PyPI](https://pypi.org/project/struphy/3.1.0)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.1.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.4...v3.1.0)


### Headlines

* Refactoring of the `Derham` class: https://github.com/struphy-hub/struphy/pull/212
    - Renamed `Nel -> num_elements`, `p -> degree`, `nq_pr -> nquads_proj`, `polar_ck -> polar_splines` 
    - The type of spline basis functions is now set via the argument `bcs`, which is a tuple of length three. It holds the type for each direction: `None` for periodic, or a tuple of length two with either `free` or `dirichlet` to indicate the type of clamped splines. `bcs` replaces the two arguments `spl_kind` and `dirichlet_bc`
    - The lifting is defined for each `FEECVariable` separately through the new attribute `lifting_function`
* Refactoring of `WeightedMassOperators.create_weighted_mass` method: https://github.com/struphy-hub/struphy/pull/227
    - The signature remains largely the same, except that list input for weights has been replaced by tuple input. Moreover, within the tuple, one can now pass a `SplineFunction` object, either from H1 or L2 space, which will be multiplied to the constant weights during `.assemble()`. **This allows for time dependent mass operators in nonlinear simulations (formerly treated with workarounds).**
    - In the constructor of `create_weighted_mass`, we now evaluate all callables derived from `weights: tuple[str]` on the integration grid, multiply them together and then pass them as xp.arrays to `WeightedMassOperator` for building the object. This leads to much simpler code than in the previous version, where we built composed functions and passed them as weights.
    - In `MassMatrixPreconditioner`, to build the Kronecker matrices from 1D matrices without domain decomposition, the values of the weights are now retrieved via an **MPI sub-communicator**, because the weights in M0, M1 etc. are now given as local xp.arrays, not as callables anymore.
* Use `logging` instead of print statements; use function `set_logging_level` from the API to set the logging level of all handlers. With pytest you can use `pytest --logging-level DEBUG src/struphy` to set the loggong level: https://github.com/struphy-hub/struphy/pull/199 and https://github.com/struphy-hub/struphy/pull/219
* New user guide: https://github.com/struphy-hub/struphy/pull/200
* Remove two submodules `struphy-parameter-files` and `struphy-tutorials` in favor of the new folders `examples/` or `tutorials/`: https://github.com/struphy-hub/struphy/pull/206

### API changes

* `base_units` is removed from the `StruphyModel` constructor; all equation parameters that can be seen in the model docstring can be passed to the model constructor: https://github.com/struphy-hub/struphy/pull/222


### User news

* Add `to_dict` and `from_dict` methods to the `Simulation` class: https://github.com/struphy-hub/struphy/pull/186
* Add `__repr__` and `__repr_no_defaults__` methods to most classes in the API: https://github.com/struphy-hub/struphy/pull/193
* Using `pyvista`; ddd `show_3d` and `create_geometry_mesh` methods to `Domain` class: https://github.com/struphy-hub/struphy/pull/195
* Add iterators to models and domains: https://github.com/struphy-hub/struphy/pull/196
* Added export and from_file methods to `SimulationBase` class: https://github.com/struphy-hub/struphy/pull/197
* Added name and description to the Simulation class: https://github.com/struphy-hub/struphy/pull/198
* New model `ToyGyrokinetic` to simulate the diocotron instability: https://github.com/struphy-hub/struphy/pull/201
* New classes for scalar quantities tracked during simulation (via new type `Scalar`): https://github.com/struphy-hub/struphy/pull/220
* The components of a model docstring are now available as class methods: https://github.com/struphy-hub/struphy/pull/229


### Bug fixes

* Weights in initial Poisson solves of kinetic models without control variate fixed: https://github.com/struphy-hub/struphy/pull/192




## Struphy 3.0.4 - 2026-02-27

* [PyPI](https://pypi.org/project/struphy/3.0.4)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.4)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.3...v3.0.4)

### Bug fixes

* New lower bound on pyccel is set to 2.2.0, due to this pyccel bug fix: https://github.com/pyccel/pyccel/pull/2567 




## Struphy 3.0.3 - 2026-02-23

* [PyPI](https://pypi.org/project/struphy/3.0.3)
* [GitHub Pages](https://struphy-hub.github.io/struphy/index.html)
* [GitHub release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.3)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.2...v3.0.3)

### Headlines

1. New class `Simulation` inherits the generic `SimulationBase`, both are in the new folder `struphy/simulations/`. The most important methods are:
    * `Simulation.run()`
    * `Simulation.pproc()` 
    * `Simulation.load_plotting_data()`
    * `Simulation.spawn_sister()` (my new favorite!)
    
These are tested in the new tutorials: https://github.com/struphy-hub/struphy-tutorials/tree/use-species-properties

The file `main.py` has been deleted.

The `Simulation` takes a model as input. Other API classes are passed as well (see tutorials). 
The model is viewed as everything related to the PDE, i.e. its variables, initial conditions etc. The simulation deals with the rest (geometry, derham, environment etc.)

Some important changes to the logic: The model does not have access to `derham`, `mass_ops` etc. anymore, these can be called from `Propagator` when needed. Solves that need to happen before the time stepping (like initial Poisson solves) are moved to `model.allocate_helpers()`.

2. The default launch file has been improved. See Tutorial 2 or test with `struphy params MODEL`. The files in the submodule `struphy-parameter-files` have been adapted.

3. Several new classes have been introduced for post processing and plotting data, see `post_processing_tools.py`. The most important ones are `PostProcessor` and `PlottingData`. Dictionaries in the plotting data have been replaced by classes. Many classes now feature the `__repr__` dunder for customized printing.


### API changes

New classes exposed: `Simulation`, `PostProcessor` and `PlottingData`.


### User news

* Add `set_zero_velocity` argument into `LoadingParameters`, enforcing velocities of all particles along specified axis to always be zero: https://github.com/struphy-hub/struphy/pull/176 
* New model `ViscousEulerSPH` replaces `EulerSPH`. The evaluation of the viscosity tensor has been implemented and tested for SPH methods. Unit tests for evaluation of the fluid velocity and its gradients (needed in the viscosity tensor) have been improved: https://github.com/struphy-hub/struphy/pull/160



## Struphy 3.0.2 - 2026-02-06

* [PyPI](https://pypi.org/project/struphy/3.0.2)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.2)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.1...v3.0.2)

### Headlines

* Added a public API. This allows imports like `from struphy import equils`: https://github.com/struphy-hub/struphy/pull/168
* New default compile language is Fortran: https://github.com/struphy-hub/struphy/pull/158
* Moved each model to its own file. Calling sub-processes must be avoided in the future because of incompatibility with MPI: https://github.com/struphy-hub/struphy/pull/152

### User news

* Added binning of higher order moments (current density, energy tensor) of f and delta f: https://github.com/struphy-hub/struphy/pull/162 

### Developer news

* Use `pyccel 2.1`: https://github.com/struphy-hub/struphy/pull/153
* Added three submodules: `struphy-parameter-files`, `struphy-tutorials` and`feectools`. The Struphy repo should be cloned with `git clone --recurse-submodules https://github.com/struphy-hub/struphy.git` to init and update the submodules. Also, run `git submodule update` regularly to get updates from the submodules. See https://github.com/struphy-hub/struphy/pull/154
* Introduced class `options.LiteralOptions` for parsing literals. Moved `Units` to `physics.py`: https://github.com/struphy-hub/struphy/pull/167


### Bug fixes

* Use `struphy.io.options.Units` in equils. This enables the use of GVEC, EQDSK and DESC in the new framework: https://github.com/struphy-hub/struphy/pull/158



## Struphy 3.0.1 - 2025-12-11

* [PyPI](https://pypi.org/project/struphy/3.0.1)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.1)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.0...v3.0.1)

### Headlines

`Psydac` is now installed via `pip` from our fork renamed to `feectools`, which is published on PyPI. This avoids installing from a `.whl` file.
Functionality remains unchanged, but future releases of Struphy are much easier and quicker. We have kept the option to install the upstream `psydac`, once it is also on PyPI.
See https://github.com/struphy-hub/struphy/pull/147.

### User news

None

### Developer news

* Removed legacy code (eigenvalue solver): https://github.com/struphy-hub/struphy/pull/129
* Add context manager to h5py.File() calls: https://github.com/struphy-hub/struphy/pull/135
* Fix undefined variables: https://github.com/struphy-hub/struphy/pull/141

### Bug fixes

* Fix setter in DESCequilibirum, update quickstart guide: https://github.com/struphy-hub/struphy/pull/132
* Set defaults for given_in_basis: "0" for scalar and "v" for vector-valued: https://github.com/struphy-hub/struphy/pull/136
* Fix the restart function: https://github.com/struphy-hub/struphy/pull/143


## Struphy 3.0.0 - 2025-11-13

* [PyPI](https://pypi.org/project/struphy/3.0.0)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v2.5.0...v3.0.0)

### Headlines

Struphy 3 represents a major refactoring with breaking changes with respect to Struphy 2, in particular:

* The `.yml` parameter files cannot be used anymore. Simulation parameters have to be transferred to the new `.py` launch files that are generated from `struphy params MODEL`. See the [Struphy README](https://github.com/struphy-hub/struphy) for a quick introduction.
* The console command `struphy run ...` has been deprecated. The new way to launch simulations is by executing the `.py` launch file, for instance with `python params_MODEL.py`.
* Other deprecated console commands are `struphy pproc` and `struphy units`. Post-processing is now done through the API via `main.pproc()`.
* The Struphy repo has moved to [Github](https://github.com/struphy-hub/struphy). The [old Gitlab repo](https://gitlab.mpcdf.mpg.de/struphy/struphy) will persist but not be maintained any longer. Issues, discussion and PRs will solely take place on the new Github repo.

### User news

* Please consult the [Struphy README](https://github.com/struphy-hub/struphy) and links therein to get familiar with the new workflows. 
* New tutorials can be found on [mybinder](https://mybinder.org/v2/gh/struphy-hub/struphy-tutorials/main).

### Developer news

Struphy has been refactored with the following principles in mind:

* get rid of console commands and increase the use of the Struphy API wherever possible
* become even more object-oriented
* use `Classes` instead of `dicts` wherever possible
* use `Literals` to show options for string arguments

In Struphy 3, models feature the following important objects:

* `ParticleSpecies`, `FieldSpecies`, `FluidSpecies`

Each species is a collection of Variables:

* `PICVariable`, `FEECVariable`, `SPHVariable`

These variables are updated by `Propagators`. All options for a simluation can be set in the new `.py` launch file.

### Bug fixes

* Incorporate psydac updates: https://github.com/struphy-hub/struphy/pull/109
* Auto install Psydac on first Struphy import: https://github.com/struphy-hub/struphy/pull/118
* Remove MPI Barrier responsible for deadlock: https://github.com/struphy-hub/struphy/pull/121 


## Struphy 2.6.0 - 2025-11-12

* [PyPI](https://pypi.org/project/struphy/2.6.0)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v2.6.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v2.5.0...v2.6.0)

### Headlines

* This is a test run for the relaease of Struphy 3.0 from the new Github repo


## Struphy 2.5.0 and prior releases

* See [Gitlab](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/releases)
