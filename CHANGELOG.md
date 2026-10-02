# Changelog


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
