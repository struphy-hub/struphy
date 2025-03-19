## Version 2.4.1

### Headlines

* Implemetation of the SPH method - see Tutorial 02 and the new model `IsothermalEulerSPH`. This required refactoring `Particles.__init__` and other parts of the base class !632

* Sound wave verification test for `IsothermalEulerSPH` !636

* New class `Tesselation`. The new marker loading mode "tesselation" allows to draw markers on a regular grid given by the center-of-mass points of a tesselation; unit tests were added. New entries for initializing weights to parameters.yml and in Particle base class: 
```
    weights : 
        reject_weights : False # reject particles with a weight < threshold
        threshold : 0.0
        from_tesselation : False # compute weights from cell averages over a tesellation
```
!636


### User news

* New variational MHD model using the pressure variable instead of the entropy.
The model equivalent at the continuous level and have only small changes at the discrete level, requiring only the addition of a `VariationalPressureEvolve` and minor changes to the dissipative propagators !613

* Updated docker images !619

* More general fluid backgrounds: `FluidEquilibrium`, `FluidEquilibriumWithB` and `MHDequilibrium` !627

* Use optional dependencies for test, dev and doc. The command `pip install .` will just make a base installation for running code; unit testing can be enabled with `pip install .[test]`; usual development (testing + linting + formatting) is enabled by `pip install .[dev]`; building the doc is enabled by `pip install .[doc]`. These can be also combined - for example: you get the full version (as until now) via `pip install .[dev, doc]`. See also the updated install doc !609

* Restructure how parameters are passed to `background:` and `perturbation:`. The `type` keyword is removed from the parameter file; `comps` is replaced by `given_in_basis`: Moreover, allow passing `bckgr_params` and `pert_params` to `initialize_weights()` !626

* Improved console diagnostics !639

* New Particles sub-class `DeltaFParticles6D`. The Particle type is now set automatically at the init of the Particles class. In the parameter file we replaced `markers["type"]` keyword by `markers["control_variate"]` boolean !636


### Developer news

* Added pre-commit hooks !573

* Cleanup residual code from old versions of Vlasov Ampere / Maxwell !630

* Added the flag `--time-execution` to struphy compile. This enables the timings of each step of the pyccelization of the kernels in struphy.

* New option `--verification` in `struphy test`.
This option allows to call parameter files that are specified in io/inp/verification/. Added weak Landau damping verification tests for `VlasovAmpereOneSpecies` and `LinearVlasovAmpereOneSpecies`. `struphy test <model_name>` can now be called with `--mpi N` and any other option from test !633

* Refactored domain cloning !634

* Re-factoring of `geometry.evaluation_kernels`: `det_df`, `df_inv`, `g` and `ginv` can now be directly used in particle kernels


### Bug fixes

* Landau damping: pass correct equation parameters to propagator `VlasovAmpere` !628

