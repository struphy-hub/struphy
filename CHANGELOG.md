## Version 2.0.2

### Core changes

* Implement boundary transfer function. When we choose `remove` as the kinetic boundary condition, optionally we can use
`boundary transfer` function: When the PIC particle reaches close to the polar axis (eta1 < rmin), the function transfers
the particle to the other side of the rmin circle at the polar axis. !377

* Added compile flag `--conda-warnings off` only if pyccel version is 1.8.0 or greater. !380

* Split `pusher_kernels.py` into itself (but with less content) and `pusher_kernels_gc.py` and split `accum_kernels.py` into itself (but with less content) and `accum_kernels_gc.py`. !380

* Added three new methods to `StruphyModel` base class: !380
    - `add_propagator`
    - `add_scalar`
    - `update_scalar`
    
    These methods improve the process of adding a new model.
    The `Propagator` class attributes `derham`, `domain`, `mass_ops` and `basis_ops`
    are now set in the base class. All models (except `VlasovMasslessElectrons`) now use the new methods of the StruphyModel base class.

* Throw only a warning if p=1 with BasisProjectionOperators, not assert !380

* Added the atttribute `pointer` to the StruphyModel base class. !382

    Pointer creation does not have to be done in the models anymore; rather, at `__init__()`,
    the base class crates a dictionary that can be called with `self.pointer[name]`, where `name`
    is the name of the species, or `species_variable` in the case of a fluid species.
    The old pointers have been removed from all models.

* Added `__init__(self, *vars)` to the `Propagator` base class. !382

    The init creates the two lists `self.feec_vars` (former `variables`)
    and `self.particles`, which serve as pointers within the scope
    of the Propagator. `isinstance` checks also happen in the new init.
    In the child Porpagators, the creation of pointers and asserts
    is replaced by a call to `super().__init__()`.
    All propagators have been adapted.

* Replaced `solvers.PoissonSolver` with `propagators_fields.ImplicitDiffusion`. !382

    The new propagator solves the heat equation in weak form, with implicit time stepping.
    In case the parameter `sigma=0.`, a `__call__()` with `dt=1.` solves the Poisson equation.

* Separated `Particles` base class into a `base.py` file. !382

* Added the method `_tmp_noise_for_mpi` to `Field`; this correctly initializes 1d noise with MPI (same noise regardless of number of processes). Not correct for 2d/3d noise yet. Added `seed` parameter noise input files under noise. !385

* Added wall clock and duration of last time step to screen output. !385

* Updated all docker images to use `:latest` Linux distro. Commented out the Fedora test for the moment, raised an [issue](https://github.com/pyccel/psydac/issues/323) with psydac import. !386

* Incorporate newest psydac changes, allow for `Python 3.11`. New psydac `v0.1.2` (our personal versioning, generated from (stefan-psydac fork)[https://github.com/spossann/stefan-psydac]). !386

* Now using `numpy 1.24.4`. Set solvability condition for Poisson in all codes to <1e-11. Added abstract method `conjugate` to `PolarVector`. 
Comment loop through `dir` of StencilMatrix in psydac_basics test. !386

* Setting struphy in/out/batch path !388

    `struphy --set-io .` was replaced with following three separate commands

        - struphy --set-i . (for input path)
        - struphy --set-o . (for the output path)
        - struphy --set-b . (for the batch path)

    `struphy --set-i d`, `struphy --set-o d` and `struphy --set-b d` are for returning back to our default paths.

* `struphy profile` can now save the output figure of the profiling, e.g. `struphy profile sim01 --savefig-dir sim01_profilefig.` !388

* Add a documentation for Performance_tests results !388


### Model specific changes

* New model: kinetic `VlasovMaxwell`, `VlasovMaxwell` coupling propagator and corresponding accumulation kernels. !378

* New model: hybrid `ColdPlasmaVlasov` !384

* generalize `propagators_coupling.VlasovMaxwell` by passing two arbitrary constants c1,c2​ instead of alpha and epsilon !384


### Documentation, tutorials, etc.

* Add field/fluid initial conditions in doc !379

* Added some more basics to doc of PIC discretization !380

* Improved developers documentation: info on Data structures, better guide to adding a new model. !382

### Repo struphy-simulations, new files:

* Scaling tests of `Maxwell` model with mpcdf cobra 

* Test simulations for `ColdPlasma`, `VlasovMaxwell`, and `ColdPlasmaVlasov` 