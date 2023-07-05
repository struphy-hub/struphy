## Version 2.0.1

### Core changes

* Tracing of `removed` particles.

    When we choose `remove` as one of kinetic_bc, removed particles are saved to the array which is the property of `Particles` class, i.e. `self._e_ions.lost_markers` `self._e_ions.n_lost_markers`. !362

* `StruphyModel.print_plasma_params` now returns plasma parameters of ALL species. These are saved as the property `pparams` for each model. For kinetic species, the density and pressure are computed from temporary objects of `init` or `background` distribution, to capture default parameters of Maxwellians. !365

* New property `StruphyModel.eq_params` contains equation/coupling parameters (`alpha`, `kappa` etc) for each species. !365

* Clarified the definition of `beta` (actual ratio instead of percent) and the units used in `mhd_equils`. `EQDSKequilibrium` has to get the units for rescaling of output to Struphy units. !365

* Renamed `models.utilities.py` to `models.setup.py` !365

* Removed basic unit of mass, only return one `units` dictionary holding all Struphy units. !365

* Added two new properties of `Particles` base class: `f_init` and `f_backgr`. !365

* Modified the call function of the `Field` class. Now it has an additional parameter named `local`. If set to `True`, the Field will be evaluated on the local domain corresponding to the MPI rank, i.e. it returns a numpy array just of local FE size to the each process. !369

* Added 2 in the definition of the Gaussian. !371

* Add `spatial` keyword to kinetic parameter files with the two possible values `uniform` and `disc`. `uniform` draws markers uniformly in `eta`, while `disc` draws uniformly on a disc that has `eta1` as the radial coordinate. This option is used in `Particles.draw_markers()`, [here](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/pic-doc/src/struphy/pic/particles.py#L393). We use the inversion method with the cumulative distribution function to draw according to the pdf `s^n = 2*eta1`. !371

* - `kinetic_backgrounds.analytical` has been split into two new modules: `kinetic_backgrounds.base` and `kinetic_backgrounds.maxwellians`. !372

### Model specific changes

* New particle pushing routines are implemented for `LinearMHDDriftkineticCC` model:

    Accordingly, pusher wrapper class `Pusher_iteration` is splitted into two classes `Pusher_iteration_Gonzalez` and `Pusher_iteration_Itoh`.

    In order to calculate inverse of the 4x4 Jacobian which is needed for the new scheme, basic linear algebra calculation functions for 4x4 matrices are added. !362

* Modified model `LinearExtendedMHD` so now it conserves helicity. ! 366

* Added model `ColdPlasma` including analytical dispersion relation and one simple test case. !367

* Only do sg in `push_vxb_analytic` if b-field non-zero. !370

### Documentation, tutorials, etc.

* Added `Note` blocks to domain, MHD equilibria and kinetic background docstrings for copy/paste of needed parameters in parameters.yml file. !359

 * Now using the sphinx extension `:autosummary:` in doc. This gives a list of all classes/functions in a module.