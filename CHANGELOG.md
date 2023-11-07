## Version 2.0.5

### Core changes

* Allow flexible choice of differential forms of initial distribution function and its binning !415

    By specifying `pforms : ['0', 'vol']`, i.e. the differential forms (0- or vol-form) of the distribution function in `[domain-space, velocity-space]`, one can now initialize the particle distribution as a volume-form. (So far, we are only allowed to initialize as 0-form.)

* Console argument completion enabled, using [argcomplete](https://github.com/kislyuk/argcomplete) !416

* Added class `L2_Projector` in `struphy.feec.projectors.py` (so far only in `H1` space) !417

* Added the method `add_option` to `StruphyModel` base class, more slick than adding every dict by hand !417

* Implemented the method `toarray_struphy` in `LinOpWithTransp` that uses the `dot` product to assemble the matrix corresponding to the linear operator as a `np.ndarray`. Asserts an MPI size of 1 for the moment !418

* Added new setter methods for marker array entries like `positions`, `velocities` !421

* Added the dictionary `Particles.index` for storing column indices of marker values !421

* Added Boolean `Particles.control_variate` and don't pass f0 to propagators - all info is in particles !421

* Moved `fields.Field` to `Derham.Field` as an inner class. A field can be created by calling the factory function `Derham.create_field()`. The file `fields.py` has been deleted !422

* Field initialization: the `init/comps` dict does not need to feature ALL variable names of a species anymore, but only those that are actually initialized !422

* Delete `Derham.send_ghost_regions`, has been replaced by `StencilVector.exchange_assembly_data` some time ago !422

* Replace `Derham.E` by `Derham.extraction_ops` !422

* Replace `Derham.B` by `Derham.boundary_ops` !422

* Replace `Derham.spaces_dict` by `Derham.space_to_form` !422

* Delete `Derham.V` and `Derham.forms_dict` !422

* Change `Derham.V` to `Derham.Vnames` !422

* Replace `quad_order` by `nquad` !422

* Re-factoring of Dirichlet boundary conditions ! 424

    In the parameter file we renamed `grid/bc` to `grid/dirichlet_bc` for clarity.
    Moreover, the entries must now be boolean (`True` if hom. Dirichlet are to be applied).
    It is possible to set `dirichlet_bc = None`, in which case the boundary operators in `Derham.__init__`
    are set as `IdentityOperators`.

* Renamed `geom_projectors.Projector` to `geom_projectors.PolarCommutingProjector` !424

* All input files in `io/inp/tests` have been removed !424

* Renamed `struphy/psydac_api` to `struphy/feec` !424

* Added the options `--short-help`, `--fluid`, `--kinetic`, `--hybrid` and `--toy` to the command line !424

* Expose `dofs_extraction_ops` in Derham !424


### Model specific changes

* Added control variate for model `VlasovMaxwell` !417 

* Implemented the new toy model `ShearAlfven`. It is composed of only the `ShearAlfven` propagator !418


### Documentation, tutorials, etc.

* Updated doc-strings for guiding-center models model (models, propagators, accumulation kernels, pusher kernels ...) !419

* Added `Tutorial 08` on Struphy data structures !421

* Added `Tutorial 09` on the discrete deRham sequence !422


### Struphy-simulations, new files:

None.
