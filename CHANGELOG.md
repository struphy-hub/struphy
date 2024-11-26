## Version 2.3.3

Diff to previous release: Merge requests !568 - !591 


### User news

* Bug fixes !577, !582, !587.

* More flexible kinetic initialization: The parameters file now allows for perturbations for each kinetic background when multiple backgrounds are given. !559

* Enbale addition of constant pressure to GVEC equilibrium. ! 569

* Enhanced post-processing for 6D orbit data (calculation of guiding center positions, constants of motion). !550

* New particle sorting: done by assigning particles to boxes that subdivide the domain and then sorting according to the box number. !565

* FEEC diagnotics and Accumulation filtering: By adding staticmethod `diagnostics_dct()` at the model class, we can save a FEEC variable for a specific diagnostics. !534

* Automatic sampling parameters: Keyword moments can now be omitted from the parameters file for an automatic importance sampling. !583

* New notebook Tutorials. !589 


### Developer news

* New factory function `WeightedMassOperators.create_weighted_mass()`. This is to be used instead of `WeightedMassOperator()` when creating a new one. !545

*  Add pre-commit hooks. !573

* Two new CLI commands, `struphy lint` and `struphy format`, to check code statistics and enforce code formatting compliance. !574

* Remove Derham dependency from Particles: re-factoring of `Particles` class and `Pusher` class. !584 

* Preparation for `LinearMHDVlasovCC` and `PC` with `CanonicalMaxwellian`, and renewing Particle columns indexing. !580

