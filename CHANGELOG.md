## Version 2.0.4

### Core changes

* Allowing for a selection of FEEC coefficients saving !408

    We can save FEEC data selectively by inserting these lines at `parameters.yml`

    ```
    em_fields :
        save_data :
            comps :
                e1 : False
                b2 : True
    ```
    and similar fpr fluid species.

* Abstract methods `species(cls)` and `options(cls)` in StruphyModel !409

    This enables the new console command `struphy params MODEL` and the automatic testing of all models with their options
    via `struphy test`.

* Updated the `Pusher` class, removed the separate Pusher classes for `Itoh` and `Gonzalez` !409

* Changed the GVECequilibirum to GVECunit interaction. The former now calls the latter at init !409

* Added the parameter `rmin` to `GVECequilibirum`, which allows to have a hole around the magnetic axis !409

* Removed Psydac mappings entirely !409

* Removed the option `mhd_u_space` from all models solving `LinearMHD`. The associated Propagators still have this option though (for future use.) !409

* Improve exit condition for iterative pusher !410

    By counting and communicating number of not converged particles `n_not_converged`, iteration can exit the loop when all the particles are converged (`n_not_converged = 0`).


### Model specific changes

* Small adaptions in `LinearVlasovMaxwell` and `DeltaFVlasovMaxwell`. !396 


### Documentation, tutorials, etc.

* Added Tutorial 06 and added `field_line_tracing` docstring !409

* Many small improvements in doc


### Struphy-simulations, new files:

None.
