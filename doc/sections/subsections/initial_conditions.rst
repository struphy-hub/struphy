.. _initial_conditions:

Initial conditions
------------------

A default parameter fiel for each model ca be created with::

    struphy params MODEL

Suppose a model featuring a fluid species ``mhd`` with three variables ``n3`` (density as 3-form),
``u2`` (velocity as 2-form) amd ``p0`` (pressure as 0-form). In the parameter file, the section
for setting their initial conditions reads as follows::

    fluid:
        mhd:
            background:
                n3:
                    BACKGR_NAME_1:
                        BACKGR_PARAMS_1
                    BACKGR_NAME_2:
                        BACKGR_PARAMS_2
                u2:
                    BACKGR_NAME_1:
                        BACKGR_PARAMS_1
                    BACKGR_NAME_2:
                        BACKGR_PARAMS_2
                p0:
                    BACKGR_NAME_1:
                        BACKGR_PARAMS_1
                    BACKGR_NAME_2:
                        BACKGR_PARAMS_2
            perturbation:
                n3:
                    PERT_NAME_1:
                        PERT_PARAMS_1
                    PERT_NAME_2:
                        PERT_PARAMS_2
                u2:
                    PERT_NAME_1:
                        PERT_PARAMS_1
                    PERT_NAME_2:
                        PERT_PARAMS_2
                p0:
                    PERT_NAME_1:
                        PERT_PARAMS_1
                    PERT_NAME_2:
                        PERT_PARAMS_2

Available ``BACKR_NAMES`` along with their available ``BACKGR_PARAMS`` are listed in :ref:`fluid_backgrounds`.
Available ``PERT_NAMES`` along with their available ``PERT_PARAMS`` are listed in :ref:`avail_inits`.
Note the following:

* If a variable (e.g. ``p0``) is removed from the dictionary it will be initialized as zero.
* The initial condition of each appearing variable is the sum of all backgrounds and all perturbations listed under its name.
* If **the same (!)** ``BACKR_NAME`` appears multiple times under one varaible one must append ``_1``, ``_2`` to differentiate them in the code.
* One or both of the sections ``background`` and ``perturbation`` can be removed.

A typical example of a ``kinetic`` initialization looks as as follows::

    background : # background is mandatory for kinetic species
        type : [Maxwellian3D_1, Maxwellian3D_2]
        Maxwellian3D_1 :
            n  : 0.5
            u1 : 3.0
        Maxwellian3D_2 :
            n  : 0.5
            u1 : -3.0
    perturbation :
        type : TorusModesCos
        TorusModesCos :
            comps :
                n : '0' # perturbation function given as 0-form 
            ms : # poloidal mode numbers
                n : [1, 3] # two poloidal modes for the density

* Available kinetic backgrounds can be found in :ref:`kinetic_backgrounds`
* Available perturbations can be found in :ref:`avail_inits`

For ``kinetic`` species, the ``background`` is mandatory. 
The moments of :mod:`~struphy.kinetic_background.maxwellians` 
can be initialized with MHD equilibrium quantities. For this, the value
of the respective moment must be set to ``mhd``. For example::

    background : # background is mandatory for kinetic species
        type : Maxwellian3D
        Maxwellian3D :
            n  : 0.05
            u1 : mhd
            u2 : 2.5
            vth1 : mhd

In the above case, the first component of the mean- and thermal velocity are
initialized with MHD quantities. An ``mhd_equilibrium`` must be specified
in the parameter file in this case.

Multiple ``background`` and ``perturbation`` types can be given as in the above fluid case.

Check out the `Maxwellian3D source code <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/kinetic_background/maxwellians.py?ref_type=heads#L119>`_ for more details.

Check out `this unit test <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/kinetic_background/tests/test_maxwellians.py?ref_type=heads>`_ for more information.


