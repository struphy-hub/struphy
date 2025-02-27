.. _initial_conditions:

Initial conditions
------------------

A default parameter file for each model can be created with::

    struphy params MODEL

Assume a model features the fluid species ``mhd`` with three variables ``n3`` (density as 3-form),
``u2`` (velocity as 2-form) amd ``p0`` (pressure as 0-form). In the parameter file, the section
for setting the corresponding initial conditions reads as follows::

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

Available ``BACKR_NAMES`` along with their available ``BACKGR_PARAMS`` are listed in :ref:`equils_avail`.
Available ``PERT_NAMES`` along with their available ``PERT_PARAMS`` are listed in :ref:`avail_inits`.
Note the following:

* If a variable (e.g. ``p0``) is removed from the dictionary it will be initialized as zero.
* The initial condition of each appearing variable is the sum of all backgrounds and all perturbations listed under its name.
* If **the same (!)** ``BACKR_NAME`` appears multiple times under one varaible one must append ``_1``, ``_2`` to differentiate them in the code.
* One or both of the sections ``background`` and ``perturbation`` can be removed.

A valid example of the above structure reads as follows::

    fluid:
        mhd:
            perturbation:
                n3:
                    ModesCos:
                        given_in_basis: '0'
                        ls: [0, 1, 2, 4] 

Here, only the variable ``n3`` is initialized with a pertubration composed of 
cosines with the mode numbers 0, 1, 2, and 4 in the first direction, given as a 0-form.

A typical example of a ``kinetic`` initialization looks as follows::

    background : # at least one background is mandatory for kinetic species
        Maxwellian3D_1 :
            n  : 0.5
            u1 : 3.0
        Maxwellian3D_2 :
            n  : 0.5
            u1 : -3.0
    perturbation :
        n : 
            TorusModesCos :
                given_in_basis : '0' 
                ms : [1, 3] # two poloidal modes for the density

* Available kinetic backgrounds can be found in :ref:`kinetic_backgrounds`
* Available perturbations can be found in :ref:`avail_inits`

For ``kinetic`` species, the ``background`` is mandatory. 
The moments of :mod:`~struphy.kinetic_background.maxwellians` 
can be initialized with fluid equilibrium quantities. For this, the value
of the respective moment must be set to ``fluid_background``. For example::

    background : 
        Maxwellian3D :
            n  : 0.05
            u1 : fluid_background
            u2 : 2.5
            vth1 : fluid_background

In the above case, the first component of the mean- and thermal velocity are
initialized with fluid background quantities. The ``fluid_background`` specified
in the parameter file is then taken for initialization of the respective Maxwellian moment.

The moments of :mod:`~struphy.kinetic_background.maxwellians` 
can be also initialized with functions defined in :ref:`moment_functions`.
In this case the value of the respective moment must be a dictionary with the
function parameters, for instance:: 

    background : 
        Maxwellian3D :
            n  : 
                ITPA_density :
                    given_in_basis : '0'
                    n0 : 0.00720655
                    c : [0.491230, 0.298228, 0.198739, 0.521298]

Multiple ``background`` and ``perturbation`` types can be given as in the above fluid case.

Check out the `Maxwellian3D source code <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/kinetic_background/maxwellians.py?ref_type=heads#L119>`_ for more details.

Check out `this unit test <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/kinetic_background/tests/test_maxwellians.py?ref_type=heads>`_ for more information.


