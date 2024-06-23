.. _initial_conditions:

Initial conditions
------------------

Initial conditions in Struphy must be set via the ``.yml`` parameter file,
see :ref:`params_yml` under "Species parameters".

Initial conditions are always the sum of ``background`` + ``perturbation``, regardless
whether the species is ``kinetic``, ``fluid``, or ``em_fields``.

For ``kinetic`` species, the ``background`` is mandatory!

For ``fluid`` or ``em_fields`` species, when neither ``background`` nor ``perturbation``
is given, the species is initialized with zero. 


Fluid initialization
^^^^^^^^^^^^^^^^^^^^

A typical example of a ``fluid`` or ``em_fields`` initialization looks as as follows::

    background: 
        type : LogicalConst
        LogicalConst :
            comps :
                potential_name : 1.3 # scalar-valued variable
                field_name : [.3, .15, null] # vector-valued variable
    perturbation :
        type : TorusModesCos
        TorusModesCos :
            comps : # components to be initialized
                potential_name : '0' # perturbation function given as 0-form 
                field_name : [null, 'v', null] # second component given as vector field, others zero
            ms : # poloidal mode numbers
                potential_name : [1] # one poloidal mode
                field_name : [null, [1, 3], null] # two poloidal modes for the second component 

* Available fluid backgrounds can be found in :ref:`fluid_backgrounds`
* Available perturbations can be found in :ref:`avail_inits`

Multiple ``background`` and ``perturbation`` types can be given by passing a list to ``type``.
If the same ``background`` type appears multiple times in the list, ``_1``, ``_2`` etc. 
must be appended to the type to differentiate the corresponding dictionaries holding the parameters.
This feature does not yet work for ``perturbation``, thus multiple perturbations of the same
type are not yet supported.
The contributions are summed up to give the initial value of the field. 
For example::

    background: 
        type : [LogicalConst_1, LogicalConst_2, MHD]
        LogicalConst_1 :
            comps :
                potential_name : 1.3 
                field_name : [.3, .15, null] 
        LogicalConst_2 :
            comps :
                other_name : 0.2 
        MHD :
            comps :
                potential_name : n0
    perturbation :
        type : [TorusModesCos, TorusModesSin]
        TorusModesCos :
            comps : 
                potential_name : '0' 
                field_name : [null, 'v', null] 
            ms : 
                potential_name : [1] 
                field_name : [null, [1, 3], null] 
        TorusModesSin :
            comps : 
                potential_name : '0'  
            ns : 
                potential_name : [2, 4] 


Kinetic initialization
^^^^^^^^^^^^^^^^^^^^^^

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


