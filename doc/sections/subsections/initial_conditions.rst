.. _initial_conditions:

Initial conditions
------------------

Initial conditions in Struphy must be set via the ``.yml`` parameter file,
see :ref:`params_yml` under "3. Species parameters".

Initial conditions are always the sum of ``background`` + ``perturbation``.
For ``kinetic`` species, the ``background`` is mandatory. 
For ``fluid`` or ``em_fields`` species, when neither ``background`` nor ``perturbation``
is given, the species is initialized with zero. 

For a chosen ``perturbation``, one has to declare the components of the :ref:`model variables <species>`
that will be inititialized with it. This is done under the dictionary ``comps`` (see below).
The keys in ``comps`` must be variable names defined in the model, or
the :ref:`names of moments <kinetic_backgrounds>` of the background distribution in case of ``kinetic`` species . 
The values in ``comps`` are either a string (for scalar-valued variables) or a list of strings (for vector-valued variables).
Such a string indicates the basis in which the perturbation function is described, see :ref:`pullback`, for each 
component in ``comps``. For scalar-valued functions, one can choose 

* ``'0'``: perturbation is a regular function :math:`\hat n(t=0, \boldsymbol \eta)`.
* ``'3'``: perturbation is 3-form (volume-form) :math:`\hat n^3(t=0, \boldsymbol \eta) = \sqrt g\,\hat n(t=0, \boldsymbol \eta)`.
* ``'physical'``: perturbation is a function on the physical domain :math:`n(t= 0, \mathbf x) = n(t=0, F(\boldsymbol \eta))`.

For vector-valued functions, for each component one can choose

* ``null``: component is initialized as zero.
* ``'v'``: perturbation is the component of contra-variant function (vector field) :math:`\hat E^i(t=0, \eta)`.
* ``'1'``: perturbation is the component of a co-variant function (1-form) :math:`\hat E_i(t=0, \eta)`.
* ``'2'``: perturbation is the component of a pseudo-vector (2-form) :math:`\hat E^{2}_i(t=0, \eta)`.
* ``'physical'``: perturbation is the Cartesian component of a function on the physical domain :math:`E_i(t= 0, \mathbf x) = E_i(t=0, F(\boldsymbol \eta))`.
* ``'physical_at_eta'``: perturbation is the Cartesian component of a function on the logical domain :math:`E_i(t= 0, \boldsymbol \eta)`.
* ``norm``: the perturbation is the component of a function given in the normalized contra-variant basis (:math:`\delta_i / |\delta_i|`).

The transformation from the basis specified in ``comps`` to the basis of the variable's solution space is done internally.
More information and plots can be obtained by running `this unit test <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/initial/tests/test_init_perturbations.py?ref_type=heads>`_.


Fluid initialization
^^^^^^^^^^^^^^^^^^^^

A typical example of a ``fluid`` or ``em_fields`` initialization looks as as follows::

    background: 
        type : LogicalConst
        LogicalConst :
            comps :
                density : 1.3 
                velocity : [.3, .15, null] 
    perturbation :
        type : TorusModesCos
        TorusModesCos :
            comps : 
                density : '0' 
                velocity : [null, 'v', null] # second component given as vector field, others zero
            ms : # poloidal mode numbers
                density : [1] # one poloidal mode
                velocity : [null, [1, 3], null] # two poloidal modes for the second component 

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
                density : 1.3 
                velocity : [.3, .15, null] 
        LogicalConst_2 :
            comps :
                other_name : 0.2 
        MHD :
            comps :
                density : n0
    perturbation :
        type : [TorusModesCos, TorusModesSin]
        TorusModesCos :
            comps : 
                density : '0' 
                velocity : [null, 'v', null] 
            ms : 
                density : [1] 
                velocity : [null, [1, 3], null] 
        TorusModesSin :
            comps : 
                density : '0'  
            ns : 
                density : [2, 4] 


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


