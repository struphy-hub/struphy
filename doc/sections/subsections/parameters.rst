.. _params_yml:

Simulation parameters
---------------------

All Struphy simulation parameters can be set in the ``.yml`` parameter file. For each model,
a default file can be created via::

    struphy params MODEL

The file is saved in the current input path (``struphy -p``, under ``/inp``) 
and can be modified and renamed. An example of such a file is given 
`here <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/io/inp/parameters.yml?ref_type=heads>`_.

Struphy simulation parameters fall into three classes, addressed by
up to 8 different keys in the ``.yml`` file:

1. Discretization parameters:

* :ref:`grid` (under the key ``grid``)
* :ref:`time` (under the key ``time``)

2. Physics parameters:

* :ref:`units` (under the key ``units``)
* :ref:`geometry` (under the key ``geometry``)
* :ref:`mhd_equilibrium` (under the key ``mhd_equilibrium``)

3. Species parameters:

* :ref:`em_fields` (under the key ``em_fields``)
* :ref:`fluid` (under the key ``fluid``)
* :ref:`kinetic` (under the key ``kinetic``)

The ``.yml`` parameter file has a generic structure for all Struphy models.
The keys ``grid``, ``time``, ``units`` and ``geometry`` are present 
in every model; moreover, at least one of the three
species types ``em_fields``, ``fluid`` and ``kinetic`` must be present 
in each model, relating to the model variable(s). An ``mhd_equilibrium``
is not mandatory.

The structure of the dictionaries under one of the 8 top-level keys
is discussed below. Special information is available on how to set :ref:`initial_conditions`
and :ref:`boundary_conditions`. 

Some hints for editing a parameter file:

* Strings can either be set as e.g. ``'Cuboid'`` or ``Cuboid``, i.e. with or without quotes - both works.
* The parameter ``null`` will be transformed to Python's ``None`` type.
* Available geometries can be found in :ref:`avail_mappings`
* Available MHD equilibria can be found in :ref:`mhd_equil`
* Available kinetic backgrounds can be found in :ref:`kinetic_backgrounds`
* Available fluid backgrounds can be found in :ref:`fluid_backgrounds`
* Available perturbations can be found in :ref:`avail_inits`

.. _grid:

Space grid parameters
^^^^^^^^^^^^^^^^^^^^^

See :ref:`boundary_conditions` for how to set boundary conditions.

::

    grid :
        Nel          : [12, 14, 4] # number of grid cells, >p
        p            : [3, 4, 2]  # spline degree
        spl_kind     : [False, True, True] # spline type: True=periodic, False=clamped
        dirichlet_bc : [[False, False], [False, False], [False, False]] # hom. Dirichlet boundary conditions for N-splines (spl_kind must be False)
        dims_mask    : [True, True, True] # True if the dimension is to be used in the mpi domain decomposition (=default for each dimension).
        nq_el        : [2, 2, 2] # quadrature points per grid cell
        nq_pr        : [2, 2, 2] # quadrature points per histopolation cell (for commuting projectors)
        polar_ck     : -1 # C^k smoothness at polar singularity at eta_1=0 (default: -1 --> standard tensor product, 1 : polar splines)

.. _time:

Time stepping parameters
^^^^^^^^^^^^^^^^^^^^^^^^

::

    time :
        dt         : 0.005 # time step
        Tend       : 0.015 # simulation time interval is [0, Tend]
        split_algo : LieTrotter # LieTrotter | Strang

.. _units:

Units
^^^^^

::

    units : 
        x : 1. # length scale unit in Meter (m)
        B : 1. # magnetic field unit in Tesla
        n : 1. # number density unit in 10^20 m^(-3)

.. _geometry:

Geometry
^^^^^^^^

See :ref:`avail_mappings` for possible geometry ``type``.

::

    geometry :
        type : Cuboid 
        Cuboid : {}

The empty dictionary means that ``Cuboid`` is instantiated with its default parameters.

.. _mhd_equilibrium:

MHD equilibrium
^^^^^^^^^^^^^^^

See :ref:`mhd_equil` for possible MHD equilibrium ``type``.

::

    mhd_equilibrium :
        type : HomogenSlab 
        HomogenSlab : {}

The empty dictionary means that ``HomogenSlab`` is instantiated with its default parameters.

.. _em_fields:

Electromagnetic fields
^^^^^^^^^^^^^^^^^^^^^^

Initial conditions are the sum of ``background`` + ``perturbation``, see :ref:`initial_conditions`.
Check out :ref:`fluid_backgrounds` and :ref:`avail_inits` for available ``type``, respectively.

::

    em_fields :
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
.. _fluid:

Fluid variables
^^^^^^^^^^^^^^^

``phys_params`` provides the mass and charge number of the fluid species.

Initial conditions are the sum of ``background`` + ``perturbation``, see :ref:`initial_conditions`.
Check out :ref:`fluid_backgrounds` and :ref:`avail_inits` for available ``type``, respectively.

::

    fluid :
        species_name :
            phys_params:
                A : 1 # mass number in units of proton mass
                Z : 1 # signed charge number in units of elementary charge
            background: 
                type : LogicalConst
                LogicalConst :
                    comps :
                        var_name : [null, 1.5, null] # vector-valued variable
                        another_var_name : 2.3 # scalar-valued variable
            perturbation :
                type : TorusModesCos
                TorusModesCos :
                    comps :
                        var_name : '0' # perturbation function given as 0-form 
                    ms : # poloidal mode numbers
                        var_name : [1, 3] # two poloidal modes

.. _kinetic:

Kinetic variables
^^^^^^^^^^^^^^^^^

``phys_params`` provides the mass and charge number of the kinetic species.

``markers`` provides all sorts of information for the marker (particle) initialization and boundary conditions.

``save_data`` allows one to choose the particle binning options (under ``f``) and single particle tracking.

Initial conditions are the sum of ``background`` + ``perturbation``, see :ref:`initial_conditions`.
Check out :ref:`kinetic_backgrounds` and :ref:`avail_inits` for available ``type``, respectively.

::

    kinetic :
        species_name :
            phys_params :
                A : 1  # mass number in units of proton mass
                Z : 1 # signed charge number in units of elementary charge
            markers :
                type    : full_f # full_f, control_variate, or delta_f
                ppc     : null  # number of markers per 3d grid cell
                Np      : 1000 # alternative if ppc = null, total number of markers
                eps     : .25 # MPI send/receive buffer (0.1 <= eps <= 1.0)
                bc : 
                    type    : [periodic, periodic, periodic] # marker boundary conditions: remove, reflect or periodic
                loading :
                    type          : pseudo_random # particle loading mechanism 
                    seed          : 1234 # seed for random number generator
                    moments       : [0., 0., 0., 1., 1., 1.] # moments of Gaussian s3, see background/moms_spec
                    spatial       : uniform # uniform or disc
                    dir_particles : 'path_to_particles' # directory of particles if loaded externally
            save_data :
                n_markers : 3 # number of markers to be saved during simulation
                f :
                    slices : [v1, e1_v1] # in which directions to bin (e.g. [e1_e2, v1_v2_v3])
                    n_bins : [[32], [32, 32]] # number of bins in each direction (e.g. [[16, 20], [16, 18, 22]])
                    ranges : [[[-3., 3.]], [[0., 1.], [-5., 5.]]] # bin range in each direction (e.g. [[[0., 1.], [0., 1.]], [[-3., 3.], [-4., 4.], [-5., 5.]]])
            background : # background is mandatory for kinetic species
                type : Maxwellian3D
                Maxwellian3D :
                    n  : 0.05
                    u2 : 2.5
            perturbation :
                type : TorusModesCos
                TorusModesCos :
                    comps :
                        n : '0' # perturbation function given as 0-form 
                    ms : # poloidal mode numbers
                        n : [1, 3] # two poloidal modes for the density
