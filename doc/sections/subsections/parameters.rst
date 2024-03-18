.. _params_yml:

Simulation parameters
---------------------

Aside from the console, the ``.yml`` parameter file is the main point of interaction for a Struphy user. 
It stores the dictionary that is read-in by the Struphy main execution file 
at the start of a simulation, see :ref:`Tutorial 1 - Run Struphy main file in a notebook  <tutorials>`.
All information relevant to the disretization and the physics of the PDE model are in this file.

The ``.yml`` parameter file has some generic structure that is uniform for all (present and future) Struphy models.
In particular, there are **9 top-level keys** that can be present in such a file:

* :ref:`grid`
* :ref:`time`
* :ref:`units`
* :ref:`geometry`
* :ref:`mhd_equilibrium`
* :ref:`electric_equilibrium`
* :ref:`em_fields`
* :ref:`fluid`
* :ref:`kinetic`

Of these, only ``grid``, ``time``, ``geometry`` and ``units`` are mandatory, and at least one of the three
species types ``em_fields``, ``fluid`` and ``kinetic`` will be present in each model.
Electromagnetic background fields can be provided through ``mhd_equilibrium`` and ``electric_equilibrium``.
The particular structure of these dictionaries can be inspected in the default parameter files
created with::

    struphy params MODEL

An example is given in the file ``<install_path>/io/inp/parameters.yml``, where the install path
is obtained from ``struphy -p``. We list the content of this file below.

The dictionary structure for a particular input, for example a mapping or an MHD equilibrium,
can be obtained from the respective section in this documentation.

Some hints for editing a parameter file:

1. Strings can either be set as e.g. ``'Cuboid'`` or ``Cuboid``, i.e. with or without quotes - both works
2. The parameter ``null`` will be transformed to Python's ``None`` type
3. Available geometries: https://struphy.pages.mpcdf.de/struphy/sections/domains.html
4. Available MHD equilibria: https://struphy.pages.mpcdf.de/struphy/sections/mhd_equils.html
5. Available kinetic backgrounds/initial conditions: https://struphy.pages.mpcdf.de/struphy/sections/kinetic_backgrounds.html
6. Available FEEC initial conditions: https://struphy.pages.mpcdf.de/struphy/sections/inits.html

.. _grid:

grid
^^^^

::

    Nel          : [12, 14, 4] # number of grid cells, >p
    p            : [3, 4, 2]  # spline degree
    spl_kind     : [False, True, True] # spline type: True=periodic, False=clamped
    dirichlet_bc : [[False, False], [False, False], [False, False]] # hom. Dirichlet boundary conditions for N-splines (spl_kind must be False)
    dims_mask    : [True, True, True] # True if the dimension is to be used in the mpi domain decomposition (=default for each dimension).
    nq_el        : [2, 2, 2] # quadrature points per grid cell
    nq_pr        : [2, 2, 2] # quadrature points per histopolation cell (for commuting projectors)
    polar_ck     : -1 # C^k smoothness at polar singularity at eta_1=0 (default: -1 --> standard tensor product, 1 : polar splines)

.. _time:

time
^^^^

::

    dt         : 0.005 # time step
    Tend       : 0.015 # simulation time interval is [0, Tend]
    split_algo : LieTrotter # LieTrotter | Strang

.. _units:

units
^^^^^

::

    x : 1. # length scale unit in m
    B : 1. # magnetic field unit in T
    n : 1. # number density unit in 10^20 m^(-3)

.. _geometry:

geometry
^^^^^^^^

See :ref:`avail_mappings` for possible mapping ``type``.

::

    type : Cuboid 
    Cuboid : {}

The empty dictionary means that ``Cuboid`` is instantiated with its default parameters.

.. _mhd_equilibrium:

mhd_equilibrium
^^^^^^^^^^^^^^^

See :ref:`mhd_equil` for possible MHD equilibrium ``type``.

::

    type : HomogenSlab 
    HomogenSlab : {}

The empty dictionary means that ``HomogenSlab`` is instantiated with its default parameters.


.. _electric_equilibrium:

electric_equilibrium
^^^^^^^^^^^^^^^^^^^^

::

    type : HomogenSlab # (possible choices seen below)
    HomogenSlab :
        phi0  : 1. # constant electric potential

.. _em_fields:

em_fields
^^^^^^^^^

See :ref:`avail_inits` for possible ``type`` of initial condition ``init``.

::

    init :
        type : TorusModesCos
        TorusModesCos :
            comps :
                e1 : [False, True, True]  # components to be initialized (for scalar fields: no list)
                b2 : [False, True, False] # components to be initialized (for scalar fields: no list)
            ms : [3] # poloidal mode numbers
            ns : [1] # toroidal mode numbers
            amps : [0.001] # amplitudes of each mode
            pfuns : ['sin'] # profile function in eta1-direction ('sin' or 'cos' or 'exp')
            pfun_params : [null] # Provides [r_0, sigma] parameters for each "exp" profile fucntion, and null for "sin" and "cos"

.. _fluid:

fluid
^^^^^

See :ref:`avail_inits` for possible ``type`` of initial condition ``init``.

::

    mhd :
        phys_params:
            A : 1 # mass number in units of proton mass
            Z : 1 # signed charge number in units of elementary charge
        mhd_u_space : H1vec # Hdiv | H1vec
        init :
            type : TorusModesCos
            TorusModesCos :
                comps :
                    n3 : False                # components to be initialized (for scalar fields: no list)
                    u2 : [False, True, False] # components to be initialized (for scalar fields: no list)
                    p3 : False                # components to be initialized (for scalar fields: no list)
                ms : [3] # poloidal mode numbers
                ns : [1] # toroidal mode numbers
                amps : [0.001] # amplitudes of each mode
                pfuns : ['sin'] # profile function in eta1-direction ('sin' or 'cos' or 'exp')
                pfun_params : [null] # Provides [r_0, sigma] parameters for each "exp" profile fucntion, and null for "sin" and "cos"   

.. _kinetic:

kinetic
^^^^^^^

Initial conditions are given by a ``background`` (see :ref:`kinetic_backgrounds` for available choices) and a possible
``perturbation`` added on top of that (see :ref:`avail_inits` for available choices).

::

    ions :
        phys_params :
            A : 1  # mass number in units of proton mass
            Z : 1 # signed charge number in units of elementary charge
        markers :
            type    : full_f # full_f, control_variate, or delta_f
            ppc     : 10  # number of markers per 3d grid cell
            Np      : 3 # alternative if ppc = null (total number of markers, must be larger or equal than # MPI processes)
            eps     : .25 # MPI send/receive buffer (0.1 <= eps <= 1.0)
            bc : 
                type    : [periodic, periodic, periodic] # marker boundary conditions: remove, reflect or periodic
            loading :
                type          : pseudo_random # particle loading mechanism 
                seed          : 1234 # seed for random number generator
                moments       : [0., 0., 0., 1., 1., 1.] # moments of Gaussian s3, see background/moms_spec
                spatial       : uniform # uniform or disc
                dir_particles : 'path_to_particles' # directory of particles if loaded externally
        background :
            type : Maxwellian6D
            Maxwellian6D :
                n  : 0.5
                u1 : 0.
                u2 : 2.5
                u3 : 0.
                vth1 : 1.
                vth2 : 1.
                vth3 : 1.
        perturbation:
            type: ModesCos
            ModesCos:
                comps:
                    n: '0'
                ls:
                    n: [3]
                amps:
                    n: [0.001]
        save_data :
            n_markers : 3 # number of markers to be saved during simulation
            f :
                slices : [v1] # in which directions to bin (e.g. [e1_e2, v1_v2_v3])
                n_bins : [[32]] # number of bins in each direction (e.g. [[16, 20], [16, 18, 22]])
                ranges : [[[-3., 3.]]] # bin range in each direction (e.g. [[[0., 1.], [0., 1.]], [[-3., 3.], [-4., 4.], [-5., 5.]]])
