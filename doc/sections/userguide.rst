.. _userguide:

Userguide
=========

Struphy can be conveniently used from the console. The overall help is displayed by typing::

    struphy -h

To get more information on the sub-commands::

    struphy COMMAND -h

The installed version is obtained by::

    struphy -v


.. _running_codes:

Solving PDEs with Struphy
-------------------------

With Struphy it is simple to solve PDEs from the console.
A list of available model PDEs and run options can be accessed with::

    struphy run -h

The list of models is automatically updated whenever a developer adds a new model PDE.
All models are continuously auto-tested with many different input options. 

The basic command for solving a PDE is::

    struphy run [OPTIONS] MODEL 

If no ``[OPTIONS]`` are specified, a default parameter file ``params_MODEL.yml``
is created in the current input path (on prompt).
If we choose to do so, the file is created (or overwritten) and the code exits. 
To run the model with default parameters we can now type::

    struphy run MODEL

The current I/O paths can be obtained from::

    struphy -p

The I/O paths can be changed with the commands:: 

    struphy --set-i <path>
    struphy --set-o <path>

If ``<path>`` is ``.``, the current working directory is selected. 
The default I/O paths are ``<install_path>/io/inp`` and ``<install_path>/io/out``, respectively.
We can always revert back to the default I/O paths::

    struphy --set-i d
    struphy --set-o d

All parameters for a Struphy run are set in the parameter file ``params_MODEL.yml``.
We can edit/copy/rename this file to our liking and pass it via ``-i <new_name>.yml`` to the run command.
All models in Struphy share a set of "core" parameters which are discussed in :ref:`params_yml`.
Model specific parameters can be seen under the keyword ``options`` in the parameter file.
Each model species has its own ``options``, which are hard-coded as class methods.
We can inspect the model-specific options from the console::

    struphy params MODEL --options

Available options appear as lists in the options dictionary. 
The first list entry is the respective default option. 
The key structure of the options dictionary is exactly the same as in the parameter file.

By default, Struphy simulation data is written to ``sim_1/`` in the current output path. 
Different input files and/or output folders with respect the current I/O paths can be specified
with the ``-i`` and/or ``-o`` options, respectively::

    struphy run MODEL -i my_params.yml -o my_folder

Absolute paths (unrelated to the current I/O paths) can also be specified::

    struphy run MODEL --input-abs path/to/file.yml --output-abs path/to/folder

Small parallel runs (for debugging) can be called via::

    struphy run MODEL --mpi <int>

where ``<int>`` denotes the number of mpi processes. 
`Slurm <https://slurm.schedmd.com/documentation.html>`_ jobs can be submitted via batch scripts. 
The path to your batch scripts can be set via::

    struphy --set-b <path>

The logic of this command is the same as for ``--set-i`` above.
A model is run as a slurm job with the ``-b`` flag::

    struphy run MODEL -b my_batch_script.sh

Again, an absolute path to a batch script can be specified::

    struphy run MODEL --batch-abs path/to/batch.sh

Struphy ``[OPTIONS]`` can be combined (the order is not important).


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

See :ref:`kinetic_backgrounds` for possible ``type`` of initial condition ``init`` and ``background``.

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
        init :
            type : Maxwellian6DPerturbed
            Maxwellian6DPerturbed :
                n :
                    n0 : 0.05
                    perturbation :
                        l : [0]
                        m : [0]
                        n : [0]
                        amps_sin : [0.]
                        amps_cos : [0.]
                u1 :
                    u10 : 0.
                u2 :
                    u20 : 2.5
                u3 :
                    u30 : 0.
                vth1 :
                    vth10 : 1.
                vth2 :
                    vth20 : 1.
                vth3 :
                    vth30 : 1.
        background :
            type : Maxwellian6DUniform
            Maxwellian6DUniform :
                n  : 1.
                u1 : 0.
                u2 : 0.
                u3 : 0.
                vth1 : 1.
                vth2 : 1.
                vth3 : 1.
        save_data :
            n_markers : 3 # number of markers to be saved during simulation
            f :
                slices : [v1] # in which directions to bin (e.g. [e1_e2, v1_v2_v3])
                n_bins : [[32]] # number of bins in each direction (e.g. [[16, 20], [16, 18, 22]])
                ranges : [[[-3., 3.]]] # bin range in each direction (e.g. [[[0., 1.], [0., 1.]], [[-3., 3.], [-4., 4.], [-5., 5.]]])


.. _profiling:

Code profiling
--------------

Each Struphy run is by default profiled with the Python profiler `cProfile <https://docs.python.org/3/library/profile.html>`_.
In order to see profiling results type::

    struphy profile [OPTIONS] sim_1 [sim_2 ...]

Here, ``sim_1``, ``sim2`` etc. are relative to the current output path. If more than one simulation is profiled, 
they all have to be from the same ``MODEL``. To get more info on possible ``OPTIONS`` type::

    struphy profile -h
   

.. _pproc:

Post-processing
---------------

The raw data generated by a Struphy run is stored in the folder ``<sim_name>/`` 
(passed with ``-o <sim_name>`` to the run command), 
relative to the output path at the time of the simulation.

The basic command for Struphy post-processing is::

    struphy pproc -d <sim_name> 

Here, ``<sim_name>`` is relative to the current output path. 
In the latter, the generated output data can be inspected at:: 
    
    cd <sim_name>/post_processing/

Struphy post-processing does several things:

* save the time grid as a numpy array under ``t_grid.npy``
* evaluate FEEC fields at the grid points (or a refined grid, with the option ``--celldivide N``)
* save the evaluation grids in physical and logical space
* generate ``.vtk`` files for use in :ref:`paraview`
* compute marker trajectories in physical (x, y, z) coordinates, saved as ``.npy`` and ``.txt`` (for use in :ref:`paraview`)
* save histograms of the distribution function is phase space.

Access further help::

    struphy pproc -h


.. _paraview:

Paraview
--------

Coming soon!


