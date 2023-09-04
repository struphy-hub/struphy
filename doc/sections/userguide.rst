.. _userguide:

Userguide
=========


.. _running_codes:

Solving PDEs with Struphy - running models
------------------------------------------

The help for running Struphy models can be accessed with::

    struphy run --help

The basic command is::

    struphy run [OPTIONS] MODEL 

See :ref:`models` for more information such as implemented PDEs and normalization.
If no ``[OPTIONS]`` are specified, the input is taken from ``<install_path>/parameters.yml``,
where ``<install_path>`` is the current input path obtained from::

    struphy -p

The I/O paths can be changed with the commands:: 

    struphy --set-i <path>
    struphy --set-o <path>

If ``<path>`` is ``.`` the current working directory is selected.
The default parameter file ``<install_path>/parameters.yml`` provides an overview of simulation parameters
that can be passed to Struphy models. Model specific parameter templates can be found under ``<install_path>/params_*.yml``.
Possible parameters are discussed in more detail in :ref:`params_yml`.

By default, simulation data is written to ``<install_path>/sim_1/``. 
Different input files and/or output folders in the current I/O paths can be specified
with the ``-i`` and/or ``-o`` flags, respectively::

    struphy run MODEL -i my_params.yml -o my_folder

Absolute paths (unrelated to the current I/O paths) can also be specified::

    struphy run MODEL --input-abs path/to/file.yml --output-abs path/to/folder

Small parallel runs for testing can be called via::

    struphy run MODEL --mpi <int>

where ``<int>`` denotes the number of mpi processes. 
`Slurm <https://slurm.schedmd.com/documentation.html>`_ jobs can be submitted via batch scripts. 
The path to your batch scripts can be set via::

    struphy --set-b <path>

If ``<path>`` is ``.`` the current working directory is selected.
A model is run as a slurm job with the ``-b`` flag::

    struphy run MODEL -b cobra_0160proc.sh

Again, an absolute path to a batch script can be specified::

    struphy run MODEL --batch-abs path/to/batch.sh

Struphy ``[OPTIONS]`` can be combined (the order is not important).


.. _profiling:

Code profiling
--------------

Access help::

    struphy profile -h

Each Struphy run is by default profiled with the Python profiler `cProfile <https://docs.python.org/3/library/profile.html>`_.
In order to see profiling results type::

    struphy profile [OPTIONS] sim_1 [sim_2 ...]

Here, ``sim_1``, ``sim2`` etc. are relative to the current output path. If more than one simulation is profiled, 
they all have to be from the same ``MODEL``. To get more info on possible ``OPTIONS`` type::

    struphy profile -h
   

.. _pproc:

Post processing
---------------

Access help::

    struphy pproc -h

The basic command for Struphy post-processing is::

    struphy pproc -d <sim_name> 

Here, ``<sim_name>`` is relative to the current output path. 
In the latter, the generated output data can be inspected at:: 
    
    cd <sim_name>/post_processing/


.. _params_yml:

Setting simulation parameters
-----------------------------

The default parameter file can be inspected under ``<install_path>/parameters.yml``.
In a Struphy parameter file, the simulation parameters must be categorized into the following top level keywords
(only ``grid``, ``time`` and ``geometry`` are mandatory):

* :ref:`grid`
* :ref:`time`
* :ref:`geometry`
* :ref:`mhd_equilibrium`
* :ref:`electric_equilibrium`
* :ref:`em_fields`
* :ref:`fluid`
* :ref:`kinetic`
* :ref:`solvers`

The input must be structured as follows:

.. _grid:

grid
^^^^

::

    Nel       : [16, 32, 32] # number of grid cells, >p
    p         : [2, 3, 4]  # spline degree
    spl_kind  : [False, True, True] # spline type: True=periodic, False=clamped
    bc        : [[null, null], [null, null], [null, null]] # boundary conditions for N-splines (homogeneous Dirichlet='d')
    dims_mask : [True, True, True] # True if the dimension is to be used in the mpi domain decomposition (=default for each dimension).
    nq_el     : [2, 2, 2] # quadrature points per grid cell
    nq_pr     : [2, 2, 2] # quadrature points per histopolation cell (for commuting projectors)
    polar_ck  : -1 # C^k smoothness at polar singularity at eta_1=0 (default: -1 --> standard tensor product, 1 : polar splines)

.. _time:

time
^^^^

::

    dt         : 0.05 # time step
    Tend       : 0.5 # simulation time interval is [0, Tend]
    split_algo : LieTrotter # LieTrotter | Strang

.. _geometry:

geometry
^^^^^^^^

See :ref:`avail_mappings` for possible mapping ``type``.

::

    type : Tokamak # mapping F   
    Tokamak :
        Nel        : [8, 32] # number of poloidal grid cells, >p
        p          : [3, 3] # poloidal spline degrees, >1
        psi_power  : 0.7 # parametrization of radial flux coordinate eta1=psi_norm^psi_power, where psi_norm is normalized flux
        psi_shifts : [2., 2.] # start and end shifts of polidal flux in % --> cuts away regions at the axis and edge
        xi_param   : equal_angle # parametrization of angular coordinate (equal_angle, equal_arc_length or sfl (straight field line))
        r0         : 0.3 # initial guess for radial distance from axis used in Newton root-finding method for flux surfaces
        Nel_pre    : [64, 256] # number of poloidal grid cells of pre-mapping needed for equal_arc_length and sfl
        p_pre      : [3, 3] # poloidal spline degrees of pre-mapping needed for equal_arc_length and sfl
        tor_period : 1 # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period

.. _mhd_equilibrium:

mhd_equilibrium
^^^^^^^^^^^^^^^

See :ref:`mhd_equil` for possible MHD equilibrium ``type``.

::

    type : AdhocTorus # MHD equilibirum
    AdhocTorus :
        a      : 1. # minor radius
        R0     : 3. # major radius
        B0     : 1. # on-axis toroidal magnetic field
        q0     : 1.05 # safety factor at r=0
        q1     : 1.80 # safety factor at r=a
        n1     : 0. # shape factor for number density profile 
        n2     : 0. # shape factor for number density profile 
        na     : 1. # number density at r=a
        p_kind : .5 # kind of pressure profile (0 : cylindrical limit, 1 : ad hoc)
        p1     : .1 # shape factor for ad hoc pressure profile
        p2     : .1 # shape factor for ad hoc pressure profile
        beta   : .01 # plasma beta in % for flat safety factor (ratio of kinetic pressure to magnetic pressure)


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
        type : ModesSin # type of initial condition (possible types seen below)
        ModesSin : 
            coords : 'logical' # in which coordinates (logical or physical)
            comps :
                e1 : [True, False, False]  # components to be initialized (for scalar fields: no list)
                b2 : [False, False, False] # components to be initialized (for scalar fields: no list)
            ls : [0] # Integer mode numbers in x or eta_1 (depending on coords)
            ms : [0] # Integer mode numbers in y or eta_2 (depending on coords)
            ns : [1] # Integer mode numbers in z or eta_3 (depending on coords)
            amps : [0.001] # amplitudes of each mode

.. _fluid:

fluid
^^^^^

See :ref:`avail_inits` for possible ``type`` of initial condition ``init``.

::

    mhd :
        mhd_u_space : H1vec # Hdiv | H1vec
        init :
            type : ModesSin # type of initial condition (possible types seen below)
            ModesSin :
                coords : 'logical' # in which coordinates (logical or physical)
                comps :
                    n3 : False              # components to be initialized (for scalar fields: no list)
                    uv : [True, True, True] # components to be initialized (for scalar fields: no list)
                    p3 : False              # components to be initialized (for scalar fields: no list)
                ls : [0] # Integer mode numbers in x or eta_1 (depending on coords)
                ms : [0] # Integer mode numbers in y or eta_2 (depending on coords)
                ns : [1] # Integer mode numbers in z or eta_3 (depending on coords)
                amps : [0.001] # amplitudes of each mode    

.. _kinetic:

kinetic
^^^^^^^

See :ref:`kinetic_backgrounds` for possible ``type`` of initial condition ``init`` and ``background``.

::

    hot_ions :
        markers :
            type    : full_f # full_f, control_variate, or delta_f
            ppc     : 100  # number of markers per 3d grid cell
            Np      : 3 # alternative if ppc = null (total number of markers, must be larger or equal than # MPI processes)
            eps     : .25 # MPI send/receive buffer (0.1 <= eps <= 1.0)
            bc_type : [periodic, periodic, periodic] # remove, reflect or periodic
            loading :
                type          : pseudo_random # particle loading mechanism 
                seed          : 1234 # seed for random number generator
                dir_particles : 'path_to_particles' # directory of particles if loaded externally
                moments       : [0., 0., 0., 1., 1., 1.] # moments of Gaussian s3, see background/moms_spec
        init :
            type : Maxwellian6DPerturbed
            Maxwellian6DPerturbed :
                n :
                    n0 : 1.
                    perturbation :
                        l : [0]
                        m : [0]
                        n : [0]
                        amps_sin : [0.]
                        amps_cos : [0.]
                ux :
                    ux0 : 0.
                uy :
                    uy0 : 0.
                uz :
                    uz0 : 0.
                vthx :
                    vthx0 : 1.
                vthy :
                    vthy0 : 1.
                vthy :
                    vthz0 : 1.
        background :
            type : Maxwellian6DUniform
            Maxwellian6DUniform :
                n  : 1.
                ux : 0.
                uy : 0.
                uz : 0.
                vthx : 1.
                vthy : 1.
                vthz : 1.
        save_data :
            n_markers : 3 # number of markers to be saved during simulation
            f :
                slices : [vz] # in which directions to bin (e.g. [e1_e2, vx_vy_vz])
                n_bins : [[32]] # number of bins in each direction (e.g. [[16, 20], [16, 18, 22]])
                ranges : [[[-3., 3.]]] # bin range in each direction (e.g. [[[0., 1.], [0., 1.]], [[-3., 3.], [-4., 4.], [-5., 5.]]])
        push_algos :
            vxb : analytic # possible choices: analytic, implicit
            eta : rk4 # possible choices: forward_euler, heun2, rk2, heun3, rk4
        use_perp_model : True # for pressure coupling 


.. _solvers:

solvers
^^^^^^^

::

    solver_1 : 
        type : PConjugateGradient
        pc : MassMatrixPreconditioner # null or name of preconditioner class
        tol : 1.e-8
        maxiter : 3000
        info : False
        verbose : False




