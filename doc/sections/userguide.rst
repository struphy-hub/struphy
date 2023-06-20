.. _userguide:

Userguide
=========


.. _running_codes:

Running Struphy models
----------------------

The help for running struphy models can be accessed with::

    struphy run --help

The basic command is::

    struphy run [OPTIONS] MODEL 

See :ref:`models` for more information such as implemented equations and normalization.
If no ``[OPTIONS]`` are specified, the input is taken from ``<install_path>/io/inp/parameters.yml``,
where ``<install_path>`` is obtained from::

    struphy -p

The default parameter file ``<install_path>/io/inp/parameters.yml`` provides an overview of simulation parameters
that can be passed to Struphy models. Model specific parameter templates can be found under ``<install_path>/io/inp/params_*.yml``.
Possible parameters are discussed in more detail in :ref:`params_yml`.

By default, simulation data is written to ``<install_path>/io/out/sim_1/``. 
Different input files and/or output folders in ``<install_path>/io/`` can be specified
with the ``-i`` and/or ``-o`` flags, respectively::

    struphy run MODEL -i my_params.yml -o my_folder

Absolute paths can also be specified::

    struphy run MODEL --input-abs path/to/file.yml --output-abs path/to/folder

Small parallel runs for testing can be called via::

    struphy run MODEL --mpi <int>

where ``<int>`` denotes the number of mpi processes. 
`Slurm <https://slurm.schedmd.com/documentation.html>`_ jobs can be submitted via batch scripts. 
Some default batch scripts are provided in ``<install_path>/io/batch``. 
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

Here, ``sim_1``, ``sim2`` etc. are relative to ``<install_path>/io/out/``. If more than one simulation is profiled, 
they all have to be from the same ``MODEL``. To get more info on possible ``OPTIONS`` type::

    struphy profile -h
   

.. _pproc:

Post processing
---------------

Access help::

    struphy pproc -h

The basic command for Struphy post-processing is::

    struphy pproc -d <sim_name> 

Here, ``<sim_name>`` is relative to ``<install_path>/io/out/``. 
The generated output data can be inspected at:: 
    
    cd <install_path>/io/out/<sim_name>/post_processing/


.. _params_yml:

Setting simulation parameters
-----------------------------

The default parameter file ``<install_path>/io/inp/parameters.yml`` can be used for all available :ref:`models`. 
The user finds the simulation parameters categorized into the following top level keywords:

* :ref:`grid`
* :ref:`time`
* :ref:`geometry`
* :ref:`mhd_equilibrium`
* :ref:`electric_equilibrium`
* :ref:`em_fields`
* :ref:`fluid`
* :ref:`kinetic`
* :ref:`solvers`

Model specific parameter files can be much shorter. It is enough to include one geometry, 
mhd_equilibirum, initial condition, etc. for a run.

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

::

    type : Cuboid # mapping F (possible types seen below)  
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
    GVECunit :
        rel_path  : True # whether to state dat_file (json_file) relative to "<struphy_path>/fields_background/mhd_equil/gvec", or give the absolute path
        dat_file  : '/ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat' # path to gvec .dat output file 
        json_file : Null # give directly the parsed json file, if it exists (then dat_file is not used)
        use_pest  : False # Whether to use straight-field line coordinates (PEST)
        use_nfp   : True # Whether to use the field periods of the stellarator in the mapping, i.e. phi = 2*pi*eta3 / nfp (piece of cake).
        Nel       : [32, 32, 32] # Number of cells in each direction used for interpolation of the mapping.
        p         : [3, 3, 3] # Spline degree in each direction used for interpolation of the mapping.
    IGAPolarCylinder :
        Nel : [8, 24] # number of poloidal grid cells, >p
        p   : [3, 3] # poloidal spline degree, >1
        Lz  : 6. # Length in third direction
        a   : 1. # minor radius
    IGAPolarTorus :
        Nel        : [8, 24] # number of poloidal grid cells, >p
        p          : [3, 3] # poloidal spline degree, >1
        a          : 1. # minor radius
        R0         : 3. # major radius
        tor_period : 2 # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
        sfl        : False # whether to use straight field line coordinates (particular theta parametrization) 
    Cuboid : 
        l1 : 0. # start of interval in eta1
        r1 : 2. # end of interval in eta1, r1>l1
        l2 : 0. # start of interval in eta2
        r2 : 2. # end of interval in eta2, r2>l2
        l3 : 0. # start of interval in eta3
        r3 : 1. # end of interval in eta3, r3>l3
    Orthogonal :
        Lx    : 2. # length in x-direction
        Ly    : 2. # length in y-direction
        alpha : .1 # x-distortion and y-distortion
        Lz    : 1. # length in third direction
    Colella :
        Lx    : 2. # length in x-direction
        Ly    : 2. # length in y-direction
        alpha : .1 # distortion factor
        Lz    : 1. # length in third direction
    HollowCylinder :
        a1 : .2 # inner radius
        a2 : 1. # outer radius
        Lz : 4. # length in third direction   
    PoweredEllipticCylinder :
        rx : 1. # axis length
        ry : 2. # axis length
        Lz : 4. # length in third direction
        s  : .5 # power of radial coordinate
    HollowTorus :
        a1 : .2 # inner radius
        a2 : 1. # minor radius
        R0 : 3. # major radius
        sfl : False # straight field line coordinates?
        tor_period : 2 # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
    ShafranovShiftCylinder :
        rx    : 1. # axis length
        ry    : 1. # axis length
        Lz    : 4. # length in third direction
        delta : .2 # shift factor, should be in [0, 0.1]
    ShafranovSqrtCylinder :
        rx    : 1. # axis length
        ry    : 1. # axis length
        Lz    : 4. # length in third direction
        delta : .2 # shift factor, should be in [0, 0.1]
    ShafranovDshapedCylinder :
        R0         : 2. # base radius
        Lz         : 4. # length in third direction
        delta_x    : .05 # Shafranov shift in x-direction
        delta_y    : .025 # Shafranov shift in y-direction
        delta_gs   : .05 # delta = sin(alpha): triangularity, shift of high point
        epsilon_gs : .5 # epsilon: inverse aspect ratio a/r0
        kappa_gs   : 2. # Kappa: ellipticity (elongation)

.. _mhd_equilibrium:

mhd_equilibrium
^^^^^^^^^^^^^^^

::

    type : HomogenSlab # (possible choices seen below)
    HomogenSlab :
        B0x  : 0. # magnetic field in x
        B0y  : 0. # magnetic field in y
        B0z  : 1. # magnetic field in z
        beta : .1 # plasma beta = 2*p*mu_0/B^2
        n0   : 1. # number density
    ShearedSlab :
        a    : 1. # minor radius (Lx=a, Ly=2*pi*a) 
        R0   : 3. # major rarius (Lz=2*pi*R0)
        B0   : 1. # magnetic field in z-direction    
        q0   : 1.05 # q-value at eta_1 = 0
        q1   : 1.80 # q-value at eta_1 = 1
        n1   : 0. # shape factor for number density profile
        n2   : 0. # shape factor for number density profile
        na   : 1. # number density at r=a
        beta : .01 # plasma beta = 2*p*mu_0/B^2
    ScrewPinch :
        a    : 1. # minor radius (radius of cylinder)
        R0   : 3. # major radius (shift of magnetic axis, length of pinch Lz=2*pi*R0)
        B0   : 1. # magnetic field in z-direction
        q0   : 1.05 # safety factor at r=0
        q1   : 1.80 # safety factor at r=a
        n1   : 0. # shape factor for number density profile 
        n2   : 0. # shape factor for number density profile 
        na   : 1. # number density at r=a
        beta : .01 # plasma beta in % for flat safety factor (ratio of kinetic pressure to magnetic pressure)
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
    EQDSKequilibrium :
        rel_path        : True # whether eqdsk file path relative to "<struphy_path>/fields_background/mhd_equil/gvec", or the absolute path
        file            : 'AUGNLED_g031213.00830.high' # path to eqdsk file
        data_type       : 0 # 0: there is no space between data, 1: there is space between data
        p_for_psi       : [3, 3] # spline degrees used in interpolation of poloidal flux function grid data
        psi_resolution  : [25., 6.25] # resolution used in interpolation of poloidal flux function grid data in %, i.e. [100., 100.] uses all grid points
        p_for_flux      : 3 # spline degree used in interpolation of 1d functions f=f(psi) (e.g. toroidal field function)
        flux_resolution : 50. # resolution used in interpolation of of 1d functions f=f(psi) in %
        n1              : 0. # 1st shape factor for number density profile n(psi) = (1-na)*(1 - psi_norm^n1)^n2 + na
        n2              : 0. # 2nd shape factor for number density profile n(psi) = (1-na)*(1 - psi_norm^n1)^n2 + na
        na              : 1. # number density at last closed flux surface
    GVECequilibrium : 
        rel_path : True # whether file path is relative to "<struphy_path>/fields_background/mhd_equil/gvec", or the absolute path
        dat_file : '/ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat' # path to gvec .dat output file 
        json_file : Null # give directly the parsed json file, if it exists (then dat_file is not used)
        use_pest : False # whether to use straight-field line coordinates (PEST)
        use_nfp : True # whether to use the field periods of the stellarator in the mapping, i.e. phi = 2*pi*eta3 / nfp (piece of cake).
        Nel : [32, 32, 32] # number of cells in each direction used for interpolation of the mapping.
        p : [3, 3, 3] # spline degree in each direction used for interpolation of the mapping.

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

::

    init :
        type : ModesSin # type of initial condition (possible types seen below)
        noise : 
            comps :
                e1 : [True, False, False]  # components to be initialized (for scalar fields: no list)
                b2 : [False, False, False] # components to be initialized (for scalar fields: no list)
            variation_in : e3 # noise variation (logical space): e1, e2, e3 (1d), e1e2, e1e3, e2e3 (2d), e1e2e3 (3d)
            amp : 0.1   # noise amplitude
        ModesSin : 
            coords : 'logical' # in which coordinates (logical or physical)
            comps :
                e1 : [True, False, False]  # components to be initialized (for scalar fields: no list)
                b2 : [False, False, False] # components to be initialized (for scalar fields: no list)
            ls : [0] # Integer mode numbers in x or eta_1 (depending on coords)
            ms : [0] # Integer mode numbers in y or eta_2 (depending on coords)
            ns : [1] # Integer mode numbers in z or eta_3 (depending on coords)
            amp : [0.001] # amplitudes of each mode
        ModesCos :
            coords : 'logical' # in which coordinates (logical or physical)
            comps :
                e1 : [True, False, False]  # components to be initialized (for scalar fields: no list)
                b2 : [False, False, False] # components to be initialized (for scalar fields: no list)
            ls : [0] # Integer mode numbers in x or eta_1 (depending on coords)
            ms : [0] # Integer mode numbers in y or eta_2 (depending on coords)
            ns : [1] # Integer mode numbers in z or eta_3 (depending on coords)
            amp : [0.001] # amplitudes of each mode
        TorusModesSin :
            coords : 'logical' # in which coordinates (logical or physical)
            comps :
                e1 : [True, False, False]  # components to be initialized (for scalar fields: no list)
                b2 : [False, False, False] # components to be initialized (for scalar fields: no list)
            ms : [1] # poloidal mode numbers
            ns : [0] # toroidal mode numbers
            amp : [0.001] # amplitudes of each mode
            pfuns : ['sin'] # profile function in eta1-direction ('sin' or 'exp')
            pfun_params : [null] # Provides [r_0, sigma] parameters for each "exp" profile fucntion, and null for "sin"
        InitialMHDSlab :
            a  : 1. # minor radius (Lx=a, Ly=2*pi*a)
            R0 : 3. # major radius (Lz=2*pi*R0)
            m  : 0  # poloidal (y) mode number
            n  : 1  # toroidal (z) mode number
            U  : 0.1 # amplitude of Ux/Uy
            A  : 0. # amplitude of Uz
        InitialMHDAxisymHdivEigFun :
            spec : '/path_to_spec/spec.npy' # relative path (to <install_path/struphy>) of the .npy spectrum
            eig_freq_upper : 0.15 # upper search limit of squared eigenfrequency to identify eigenfunction
            eig_freq_lower : 0.14 # lower search limit of squared eigenfrequency to identify eigenfunction
            kind : r # real (r) or imaginary (i) part of eigenfunction
            scaling : 1. # scaling factor to scale the amplitude of the eigenfunction

.. _fluid:

fluid
^^^^^

::

    mhd :
        mhd_u_space : H1vec # Hdiv | H1vec
        init :
            type : ModesSin # type of initial condition (possible types seen below)
            noise :
                comps :
                    n3 : False               # components to be initialized (for scalar fields: no list)
                    uv : [True, True, True]  # components to be initialized (for scalar fields: no list)
                    p3 : False               # components to be initialized (for scalar fields: no list)
                variation_in : e3 # noise variation (logical space): e1, e2, e3 (1d), e1e2, e1e3, e2e3 (2d), e1e2e3 (3d)
                amp : 0.1   # noise amplitude
            ModesSin :
                coords : 'logical' # in which coordinates (logical or physical)
                comps :
                    n3 : False              # components to be initialized (for scalar fields: no list)
                    uv : [True, True, True] # components to be initialized (for scalar fields: no list)
                    p3 : False              # components to be initialized (for scalar fields: no list)
                ls : [0] # Integer mode numbers in x or eta_1 (depending on coords)
                ms : [0] # Integer mode numbers in y or eta_2 (depending on coords)
                ns : [1] # Integer mode numbers in z or eta_3 (depending on coords)
                amp : [0.001] # amplitudes of each mode
            ModesCos :
                coords : 'logical' # in which coordinates (logical or physical)
                comps :
                    n3 : False               # components to be initialized (for scalar fields: no list)
                    uv : [True, True, True]  # components to be initialized (for scalar fields: no list)
                    p3 : False               # components to be initialized (for scalar fields: no list)
                ls : [0] # Integer mode numbers in x or eta_1 (depending on coords)
                ms : [0] # Integer mode numbers in y or eta_2 (depending on coords)
                ns : [1] # Integer mode numbers in z or eta_3 (depending on coords)
                amp : [0.001] # amplitudes of each mode
            TorusModesSin :
                coords : 'logical' # in which coordinates (logical or physical)
                comps :
                    n3 : False              # components to be initialized (for scalar fields: no list)
                    uv : [True, True, True] # components to be initialized (for scalar fields: no list)
                    p3 : False              # components to be initialized (for scalar fields: no list)
                ms : [1] # poloidal mode numbers
                ns : [0] # toroidal mode numbers
                amp : [0.001] # amplitudes of each mode
                pfuns : ['sin'] # profile function in eta1-direction ('sin' or 'exp')
                pfun_params : [null] # Provides [r_0, sigma] parameters for each "exp" profile fucntion, and null for "sin"
            InitialMHDSlab :
                a  : 1. # minor radius (Lx=a, Ly=2*pi*a)
                R0 : 3. # major radius (Lz=2*pi*R0)
                m  : 0  # poloidal (y) mode number
                n  : 1  # toroidal (z) mode number
                U  : 0.1 # amplitude of Ux/Uy
                A  : 0. # amplitude of Uz
            InitialMHDAxisymHdivEigFun :
                spec : '/path_to_spec/spec.npy' # relative path (to <install_path/struphy>) of the .npy spectrum
                eig_freq_upper : 0.15 # upper search limit of squared eigenfrequency to identify eigenfunction
                eig_freq_lower : 0.14 # lower search limit of squared eigenfrequency to identify eigenfunction
                kind : r # real (r) or imaginary (i) part of eigenfunction
                scaling : 1. # scaling factor to scale the amplitude of the eigenfunction     

.. _kinetic:

kinetic
^^^^^^^

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
    solver_2 :
        type : PConjugateGradient
        pc : MassMatrixPreconditioner # null or name of preconditioner class
        tol : 1.e-8
        maxiter : 3000
        info : False
        verbose : False




