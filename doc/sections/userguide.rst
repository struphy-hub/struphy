.. _userguide:

Userguide
=========


.. _intro:

Introduction
------------

A main goal of :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` is to provide **easy access,
usage and development** of plasma hybrid codes: 

    - **easy access** (see :ref:`user_install`) is provided through the Python Packaging Index (PYPI).

    - **easy usage** comes from the command-line interface and the interaction through the :ref:`params_file`.

    - **easy development** because of Python modules that can be used with any code.

Currently availabe codes in :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` are

========================== =============================================================== ===================================================== =========================
Code name                  Description                                                     Features                              
========================== =============================================================== ===================================================== =========================
``lin_mhd``                3D linear, ideal MHD equations                                  choose p-form for U and p                             :ref:`model equations <lin_mhd>`
``lin_mhd_MF``                                                                             matrix-free MHD-specific operators                    :ref:`model equations <lin_mhd>`
``lin_mhd_psydac``         (not merged)                                                    MPI parallel through Psydac data structures           :ref:`model equations <lin_mhd>`     
``cc_lin_mhd_6d``          Current coupling, linear MHD, 6D Vlasov                         optimized MHD-specific operators                      :ref:`model equations <cc_lin_mhd_6d>`
``cc_lin_mhd_6d_MF``                                                                       matrix-free, choose p-form for U and p                :ref:`model equations <cc_lin_mhd_6d>` 
``cc_lin_mhd_6d_axissymm`` (not merged) 2D cc_lin_mhd_6d                                   single Fourier mode in third direction                :ref:`model equations <cc_lin_mhd_6d>`
``pc_lin_mhd_6d_MF_full``  Pressure coupling, linear MHD, 6D Vlasov                        full pressure tensor                                  :ref:`model equations <pc_lin_mhd_6d>`                                                                  
``pc_lin_mhd_6d_MF_perp``                                                                  perpendicular pressure tensor w.r.t B0                :ref:`model equations <pc_lin_mhd_6d>`                                                                    
``kinetic_extended``       (not merged) 6D Vlasov, massless electrons, extended Ohm's law  nonlinear, three different algorithms available       :ref:`model equations <kinetic_extended>`
``cold_plasma``            (not merged) 3D cold plasma equations, 6D Vlasov                electron timescales                                   :ref:`model equations <cold_plasma>`
``mhd_eig_axissymm``       (not merged) 2D MHD eigenvalue solver with single Fourier mode  polar splines                                         :ref:`model equations <mhd_eig_axissymm>`
========================== =============================================================== ===================================================== =========================

This list is expected to grow over time (anybody is welcome to contribute a new model!). 
If you are interested in adding a code please go to :ref:`developers` and visit section :ref:`add_model`.

The usage of :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` is explained in the :ref:`quickstart` guide.


.. _params_file:

Parameter file
--------------

The user interaction with :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` happens via code-specific
``parameters.yml`` files specifying the parameters of a run, such as

    * grid spacing, time step, spline degree, etc.
    * boundary conditions
    * geometry (mapped domain)
    * equilibirium states (fluid and kinetic)
    * type of initial conditions (noise, modes)
    * parameters for iterative solvers
    * choice of pre-conditioners
    * data saving

The file for a code ``<code_name>`` is located at::

    <path_to_lib>/struphy/io/inp/<code_name>/parameters.yml

where ``<path_to_lib>`` depends on the user's environment. This file should be modified, 
renamed/copied and targeted from the command line 
with the option ``struphy -i <file_name>``, as explained in the :ref:`quickstart` guide.

By using ``h5py``, the ``.yml`` input is translated to a nested Python dictionary, 
whose entries are then used to initialize :ref:`objects`.
The keys of the dictionary are ``geometry``, ``mhd_equilibrium``, ``kinetic_equilibirum``, ``mhd_init``, ``kinetic_init``,
``grid``, ``markers``, ``time`` and ``solvers``. We will describe the corresponding values in the following.


.. _params_geometry:

Parameters: ``geometry``
^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``geometry`` are used to initialize the :ref:`domain_class`.
The corresponding entry in ``parameters.yml`` reads::

   geometry : 
      type           : cuboid # geometry (see mappings_3d.py for choices -> kind_map)   
      params_cuboid :
            b1       : 1. # start of slice in eta1
            e1       : 2. # end of slice in eta1, e1 > b1
            b2       : 10.
            e2       : 20.
            b3       : 100.
            e3       : 200.
      params_orthogonal :
            Lx       : 1.
            Ly       : 1.
            alpha    : .1 # x-distortion
            Lz       : 1.
      params_colella :
            Lx       : 1.
            Ly       : 1.
            alpha    : .1 # xy-distortion
            Lz       : 1.
      params_hollow_cyl :
            a1       : .1 # inner radius
            a2       : 1. # outer radius
            Lz       : 1. # length of cylinder and major radius
      params_hollow_torus :
            a1       : .1 # inner radius
            a2       : 1. # minor radius
            R0       : 1. # major radius
      params_spline :
            file     : 'mhd_equil/gvec/sample_gvec_spline_coeffs.hdf5'
            Nel      : [6, 6, 6] # number of grid cells > 1
            p        : [2, 3, 3] # spline degree > 0
            spl_kind : [False, True, True] # spline type: True=periodic, False=clamped
      params_spline_cyl :
            a        : 1. # minor radius
            R0       : 0. # major radius (shift of pole from zero)
            Lz       : 4. # length of cylinder
            Nel      : [16, 2] # number of grid cells > 1
            p        : [3, 1] # spline degree > 0
            spl_kind : [False, True] # spline type: True=periodic, False=clamped
      params_spline_torus :
            a        : 1. # minor radius
            R0       : 10. # major radius
            Nel      : [16, 2] # number of grid cells > 1
            p        : [3, 1] # spline degree > 0
            spl_kind : [False, True] # spline type: True=periodic, False=clamped


.. _params_mhd_equil:

Parameters: ``mhd_equilibrium``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``mhd_equilibrium`` are used to initialize the :ref:`mhd_equil_p`.
The corresponding entry in ``parameters.yml`` reads::

    mhd_equilibrium :
        general :
                type        : slab # type of MHD equilibrium (slab, analytic_circular, eqdesk, gvec, vmec)
                mass_number : 1    # in units of proton mass, i.e. Hydrogen : 1, Deuterium: 2, etc.
        params_slab :
                B0x         : 1.   # magnetic field in Tesla (x)
                B0y         : 0.   # magnetic field in Tesla (y)
                B0z         : 0.   # magnetic field in Tesla (z)
                rho0        : 1.   # equilibirum mass density
                beta        : 0.   # plasma beta in %
        params_cylinder :
                a           : 1.   # minor radius in m
                R0          : 10.  # major radius in m (sets length of cylinder to 2*pi*R0)
                B0          : 1.   # constant axial magnetic field in Tesla
                q0          : 1.1  # safety factor at s=0
                q1          : 1.85 # safety factor at s=1
                rl          : 1.   # shape factor
                density profile :
                    r1    : 4.
                    r2    : 3.
                    ra    : 0.
        params_torus :
                a           : 1.   # minor radius in m
                R0          : 10.  # major radius in m
                B0          : 1.   # on-axis (s=0) toroidal magnetic field in Tesla
                q0          : 1.1  # safety factor at s=0
                q1          : 1.85 # safety factor at s=1
                rl          : 1.   # shape factor
                q_add       : 0    # add toroidal correction  
                density profile :
                    r1    : 4.
                    r2    : 3.
                    ra    : 0.
                pressure profile :
                    beta  : 0.2
                    p1    : 0.95
                    p2    : 0.05
        params_gvec :
                filepath : 'struphy/mhd_equil/gvec/'
                filename : 'GVEC_ellipStell_profile_update_State_0000_00010000.dat' # .dat or .json


.. _params_kinetic_equil:

Parameters: ``kinetic_equilibirum``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``kinetic_equilibrium`` are used to initialize the :ref:`kinetic_equil_p`.
The corresponding entry in ``parameters.yml`` reads::

    kinetic_equilibrium :
        general :
                type            : Maxwell_homogen_slab # type of kinetic equilibrium
                nuh             : 0.05 # ratio hot/bulk number densities (nuh<=0.0 is a run without particles)
                particle_charge : 1. # in units of elementary charge
                particle_mass   : 1. # in units of proton mass
                alpha           : 1. # coupling parameter alpha = Omega_{cp0}/Omega_{A0} 
        params_Maxwell_homogen_slab :
                vth_x    : 1. # thermal velocity (in units of v_A)
                vth_y    : 1.
                vth_z    : 1.
                v0_x     : 0.
                v0_y     : 0.
                v0_z     : 0.
                nh0      : 1. # density in units ?
        params_Maxwell_pitchangle :
                vth    : 1. # thermal velocity (in units of v_A)
                alpha0 : 1.
                delta  : 1.


.. _params_mhd_init:

Parameters: ``mhd_init``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``mhd_init`` are used to initialize the :ref:`mhd_class`.
The corresponding entry in ``parameters.yml`` reads::

    mhd_init   : 
        general :
                type : modes_k # initialization of MHD variables (unperturbed, modes_k, modes_mn, eigfun, noise)
                coords : physical # (physical, norm_logical)
                basis_u : 2 # representation of U (0 : all NNN, 1 : 1-form, 2 : 2-form)
                basis_p : 3 # representation of p (0 : 0-form, 3 : 3-form)
        params_modes_k : # modes_k
                target : [b3] # initial perturbation of (b1, b2, b3, u1, u2, u3, p, r)
                kx : [.8]
                ky : [0.]
                kz : [0.]
                amp : [0.001]
        params_modes_mn : # modes_mn 
                target : [b3] # initial perturbation of (b1, b2, b3, u1, u2, u3, p, r)
                modes_m : [1]
                modes_n : [1]
                amp : [0.001]
        params_eigfun : # eigenfun
                n_tor    : -1 # toroidal mode number
                profiles : False # project equilibrium profiles
                eig_kind : 11 # real (11) or imag (12) part
                eig_freq : 0. # squared eigenfreq.
        params_noise : # noise
                target : [b1, b2, b3, u1, u2, u3, p, r]
                plane : yz # plane of noise (xy, xz, yz)


.. _params_kinetic_init:

Parameters: ``kinetic_init``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``kinetic_init`` are used to initialize the :ref:`markers_class`.
The corresponding entry in ``parameters.yml`` reads::

    kinetic_init   : 
        general : 
                type   : noise # initialization of particles (noise OR modes_k, modes_mn -> density perturbations)
                coords : physical # (physical OR logical)
        params_noise :
                plane : yz # plane of noise (xy, xz, yz)
        params_modes_k :
                kx : [.8]
                ky : [0.]
                kz : [0.] 
                amp : [0.001]
        params_modes_mn :
                modes_m : [1]
                modes_n : [1]
                amp : [0.001]


.. _params_grid:

Parameters: ``grid``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``grid`` are used to initialize the :ref:`mhd_class`.
The corresponding entry in ``parameters.yml`` reads::

    grid :
        Nel      : [16, 2, 2] # number of grid cells > 1
        p        : [3, 1, 1] # spline degree > 0
        spl_kind : [True, True, True] # spline type: True=periodic, False=clamped
        bc       : [f, f] # normal component boundary conditions in eta_1 for 2-forms (homogeneous Dirichlet = d, free boundary = f)
        nq_el    : [4, 2, 2] # quadrature points per grid cell
        nq_pr    : [4, 2, 2] # quadrature points per histopolation cell (for projection)
        polar    : False # use polar splines


.. _params_markers:

Parameters: ``markers``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``markers`` are used to initialize the :ref:`markers_class`.
The corresponding entry in ``parameters.yml`` reads::

    markers :
        Np            : 32000 # total number of hot particles
        control       : False # control variate (use delta-f)?
        loading       : pseudo_random # particle loading
        seed          : 1234 # seed for random number generator
        dir_particles : path_to_particles # directory of particles if loaded externally
        n_bins        : [32, 32] # number of bins in (x,v) for 2d particle binning.   
        v_max         : 5. # maximum velocity for binning (in units of v_A)  


.. _params_time:

Parameters: ``time``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``time`` are used to initialize the :ref:`update_classes`.
The corresponding entry in ``parameters.yml`` reads::

    time : 
        dt         : 0.1
        Tend       : 150.
        max_time   : 1000. # maximum runtime of program in minutes
        split_algo : LieTrotter # 'Lie-Trotter' or 'Strang'
        loc_j_eq   : step_6 # location of j_eq X B term (either 'step_2' or 'step_6')


.. _params_solvers:

Parameters: ``solvers``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters under the key ``solvers`` are used to initialize the :ref:`update_classes`.
The corresponding entry in ``parameters.yml`` reads::

    solvers :
        PRE : FFT # pre-conditioner for linear systems (ILU or FFT)
        tol_inv : 0.1 # ILU: set tolerance for approximation (values < tol_inv are set to zero)
        drop_tol_A  : 0.0001 # ILU: set drop_tol and fill_fac (default: drop_tol=1e-4, fill_fac=10.)
        fill_fac_A  : 10.
        drop_tol_S2 : 0.0001
        fill_fac_S2 : 10.
        drop_tol_S6 : 0.0001
        fill_fac_S6 : 10.
        solver_type_2 : cg # iterative solver used
        solver_type_3 : cg
        tol1 : 0.00000001 # iterative solver tolerance
        tol2 : 0.00000001
        tol3 : 0.00000001
        tol6 : 0.00000001
        maxiter1 : 1000 # maximum number of iterations
        maxiter2 : 1000
        maxiter3 : 1000
        maxiter6 : 1000

