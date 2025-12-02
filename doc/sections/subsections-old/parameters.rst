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
* :ref:`fluid_background` (under the key ``fluid_background``)

3. Species parameters:

* :ref:`em_fields` (under the key ``em_fields``)
* :ref:`fluid` (under the key ``fluid``)
* :ref:`kinetic` (under the key ``kinetic``)

The ``.yml`` parameter file has a generic structure for all Struphy models.
The keys ``grid``, ``time``, ``units`` and ``geometry`` are present 
in every model; moreover, at least one of the three
species types ``em_fields``, ``fluid`` and ``kinetic`` must be present 
in each model, relating to the model variable(s). An ``fluid_background``
is not mandatory.

The structure of the dictionaries under each of the 8 top-level keys
is discussed below. Special information is available on how to set :ref:`initial_conditions`
and :ref:`boundary_conditions`. 

Some hints for editing a parameter file:

* Strings can either be set as e.g. ``'Cuboid'`` or ``Cuboid``, i.e. with or without quotes - both works.
* The parameter ``null`` will be transformed to Python's ``None`` type.
* Available geometries can be found in :ref:`avail_mappings`
* Available fluid equilibria can be found in :ref:`equils`
* Available kinetic backgrounds can be found in :ref:`kinetic_backgrounds`
* Available fluid backgrounds can be found in :ref:`equils_avail`
* Available perturbations can be found in :ref:`avail_inits`

.. _grid:

Space grid parameters
^^^^^^^^^^^^^^^^^^^^^

Struphy uses a tensor-product grid in three dimensions for space discretization.

Example:

::

    grid :
        Nel          : [12, 14, 4] 
        p            : [3, 4, 2]  
        spl_kind     : [False, True, True] 
        dirichlet_bc : [[False, False], [False, False], [False, False]] 
        dims_mask    : [True, True, True] 
        nq_el        : [2, 2, 2] 
        nq_pr        : [2, 2, 2] 
        polar_ck     : -1 

Parameters:

.. list-table:: 
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``Nel``
     - Number of elements (grid cells) in each direction.
     - 3-list[int]
     - 
   * - ``p``
     - Spline degree in each direction.
     - 3-list[int]
     -
   * - ``spl_kind``
     - Kind of spline in each direction (see :ref:`uni_variate_spaces`).
     - 3-list[bool]
     - * ``True``: periodic
       * ``False``: clamped
   * - ``dirichlet_bc``
     - Homogeneous Dirichlet boundary conditions at left and/or right boundary in each direction.
     - 3-list[2-list[int]]
     - * ``True``: yes
       * ``False``: no
       * See :ref:`feec_bcs` for how to set boundary conditions.
   * - ``dims_mask``
     - Allow domain decomposition of direction.
     - 3-ist[bool]
     - * ``True``: yes
       * ``False``: no
   * - ``nq_el``
     - Number of quadrature points per element (e.g. for L2-projections) in each direction.
     - List[int]
     -
   * - ``nq_pr``
     - Quadrature points between Greville points (for commuting projectors) in each direction. 
     - List[int]
     -
   * - ``polar_ck``
     - :math:`C^k` smoothness of B-splines at polar singularity :math:`\eta_1=0` for polar geometries.
     - int
     - * ``-1``: standard tensor product, not continuous at pole
       * ``1``: use :math:`C^1` :ref:`polar_splines`

These paramters are primarily used to instantiate :class:`struphy.feec.psydac_derham.Derham`.


.. _time:

Time stepping parameters
^^^^^^^^^^^^^^^^^^^^^^^^

Example:

::

    time :
        dt         : 0.005 
        Tend       : 0.015 
        split_algo : LieTrotter 

Parameters:

.. list-table:: 
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``dt``
     - Time step in units given by a model's :ref:`normalization` (inspect via ``struphy units -i <params_file> MODEL`` from the console). 
     - float
     -
   * - ``Tend``
     - End time in same units as ``dt``.
     - float
     -
   * - ``split_algo``
     - Time splitting algorithm
     - str
     - * ``LieTrotter``: first order
       * ``Strang``: second order, symmetric


.. _units:

Units
^^^^^

Example:

::

    units : 
        x : 1.  
        B : 1. 
        n : 1. 
        kBT : 1. 

Parameters:

.. list-table:: 
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``x``
     - Length scale unit in Meter (m).
     - float
     -
   * - ``B``
     - Magnetic field unit in Tesla (T).
     - float
     -
   * - ``n``
     - Number density unit in :math:`10^{20}\ m^{-3}`.
     - float
     -
   * - ``kBT`` (optional)
     - Thermal energy unit in keV (optional).
     - float
     -

Theses base units are used to derive all other units via :func:`struphy.io.setup.derive_units`.
See also :ref:`normalization` for details.


.. _geometry:

Geometry
^^^^^^^^

Example:

::

    geometry :
        type : Cuboid 
        Cuboid : {}

Parameters:

.. list-table:: 
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``type``
     - The mapping :math:`F:[0, 1]^3 \to \Omega` (its class name ``<mapping_class>``). 
     - str
     - See :ref:`avail_mappings`.
   * - ``<mapping_class>``
     - Parameters for the chosen mapping. If empty, the defaults are taken.
     - dict
     - See docstrings in :ref:`avail_mappings`.


.. _fluid_background:

Fluid background
^^^^^^^^^^^^^^^^

Example:

::

    fluid_background :
        HomogenSlab : {}

Parameters:

.. list-table:: 
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``<equil_class>``
     - Class name and parameters for the equilibrium. If empty, the defaults are taken.
     - dict
     - See :ref:`equils_avail`.


.. _em_fields:

Electromagnetic fields
^^^^^^^^^^^^^^^^^^^^^^

Initial conditions are the sum of ``background`` + ``perturbation``, see :ref:`initial_conditions`.

Example:

::

    em_fields :
        background: 
            phi : 
              LogicalConst :
                  values : 1.3 
            A : 
              LogicalConst :
                  values : [.3, .15, null] 
        perturbation :
            phi:
              TorusModesCos :
                  given_in_basis : '0' 
                    A : 
                  ms : [1] 
            A:
              TorusModesCos :
                  given_in_basis : [null, 'v', null]
                  ms : [null, [1, 3], null]  

Parameters:

.. list-table:: ``background``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``<variable_names>``
     - Name of variable to be initialized.
     - dict
     - Contains class name and parameters for the static background, see :ref:`equils_avail`. If empty, the defaults are taken.

.. list-table:: ``perturbation``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``<variable_names>``
     - Name of variable to be initialized.
     - dict
     - Contains class name and parameters for the perturbation, see :ref:`avail_inits`. If empty, the defaults are taken.



.. _fluid:

Fluid variables
^^^^^^^^^^^^^^^

There can be multiple ``fluid`` species in a Struphy model,
each of which has its paramaters under a key ``<species_name>``.
For each species, there are

* physical parameters ``phys_params``,
* initial conditions as the sum of ``background`` + ``perturbation``, see :ref:`initial_conditions`.

Example (one fluid species):

::

    fluid :
        <species_name> :
            phys_params:
                A : 1 
                Z : 1 
            background : 
                velocity :
                    LogicalConst :
                        values : [null, 1.5, null] 
                density : 
                    LogicalConst :
                        values : 2.3 
            perturbation :
                density : 
                    TorusModesCos :
                        given_in_basis : '0' 
                        ms : [1, 3] 

Parameters:

.. list-table:: ``phys_params``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``A``
     - Mass number in units of proton mass.
     - float
     - :math:`A=1/1836` for electrons.
   * - ``Z``
     - Signed charge number in units of elementary charge.
     - int
     - :math:`Z=-1` for electrons.

.. list-table:: ``background``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``<variable_names>``
     - Name of variable to be initialized.
     - dict
     - Contains class name and parameters for the static background, see :ref:`equils_avail`. If empty, the defaults are taken.

.. list-table:: ``perturbation``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``<variable_names>``
     - Name of variable to be initialized.
     - dict
     - Contains class name and parameters for the perturbation, see :ref:`avail_inits`. If empty, the defaults are taken.

.. _kinetic:

Kinetic variables
^^^^^^^^^^^^^^^^^


There can be multiple ``kinetic`` species in a Struphy model,
each of which has its paramaters under a key ``<species_name>``.
For each species, there are

* physical parameters ``phys_params``,
* information on the ``markers``,
* parameters for saving kinetic data ``save_data``,
* initial conditions as the sum of ``background`` + ``perturbation``, see :ref:`initial_conditions`.

For ``kinetic`` species, the ``background`` is mandatory.

Example (one kinetic species):

::

    kinetic :
        <species_name> :
            phys_params :
                A : 1
                Z : 1
            markers :
                type    : full_f 
                ppc     : null 
                Np      : 1000
                bufsize     : .25
                bc : 
                    type    : [remove, periodic, periodic] 
                    remove  : boundary_transfer
                loading :
                    type              : pseudo_random
                    seed              : 1234 
                    moments           : [0., 0., 0., 1., 1., 1.] 
                    spatial           : uniform
                    dir_external      : 'path_to_particles' 
                    dir_particles     : 'path_to_particles' 
                    dir_particles_abs : 'path_to_particles' 
            save_data :
                n_markers : 3
                f :
                    slices : [v1, e1_v1]
                    n_bins : [[32], [32, 32]]
                    ranges : [[[-3., 3.]], [[0., 1.], [-5., 5.]]]
            background : 
                Maxwellian3D :
                    n  : 0.05
                    u2 : 2.5
            perturbation :
                n :
                    TorusModesCos :
                        given_in_basis :  '0' 
                        ms : [1, 3] 

Parameters:

.. list-table:: ``phys_params``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``A``
     - Mass number in units of proton mass.
     - float
     - :math:`A=1/1836` for electrons.
   * - ``Z``
     - Signed charge number in units of elementary charge.
     - int
     - :math:`Z=-1` for electrons.

.. list-table:: ``markers``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``Np``
     - Total number of markers.
     - int
     - 
   * - ``ppc``
     - Number of markers per grid cell.
     - int
     - Takes effect only if ``Np`` is absent.
   * - ``ppb``
     - Number of markers per sorting box.
     - int
     - Takes effect only if both ``Np`` and ``ppc`` are absent.
   * - ``bufsize``
     - Size of MPI-buffer in markers array as fraction of markers per process.
     - float
     - 
   * - ``bc``
     - Marker boundary conditions. See :meth:`struphy.pic.base.Particles.apply_kinetic_bc`.
     - dict
     - * ``type`` (3-list[str]): marker boundary conditions in each logical direction.
     
         * ``remove``: markers outside of :math:`\Omega` are removed or re-filled.
         * ``reflect``: markers are reflected at :math:`\partial \Omega`.
         * ``periodic``: markers re-enter at other side.
       * ``remove`` (str): re-fills markers in radial direction on same flux surface if ``type`` is ``remove``.

         * ``boundary_transfer``:  when particles reach the INNER radial boundary, transfer them to the opposite poloidal angle of the same magnetic flux surface.
         * ``particle_refilling``: when particles reach the OUTER radial boundary, transfer them to the opposite poloidal angle of the same magnetic flux surface.
   * - ``loading``
     - How to load random markers.
     - dict
     - * ``type`` (str): particle loading mechanism, see :meth:`struphy.pic.base.Particles.draw_markers`.

         * ``pseudo_random`` load markers with standard random generator ``np.random.rand``.
         * ``sobol_standard`` load Sobol sequence, see :module:`struphy.pic.sobol_seq`.
         * ``sobol_antithetic`` load Sobol sequence, see :module:`struphy.pic.sobol_seq`.
         * ``external``: load markers from external .hdf5 file specified in ``dir_external``.
         * ``restart``: load markers from stopped simulation in folder ``dir_particles`` or ``dir_particles_abs``.
       * ``seed`` (int): seed for random number generator ``np.random.rand``.
       * ``moments`` (list): mean velocity and standard deviation of Gaussian that markers are drawn from, see :meth:`struphy.pic.base.Particles.draw_markers`.
       * ``spatial`` (str): uniformity of markers in position space.
         
         * ``uniform``: markers are uniform on the logical unit cube.
         * ``disc``: markers are uniform on the disc :math:`(\eta_1, \eta_2) \mapsto (r, \theta)`.
       * optional: ``dir_external`` (str): path to .hdf5 file
       * optional: ``dir_particles`` (str): simulation folder relative to current output path 
       * optional: ``dir_particles_abs`` (str): simulation folder, absolute path
   * - ``control_variate``
     - Whether to use a :ref:`control_var` for noise reduction (only if ``type`` is ``full_f`` or ``sph``).
     - bool
     -


.. list-table:: ``save_data``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``n_markers``
     - Number of markers to be saved during simulation. Markers with an ID less than ``n_markers`` are saved.
     - int
     - 
   * - ``f``
     - Binning plots of the distribution function in phase space.
     - dict
     - * ``slices`` (list): in which directions to bin (e.g. ``[e1_e2, e1_v2_v3]``)
       * ``n_bins`` (list): number of bins in each slice-direction (e.g. ``[[16, 20], [16, 18, 22]]``)
       * ``ranges`` (list): bin range in each slice-direction (e.g. [[[0., 1.], [0., 1.]], [[0, 1.], [-4., 4.], [-5., 5.]]])


.. list-table:: ``background``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``<bckgr_class>``
     - Class name and parameters for the static kinetic background. If empty, the defaults are taken.
     - dict
     - See :ref:`kinetic_backgrounds`.

.. list-table:: ``perturbation``
   :widths: 15 35 10 40
   :header-rows: 1

   * - Name
     - Description
     - Format
     - Choices
   * - ``<moment_names>``
     - Name of moment to be initialized.
     - dict
     - Contains class name and parameters for the perturbation, see :ref:`avail_inits`. If empty, the defaults are taken.