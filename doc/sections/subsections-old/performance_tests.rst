.. _performance_tests:

Scaling tests
-------------

Scaling test: MPCDF_cobra
^^^^^^^^^^^^^^^^^^^^^^^^^
    .. image:: ../../pics/performance_tests/mpcdf_cobra/mpcdf_cobra_overview.png
        :width: 1100
        :align: center


The results from the scaling test of the model, **Maxwell** :py:class:`struphy.models.toy.Maxwell`, with **MPCDF_cobra**.
The Maxwell equations are solved at different domains: :py:class:`Cuboid <struphy.geometry.domains.Cuboid>`, :py:class:`Colella <struphy.geometry.domains.Colella>` and :py:class:`HollowTorus <struphy.geometry.domains.HollowTorus>`.
At each domains, two different initialization, :ref:`Noise <avail_inits>` and :py:class:`ModesSin <struphy.initial.perturbations.ModesSin>`, are tested.
Each of exactly same problems is solved with different numbers of processes **(4, 8, 16, 32)**. 

Used parameters and batch files are available at `struphy-simulations <https://gitlab.mpcdf.mpg.de/struphy/struphy-simulations/-/tree/main/mpcdf_cobra_scaling/Maxwell>`_.

**Time step** : :math:`\Delta t = 0.05 * \hat t \quad`  (:math:`\hat t = \frac{\hat x = 1 \, m}{\hat v = c \, m/s} = 3.335 * 10^{-9} \, s`)

- **Cuboid (Noise)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/maxwell/cuboid.png
        :width: 700
        :align: center

- **Cuboid (ModesSin)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/maxwell/cuboid_sin.png
        :width: 700
        :align: center

- **Colella (Noise)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/maxwell/colella.png
        :width: 700
        :align: center

- **Colella (ModesSin)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/maxwell/colella_sin.png
        :width: 700
        :align: center

- **HollowTorus (Noise)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/maxwell/hollowtorus.png
        :width: 700
        :align: center

- **HollowTorus (ModesSin)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/maxwell/hollowtorus_sin.png
        :width: 700
        :align: center

The results from the scaling test of the model, **LinearMHD** :py:class:`struphy.models.fluid.LinearMHD`, with **MPCDF_cobra**.
The LinearMHD equations are solved at different domains: :py:class:`Cuboid <struphy.geometry.domains.Cuboid>`, :py:class:`Colella <struphy.geometry.domains.Colella>`, :py:class:`HollowTorus <struphy.geometry.domains.HollowTorus>` and :py:class:`Tokamak <struphy.geometry.domains.Tokamak>`.
At each domains, corresponding MHD equilibriums are used, :py:class:`HomogenSlab <struphy.fields_background.equils.HomogenSlab>`, :py:class:`AdhocTorus <struphy.fields_background.equils.AdhocTorus>` and :py:class:`EQDSKequilibrium <struphy.fields_background.equils.EQDSKequilibrium>`.
For all cases, two different initialization, :ref:`Noise <avail_inits>` and :py:class:`ModesSin <struphy.initial.perturbations.ModesSin>`, are tested.
Each of exactly same problems is solved with different numbers of processes **(4, 8, 16, 32)**. 

Used parameters and batch files are available at `struphy-simulations <https://gitlab.mpcdf.mpg.de/struphy/struphy-simulations/-/tree/main/mpcdf_cobra_scaling/LinearMHD>`_.

**Time step** : :math:`\Delta t = 0.05 * \hat t \quad`   (:math:`\hat t = \frac{\hat x = 1 \, m}{\hat v = v_A \, m/s}`)

- **Cuboid, HomogenSlab (Noise)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/cuboid.png
        :width: 700
        :align: center

- **Cuboid, HomogenSlab (ModesSin)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/cuboid_sin.png
        :width: 700
        :align: center

- **Colella, HomogenSlab (Noise)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/colella.png
        :width: 700
        :align: center

- **Colella, HomogenSlab (ModesSin)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/colella_sin.png
        :width: 700
        :align: center

- **HollowTorus, AdhocTorus (Noise)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/hollowtorus.png
        :width: 700
        :align: center

- **HollowTorus, AdhocTorus (ModesSin)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/hollowtorus_sin.png
        :width: 700
        :align: center

- **Tokamak, EQDSKequilibrium (Noise)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/tokamak.png
        :width: 700
        :align: center

- **Tokamak, EQDSKequilibrium (ModesSin)**

    .. image:: ../../pics/performance_tests/mpcdf_cobra/linearmhd/tokamak_sin.png
        :width: 700
        :align: center