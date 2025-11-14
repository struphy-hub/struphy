Noise
^^^^^

In the following example, the first and the third component of the 2-form ``u2`` are initialized with noise 
(in the direction ``e3`` logical space),
while the second component as well as the 3-forms ``n3`` and ``p3`` are left as zero fields::

    perturbations :
        type : noise
        noise : 
            comps :
                n3 : False                # components to be initialized 
                u2 : [True, False, True]  # components to be initialized 
                p3 : False                # components to be initialized 
            direction : e3 # noise variation (logical space): e1, e2, e3 (1d), e1e2, e1e3, e2e3 (2d), e1e2e3 (3d)
            amp : 0.0001   # noise amplitude
            seed : 1234    # seed for random number generator

Analytical perturbations
^^^^^^^^^^^^^^^^^^^^^^^^

Perturbations must be hard-coded callables in ``struphy/initial/perturbations.py``. 
In the parameter file, one can specify the basis in which these callables are given,
via the parameter ``given_in_basis``, see :ref:`pullback`.
For scalar-values variables, the choices are

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

The transformation from the basis specified in ``give_in_basis`` to the variable's solution space is done internally.
More information and plots can be obtained by running `this unit test <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/initial/tests/test_init_perturbations.py?ref_type=heads>`_.

.. automodule:: struphy.initial.perturbations
    :members:
    :show-inheritance:
    
MHD eigenfunctions
^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.initial.eigenfunctions
    :members:
    :show-inheritance:

Utilities
^^^^^^^^^

.. automodule:: struphy.initial.utilities
    :members:
    :show-inheritance: