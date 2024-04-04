.. _avail_inits:

Perturbations
-------------

Perturbations can be added on top of :ref:`fluid_backgrounds` or :ref:`kinetic_backgrounds`.
Below are examples for dictionary structures to be put under the ``perturbations``
key of a ``.yml`` parameter file.

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