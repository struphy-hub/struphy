.. _add_model:

Adding a new PDE model
----------------------

Struphy provides an abstract framework for seamless addition of new model equations.
A model consists of a set of PDEs that has been discretized within the 
:ref:`GEMPIC <gempic>` framework.

New Struphy models must be added in one of the four modules:

* `models/toy.py <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/models/toy.py>`_
* `models/fluid.py <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/models/fluid.py>`_
* `models/kinetic.py <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/models/kinetic.py>`_
* `models/hybrid.py <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/models/hybrid.py>`_

as child classes of the :class:`StruphyModel <struphy.models.base.StruphyModel>`. **Please refer to existing models for templates.**
Here is a list of points that need to be followed when creating a new model:


1. Start from a template 
^^^^^^^^^^^^^^^^^^^^^^^^

Perform the following steps:

a. In one of the four files above, copy-and-paste an existing model.
b. Change the class name to ``<newname>``.
c. Run ``struphy --refresh-models`` in the console.
d. Type ``struphy params <newname>`` and run the new model.


2. Derive Struphy discretization of your PDE 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Struphy uses the :ref:`GEMPIC <gempic>` framework on 3D mapped domains. 
This framework uses Lagrangian particle methods combined with geometric finite elements
based on differential forms. 

Please consult :ref:`disc_example`
and/or given references for a tutorial on how to apply this discretization method.

.. _species:


3. Define :code:`Species`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`species`.


4. Define ``bulk_species`` and ``velocity_scale``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These must be implemented in every Struphy model in order to :func:`struphy.io.setup.derive_units`, see also :ref:`normalization`.

.. _add_prop:

5. Add Propagators
^^^^^^^^^^^^^^^^^^

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme in :attr:`struphy.models.base.StruphyModel.integrate`.

When adding a new model to Struphy, make sure to

1. check the lists of :ref:`available propagators <propagators>` - maybe what you need is already there!
2. write your own propagators based on existing templates. 

Propagators are in one of the following modules:

* `propagators/propagators_fields.py <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/propagators/propagators_fields.py>`_
* `propagators/propagators_markers.py <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/propagators/propagators_markers.py>`_
* `propagators/propagators_coupling.py <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/propagators/propagators_coupling.py>`_

**Check out** :ref:`write_prop` **for practical details on the implementation.**

A model's propagators are defined in :class:`struphy.models.base.StruphyModel.Propagators`.
The order in which propagators are added in :class:`~struphy.models.base.StruphyModel.Propagators` matters. 
They are called consecutively according to the time splitting scheme defined in :ref:`time`.


6. Add scalar quantities
^^^^^^^^^^^^^^^^^^^^^^^^

It is often usefule to define scalar quantities that should be saved during the simulation,
e.g. for checking concervation properties. This can be done via the methods

* :meth:`struphy.models.base.StruphyModel.add_scalar`
* :meth:`struphy.models.base.StruphyModel.update_scalar_quantities`

Check out existing models for templates.


7. Overrride `generate_default_parameter_file`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If necessary due to the propagators of the model, override :meth:`struphy.models.base.StruphyModel.generate_default_parameter_file`
to generate the parameter file you intended.


8. Test
^^^^^^^

Once you added a model and re-installed struphy (``pip install -e .``), 
you can run the model with::
    
    struphy params -y <yourmodel>
    python params_<yourmodel>.py

If the model is not found::

    struphy --refresh-models