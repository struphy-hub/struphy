.. _add_model:

Adding a new PDE model
----------------------

Struphy provides an abstract framework for seamless addition of new model equations.
A model consists of a set of PDEs that has been discretized within the 
:ref:`GEMPIC <gempic>` framework.

New Struphy models must be added in one of the four modules:

* `models/toy.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/toy.py?ref_type=heads>`_
* `models/fluid.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/fluid.py?ref_type=heads>`_
* `models/kinetic.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/kinetic.py?ref_type=heads>`_
* `models/hybrid.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/hybrid.py?ref_type=heads>`_

as child classes of the :class:`StruphyModel <struphy.models.base.StruphyModel>`. **Please refer to existing models for templates.**
Here is a list of points that need to be followed when creating a new model:


1. Start from a template 
^^^^^^^^^^^^^^^^^^^^^^^^

Perform the following steps:

a. In one of the four files above, copy-and-paste an existing model.
b. Change the class name to ``<newname>``.
c. Run ``struphy --refresh-models`` in the console.
d. In the console, run ``struphy run <newname>`` which will just execute the copied model after creating a default parameter file.


2. Derive Struphy discretization of your PDE 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Struphy uses the :ref:`GEMPIC <gempic>` framework on 3D mapped domains. 
This framework uses Lagrangian particle methods combined with geometric finite elements
based on differential forms. 

Please consult :ref:`disc_example`
and/or given references for a tutorial on how to apply this discretization method.

.. _species:


3. Define :code:`species(cls)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`species(cls)` method must be implemented in every Struphy model.
It returns a dictionary that holds the information on the models' variables (i.e. the unknowns)
and their respective discrete spaces (PIC or FEEC) in which they are defined.
Let us look at the model `LinearMHDVlasovCC <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/hybrid.py?ref_type=heads#L64>`_ as an example::

    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        
        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'n3': 'L2', 'u2': 'Hdiv', 'p3': 'L2'}
        dct['kinetic']['energetic_ions'] = 'Particles6D'
        return dct    

In Struphy, three types of species can be defined:

* electromagnetic (EM) fields (under the dict key :code:`em_fields`)
* fluid species (dict key :code:`fluid`)
* kinetic species (dict key :code:`kinetic`)

Each species can be assigned an arbitrary name, chosen by the developer,
which must appear as a sub-key in one of the above dicts. The corresponding value is either

* for EM fields: the name of a FEEC space (``H1``, ``Hcurl``, ``Hdiv``, ``L2`` or ``H1vec``)
* for fluid species: a dictionary holding the fluid variable names (keys) and FEEC spaces (values)
* for kinetic species: the name of a :ref:`particle class <pic_base>`

In the example above, one field variable (``b2``), one fluid species (``mhd``) and one kinetic species (``energetic_ions``)
are initialized. The corresponding discrete spaces appear as values.
There is no limit in how many species/fields can be defined within a model.

Later, the variables defined in :code:`species(cls)` can be accessed 
via the :attr:`pointer attribute <struphy.models.base.StruphyModel.pointer>` 
of the :class:`StruphyModel <struphy.models.base.StruphyModel>` base class.
The variable names are to be used as keys, for example::

    _b2 = self.pointer['b2']
    _n3 = self.pointer['mhd_n3'] 

This returns the :ref:`data_structures` of the variable (the whole :ref:`particle class <particles>` for kinetic species).
In case of a fluid species, the naming convention is :code:`species_variable` 
with an underscore separating species name and variable name.


4. Define ``bulk_species(cls)`` and ``velocity_scale(cls)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These must be implemented in every Struphy model in order to :func:`struphy.io.setup.derive_units`::

    @classmethod
    def bulk_species(cls):
        return 'energetic_ions'

    @classmethod
    def velocity_scale(cls):
        return 'light'

The ``bulk_species`` must return the name of one of the species of the model. 

There are four options for the ``velocity_scale``:

* ``alfvén``
* ``cyclotron``
* ``light``
* ``thermal``

The choice corresponds to setting the velocity unit :math:`\hat v` of the :ref:`normalization`.
This then sets the time unit :math:`\hat t = \hat x / \hat v`, where :math:`\hat x` is the 
unit of length specified through the parameter file.


.. _add_prop:

5. Add Propagators
^^^^^^^^^^^^^^^^^^

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme in :attr:`struphy.models.base.StruphyModel.integrate`.

When adding a new model to Struphy, make sure to

1. check the lists of :ref:`available propagators <propagators>` - maybe what you need is already there!
2. write your own propagators based on existing templates. 

Propagators are in one of the following modules:

* `propagators/propagators_fields.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/propagators/propagators_fields.py?ref_type=heads>`_
* `propagators/propagators_markers.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/propagators/propagators_markers.py?ref_type=heads>`_
* `propagators/propagators_coupling.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/propagators/propagators_coupling.py?ref_type=heads>`_

**Check out** :ref:`write_prop` **for practical details on the implementation.**

A model's propagators are defined in :meth:`struphy.models.base.StruphyModel.propagators_dct`.
See `LinearMHD <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/fluid.py?ref_type=heads#L7>`_ for an example::

    @staticmethod
    def propagators_dct():
        return {propagators_fields.ShearAlfven: ['mhd_velocity', 'b_field'],
                propagators_fields.Magnetosonic: ['mhd_density', 'mhd_velocity', 'mhd_pressure']}

The keys are the :ref:`propagator classes <propagators>` themselves; the values are the names of model variales to be updated by the propagator,
as defined in :meth:`struphy.models.base.StruphyModel.species`, see above.
The updated variables must conform to the solution spaces defined in the ``__init__`` of the propagator (arguments BEFORE ``*``).

The order in which propagators are added in :meth:`~struphy.models.base.StruphyModel.propagators_dct` matters. 
They are called consecutively according to the time splitting scheme defined in :ref:`time`.

Propagator parameters (passed as keyword arguments) must be defined in the ``__init__`` of the model class
by setting the ``self._kwargs`` dictionary of the model, 
see `LinearMHD <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/fluid.py?ref_type=heads#L7>`_ for an example::

    # set keyword arguments for propagators
    self._kwargs[propagators_fields.ShearAlfven] = {'u_space': u_space,
                                                    'solver': alfven_solver}

    self._kwargs[propagators_fields.Magnetosonic] = {'b': self.pointer['b_field'],
                                                     'u_space': u_space,
                                                     'solver': sonic_solver}

The given keyword arguments must conform to the ones defined in the ``__init__`` of the propagator
(arguments AFTER ``*``).



6. Add scalar quantities
^^^^^^^^^^^^^^^^^^^^^^^^

It is often usefule to define scalar quantities that should be saved during the simulation,
e.g. for checking concervation properties. This can be done via the methods

* :meth:`struphy.models.base.StruphyModel.add_scalar`
* :meth:`struphy.models.base.StruphyModel.update_scalar_quantities`

Check out existing models for templates.


7. Add options
^^^^^^^^^^^^^^

Most of a model's options are defined within :meth:`struphy.propagators.base.Propagator.options`,
i.e within the options of the models's propagators. 
It is possible to add additional options through :meth:`struphy.models.base.StruphyModel.options`.
This is done with the method :meth:`struphy.models.base.StruphyModel.add_option`::

    @classmethod
    def options(cls):
        dct = super().options()
        cls.add_option(species=['fluid', 'mhd'], key='u_space',
                       option='Hdiv', dct=dct)
        return dct


8. Test
^^^^^^^

Once you added a model and re-installed struphy (``pip install -e .``), 
you can run the model with::
    
    struphy run <yourmodel>

If the model is not found::

    struphy --refresh-models

and run again. The parameter file of a model is created via::

    struphy params <yourmodel>


9. Add a model docstring
^^^^^^^^^^^^^^^^^^^^^^^^

The docstring should have the following form (example taken from `LinearMHD <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/fluid.py?ref_type=heads#L7>`_)::

    Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

        :ref:`normalization`:

        .. math::

            <normalization in Latex format>

        :ref:`Equations <gempic>`:

        .. math::
        
            <some equations in Latex format>

        :ref:`propagators` (called in sequence):

        1. :class:`~struphy.propagators.propagators_fields.ShearAlfven`
        2. :class:`~struphy.propagators.propagators_fields.Magnetosonic`

        :ref:`Model info <add_model>`:

The equations should be written in strong form (like in a textbook), in the chosen :ref:`normalization`.
Do not include discretized equations in the model docstring.
You can follow :ref:`change_doc` to see if your changes have been taken into
account. 

    