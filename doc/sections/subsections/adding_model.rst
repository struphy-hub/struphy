.. _add_model:

Adding a new PDE model
----------------------

Struphy provides an extensive framework for adding new model equations.
A model consists of a set of PDEs that has been discretized within the 
:ref:`GEMPIC <gempic>` framework.

New Struphy models must be added in one of the four modules

.. autosummary::
    :nosignatures:
    :toctree: STUBDIR

    struphy.models.fluid
    struphy.models.kinetic
    struphy.models.hybrid
    struphy.models.toy

as child classes of the :class:`StruphyModel <struphy.models.base.StruphyModel>`. **Please refer to existing models for templates.**
Here is a list of points that need to be followed when creating a new model:

1. Derive Struphy discretization of your PDE 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Struphy uses the :ref:`GEMPIC <gempic>` framework on 3D mapped domains. 
This framework uses Lagrangian particle methods combined with geometric finite elements
based on differential forms. In order to benefit from the full capabilities of Struphy,
you should discretize your PDE in this framework. Please consult :ref:`disc_example`
and/or given references for a tutorial on how to do this.

2. Start from a template 
^^^^^^^^^^^^^^^^^^^^^^^^

As a start for adding a new model, copy-and-paste an existing one and change its name. 
You can already run your "new" model with ``struphy run NEW_NAME``, which will just execute the copied model. 

3. Add a model docstring
^^^^^^^^^^^^^^^^^^^^^^^^

The docstring should include the model equations in Latex format and the :ref:`normalization`.
Weak equations should be written as such.
Do not include discretized equations in the model docstring.
You can follow :ref:`change_doc` to see if your changes have been taken into
account. 

.. _species:

4. Define :code:`species(cls)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`species(cls)` class method must be implemented in every Struphy model.
It returns a dictionary that holds the information on the models' variables (i.e. the unknowns)
and their respective discrete spaces (PIC or FEEC) in which they are defined.
In Struphy, three types of species can be defined:

* electromagnetic (EM) fields (under the dict key :code:`em_fields`)
* fluid species (dict key :code:`fluid`)
* kinetic species (dict key :code:`kinetic`)

Each species can be assigned an arbitrary name, chosen by the developer,
which must appear as a sub-key in one of the above dicts. The corresponding value is either

* for EM fields: the name of a FEEC space (``H1``, ``Hcurl``, ``Hdiv``, ``L2`` or ``H1vec``)
* for fluid species: a dictionary holding the fluid variable names (keys) and FEEC spaces (values)
* for kinetic species: the name of a :ref:`particle class <pic_base>`

Let us look at the model ``LinearMHDVlasovCC`` as an example::

    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        
        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'n3': 'L2', 'u2': 'Hdiv', 'p3': 'L2'}
        dct['kinetic']['energetic_ions'] = 'Particles6D'
        return dct    

Here, one field variable (``b2``), one fluid species (``mhd``) and one kinetic species (``energetic_ions``)
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


5. Define ``bulk_species(cls)`` and ``velocity_scale(cls)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These must be implemented as class methods of the model::

    @classmethod
    def bulk_species(cls):
        return 'energetic_ions'

    @classmethod
    def velocity_scale(cls):
        return 'light'

The ``bulk_species`` must return the name of one of the species of the model. 

There are three options for the ``velocity_scale``:

* ``alfvén``
* ``cyclotron``
* ``light``

The choice corresponds to setting the velocity unit :math:`\hat v` of the normalization.
This then sets the time unit :math:`\hat t = \hat x / \hat v`, where :math:`\hat x` is the 
unit of length specified through the parameter file. We refer to :ref:`normalization` and to
`Tutorial 01 <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_01_units_run_main.html#Struphy-normalization-(units)>`_ 
for more details on the Struphy normalization.

.. _add_prop:

6. Add Propagators
^^^^^^^^^^^^^^^^^^

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme of the algorithm. When adding a new model to Struphy, make sure to

1. check the lists of :ref:`available propagators <propagators>` - maybe what you need is already there!
2. write your own propagators based on existing templates. 

Propagators have to be added to a model in the :code:`__init__()`-section of the model class 
via the method :meth:`add_propagator() <struphy.models.base.StruphyModel.add_propagator>`.
The method takes one argument, namely an object from one of its class attributes

* :attr:`self.prop_fields <struphy.models.base.StruphyModel.prop_fields>` 
* :attr:`self.prop_markers <struphy.models.base.StruphyModel.prop_markers>` 
* :attr:`self.prop_coupling <struphy.models.base.StruphyModel.prop_coupling>` 

These three class attributes are pointers to the :ref:`three propagator modules <propagators>`. 
When a model is instantiated, all classes in these three modules (which are child classes of :ref:`prop_base`) 
get assigned the current :ref:`de Rham sequence <de_rham>`, :ref:`domain <avail_mappings>`, 
:ref:`mass operators <weighted_mass>` and :ref:`basis projection operators <basis_ops>`.
Any instance of a class then inherits this information.

Consider for example of the model :class:`~struphy.models.kinetic.VlasovMaxwellOneSpecies`,
where four different propagators are used for time stepping::

    self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e1'],
            self.pointer['b2'],
            **params_maxwell))

    self.add_propagator(self.prop_markers.PushEta(
        self.pointer['electrons'],
        algo=algo_eta,
        bc_type=electron_params['markers']['dirichlet_bc']['type'],
        f0=None))

    self.add_propagator(self.prop_markers.PushVxB(
        self.pointer['electrons'],
        algo=algo_vxb,
        scale_fac=1/self._epsilon,
        b_eq=self._b_background,
        b_tilde=self.pointer['b2'],
        f0=None))

    self.add_propagator(self.prop_coupling.VlasovMaxwell(
        self.pointer['e1'],
        self.pointer['electrons'],
        c1=self._alpha**2/self._epsilon,
        c2=1/self._epsilon,
        **params_coupling)) 

Here, one "field"-propagator (:class:`Maxwell <struphy.propagators.propagators_fields.Maxwell>`), 
two "marker"-propagators (:class:`PushEta <struphy.propagators.propagators_markers.PushEta>` 
and :class:`PushVxB <struphy.propagators.propagators_markers.PushVxB>`) and one "hybrid"-propagator 
(:class:`VlasovMaxwell <struphy.propagators.propagators_coupling.VlasovMaxwell>`) are added to the model.
Each Propagator takes as arguments the variables to be updated, 
usually passed via the :attr:`pointer attribute <struphy.models.base.StruphyModel.pointer>`. 
**All additional arguments MUST be passed as keyword arguments.** 

The order in which you add propagators matters. They are called consecutively in the given :ref:`time splitting scheme <time>`.


7. Add scalar quantities
^^^^^^^^^^^^^^^^^^^^^^^^

This allows to define scalar quantities that should be saved during the simulation.
We use the method :meth:`add_scalar() <struphy.models.base.StruphyModel.add_scalar>` 
of the :class:`StruphyModel <struphy.models.base.StruphyModel>` base class.
Consider the the example of the model :class:`~struphy.models.kinetic.VlasovMaxwellOneSpecies`::

    self.add_scalar('en_e')
    self.add_scalar('en_b')
    self.add_scalar('en_w')
    self.add_scalar('en_tot')
 
These commands just reserve memory for the scalars and assigns a name to them.
You must spacify an update rule via :meth:`struphy.models.base.StruphyModel.update_scalar_quantities`,
such as in the above example::

    def update_scalar_quantities(self):

        self._mass_ops.M1.dot(self.pointer['e1'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp2)
        en_E = self.pointer['e1'].dot(self._tmp1) / 2.
        en_B = self.pointer['b2'].dot(self._tmp2) / 2.
        self.update_scalar('en_e', en_E)
        self.update_scalar('en_b', en_B)

        self._tmp[0] = self.alpha**2 / (2 * self.pointer['electrons'].n_mks) * \
            np.dot(self.pointer['electrons'].markers_wo_holes[:, 3]**2 + self.pointer['electrons'].markers_wo_holes[:, 4] ** 2 +
                    self.pointer['electrons'].markers_wo_holes[:, 5]**2, self.pointer['electrons'].markers_wo_holes[:, 6])
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)
        self.update_scalar('en_w', self._tmp[0])

        self.update_scalar('en_tot', en_E + en_B + self._tmp[0])


8. Add options
^^^^^^^^^^^^^^

The :code:`options(cls)` class method can be used to add model-specific
parameters to the :ref:`core parameters <params_yml>` in the input file.
It returns a dict with the same top-level keys as :ref:`species(cls) <species>`.
The dict structure must be the same as it should appear in the parameter file.
For each species, they must appear under the key :code:`options`. 

Often, some of these options can be imported from propagators. Let us 
check the model :code:`LinearMHDVlasovCC` as an example::

    @classmethod
    def options(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        # import propagator options
        from struphy.propagators.propagators_fields import ShearAlfvén, Magnetosonic, CurrentCoupling6DDensity
        from struphy.propagators.propagators_markers import PushEta, PushVxB
        from struphy.propagators.propagators_coupling import CurrentCoupling6DCurrent
        dct['fluid']['mhd'] = {}
        dct['fluid']['mhd']['options'] = {}
        dct['fluid']['mhd']['options']['solvers'] = {}
        dct['fluid']['mhd']['options']['solvers']['shear_alfven'] = ShearAlfvén.options()['solver']
        dct['fluid']['mhd']['options']['solvers']['magnetosonic'] = Magnetosonic.options()['solver']
        dct['kinetic']['energetic_ions'] = {}
        dct['kinetic']['energetic_ions']['options'] = {}
        dct['kinetic']['energetic_ions']['options']['algos'] = {}
        dct['kinetic']['energetic_ions']['options']['algos']['push_eta'] = PushEta.options()['algo']
        dct['kinetic']['energetic_ions']['options']['algos']['push_vxb'] = PushVxB.options()['algo']
        dct['kinetic']['energetic_ions']['options']['solvers'] = {}
        dct['kinetic']['energetic_ions']['options']['solvers']['density'] = CurrentCoupling6DDensity.options()['solver']
        dct['kinetic']['energetic_ions']['options']['solvers']['current'] = CurrentCoupling6DCurrent.options()['solver']


9. Test run
^^^^^^^^^^^

Once you added a model and re-installed struphy (``pip install -e .``), 
you can run the model with::
    
    struphy run YOUR_MODEL
    
where ``YOUR_MODEL`` is the name you gave to 
the model class (it must start with a capital letter).