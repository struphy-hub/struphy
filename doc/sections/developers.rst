.. _developers:

Developer's guide
=================

Developers can contribute to Struphy in multiple ways. A common approach is by :ref:`add_model`,
but also writing :ref:`diagnostics` or physics features, or modifying core routines is possible.

The main line of communication between developers is the `Struphy-developers channel <https://chat.gwdg.de/channel/struphy-developers>`_ 
on the GWDG RocketChat. 
Besides, the Struphy developer community meets regularly at `Struphy Hackathons <https://gitlab.mpcdf.mpg.de/struphy/struphy-hackathons>`_.

In order to maintain a sustainable software framework, Struphy developers should follow:

* :ref:`git_workflow`
* :ref:`conventions` 
* `Struphy issues page <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/issues>`_

A comprehensive documentation is vital for the usability of Struphy. All developers should therefore 
check out the hints for :ref:`change_doc`.

Struphy is an object-oriented code that provides base classes (templates)
which facilitate the implementation of new features. In other words, 
Struphy provides an abstraction layer for PIC and FEEC basic algorithms and operators,
which are efficiently implemented in the low-level core routines,
or imported from other Python packages (`Psydac <https://github.com/pyccel/psydac>`_) via APIs.
The developer can use these core features to construct new instances of operators/algorithms,
suitable for his PDE model. The core routines provide MPI/OpenMP hybrid parallelization
and manage the communication between different software parts.
**This enables the developer to focus on the physical and mathematical aspects of his model.**
For further details please go to:

* :ref:`struphy_model`
* :ref:`pic_base`
* :ref:`feec_base`

In order to understand these classes, developers should get familiar with :ref:`data_structures`.

The objective of Struphy is to conserve and maintain PDE models that are added by developers.
This can be achieved by having a uniform, well-documented structure to all models.
This structure is provided by a) the above base classes and b) the :ref:`params_yml`.
New developers should always look to existing models/features/code parts for guidance.



.. _git_workflow:

Git workflow
------------

Main branches 
^^^^^^^^^^^^^

The `Struphy repository <https://gitlab.mpcdf.mpg.de/struphy/struphy/>`_ has two main branches, ``master`` and ``devel``. 
Nobody can push directly to these branches.

The ``master`` branch holds the current release of the code. 

``devel`` is the main branch for developers. :ref:`feature_branches` must be checked out and merged into ``devel``.


.. _open_dev:

Open development
^^^^^^^^^^^^^^^^

When adding code to struphy it is important that other developers can follow your plans.
For this we use the `Issue tracker <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/issues>`_.
There, you can add a short description of your plans and choose one of the following ``labels``:

* Bug
* Discussion
* Documentation
* Feature Request
* ToDo

For coding, create :ref:`feature_branches` to work an an issue or a group of issues. 

**Do not keep a feature branch too long (!)** (several weeks max).
Otherwise the deviation from ``devel`` will become too large and :ref:`merge_requests` will be increasingly difficult.

It is good practice to junk up a new feature into its "atomic" parts and to keep a log of your work via informative **commit messages**. 

.. _main_branches:


.. _forking:

Forking 
^^^^^^^

In case you are not a `member <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/project_members>`_ of the Struphy project,
you can contribute code by `forking <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html>`_ the Struphy
repository.

You must create a **public fork** to be able to merge your code into Struphy!

You can create :ref:`feature_branches` in your forked repo and create merge requests into 
the Struphy repo.

`Update your fork <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#from-the-command-line>`_
in case ``devel`` changes in the Struphy repo while you are working on your feature.


.. _feature_branches:

Feature branches 
^^^^^^^^^^^^^^^^

When implementing changes to ``devel`` (can be in your fork) you must do this via a **feature branch** in the following way::

    git checkout devel
    git checkout -b feature

This creates locally on your machine the new feature branch `feature` and checks it out.
Work on your new feature and commit changes in a timely manner::

    git commit -m 'I made this and that.'

You can make as many commits as you like in your feature branch.

From time to time, it is important to check whether the main branch has changed
(make sure your working directory is clean for this, e.g. with ``git stash``)::

    git fetch
    git checkout devel
    git status

See :ref:`here <forking>` what to do in a forked repo.

In case the main branch has changed you must perform either :ref:`rebasing` or :ref:`merging` (see `here <https://www.atlassian.com/de/git/tutorials/merging-vs-rebasing>`_ for a comparison of the two concepts). 
Merging must be done instead of rebasing if

1. Your branch is published (pushed to origin)
2. Your branch is not yet merged with main
3. The main branch has changed since your feature creation or last rebase

**The golden rule of rebasing: only rebase a private branch!** This means you rebase **before** publishing your feature branch to the repository.

When you are done coding the new feature, create a new remote branch and push your changes::

    git push -u origin feature

You can continue working locally on your feature, then use ``git push`` to update the new remote branch.


.. _merge_requests:

Merge requests 
^^^^^^^^^^^^^^

Once you are done working on the new feature, you must create a **merge request** (MR, called pull request on Github). 
There are several ways to do this, one of which is as follows: 

1. In the Struphy repo webpage go to "Merge requests" on the left panel of the page. 
2. Choose ``feature`` as the **source branch** and ``devel`` as the **target branch**.
3. Write an informative summary of the added feature (mention issues solved by this MR)
4. Select one of the Maintainers as **Assignee** for code review.
5. Check both boxes ``delete source branch`` and ``sqash commits``.
6. Click "Create merge request".

Once the merge is accepted your code is merged into ``devel``, 
the remote feature branch gets deleted and the commits are squashed.

In order to mention the merge request in issues/comments, go to its page and copy/paste the link under ``Reference:`` from the right panel.


.. _rebasing:

Rebasing
^^^^^^^^

In a team of multiple developers it often happens that the main branch ``devel`` changes while
you are working on your feature branch. In other words, **main and feature branches have diverged**. 

In this case it is advised to **rebase your feature branch** as follows
(make sure your working directory is clean for this, e.g. with ``git stash``)::

    git checkout devel
    git pull
    git checkout feature
    git rebase devel

This will add your ``feature`` commits on top of the main branch's current state.

**The golden rule of rebasing: only rebase a private branch!** 
This means you rebase **before** publishing your feature branch to the repository.

**Merge conflicts** have to be resolved manually.
``git rebase`` will let you resolve merge conflicts one-by-one for each of your feature branch commits
(which are placed on top of the diverged state of ``devel`` with ``rebase``).

``git status`` will show which files have to be looked at ("both modified").

[Visual Studio Code](https://code.visualstudio.com/) provides a very useful interface for resolving merge conflicts. When opening a file "both modified" you will see something like this:

.. image:: ../pics/vscode_rebase.png

**HEAD (current state)** is the ``devel`` branch (!) in green, and blue is from your feature commit. 
The merge conflict is resolved by clicking either "Accept Current Change" (``devel``) or "Accept Incoming Change" (``feature``).
You have to do this for each conflict in the file (indicated by a blue region in the rightmost scrolling bar), and for each file "both modified".

Save the changes in the files.

``git add`` the modified files.

``git rebase --continue`` will move you forward to the next commit to be added on top of ``devel``. 
If no files have to be changed you can move forward with ``git rebase --skip``.

In case that you made an error during the rebase process you can always go back to your local state with ``git rebase --abort``.


.. _merging:

Merging
^^^^^^^

Merging must be done instead of rebasing if

1. Your branch is published (pushed to origin)
2. Your branch is not yet merged with main
3. The main branch has changed since your feature creation or last rebase

Merging is easy::

    git checkout devel
    git pull
    git checkout feature
    git merge devel

Merging will create a meaningless merge commit in your ``git log``. 


.. _ci:

Continuous integration 
^^^^^^^^^^^^^^^^^^^^^^

`Continuous integration (CI) <https://gitlab.mpcdf.mpg.de/help/ci/index.md>`_ stands for the automatic building and testing of the code.
On gitlab this is done through the ``.gitlab-ci.yml`` file in the repository (see `quickstart ci guide <https://gitlab.mpcdf.mpg.de/help/ci/quick_start/index.md>`_).

In struphy, testing is done with Python's ``pytest`` package. 
All Struphy models (classes in the modules ``models/fluid.py``, ``models/kinetic.py``, ``models/hybrid.py`` and ``models/toy.py``)
are tested automatically (every Monday morning) for all available options and on three different mappings.

New tests have to be added in the folder ``struphy/tests``. 
The files therein have to start with ``test_`` 
and contain ONLY functions that also start with ``test_``. 
In this way they are recognized by ``pytest`` in ``.gitlab-ci.yml``.

Please consult existing tests as templates.


.. _conventions:

Struphy coding conventions
--------------------------

1. Struphy follows the `Python PEP 8 style guide <https://peps.python.org/pep-0008/>`_.
If you use VScode, the format conventions can be automatically applied by choosing "format document"
after right-click on the source.

2. Every class and functions must have a docstring that explains its functionalty.
Struphy uses numpy-style docstrings with "Parameters", "Returns" and "Note" keywords.


.. _change_doc:

Changing the documentation 
--------------------------

Struphy docstrings will automatically appear in the documentation through `Sphinx <https://www.sphinx-doc.org/en/master/>`_.

Moreover, there are source files (``.rst``) for the documentation in ``doc/sections/`` and
Tutorial notebooks in ``doc/tutorials/``. 

In order to build the ``.html`` file of the documentation,
`Pandoc <https://pandoc.org/>`_ needs to be installed on your system (for notebook conversion).
To convert the tutorial notebooks to ``.html``, you need first to generate the data by running::

    struphy test tutorials

If you make changes to docstrings, the ``.rst`` files or the tutorial notebooks, 
you can review them in your browser (e.g. firefox)::

    cd doc
    make html
    firefox _build/html/index.html

When making further changes, just do ``make html`` and refresh the window in your browser.


.. _struphy_model:

Struphy model base class
------------------------

.. autoclass:: struphy.models.base.StruphyModel
    :members:
    :undoc-members:
    :show-inheritance:


.. _feec_base:

FEEC classes
------------

.. _de_rham:

Derham sequence (3D) 
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.feec.psydac_derham.Derham
    :members:
    :undoc-members:
    :show-inheritance:

.. _weighted_mass:

Weighted mass operators
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.feec.mass.WeightedMassOperators
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: struphy.feec.mass.WeightedMassOperator
    :members:
    :undoc-members:
    :show-inheritance:


.. _projectors:

Projections into Derham
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.feec.projectors.CommutingProjector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: struphy.feec.projectors.L2Projector
    :members:
    :undoc-members:
    :show-inheritance:


.. _basis_ops:

Basis projection operators
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.feec.basis_projection_ops.BasisProjectionOperators
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: struphy.feec.basis_projection_ops.BasisProjectionOperator
    :members:
    :undoc-members:
    :show-inheritance:



.. _pic_base:

Particle classes
----------------

.. inheritance-diagram:: struphy.pic.particles
    :parts: 1

.. autoclass:: struphy.pic.base.Particles
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: struphy.pic.particles
    :members:
    :undoc-members:
    :show-inheritance:



.. _data_structures:

Struphy data structures
-----------------------

Check out `Tutorial 08 - Struphy data structures <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_08_data_structures.html>`_
for a hands-on introduction.


FEEC variables
^^^^^^^^^^^^^^

Struphy uses the FEEC data structures provided by the open source package `Psydac <https://github.com/pyccel/psydac>`_ for its
fluid/EM-fields variables. FE coefficients are stores as

* a :class:`StencilVector <psydac.linalg.stencil.StencilVector>` for scalar-valued variables (:code:`H1` or :code:`L2`)
* a :class:`BlockVector <psydac.linalg.block.BlockVector>` for vector-valued variables (:code:`Hcurl`, :code:`Hdiv` or :code:`H1vec`)

A BlockVector is just a 3-list of StencilVectors. 

.. autoclass:: psydac.linalg.stencil.StencilVector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: psydac.linalg.block.BlockVector
    :members:
    :undoc-members:
    :show-inheritance:

Kinetic variables
^^^^^^^^^^^^^^^^^

All information pertaining to markers in Struphy is stored in the :ref:`particle_base`. 
In particular, the data structure holding the values of each marker is under :meth:`struphy.pic.base.Particles.markers`.


.. _utilities:

Utilities
---------

Linear algebra
^^^^^^^^^^^^^^

.. automodule:: struphy.linear_algebra.linalg_kernels
    :members: 
    :undoc-members:

.. automodule:: struphy.linear_algebra.iterative_solvers
    :members: 
    :undoc-members:

.. automodule:: struphy.linear_algebra.schur_solver
    :members: 
    :undoc-members:

Linear operators
^^^^^^^^^^^^^^^^

.. automodule:: struphy.feec.linear_operators
    :members: 
    :undoc-members:

Preconditioners
^^^^^^^^^^^^^^^

.. automodule:: struphy.feec.preconditioner
    :members: 
    :undoc-members:

Stencil data objects
^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.feec.utilities
    :members: 
    :undoc-members:

PIC
^^^

.. automodule:: struphy.pic.utilities
    :members: 
    :undoc-members:


.. _add_model:

Adding a new PDE model
----------------------

Struphy provides an extensive framework for adding new model equations.
A model consists of a set of PDEs that has been discretized within the 
:ref:`GEMPIC <gempic>` framework.

Struphy models must be added under ``src/struphy/models/`` in one of the four modules

* ``fluid.py``  
* ``kinetic.py``
* ``hybrid.py``
* ``toy.py``

as child classes of the :class:`StruphyModel <struphy.models.base.StruphyModel>`. **Please refer to existing models for templates.**
Here is a list of points that need to be followed when creating a new model:

1. Write a docstring 
^^^^^^^^^^^^^^^^^^^^

The docstring should include the model equations (Latex format) and the :ref:`normalization`.
Do not include discretized equations in the model docstring.

**Suggestion:** 

    As a start for adding a new model, copy-and-paste an existing one, change its name 
    and make some small change in the docstring. Then follow :ref:`change_doc` to see if your changes have been taken into
    account. You can already run your "new" model with ``struphy run NEW_NAME``, which will just execute the copied model.  

.. _species:

2. Define :code:`species(cls)`
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


3. Define ``bulk_species(cls)`` and ``velocity_scale(cls)``
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
unit of length specified through the parameter file. We refer to `Tutorial 01 <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_01_units_run_main.html#Struphy-normalization-(units)>`_ for more details
on the Struphy normalization.

.. _add_prop:

4. Add Propagators
^^^^^^^^^^^^^^^^^^

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme of the algorithm. When adding a new model to Struphy, make sure to

1. check the lists of :ref:`available propagators <propagators>` - maybe what you need is already there!
2. :ref:`write your own propagators <add_props>` in case you need to. 

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

Consider for example of the model :class:`VlasovMaxwell <struphy.models.kinetic.VlasovMaxwell>`,
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


5. Add scalar quantities
^^^^^^^^^^^^^^^^^^^^^^^^

This allows to define scalar quantities that should be saved during the simulation.
We use the method :meth:`add_scalar() <struphy.models.base.StruphyModel.add_scalar>` 
of the :class:`StruphyModel <struphy.models.base.StruphyModel>` base class.
Cosider the the example of the model :class:`VlasovMaxwell <struphy.models.kinetic.VlasovMaxwell>`::

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


6. Add options
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


6. Test run
^^^^^^^^^^^

Once you added a model and re-installed struphy (``pip install -e .``), 
you can run the model with::
    
    struphy run YOUR_MODEL
    
where ``YOUR_MODEL`` is the name you gave to 
the model class (it must start with a capital letter).


.. _add_props:

Writing a new propagator
------------------------

Coming soon - for now, please check out :ref:`existing propagators <propagators>` for templates.


.. include:: performance_tests.rst



















