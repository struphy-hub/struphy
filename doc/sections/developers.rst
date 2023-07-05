.. _developers:

Developer's guide
=================

Struphy is **open source** and **open development**, and thus thrives on the contributions of its users.

In order to maintain a sustainable software framework, :ref:`conventions` have to be followed
to achieve successful pull requests. 

Suggestions for a standard :ref:`git_workflow` are also available.

A comprehensive documentation is vital for the usability of Struphy; any developer should therefore 
check out the hints for :ref:`change_doc`.

Developers can contribute to Struphy in multiple ways.
Single physics features or algorithmic novelties can be added 
via the available :ref:`base_classes`. For FEEC discretizations,
the most relevant classes are

    * :ref:`weighted_mass`
    * :ref:`basis_ops`

For PIC, the most relevant classes are

    * :ref:`pusher`
    * :ref:`Accumulator <accumulator>`

Useful models for linear algebra, preconditioners, stencil data objects
and PIC routines can be found under :ref:`utilities`.

In case you want to add a new solver for a PDE or a system of PDEs, 
please follow :ref:`add_model`.


.. _conventions:

Struphy coding conventions
--------------------------

Struphy follows the `Python PEP 8 style guide <https://peps.python.org/pep-0008/>`_.
If you use VScode, the format conventions can be automatically applied by choosing "format document"
after right-click on the source.

Every class and functions must have a docstring that explains its functionalty.
Struphy uses numpy-style docstrings with "Parameters", "Returns" and "Note" keywords.


.. _base_classes:

Struphy base classes
--------------------

Derham sequence (3D) 
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.psydac_api.psydac_derham.Derham
    :members:

FE field 
^^^^^^^^

.. autoclass:: struphy.psydac_api.fields.Field
    :members:

.. _weighted_mass:

Weighted mass operators
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.psydac_api.mass.WeightedMassOperators
    :members:

.. _basis_ops:

Basis projection operators
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.psydac_api.basis_projection_ops.BasisProjectionOperators
    :members:

.. _particles:

Particles
^^^^^^^^^

.. autoclass:: struphy.pic.particles.Particles
    :members:

.. _pusher:

Particle pusher
^^^^^^^^^^^^^^^

.. autoclass:: struphy.pic.pusher.Pusher
    :members:


.. _utilities:

Utilities
---------

Linear algebra
^^^^^^^^^^^^^^

.. automodule:: struphy.linear_algebra.core
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

.. automodule:: struphy.psydac_api.linear_operators
    :members: 
    :undoc-members:

Preconditioners
^^^^^^^^^^^^^^^

.. automodule:: struphy.psydac_api.preconditioner
    :members: 
    :undoc-members:

Stencil data objects
^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.psydac_api.utilities
    :members: 
    :undoc-members:

PIC
^^^

.. automodule:: struphy.pic.utilities
    :members: 
    :undoc-members:


.. _add_model:

Adding a new Struphy model
--------------------------

Struphy provides an extensive framework for adding new model equations.
A model consists of a set of PDEs togehter with a chosen discretization scheme.

Struphy models must be added under ``src/struphy/models/`` in one of the four modules

* ``fluid.py``  
* ``kinetic.py``
* ``hybrid.py``
* ``toy.py``

as child classes of the :class:`struphy.models.base.StruphyModel`. **Please refer to existing models for templates.**

A Struphy model s defined by 

    a. unknowns: *field* (FEEC), *fluid* (FEEC), or *kinetic* (PIC) variables
    b. :ref:`propagators` to update the unknowns 
    c. scalar quantities depending on the uknowns tracked during the simulation (e.g. total energy)
 
These three categories must be provided to :class:`struphy.models.base.StruphyModel`.  
A typical initialization of the unknowns looks like::

    super().__init__(params, comm, b2='Hdiv', mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'}, energetic_ions='Particles5D')

This example is taken from :class:`struphy.models.hybrid.LinearMHDDriftkineticCC`. Here,
one field variable (``b2``), one fluid species (``mhd``) and one kinetic species (``energetic_ions``)
are initialized.

Regarding the propagators, it is important to expose the base class :class:`struphy.propagators.base.Propagator`
to the current instances of the de Rham complex, domain, weighted mass operators and (if needed)
basis projection operators, BEFORE instantiating the model-specific propagators::

    Propagator.derham = self.derham
    Propagator.domain = self.domain
    Propagator.mass_ops = self.mass_ops
    Propagator.basis_ops = BasisProjectionOperators(
        self.derham, self.domain, eq_mhd=self.mhd_equil)

**Please refer to** :ref:`propagators` **for propagator templates.**

Once you added a model and re-installed struphy (``pip install -e .``), 
you can run the model with ``struphy run YOUR_MODEL``, where ``YOUR_MODEL`` is the name you gave to 
the model class (it must start with a capital letter).

.. autoclass:: struphy.models.base.StruphyModel
    :members:
    :undoc-members:

.. autoclass:: struphy.propagators.base.Propagator
    :members:
    :undoc-members:


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
Tests have to be added in the folder ``struphy/tests`` or ``struphy/tests_mpi`` (for parallel testing). 
The files therein have to start with ``test_`` 
and contain ONLY functions that also start with ``test_``. 
In this way they are recognized by ``pytest`` in ``.gitlab-ci.yml``:

Please consult existing tests as templates.


.. _change_doc:

Changing the documentation 
--------------------------

The source files (``.rst``) for the documentation are in ``/doc/sections`` in the repository. 
If you make changes to these files, you can review them in your browser (e.g. firefox)::

    cd doc
    make html
    firefox _build/html/index.html

When making further changes, just do ``make html`` and refresh the window in your browser.
















