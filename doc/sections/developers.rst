.. _developers:

Developer's guide
=================

.. _repo:

Git repository
--------------

The Struphy repository is on `gitlab.mpcdf.mpg.de <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_.

For access please contact `stefan.possanner@ipp.mpg.de <stefan.possanner@ipp.mpg.de>`_.


.. _adding_code:

Shared development
------------------

.. _workflow:

Workflow 
^^^^^^^^

When adding code to Struphy it is important that other developers can follow what you are planning/doing.
For this we use the ``Issue`` tracker on the left panel of the `repo webpage <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_.

Create a :ref:`feature branch <feature_branches>` to work an an issue or a group of issues. 
**Delete** the feature branch after the :ref:`merge_requests`.
**Do not keep a feature branch too long (!)**, usually a couple of days or 1-2 weeks max.
Otherwise the code review will be too cumbersome and time consuming.
It is good practice to junk up a new feature into its "atomic" parts and to keep a log of the work via informative **commit messages**. 

When closing an issue you can mention it via its link (bottom of the right side panel of the issue's page) in the corresponding merge request.
This helps to keep a better overview for other developers.

.. _main_branches:

Main branches 
^^^^^^^^^^^^^

There are two main branches, ``master`` and ``devel``. Nobody can push directly to these branches. 
The master branch holds the current release of the code. 
``devel`` is the main branch for developers. The code in ``devel`` can be modified via :ref:`feature_branches`.

.. _feature_branches:

Feature branche 
^^^^^^^^^^^^^^^

When implementing changes to ``devel`` you must do this via a **feature branch** in the following way::

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

    1. On the gitlab webpage go to "Merge requests" on the left panel of the page. 
    2. Choose ``feature`` as the **source branch** and ``devel`` as the **target branch**.
    3. Write an informative summary of the added feature (mention issues solved by this MR)
    4. Select either *Stefan Possanner* or *Florian Holderied* as **Assignee** for code review.
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

In Struphy, testing is done with Python's ``pytest`` package. 
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


.. _how_to_add:

How to add ... 
--------------

.. _add_model:

A new model
^^^^^^^^^^^

    1. Check if your model satisfies the follwing criteria:

        a. 3d in space
        b. Initial-boundary value problem or eigenvalue problem
        c. Smooth solutions
        d. Geometry can be described with a single patch mapping

    2. Write down the numerical scheme for your model:

        a. Field variables depending on :math:`(\mathbf x,t)` are discretized within the :ref:`3d_derham_complex`
        b. (Gyro-)kinetic variables depending in :math:`(\mathbf x, \mathbf v,t)` are discretized with a particle method
        c. Time discretization: splitting schemes (Lie-Trotter, Strang) composed of multiple **propagators** allowed

    3. Implement the **propagators** of your model.

        First, check the list of existing :ref:`propagators` (you might be lucky and save some work). 
        If necessary, add propagators to ``struphy/models/propagators.py``. The name must start with ``Step`` and 
        the :ref:`prop_base_class` must be inherited.
        Use existing classes as templates. 
        When adding propagators, you should benefit from STRUPHY's :ref:`toolkit` that provides many useful modules and templates.

    4. Add your model to ``struphy/models/models.py``. 
    
        Use existing classes as templates.
        Your model's class must inherit the :ref:`model_base_class` and define the abstract properties 
        ``propagators``, ``scalar_quantities`` as well as the abstract method  ``update_scalar_quantities``.

    5. Re-install Struphy, compile and run your model via the command line: ``struphy run <your_model_class_name>``.
    
        The name of your model is the same as the name you give to the class (it must start with a capital letter).

Let us look at the example of the model ``Maxwell`` (:ref:`models`). 
The model has two field variables (FE coefficients) :math:`\mathbf e \in \mathbb R^{N_1}` and :math:`\mathbf b \in \mathbb R^{N_2}` that 
are updated with a single propagator derived from a Crank-Nicolson discretization:

    .. math::
        \begin{bmatrix}
        \mathbf e^{n+1} - \mathbf e^n \\[2mm] \mathbf b^{n+1} - \mathbf b^n
       \end{bmatrix} = 
       \frac{\Delta t}{2} 
       \begin{bmatrix}
        0 & \mathbb{M}_1^{-1}\mathbb{C}^\top 
        \\[2mm] 
        - \mathbb{C}\mathbb{M}_1^{-1}  & 0
       \end{bmatrix}
       \begin{bmatrix}
        \mathbb{M}_1 (\mathbf e^{n+1} + \mathbf e^n) \\[2mm] \mathbb M^2 (\mathbf b^{n+1} + \mathbf b^n)
       \end{bmatrix} \,.

.. literalinclude:: ../../struphy/models/models.py
    :language: python
    :linenos: 
    :lineno-start: 6
    :lines: 6-77

First, we note that the docstring contains the model equations 
and their normalization (in latex format). This is necessary for the correct documentation of the model.

The class ``Maxwell`` inherits all members of the base class ``StruphyModel``. 
As we can see, the abstract properties  ``propagators``, ``scalar_quantities`` as well as the abstract method  
``update_scalar_quantities`` have been implemented.
Otherwise, an error message would occur.

The ``__init__`` function of the base class is called via ``super().__init__``.
The field variables of the model are specified as in ``e_field='Hcurl'``, where the keyword ``e_field`` 
is the name of the variable used for saving and the value ``'Hcurl'`` is the space of the variable 
(can also be ``'H1'``, ``'Hdiv'``, ``'L2'`` or ``'H1vec'``).

Mass matrices have to be assembled model-specific. In this case we only need :math:`\mathbb M_1` and :math:`\mathbb M_2`:
The actual Stencil-/BlockVectors holding the FE coefficients have been created by ``super().__init__``
and can be retrieved from the ``fields`` property defined in the base class.

The lists ``self._propagators`` and ``self._scalar_quantities`` returned by the abstract properties
have to be filled with model specific propagator objects 
(just one in the case of ``Maxwell``) and scalar quantities of interest.

The one propagator used in the model ``Maxwell`` is listed in :ref:`propagators` and called ``StepMaxwell``:

.. literalinclude:: ../../struphy/propagators/propagators.py
    :language: python
    :linenos: 
    :lineno-start: 13
    :lines: 13-88

First, we note that the docstring contains the 
propagator equations (in latex format). This is necessary for the correct documentation of the propagator.

The ``__init__`` call is model-specific (not done by the base class).
The present propagator needs a :ref:`preconditioner`, one 2x2 block matrix and a :ref:`schur_solver`

The definition of the abstract property ``variables`` is a must in each propagator.
It returns a list of the variables updated by the propagator.

Finally, also the abstract method ``__call__`` must be defined in each propagator.
It only takes the time step ``dt`` as argument and updates the variables specified in ``self.variables``.
Note that the function ``self.in_place_update``  from the base class ``Propagators`` can be used 
to perform the in-place update of the variables.


.. _add_mapping:

A new mapping 
^^^^^^^^^^^^^

Implemented mappings are listed in :ref:`avail_mappings`. 

The addition of a new mapping starts at the domain class:

.. autoclass:: struphy.geometry.domain_3d.Domain

A new mapping must have a **string name** and an **integer identifier**.
These must be added to the domain class. For example, the mapping 
``cuboid`` appears as follows:

.. literalinclude:: ../../struphy/geometry/domain_3d.py
    :language: python
    :linenos: 
    :lineno-start: 46
    :lines: 46-60

The string name is checked in the if-clause and the integer id is assigned to ``self._kind_map``. 
Add your mappings according to this example.

Moreover, the mapping and its Jacobian have to added to the functions ``struphy.geometry.map_eval.f``
and ``struphy.geometry.map_eval.df``, respectively. Finally, the actual implementation of 
the mapping and its Jacobian is done in ``struphy.geometry.mappings_3d_bis``. 


.. _add_equil:

A new background
^^^^^^^^^^^^^^^^

Implemented backgrounds listed in :ref:`backgrounds`. 


.. _add_dispersion:

A new dispersion relation 
^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented dispersion relations that inherit the base class are listed in :ref:`dispersions`. 

.. autoclass:: struphy.dispersion_relations.base.DispersionRelations1D

As an example, consider the dispersion relation for light waves in vacuum:

.. literalinclude:: ../../struphy/dispersion_relations/analytic.py
    :language: python
    :linenos: 
    :lineno-start: 7
    :lines: 7-35  

The ``__init__`` is done via the base class by calling ``super().__init__``. 
One has to provide a name for each branch of the spectrum (here just the ``light wave``) and all parameters necessary for
computing the dispersion relation (here just the speed of light ``c``, which is 1 in Struphy normalization, see :ref:`models`).

The abstract method ``spectrum`` must be defined; the only really model-specific part is the definition 
of the computation of each branch. Here, there is just one branch, namely the light wave in vacuum which propagates at the speed ``c``.


.. _add_pusher:

A new particle pusher
^^^^^^^^^^^^^^^^^^^^^


.. _add_accum:

A new PIC accumulation routine 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented accumulation functions are listed in :ref:`accumulators`. 

.. autoclass:: struphy.pic.particles_to_grid.Accumulator

.. autofunction:: struphy.pic.accum_kernels._docstring


.. _add_weighted_mass:

A new weighted mass matrix 
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.psydac_api.mass_psydac
    :members:


.. _add_mhd_ops:

A new MHD operator 
^^^^^^^^^^^^^^^^^^

Implemented MHD operators are listed in :ref:`mhd_ops`.











