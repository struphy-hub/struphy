.. _developers:

Developer's guide
=================


.. _repo:

Git repository
--------------

The Struphy repo is on `gitlab.mpcdf.mpg.de <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_.

You need a developer to grant access.


.. _adding_code:

Adding code
-----------

.. _workflow:

Workflow 
^^^^^^^^

When adding code to Struphy it is important that other developers can follow what you are planning/doing.
For this we use the ``Issue`` tracker on the left panel of the `repo webpage <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_.

If you plan to add a small feature (finished in a couple of days):

1. Create an ``Issue`` by clicking on the blue button ``New issue`` in the top-right corner.
2. Insert informative title and description. 
3. Assign to yourself.
4. Select a corresponding Milestone if the issue fits.
5. [Optional: select a due date.]
6. Click ``Create Issue``.

If you have a bigger code change in mind that might take several days or weeks::

1. Click ``New milestone`` in ``Issues/Milestone`` from the left panel.
2. Insert informative title and description (point by point).
3. [Optional: select a due date.]
4. Click ``Create milestone``.
5. Add issues that correspond to your milestone (see above).

Regularly comment on isuues that you are working on. Close an issue by mentioning the corresponding :ref:`merge_request`.

.. _main_branches:

Main branches 
^^^^^^^^^^^^^

There are two main branches, ``master`` and ``devel``. Nobody can push directly to these branches. 
The master branch holds the current release of the code. 
``devel`` is the main branch for developers. The code in ``devel`` can be modified via :ref:`feature_branches`.

.. _feature_branches:

Feature branches 
^^^^^^^^^^^^^^^^

When implementing changes to ``devel`` you must do this via a **feature branch** in the following way::

    git checkout devel
    git checkout -b <name_of_feature>

This creates locally on your machine the new feature branch `<name_of_feature>` and checks it out.
Work on your new feature and commit changes in a timely manner::

    git commit -m 'I made this and that.'

You can make as many commits as you like in your feature branch.

From time to time, it is important to check whether the main branch has changed
(make sure your working directory is clean for this, e.g. with ``git stash``)::

    git fetch
    git checkout devel
    git status

In case the main branch has changed you must perform :ref:`rebasing`. 

**The golden rule of rebasing: only rebase a private branch!** This means you rebase **before** publishing your feature branch to the repository.

When you are done coding the new feature, create a new remote branch and push your changes::

    git push -u origin <name_of_feature>

You can continue working locally on your feature, then use ``git push`` to update the new remote branch.

.. _merge_request:

Merge requests 
^^^^^^^^^^^^^^

Once you are done working on the new feature, you must create a **merge request** (called pull request on Github). 
There are several ways to do this, one of which is as follows: 

    1. On the gitlab webpage go to "Merge requests" on the left panel of the page. 
    2. Choose ``<name_of_feature>`` as the **source branch** and ``devel`` as the **target branch**.
    3. Select *Stefan Possanner* as **Assignee** and *Florian Holderied* as **Reviewer**.
    4. Check both boxes ``delete source branch`` and ``sqash commits``.
    5. Click "Create merge request".

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
    git checkout <name_of_feature>
    git rebase devel

This will add your ``<name_of_feature>`` commits on top of the main branch's current state.

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


.. _change_doc:

Changing the documentation 
--------------------------

The source files (``.rst``) for the documentation are in ``/doc/sections`` in the repository. 
If you make changes to these files, you can review them in your browser (e.g. firefox)::

    cd doc
    make html
    firefox _build/html/index.html

When making firther changes, just do ``make html`` and refresh the window in your browser.

**Please keep the documentation up-to-date with your added features.**


.. _add_model:

Adding new models 
-----------------

.. _model_requirements:

Requirements
^^^^^^^^^^^^

Struphy can accomodate only models that satisfy the following criteria:

* 3d in space
* Initial-boundary value problem or eigenvalue problem
* Smooth solutions
* Field variables depending on :math:`(\mathbf x,t)` are discretized within the :ref:`3d_derham_complex`
* (Gyro-)kinetic variables depending in :math:`(\mathbf x, \mathbf v,t)` are discretzed with a particle method
* Single patch mapping (see :ref:`add_mapping`)

Moreover, the time stepping scheme must be composed of several well-defined **propagators**.
Assuming our model features the unknowns :math:`vars(t)`, the overall propagator
can be denoted as :math:`\Phi_{\Delta t}[vars(t)] = vars(t + \Delta t)`.
Struphy can handle models where :math:`\Phi_{\Delta t}` is decomposed (or split):  

.. math::
    \Phi_{\Delta t}[vars(t)] = \Phi^1_{\Delta t}[\textnormal{subset1}(vars(t))] \circ \Phi^2_{\Delta t}[\textnormal{subset2}(vars(t))] \circ ...

with substeps :math:`\Phi^1_{\Delta t}`, :math:`\Phi^2_{\Delta t}`, etc. that update a subset of (or all) :math:`vars(t)`. 
More refined splitting schemes than Lie-Trotter are available in Struphy (see ``struphy/models/codes/exec.py`` lines 162-182):

.. literalinclude:: ../../struphy/models/codes/exec.py
    :language: python
    :linenos: 
    :lineno-start: 162
    :lines: 162-182

.. _base_classes:

Base classes
^^^^^^^^^^^^

Struphy models are built upon two base classes:

.. autoclass:: struphy.models.codes.models.StruphyModel
   :members: 
   :undoc-members:

.. autoclass:: struphy.models.codes.propagators.Propagator
   :members: 
   :undoc-members:

| All Struphy models are subclasses of ``StruphyModel`` and should be added to ``struphy/models/code/models.py``. 
| ``StruphyModel`` demands a list of propagators in the sense explained in :ref:`model_requirements`. 
| All Struphy propagators are subclasses of ``Propagator`` and should be added to ``struphy/models/code/propagators.py`` 


.. _maxwell_example:

Example: Maxwell equations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us look at the example of the model ``maxwell`` (:ref:`maxwell`). 
The model has two field variables (FE coefficients) :math:`\mathbf e \in \mathbb R^{N_1}` and :math:`\mathbf b \in \mathbb R^{N_2}` that 
are updated with a single propagator derived from a Cranck-Nicolson discretization:

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

The corresponding ``StruphyModel`` is

.. autoclass:: struphy.models.codes.models.Maxwell
    :members: a
    :undoc-members:

.. literalinclude:: ../../struphy/models/codes/models.py
    :language: python
    :linenos: 
    :lineno-start: 139
    :lines: 139-182

Let us go through the source code one-by-one.
The ``__init__`` function is called from the base class via ``super()``:

.. literalinclude:: ../../struphy/models/codes/models.py
    :language: python
    :linenos: 
    :lineno-start: 159
    :lines: 159

The field variables of the model are specified as in ``e_field='Hcurl'``, where the keyword ``e_field`` 
is the name of the variable used for saving and the value ``'Hcurl'`` is the space of the variable 
(can also be ``'H1'``, ``'Hdiv'`` or ``'L2'``).

Mass matrices have to be assembled model-specific. In this case we only need :math:`\mathbb M_1` and :math:`\mathbb M_2`:

.. literalinclude:: ../../struphy/models/codes/models.py
    :language: python
    :linenos: 
    :lineno-start: 162
    :lines: 162-163

The actual ``Stencil-/BlockVectors`` holding the FE coefficients have been created by ``super().__init__``
and can be retrieved from the ``fields`` property:

.. literalinclude:: ../../struphy/models/codes/models.py
    :language: python
    :linenos: 
    :lineno-start: 166
    :lines: 166-167

The only ``@abstractmethod`` to be implemented is ``update_scalar_quantities``:

.. literalinclude:: ../../struphy/models/codes/models.py
    :language: python
    :linenos: 
    :lineno-start: 178
    :lines: 178-182

It remains to fill the lists ``self._propagators`` and ``self._scalar_quantities``:

.. literalinclude:: ../../struphy/models/codes/models.py
    :language: python
    :linenos: 
    :lineno-start: 170
    :lines: 170-176

The first list holds all propagators of the model, which amounts to just one in this example:

.. autoclass:: struphy.models.codes.propagators.StepMaxwell
    :members: 
    :undoc-members:

.. literalinclude:: ../../struphy/models/codes/propagators.py
    :language: python
    :linenos: 
    :lineno-start: 84
    :lines: 84-152

Let us go through the propagator's source one-by-one. The instantiation is model-specific (not done by the base class):

.. literalinclude:: ../../struphy/models/codes/propagators.py
    :language: python
    :linenos: 
    :lineno-start: 106
    :lines: 106-131

Model-specific objects needed in this propagator are
a :ref:`preconditioner`, one 2x2 block matrix and a :ref:`schur_solver`.

The definition of the abstract property ``variables`` is a must in each propagator:

.. literalinclude:: ../../struphy/models/codes/propagators.py
    :language: python
    :linenos: 
    :lineno-start: 133
    :lines: 133-135

It returns a list of the variables updates by the propagator.
Finally, also the abstract method ``push`` must be defined in each propagator:

.. literalinclude:: ../../struphy/models/codes/propagators.py
    :language: python
    :linenos: 
    :lineno-start: 137
    :lines: 137-152

It ony takes the time step ``dt`` as an argument and updates the variabels specified in ``self.variables``.
Note that the function ``self.in_place_update`` can be used to perform the in-place update of the variables:

.. literalinclude:: ../../struphy/models/codes/propagators.py
    :language: python
    :linenos: 
    :lineno-start: 42
    :lines: 42-81


.. _to_do_list_model:

TODO for adding a model
^^^^^^^^^^^^^^^^^^^^^^^

The following steps must be taken to add a new model to Struphy:

1. Write a new model class in ``struphy/models/codes/models.py`` based on ``StruphyModel`` (add the name to ``__all__``)
2. Possibly add new propagators in ``struphy/models/codes/propagators.py`` based on ``Propagator`` (add the name to ``__all__``). The name must start with ``Step``.
3. Change the default input file ``struphy/io/inp/parameters.yml`` if needed.
4. Add a new if-clause for your model in ``exec.py`` after line 81::

    if code=='maxwell':
        MODEL = models.Maxwell(DR, DOMAIN, params['solvers']['pcg_1'])
    else:
        raise NotImplementedError(f'Model {code} not implemented.')
        exit()

5. Test your model and optionally add an ``example_`` bash script in ``scripts/`` and/or ``struphy/examples``.
6. Add a description of your model in

    a. the help function of ``scripts/struphy``
    b. the doc under ``doc/section/userguide.rst`` in the table of section :ref:`running_codes`
    c. the doc under ``doc/section/models.rst``


.. _add_dispersion:

Adding new dispersion relations 
-------------------------------


.. _add_mapping:

Adding new mappings 
--------------------------------

.. include:: mappings.rst


.. _add_equil:

Adding new equilibria 
---------------------


.. _add_accum:

Adding new PIC accumulation routines 
------------------------------------






