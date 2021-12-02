.. _developers:

Developer's guide
=================


.. _repo:

GitLab repository
-----------------

.. _main_branches:

Main branches 
^^^^^^^^^^^^^

There are two main branches, ``master`` and ``devel``. No one can push directly to these branches. The master branch holds the current pre-release of the code. 
``devel`` is the main branch for developers. Merge requests must always go into ``devel``.

.. _feature_branches:

Feature branches 
^^^^^^^^^^^^^^^^

When implementing changes to ``devel`` you should do this via a **feature branch** in the following way::

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
Once you are done working on the new feature, you must create a **merge request** (called pull requeston Github) 
such that your changes can be incorporated into ``devel``. There are several ways to do this, one of which is as follows: 

    1. On the gitlab webpage go to "Merge requests" on the left side of the page. 
    2. Choose ``<name_of_feature>`` as the **source branch** and ``devel`` as the **target branch**.
    3. Select *Stefan Possanner* as **Assignee** and *Florian Holderied* as **Reviewer**.
    4. Click "Create merge request" and you are done.

Once the merge is accepted your code is merged into ``devel``, 
the remote feature branch gets deleted and the commits are squashed.


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


.. _coding_style:

Coding style
------------

- **Always stay close to the main branch (``devel`` for developers).** Make a feature branch when you need to add functionalty. Never work more than several days on a feature branch before merging.

- Make regular commits when working on a feature, such that your work can be recapitulated easily.

- Choose self-explaining module and variable names (Classes start with a capital letter).

- **Always include a docstring** (`numpy style via Napolean extension <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#module-sphinx.ext.napoleon>`_): 
    1. one or two lines explaining what the function does.
    2. give "Parameters" (input vars) and "Returns" (output vars), and maybe "Attributes" (for classes) and/or "Notes".
    3. See :ref:`linear_operators` for an example of how to write docstrings.

- Include ``test_`` files (= unit tests) in the folder ``struphy/tests``. It is advisable that new features come with a test file, where users can understand the functionality and CI can automate testing.


.. _add_model:

Adding a new model 
------------------

In :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` it is pretty easy to add new model equations, 
because all building blocks for discrete differential forms (FEEC, de Rham diagram), particle-in-cell (PIC)
discretization and mapped domains are provided by the package.

First you have to create a **feature branch** (see :ref:`above <feature_branches>`).
In order to add a new model it is best to follow an existing code as a template; the main files are located in::

    struphy/models/codes

You have to create a new file for your code in this folder (using for instance ``cc_lin_mhd_6d`` as template)::

    cp cc_lin_mhd_6d.py <my_new_model>.py

Now you can adapt the file ``<my_new_model>.py`` to implement the new model. 
Check out :ref:`objects` to get an overview of the code structure.
In order to run your code you need to perform two more steps:

1. Add a new if-clause in ``struphy/models/codes/exec.py``::

    elif code=='<my_new_model>':
        from struphy.models.codes import <my_new_model> 
        <my_new_model>.execute(file_in, path_out, mode=='a')

2. Add an input folder and a parameter file::

    cd struphy/io/inp
    mkdir <my_new_model>/
    cp cc_lin_mhd_6d/parameters.yml <my_new_model>/parameters.yml

3. Add the parameter file to the ``package_data`` dictionary in ``setup.py``, under the key ``'struphy.io.inp'``.

Adapt the file ``parameters.yml`` to your model, see :ref:`params_file`.
Moreover, there are certain folders where new modules (``.py``-files) should be put:

    * :ref:`linear_operators` go in ``struphy/linear_operators/<code_name>``
    * PIC accumulation/deposition routines go in ``struphy/pic/<code_name>``
    * MHD equilibria go in ``struphy/mhd_equil/analytical``
    * Kinetic equilibria go in ``struphy/kinetic_equil/analytical``
    * dispersion relation solvers go in ``struphy/models/dispersion_relations``
    * diagnostic files go in ``struphy/diagnostics``

Last but not least, it is important to add a minimal documentation for your model.
For this, add a line in ``bin/struphy.sh`` in the help ``h)``, namely under the line::

    echo "   -r <code>          Specify code to run. Available codes are:"

Moreover, please add your model and code name to the section :ref:`models`. 
Finally, you could add a line to the table in :ref:`intro` of the :ref:`userguide`.


.. _objects:

STRUPHY objects
---------------

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` is an object-oriented code, which means it relies heavily on *Python classes*. 
From the first paragraph of the `official Python documentation <https://docs.python.org/3/tutorial/classes.html>`_:

    Classes provide a means of bundling data and functionality together. 
    Creating a new class creates a new type of object, allowing new instances of that type to be made. 
    Each class instance can have attributes attached to it for maintaining its state. 
    Class instances can also have methods (defined by its class) for modifying its state.

Classes are very efficient for interchanging information. 
Instead of passing many variables and/or functions one-by-one to a function, 
these are first grouped together in an instance of a class (the "object"), and then the object is passed. 
Moreover, each instance of a class can be different, depending on the parameters passed to the class at initialization.
For example, in :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` there is an object called ``EQ_MHD_P`` 
which holds all information about the MHD equilibirum in Cartesian space. 
This information consists of Physics parameters but also of callable functions such as the equilibirum pressure and the magnetic field.

Each code in :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` is based on more or less the same set of Classes, 
which are listed in the following table:

============================ ============================================ ===================== ========================================== =========
Class name                   Location in ``struphy.``                     Instance name in code Description                        
============================ ============================================ ===================== ========================================== =========
Domain                       ``geometry.domain_3d``                       DOMAIN                metric coefficients, push, pull, transform :ref:`more info <domain_class>`            
Tensor_spline_space          ``feec.spline_space``                        SPACES                discrete De Rham sequence and projectors   :ref:`more info <derham>`
Equilibrium_mhd_physical     ``mhd_equil.mhd_equil_physical``             EQ_MHD_P              MHD eqilibrium functions in physical space :ref:`more info <mhd_equil_p>`
Equilibrium_mhd_logical      ``mhd_equil.mhd_equil_logical``              EQ_MHD_L              MHD eqilibrium functions in logical space  :ref:`more info <mhd_equil_l>`
Initialize_mhd               ``mhd_init.mhd_init``                        MHD                   MHD variables                              :ref:`more info <mhd_class>` 
MHD_operators                ``feec.mhd_operators.linear``                MHD_OPS               MHD projection operators                   :ref:`more info <linear_operators>` 
Equilibrium_kinetic_physical ``kinetic_equil.kinetic_equil_physical``     EQ_KINETIC_P          kinetic equilibirum in physical space      :ref:`more info <kinetic_equil_p>` 
Equilibrium_kinetic_logical  ``kinetic_equil.kinetic_equil_logical``      Q_KINETIC_L           kinetic equilibirum in logical space       :ref:`more info <kinetic_equil_l>` 
Initialize_markers           ``kinetic_init.kinetic_init``                KIN                   marker information                         :ref:`more info <markers_class>`  
Accumulation                 ``pic.<code_name>.accumulation``             ACCUM                 pic accumulation/deposition routines       :ref:`more info <accum>` 
Linear_mhd                   ``struphy.models.substeps.push_linear_mhd``  UPDATE_MHD            MHD propagators (split steps)              :ref:`more info <push_mhd>` 
Push                         ``struphy.models.substeps.push_markers``     UPDATE_MARKERS        marker propagators (split steps)           :ref:`more info <push_markers>`
<code_dependent>             ``struphy.models.substeps.push_<code_name>`` UPDATE_COUPL          coupling terms propagators (split steps)   :ref:`more info <push_coupl>` 
============================ ============================================ ===================== ========================================== =========

The dependencies between these objects is depicted in the Figure below.
We see for example that ``DOMAIN`` has no dependencies and is therefore the lowest level.
``SPACES`` depends on ``DOMAIN`` because of polar splines bases, which rely on the iso-geometroc approach (IGA). 
Moreover, ``UPDATE_COUPL`` depends on five lower-level objects.

.. image:: ../pics/obj_network.png


.. include:: mappings.rst
.. include:: derham.rst
.. include:: mhd_equil.rst
.. include:: mhd.rst
.. include:: linear_ops.rst
.. include:: kinetic_equil.rst
.. include:: markers.rst
.. include:: accum.rst
.. include:: update.rst
.. include:: data.rst

