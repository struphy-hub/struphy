.. _developers:

Developer's guide
=================


.. _intro_devel:

Introduction
------------



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

- **Use the ``devel`` version when running simulations for your thesis/paper/physics work.** We want your features in struphy for goor maintenance, documentation, discussion and future improvements. 

- Choose self-explaining module and variable names (Classes start with a capital letter).

- Always include a docstring (numpy style): 
    1. one or two lines explaining what the function does.
    2. give "Parameters" (input vars) and "Returns" (output vars).

- Include ``test_`` files (= unit tests) in the folder ``struphy/tests``. It is advisable that new feature come with a test file, where users can understand the functionality and CI can automate testing.


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

Adapt the file ``parameters.yml`` to your model.

Last but not least, it is important to add a minimal documentation for your model.
For this, add a line in ``bin/struphy.sh`` in the help ``h)``, namely under the line::

    echo "   -r <code>          Specify code to run. Available codes are:"

Moreover, please add your model and code name to the section :ref:`models`. 
Finally, you could add a line to the table in :ref:`intro` of the :ref:`userguide`.

