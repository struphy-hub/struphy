.. _intro:

Introduction
============

**Struphy provides easy access to partial differential equations (PDEs) in plasma physics.
The package provides** *performance* **(can be used for HPC),** *flexibility* **(various models and features)
and** *usabilty* **(-> Python).**  

Struphy provides different levels of abstraction. A **common user** will run struphy code
for a physics problem and take davantage of the many features which are implemented
and documented. In this case the :ref:`userguide` is the place to go.

A **developer** will aim to add a new model or physics features to struphy. He can benefit
from the abstraction layers provided by struphy; in this way the focus can be put on the
numerical algorithm itself, rather than its technical implementation and optimization.
A developer interacts with the "middle layer" of struphy, without touching the core routines. 
In this case the :ref:`developers` is the place to go.

A **core developer** will work on the base layer of struphy. The core routines of the base layer
provide the above-mentioned abstraction and should be changed only very rarely. Such core
routines are for instance particle- and field initialization, mpi communication or linear solvers.
There is no particular documentation for core developers because the source itself should be sufficiently well documented.

Struphy's Python code is object-oriented and makes use of *class inheritance*. 
The class :class:`struphy.models.base.StruphyModel` for instance enables the implemetation of new models
with a high level of abstraction. The user can choose from a large number of :ref:`propagators`
which can be combined in a straight-forward way to create a new time stepping scheme.
Other features like mapped domains, background distribution functions or particle 
accumulation routines, among many others, can be added in a simliar fashion.

Struphy makes extensive use of the Python packages `pyccel <https://github.com/pyccel/pyccel>`_ 
and `psydac <https://github.com/pyccel/psydac>`_.