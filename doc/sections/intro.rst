.. _intro:

Introduction
============

Struphy aims to provide easy access to partial differential equations (PDEs) used in plasma physics.
By abstracting away tasks like *array allocation*, *marker initialization* or *mpi communication*, to name a few,
the user can focus on the physics problem instead of dealing with segmentation faults and debugging all the time.

Struphy's Python interface leads to an intuitive access to plasma physics equations. 
The class :class:`struphy.models.base.StruphyModel` enables the implemetation of new models in a streamlined fashion,
with a high level of abstraction. Other features like mapped domains, background distribution functions or particle 
accumulation routines, among many others, can be added in a simliar fashion.

There are three basic ways to use struphy:

    1. as a **regular user** 
    2. as a **developer**
    3. as a **core developer**

The level of abstraction is different for the three. As a **regular user**, you will use struphy as is, 
namely running existing models with different physics parameters or features which are already implemented
and well documented. In this case you should consult the :ref:`userguide`.

As a **developer** your aim is to add new models or physics features to struphy. You will benefit
from the abstraction layers provided by struphy through various base classes and modules that guide the development process.
A developer interacts with the "middle layer" of struphy, without touching the lower level core routines. 
In this case you should consult the :ref:`developers`.

Finally, as a **core developer** you will work on the base layer of struphy. The core routines of the base layer
provide the abstraction for the other two user classes and should be changed only very rarely. Such core
routines are for instance *particle and field initialization*, *mpi process communication* or *linear solvers*.
There is no particular documentation for core developers because the source itself should be sufficiently well documented. 

Regardless of the way YOU want to interact with struphy, we hope you will enjoy the experience and become a contributor!
