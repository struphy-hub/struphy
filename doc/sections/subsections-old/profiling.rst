.. _profiling:

Code profiling
--------------

Struphy runs can profiled with the Python profiler `cProfile <https://docs.python.org/3/library/profile.html>`_
by adding the flag ``--cprofile`` to your run command.
In order to see profiling results type::

    struphy profile [OPTIONS] sim_1 [sim_2 ...]

Here, ``sim_1``, ``sim2`` etc. are relative to the current output path. If more than one simulation is profiled, 
they all have to be from the same ``MODEL``. To get more info on possible ``OPTIONS`` type::

    struphy profile -h