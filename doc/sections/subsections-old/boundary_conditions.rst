.. _boundary_conditions:

Boundary conditions
-------------------

Boundary conditions in Struphy must be set via the ``.yml`` parameter file.
Both PIC (kinetic) and FEEC boundary conditions are set in this way.


.. _kinetic_bcs:

Kinetic boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The behavior of particles
at the boundary can be set via :ref:`kinetic` under the key ``markers/bc``.
The latter takes a list of three string entries (referring to the three spatial directions);
possible string values are:

* ``periodic``: particles re-enter the domain on the other side
* ``reflect``: particles get reflected with respect to the surface normal
* ``remove``: particles get removed from the simulation upon hitting the boundary

More kinetic boundary conditions will be added in the future.


.. _feec_bcs:

FEEC boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^

Periodic or clamped Splines
"""""""""""""""""""""""""""

In :ref:`grid`, the boundary conditions for FEEC variables are steered via the 
following two keys::

    spl_kind     : [False, True, True] # spline type: True=periodic, False=clamped
    dirichlet_bc : [[False, False], [False, False], [False, False]] # hom. Dirichlet boundary conditions for N-splines (spl_kind must be False)

The first key ``spl_kind`` takes a list of three Boolean entries, 
referring to the three spatial directions; ``True`` means that a periodic spline basis
is chosen in the respective direction, whereas ``False`` signifies the choice 
of a "clamped" basis (see :ref:`uni_variate_spaces`).

In case of a "clamped" basis in $i$-direction (``spl_kind[i] = False``), 
the second key ``dirichlet_bc`` becomes important. Again, it takes a list
of length three referring to the three spatial directions; the entries are
themselves lists containing two Boolean values, where the first entry refers 
to the left boundary (:math:`\eta_i = 0.0`) and the 
second entry refers to the right (:math:`\eta_i = 1.0`) boundary.
A value ``dirichlet_bc[i][0] = True`` means that homogeneous Dirichlet conditions
are applied at the left boundary of the $i$-direction.
A value ``dirichlet_bc[i][1] = False`` means that values are free at the right
boundary of the $i$-direction.
It is possible to set ``dirichlet_bc : null`` which is equivalent to all values being ``False``.


Traces of de Rham spaces
""""""""""""""""""""""""

It is important to note that the values of ``dirichlet_bc`` affect 
only :math:`N`-splines, see :ref:`feec_basics`.
This reflects the correct traces (possible boundary conditions) of the 
FE spaces :math:`H^1`, :math:`H(\textrm{curl})`, :math:`H(\textrm{div})` 
and :math:`L^2`. Indeed, the basis of :math:`H^1` is an ``NNN``-basis,
which means that ``dirichlet_bc`` affects all three directions.  
The bases of the three components of :math:`H(\textrm{curl})` are 
``(DNN, NDN, NND)``, which has the following implications: suppose the 
outward normal to the boundary is :math:`\mathbf n = (0, 0, 1)`, being normal
to the boundary :math:`\eta_3 = 1.0`. In this case it is not possible to set
the value of the third component :math:`E_3` of a function 
:math:`\mathbf E = (E_1, E_2, E_3) \in H(\textrm{curl})`, because
its basis is ``NND``. However, it is possible to set :math:`E_1` and :math:`E_2`
at this boundary. Therefore, the space :math:`H(\textrm{curl})` allows for setting
Dirichlet boundary conditions of the form

.. math::

    \mathbf E \times \mathbf n = 0\quad \textrm{on} \quad \partial \Omega\,,\qquad \mathbf E \in H(\textrm{curl})\,.  

With the same reasoning, it is easy to see that for a function 
:math:`\mathbf B = (B_1, B_2, B_3) \in H(\textrm{div})` with bases ``(NDD, DND, DDN)``
one can set the following type of Dirichlet boundary conditions:

.. math::

    \mathbf B \cdot \mathbf n = 0\quad \textrm{on} \quad \partial \Omega\,,\qquad \mathbf B \in H(\textrm{div})\,.  

Dirichlet boundary conditions cannot be set for :math:`p \in L^2` with basis ``DDD``.


Neumann boundary conditions
"""""""""""""""""""""""""""

Derivative boundary conditions are so-called "natural" boundary conditions,
which are set through integration by parts. 
Consider for example the Poisson equation

.. math::

    -\nabla \cdot \nabla \phi(\mathbf x) = \rho(\mathbf x)\,,

with :math:`\phi \in H^1(\Omega)` and a given right-hand side :math:`\rho(\mathbf x)`.
Multiplying with a test function :math:`\psi \in H^1(\Omega)`, then
integrating over the (physical) domain :math:`\Omega` and using Stokes' Theorem
leads to

.. math::

    \int_\Omega \nabla \phi \cdot \nabla \psi\,\textrm d \mathbf x - \int_{\partial\Omega} \psi\,\nabla \phi \cdot \mathbf n \,\textrm d \mathbf x = \int_\Omega \rho\,\psi \,\textrm d \mathbf x\,.

The second term is a boundary integral. By setting it to zero, 
we imply one of two things: either :math:`\psi = 0` or 
:math:`\nabla \phi \cdot \mathbf n = 0` on the boundary :math:`\partial\Omega`.
If we set ``dirichlet_bc`` to ``False`` on the desired boundary, this means 
:math:`\psi \neq 0` on that boundary, enforcing thus Neumann boundary conditions.

In summary, on each of the six boundary surfaces of the unit cube :math:`(0, 1)^3`,
one has the following possibilities to set boundary conditions:

* ``spl_kind = True``: periodic bcs.
* ``spl_kind = False`` and ``dirichlet_bc = True``: homogeneous Dirichlet bcs.
* ``spl_kind = False`` and ``dirichlet_bc = False``: free bcs, possible "natural" Neumann boundary conditions through the equation.

Check out `the Poisson unit test <https://github.com/struphy-hub/struphy/blob/devel/src/struphy/propagators/tests/test_poisson.py>`_ for an example.

More FEEC boundary conditions will be added in the future.