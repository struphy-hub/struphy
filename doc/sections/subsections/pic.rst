.. _particle_discrete:

Particle-in-cell methods (PIC)
------------------------------

Basics
^^^^^^

PIC methods are efficient for the discretization of transport (advection) equations in high-dimensional spaces. 
They do not suffer from the curse of dimensionality and can be easily parallelized. However, a disadvantage is their slow
convergence :math:`\sim 1/\sqrt N` with the number of particles (also called markers) and the associated noise
that is present in simulations. Often times noise-reduction techniques such as the :ref:`control_var` have to be employed
to reduce simulation cost.

Low-density plasma can be described in terms of kinetic equations for its constituents (usually ions and electrons),
coupled to Maxwell's equations modeling their mean-field interaction. The solution of a kinetic equation is 
a phase space volume density :math:`f^\textrm{vol}(t, q)`, also called *distribution function*, 
with :math:`t` denoting time and :math:`q \in \mathbb R^n` denoting phase space coordinates; 
it thus satisfies the classical conservation law

.. math::
    :label: eq:kin:n

    \partial_t f^\textrm{vol} + \nabla_q \cdot (G(t, q) \, f^\textrm{vol}) = 0 \,.

Here, :math:`G(t,q)` denotes the vector field in the ODE describing single particle motion in phase space,

.. math::
    :label: chars

    \dot q(t) = G(t,q(t))\qquad q(0) = q_0\,.

Conventionally, :math:`q=(x,v) \in \mathbb R^n` where :math:`x` stands for the position and :math:`v` stands for
the velocity; in this case :math:`G = (v, a)` with :math:`a` being the acceleration of the particle due to forces.
PIC methods employ the method of characteristics to solve :math:numref:`eq:kin:n`. The distribution function 
is approximated by a sum of :math:`N\gg 1` delta functions,

.. math::
    :label: pic:ansatz

    f^\textrm{vol} \approx f^\textrm{vol}_h(t, q) = \frac 1N\sum_{k=1}^N w_k\, \delta(q - q_k(t))\,,

where :math:`q_k(t)` denote the position of "markers", each of which satisifes :math:numref:`chars` 
(with different initial condition), and :math:`w_k \in \mathbb R` stands for a marker's "weight", to be discussed below.

In general, :math:`q` are not Cartesian coordinates, and :math:`\nabla_q \cdot G \neq 0` such that :math:numref:`eq:kin:n`
cannot be directly written as a transport
equation. In this case :math:`f^\textrm{vol}` is not constant along the solutions of :math:numref:`chars`. Indeed,
it can be easily shown that the kinetic equation :math:numref:`eq:kin:n` is equivalent to the following two statements:

.. math::
    :label: int
    
     \frac{\textnormal d}{\textnormal d t} \int_{q(t,\mathcal V_0)} f^\textrm{vol}(t, q)\,\textnormal d q = 0  \qquad \Leftrightarrow\qquad \frac{\textnormal d}{\textnormal d t} \left( f^\textrm{vol}(t, q(t,q_0)) |J(t,q_0)| \right) = 0\,,  

where :math:`\mathcal V_0` is an arbitrary phase space volume and
:math:`J(t,q_0) = \textnormal{det}\, \partial q(t,q_0)/\partial q_0` is the determinant of 
the Jacobian matrix of the flow map :math:`q_0 \mapsto q(t,q_0)` corresponding to the ODE :math:numref:`chars`.
These equations express the continuity (or conservation) of probability in phase space, in any coordinates :math:`q`.
Correspondingly, the second term in the kinetic equation :math:numref:`eq:kin:n` is the Lie-derivative of a volume form.

Suppose we describe the system in a set of Cartesian coordinates :math:`z\in\mathbb R^n`, related to some other,
possibly curvilinear coordinates :math:`q\in\mathbb R^n`
via an invertible, differentiable mapping :math:`F: \mathbb R^n \to \mathbb R^n,\, z = F(q)`. Moreover, let :math:`f(t,z)` denote the
distribution function in Cartesian phase space and define :math:`f^0(t,q) := f(t,F(q))`. The first statement of :math:numref:`int`
can then be written as

.. math::

    \frac{\textnormal d}{\textnormal d t} \int_{z(t,F(\mathcal V_0))} f(t, z)\,\textnormal d z = \frac{\textnormal d}{\textnormal d t} \int_{q(t,\mathcal V_0)} f^0(t, q) |J_F(q)|\,\textnormal d q = 0\,, 

where :math:`J_F = \textnormal{det}\,\partial F/\partial q` stands for the Jacobian determinant of the mapping.
It follows that :math:`f^\textrm{vol} = f^0 |J_F|`. The equation for :math:`f^0` can be derived from :math:numref:`eq:kin:n` 
by using the **Liouville theorem**,

.. math::

    \partial_t |J_F| + \nabla_q \cdot(G(t,q) |J_F|) = 0\,.

Indeed, :math:`f^0` satisfies a transport equation, since

.. math::
    :nowrap:

    \begin{align}
    &\partial_t f^\textrm{vol} + \nabla_q \cdot (G(t, q) \, f^\textrm{vol}) = 0 \\[1mm]
    
    \Leftrightarrow \quad &\partial_t (|J|f^0) + \nabla_q \cdot (G |J| f^0) = 0 \\[1mm]
    
    \Leftrightarrow \quad & |J|\partial_t f^0 + |J| G \cdot \nabla_q f^0 + f^0 \partial_t |J| + f^0 \nabla_q \cdot (G |J| ) = 0 \\[1mm]
    
    \Leftrightarrow \quad & \partial_t f^0 + G \cdot \nabla_q f^0 = 0 \,.
    \end{align}

The 0-form :math:`f^0` is thus constant along the characteristics :math:numref:`chars`,

.. math::

    \frac{\textnormal d}{\textnormal d t} \left( f^0(t, q(t,q_0)) \right) = 0\,.


.. _monte_carlo:

Monte-Carlo integrals
^^^^^^^^^^^^^^^^^^^^^

At the core of PIC methods is the approximation of phase space integrals 

.. math::

    I = \int_{\mathbb R^n} f (t,z)\, A(t,z) \,\textnormal d z

by Mont-Carlo integration. Here, :math:`z=(x,v) \in \mathbb R^n` are Cartesian phase space coordinates
and :math:`A(t,z)` is a given function. In Struphy, such integrals are calculated in non-Cartesian
coordinates :math:`q \in \mathbb R^n`, related via a diffeomorphism :math:`z = F(q)`,

.. math::
    :label: int:1

    I = \int_{\mathbb R^n} f^0 (t,q)\, A^0(t,q) |J_F(q)|\,\textnormal d q\,,

where we use the same notation as in the previous section. We know that :math:`f^0(t,q)` satisfies a transport equation

.. math::
    :label: kin:f0

    \partial_t f^0 + G \cdot \nabla_q f^0 = 0\qquad f^0(t=0) = f^0_{\textnormal{in}}\,,

with :math:`f^0_{\textnormal{in}}` denoting the initial condition. 
Let us now introduce another function :math:`s^0(t,q)`,
which we assume also satisfies the transport equation,

.. math::

    \partial_t s^0 + G \cdot \nabla_q s^0 = 0\qquad s^0(t=0) = s^0_{\textnormal{in}}\,,

but with potentially different initial condition :math:`s^0_{\textnormal{in}}`. 
We moreover assume that the corresponding :math:`n`-form or volume density :math:`s^\textrm{vol} = s^0 |J_F|` is
a probability distribution function (PDF), i.e.

.. math::

    \int_{\mathbb R^n} s^\textrm{vol}(t,q)\,\textnormal d q = 1 \qquad \forall t\,, \qquad \quad s^\textrm{vol} >0\,.

The integral :math:numref:`int:1` can now be interpreted as the expectation value of the random
variable :math:`A^0\,f^0/s^0` distributed according to the PDF :math:`s^\textrm{vol}`:

.. math::

    I = \int_{\mathbb R^n} \frac{f^0 (t,q)}{s^0 (t,q)}\, A^0(t,q) s^\textrm{vol}(t,q)\,\textnormal d q = \mathbb E \left( \frac{f^0 }{s^0 }\, A^0 \right)_{s^\textrm{vol}}\,.

In order to approximate the expectation value, we draw :math:`N\gg 1` markers :math:`(q_{k0})_k` according to the PDF :math:`s^\textrm{vol}_{\textnormal{in}}`
and evolve them along the characteristics :math:numref:`chars`. When denoting the ODE solutions by :math:`(q_{k}(t))_k`, 
the unbiased estimator for the expectation value reads

.. math::

    I \approx \frac{1}{N} \sum_{k=1}^N \frac{f^0(t, q_k(t)) }{s^0(t, q_k(t)) }\, A^0 (t, q_k(t)) \,.

If we further exploit that both :math:`f^0` and :math:`s^0` are constant along the characteristics, we arrive at 

.. math::
    :label: mcint

    I \approx \frac{1}{N} \sum_{k=1}^N w_{k0}\, A^0 (t, q_k(t)) \,,

where we defined the time-independent **weights**

.. math::
    :label: weights

    w_{k0} := \frac{f^0(t, q_k(t)) }{s^0(t, q_k(t)) } = \frac{f^0(0, q_k(0)) }{s^0(0, q_k(0)) } = \frac{f^0_{\textnormal{in}}(q_{k0}) }{s^0_{\textnormal{in}}(q_{k0}) }\,.

These weights are implemented in :meth:`struphy.pic.base.Particles.initialize_weights`.
Equation :math:numref:`mcint` can be obtained directly from :math:numref:`int:1` by inserting
the PIC ansatz :math:numref:`pic:ansatz`, with :math:`f^0|J_F| \approx f^\textrm{vol}_h`.

In Struphy, Monte-Carlo integrals of the form :math:numref:`mcint` are implemented via :ref:`accums`. 


.. _pic_algo:

PIC algorithm
^^^^^^^^^^^^^

The PIC algorithm can be summarized in the following steps:

1. Draw :math:`N\gg 1` markers :math:`(q_{k0})_k` according to :math:`s^\textrm{vol}_{\textnormal{in}}`. The initial marker distribution is implemented in :meth:`struphy.pic.base.Particles.draw_markers`. In Struphy, it is always a Gaussian in velocity space (see docstring).

2. Compute the weights :math:`(w_{k0})_k` according to equation :math:numref:`weights`. The initial condition of the kinetic equation :math:numref:`kin:f0` enters only here. Note that :math:`s^0_{\textnormal{in}} = s^\textrm{vol}_{\textnormal{in}} / |J_F|`.

3. Solve the characteristic equations :math:numref:`chars` with initial condition :math:`q_{k0}` for each marker.

4. Whenever necessary, compute integrals according to :math:numref:`mcint`.
 

.. _binning:

Particle binning
^^^^^^^^^^^^^^^^

The aim of particle binning is to obtain a "grid-representation" of either :math:`f^0(t, q)` or :math:`f^\textrm{vol}(t, q)`,
which are represented by markers in Struphy. The approximation is obtained by integration of :math:`f^{0/\textrm{vol}}(t, q)`
over small volumes of phase space, called "bins". These bins are not necessarily related to the FE grid used for fluid/field variables
(but they can be). Moreover, bins do not have to be defined in all :math:`n` directions of the phase space; one could do just 1D or
2D binning, integrating over the other coordinates entirely. 
 
For example, let's say we want to do 2D binning in the :math:`(\eta_1,v_1)` subspace of the full :math:`(\eta,v) \in \mathbb R^6` phase space.
In this case we want an approximate representation of the "reduced" 2D density

.. math::

    f^{k,\textnormal{red}}(t, \eta_1, v_1) := \int_{(0, 1)^2}\int_{\mathbb R^2} f^k(t, \eta, v) \,\textnormal d \eta_2 \textnormal d \eta_3 \textnormal d v_2 \textnormal dv_3\,,

where :math:`k \in \{0, \textrm{vol}\}` such that :math:`f^k` is either the 0-form or the volume form in logical space (here 
integrated over a submanifold, which is technically not allowed - but serves to get an approximation via numerical binning).
If we define bin edges :math:`(\eta_{1i})_i` and :math:`(v_{1j})_j` in the :math:`(\eta_1, v_1)` subspace, 
we can approximate :math:`f^{k,\textnormal{red}}` restricted to a bin
:math:`\Omega_{ij} = (\eta_{1i}, \eta_{1i+1}) \times (v_{1j}, v_{1j+1})` by its mean value over over that bin:

.. math::
    :label: f_approx

    f^{k,\textnormal{red}}(t, \eta, v)\Big|_{\Omega_{ij}} \approx R^k_{ij}(t) := \frac{1}{|\Omega_{ij}|} \int_{\Omega_{ij}} f^{k,\textnormal{red}}(t, \eta_1, v_1)\,\textnormal d\eta_1 \textnormal d v_1\,.

where :math:`|\Omega_{ij}|` is the volume of the :math:`ij`-th bin (in 2D):

.. math::

    |\Omega_{ij}| = \int_{\Omega_{ij}} \textnormal d\eta_1 \textnormal d v_1 = (\eta_{1i+1} - \eta_{1i})(v_{1j+1} - v_{1j})\,.

Note that :math:`R^k_{ij}` has the same dimension as :math:`f^{k,\textnormal{red}}`,
due to the one-over-volume prefactor. The integrals :math:`R^k_{ij}` are then approximated by Monte-Carlo integration,
using the indicator function :math:`\textnormal{id}_{ij}(x_1, v_1)` of the :math:`ij`-th bin 
(it is 1 inside :math:`\Omega_{ij}` and zero otherwise). Depending on the value of :math:`k \in \{0, \textrm{vol}\}`, 
there are two different integrals to approximate:

.. math::
    :nowrap:
    :label: def:Rij

    \begin{align}
    R^0_{ij}(t) &= \frac{1}{|\Omega_{ij}|}\int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, f^0(t, \eta, v)\,\textnormal d\eta\, \textnormal d v \\

    &= \frac{1}{|\Omega_{ij}|} \int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, \frac{f^0}{s^0 |\textnormal{det} J_F|}\,s^\textrm{vol}\,\textnormal d\eta\, \textnormal d v \\
    
    &\approx \frac{1}{N|\Omega_{ij}|} \sum_{k \in \Omega_{ij}} w_{k} \frac{1}{|\textnormal{det} J_F|(\eta_k(t), v_k(t))}\,, \\[2mm] 
    
    R^\textrm{vol}_{ij}(t) &= \frac{1}{|\Omega_{ij}|}\int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, f^\textrm{vol}(t, \eta, v)\,\textnormal d\eta\, \textnormal d v \\

    &= \frac{1}{|\Omega_{ij}|} \int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, \frac{f^0}{s^0}\,s^\textrm{vol}\,\textnormal d\eta\, \textnormal d v \\
    
    &\approx \frac{1}{N|\Omega_{ij}|} \sum_{k \in \Omega_{ij}} w_{k} \,.
    \end{align}

Here, the sum runs over all markers that are in the bin :math:`\Omega_{ij}`. 
The values :math:`R^0_{ij}` computed in :math:numref:`def:Rij` are discrete approximations of :math:`f^0`,
and values :math:`R^\textrm{vol}_{ij}` are discrete approximations of :math:`f^\textrm{vol}`.
This binning algorithm is implemented in :meth:`struphy.pic.base.Particles.binning`.


.. _control_var:

Control variate method
^^^^^^^^^^^^^^^^^^^^^^

This noise reduction technique relies on 

1. The ability to compute (without Monte-Carlo) the integral :math:numref:`int:1` for a known distribution function :math:`f^0 = \mathcal M^0`:

.. math::

    I_{\mathcal M} := \int_{\mathbb R^n} \mathcal M^0 (t,q)\, A^0(t,q) |J_F(q)|\,\textnormal d q = \int_{\mathbb R^n} \mathcal M^n (t,q)\, A^0(t,q) \,\textnormal d q\,.

2. The assumption that :math:`f^0(t)` is close to :math:`\mathcal M^0(t)` for all :math:`t`.

In that case we can decompose :math:numref:`int:1` as

.. math::
    :nowrap:

    \begin{align}
    I &= \int_{\mathbb R^n} f^0 \, A^0 |J_F|\,\textnormal d q\,, \\[1mm]

    &= \int_{\mathbb R^n} (f^0 - \mathcal M^0) \, A^0 |J_F|\,\textnormal d q + I_{\mathcal M}\,, \\[1mm]

    &= \int_{\mathbb R^n} \frac{(f^0 - \mathcal M^0)}{s^0} \, A^0 s^\textrm{vol}\,\textnormal d q + I_{\mathcal M}\,, \\[1mm]

    &\approx \frac{1}{N} \sum_{k=1}^N w_{k}(t)\, A^0 (t, q_k(t)) + I_{\mathcal M}\,,
    \end{align}

where we have introduced the time-dependent weights

.. math::

    w_k(t) = w_{k0} - \frac{\mathcal M^0(t, q_k(t))}{s^0_{\textnormal{in}}(q_{k0})}\,.

This algorithm is implemented in :meth:`struphy.pic.base.Particles.update_weights`.
