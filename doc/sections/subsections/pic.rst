.. _particle_discrete:

Particle-in-cell methods (PIC)
------------------------------

Basics
^^^^^^

PIC methods are well-suited for the discretization of conservation laws of the form

.. math::
    :label: conservelaw

    \partial_t f + \nabla_q \cdot (\mathbf u(t, q) \, f) &= S(f) \qquad q \in \Omega \subset \mathbb R^n\,,
    \\[2mm]
    f(t=0) &= f_\textnormal{in}\,,

where :math:`\Omega\subset \mathbb R^n` is open, :math:`q` denotes a set of suitable coordinates in :math:`\Omega`,
and we assume appropriate boundary conditions on :math:`\partial\Omega`.
Here, :math:`f: (t,q) \mapsto \mathbb R^+` 
is a positive density or "volume form", :math:`\mathbf u: (t,q) \mapsto \mathbb R^n` is the vector-field defining the flow, 
and :math:`S(f)` is a (nonlinear) source term. 
Equation :math:numref:`conservelaw` can be re-written as 

.. math::

    \partial_t f + \mathbf u \cdot \nabla_q f = S(f) - f \nabla_q \cdot \mathbf u \,,

which represents an advection equation with a source term.
The PIC ansatz for solving equation :math:numref:`conservelaw` is 

.. math::
    :label: pic:ansatz

    f \approx f_h(t, q) = \frac 1N \sum_{k=1}^N w_k(t)\, \delta(q - q_k(t))\,,

where :math:`N \gg 1` is a large number and :math:`\delta(q)` stands for the Dirac delta-distribution.
Moreover, :math:`q_k: \mathbb R \to \Omega` denotes the
trajectory of marker :math:`k` in the domain :math:`\Omega` and :math:`w_k: \mathbb R \to \mathbb R` 
stands for its (time-dependent) "weight". The equations for :math:`q_k(t)` and :math:`w_k(t)`
are derived from the following principle:

.. math::
    :label: pic:principle

    \frac{\textrm{d}}{\textrm{d} t} \int_\Omega (f - f_h)\, \phi\,\textrm d q = O(N^{-1/2}) \qquad \forall \ \phi \in C^\infty(\Omega)\,.

The time derivative of the :math:`f_h`-integral reads

.. math::
    :label: int_fh

    \frac{\textrm{d}}{\textrm{d} t} \int_\Omega f_h \phi\,\textrm d q &= \frac{\textrm{d}}{\textrm{d} t} \frac 1N \sum_{k=1}^N w_k\, \phi(q_k)
    \\[2mm]
    &= \frac 1N \sum_{k=1}^N \left[ \dot w_k\, \phi(q_k) + w_k\, \dot q_k \cdot \nabla_q \phi(q_k) \right]\,.

Assuming that boundary terms vanish, the time derivative of the :math:`f`-integral can be written as

.. math::

    \frac{\textrm{d}}{\textrm{d} t} \int_\Omega f \phi\,\textrm d q &= - \int_\Omega \nabla_q \cdot (\mathbf u \, f) \phi\,\textrm d q + \int_\Omega S(f) \phi\,\textrm d q
    \\[2mm]
    &= \int_\Omega  f\,\mathbf u \cdot \nabla_q \phi\,\textrm d q + \int_\Omega S(f) \phi\,\textrm d q \,.

These integrals are now viewed as :ref:`monte_carlo` in the following way: 
assume :math:`s: (t,q) \mapsto \mathbb R^+` 
to be a time-dependent probability distribution function (PDF) on :math:`\Omega`, 
and let :math:`\mathbb E(X)_s` denote the expectation value of a random variable :math:`X`
distributed according to :math:`s`. Then,

.. math::
    
    \int_\Omega  f\,\mathbf u \cdot \nabla_q \phi\,\textrm d q &= \int_\Omega  \frac fs\,\mathbf u \cdot \nabla_q \phi\,s\,\textrm d q = \mathbb E\left(\frac fs\,\mathbf u \cdot \nabla_q \phi \right)_s\,,
    \\[2mm]
    \int_\Omega S(f) \phi\,\textrm d q &= \int_\Omega \frac 1s S(f) \phi\,s \,\textrm d q = \mathbb E\left(\frac 1s S(f) \phi \right)_s\,.

At each time :math:`t` we can estimate 
the above expectation values from the :math:`N` samples :math:`q_k`:

.. math::
    :label: expectvals

    \mathbb E\left(\frac fs\,\mathbf u \cdot \nabla_q \phi \right)_s &\approx \frac 1N \sum_{k=1}^N \frac{f(q_k)}{s(q_k)}\,\mathbf u(t, q_k) \cdot \nabla_q \phi(q_k) + O(N^{-1/2})\,,
    \\[2mm]
    \mathbb E\left(\frac 1s S(f) \phi \right)_s &\approx \frac 1N \sum_{k=1}^N\frac{1}{s(q_k)}\,S(f(q_k)) \phi(q_k) + O(N^{-1/2})\,.

We can compare this to equation :math:numref:`int_fh` and since equation :math:numref:`pic:principle` 
holds for any :math:`\phi`, by comparing coefficients we deduce

.. math::

    w_k\, \dot q_k &= \frac{f(q_k)}{s(q_k)}\,\mathbf u(t, q_k)\,, 
    \\[2mm]
    \dot w_k &= \frac{1}{s(q_k)}\,S(f(q_k))\,. 

Let us define the weights to be

.. math::
    :label: def:weights

    w_k := \frac{f(q_k)}{s(q_k)} \,,

which implies 

.. math::
    :label: def:dot_qw

    \dot q_k &= \mathbf u(t, q_k)\,, 
    \\[2mm]
    \dot w_k &= \frac{1}{s(q_k)}\,S(f(q_k))\,. 

and also

.. math::
    :label: def:dot_w2

    \dot w_k = \frac{\dot f(q_k)}{s(q_k)} - \frac{f(q_k)}{s(q_k)^2} \dot s(q_k) \,.

By comparing the equations :math:numref:`def:dot_qw` and :math:numref:`def:dot_w2` we find the equation 
that must be satisfied by the PDF :math:`s`:

.. math::

    \dot s(q_k) = \frac{s(q_k)}{f(q_k)} \Big[ \dot f(q_k) - S(f(q_k)) \Big]\,.

Since, from equation (1),  

.. math::

    \dot f(q_k) = S(f(q_k)) - f(q_k) \nabla_q \cdot \mathbf u(t, q_k)\,,

we deduce

.. math::

    \dot s(q_k) = - s(q_k) \nabla_q \cdot \mathbf u(t, q_k) \,.

Therefore, with the choice of the weights made in equation :math:numref:`def:weights`, 
the PDF :math:`s` must satisfy

.. math::

    \partial_t s + \nabla_q \cdot (\mathbf u(t, q) \, s) = 0\,.

This means that :math:`s` is a density (or volume form) advected by the flow of :math:`\mathbf u`. 
For incompressible flow, :math:`\nabla_q \cdot \mathbf u = 0`, the PDF 
:math:`s` is constant along the flow of :math:`\mathbf u` and the PIC solution :math:numref:`pic:ansatz` 
is obtained by solving

.. math::
    :label: pic:eqs

    \dot q_k &= \mathbf u(t, q_k) \qquad &&q_k(0) = q_{k0}\,,
    \\[2mm]
    \dot w_k &= \frac{1}{s(t=0, q_{k0})}\,S(f)  \qquad &&w_k(0) = \frac{f(t=0, q_{k0})}{s(t=0, q_{k0})}\,,

where the :math:`q_{k0}` are drawn according to the PDF :math:`s(t=0)`.
On the other hand, for compressible flow :math:`\nabla_q \cdot \mathbf u \neq 0`
the weights update is more complicated:

.. math::
    :label: pic:eqs:compressible

    \dot q_k &= \mathbf u(t, q_k) \qquad &&q_k(0) = q_{k0}\,,
    \\[2mm]
    \dot w_k &= \frac{w_k}{f(t, q_k)}\,S(f)  \qquad &&w_k(0) = \frac{f(t=0, q_{k0})}{s(t=0, q_{k0})}\,.

Note that when there is no source, :math:`S(f) = 0`, the weights are constant for compressible
and incompressible flow.

PIC methods are popular for solving these types of equations because

a. they do not suffer from the curse of dimensionality for large :math:`n` 
b. they and can be easily parallelized. 

However, a disadvantage is their slow
convergence :math:`\sim 1/\sqrt N` with the number of particles (also called markers) and the associated noise
that is present in simulations. Often times noise-reduction techniques such as the :ref:`control_var` have to be employed
to reduce simulation cost.


.. _monte_carlo:

Monte-Carlo integrals
^^^^^^^^^^^^^^^^^^^^^

At the core of the PIC method is the approximation of integrals 

.. math::

    I = \int_{\Omega} f(q)\, A(q) \,\textnormal d^n q\,.

by Mont-Carlo (MC) integration. The first expecation value in equation :math:numref:`expectvals`
is an example of this. In Struphy, such integrals are calculated in non-Cartesian
coordinates :math:`q \in \mathbb R^n`, and we denote by 

.. math::

    \textnormal d^n q = |J| \textnormal d q\,,   

the volume element, including the Jacobian determinant from the mapping to Cartesian (straight) space.
Thus, we could also write the integral as

.. math::
    :label: int:1

    I = \int_{\Omega} f(q)\, A(q) |J|\,\textnormal d q\,,

We have seen that :math:`s` satisfies the transport equation :math:numref:`s_eq`,
and that it is normalized, thus

.. math::

    \int_{\Omega} s |J|\,\textnormal d q = 1 \qquad \forall t\,, \qquad \quad s >0\,.

In MC integration, we view the integral :math:`I` as an expectation value:

.. math::
    :label: int:2

    I = \int_{\Omega} \frac{f(q)}{s(q)}\, A(q)\, s(q) |J|\,\textnormal d q = \mathbb E\left( \frac{f}{s} A\right)_s\,,

with respect to the PDF :math:`s|J|`.
Having access to :math:`N` samples :math:`q_k(t)` of :math:`s` at each time :math:`t`
allows us to estimate this expectation via

.. math::
    :label: mcint

    I = \mathbb E\left( \frac{f}{s} A\right)_s \approx \frac 1N \sum_{k=1}^N \frac{f(q_k)}{s(q_k)} A(q_k) = \frac 1N \sum_{k=1}^N w_k A(q_k)\,,

using the definition of the weights in equation :math:numref:`def:weights`. 
The error in this approximation is of order :math:`N^{-1/2}`.

The weights are implemented in :meth:`struphy.pic.base.Particles.initialize_weights`.
Equation :math:numref:`mcint` can be obtained directly from :math:numref:`int:2` by inserting
the PIC ansatz :math:numref:`pic:ansatz`, with :math:`f|J| \approx f_h`.

In Struphy, Monte-Carlo integrals of the form :math:numref:`mcint` are implemented via :ref:`accums`. 


.. _pic_algo:

PIC algorithm
^^^^^^^^^^^^^

The PIC algorithm can be summarized in the following steps:

1. Draw :math:`N\gg 1` markers :math:`(q_{k0})_k` according to :math:`s_{\textnormal{in}}|J|`. The initial marker distribution is implemented in :meth:`struphy.pic.base.Particles.draw_markers`. In Struphy, it is always a Gaussian in velocity space (see docstring).

2. Compute the weights :math:`(w_{k0})_k` according to equation :math:numref:`def:weights`. The initial condition of the kinetic equation enters here.

3. Solve the characteristic equations :math:numref:`pic:eqs` for each marker.

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
