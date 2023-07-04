.. _gempic:

GEMPIC discretization
=====================

Struphy discretization is performed according to the GEMPIC (Geometric Electro-Magnetic Particle-In-Cell) framework.
Relevant publications are (and the references therein):

    1. M. Kraus, K. Kormann, P. J. Morrison, E. Sonnendrücker, "GEMPIC: Geometric ElectroMagnetic
    Particle-In-Cell Methods", `Journal of Plasma Physics 83.4 (2017) <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gempic-geometric-electromagnetic-particleincell-methods/C32D97F1B5281878F094B7E5075D291A>`_
    
    2. A. Buffa, J. Rivas , G. Sangalli , R. Vasquez, 
    "Isogeometric discrete differential forms in three dimensions", `SIAM J. Numer. Anal. Vol. 49, No. 2 (2011) <https://epubs.siam.org/doi/10.1137/100786708>`_

    3. F. Holderied, S. Possanner, X. Wang, "MHD-kinetic hybrid code based on structure-preserving 
    finite elements with particles-in-cell", `J. Comp. Phys. 433 (2021) 110143 <https://www.sciencedirect.com/science/article/pii/S0021999121000358?via%3Dihub>`_

In Struphy, kinetic equations are discretized with the :ref:`particle_discrete`. Equations for electromagnetic fields and/or fluid species 
are discretized with :ref:`geomFE`. In what follows we provide a brief introduction to these methods. In section :ref:`disc_example`
we detail the discretization of the Vlasov-Maxwell system implemented in Struphy. 


.. _particle_discrete:

Particle-in-cell method (PIC)
-----------------------------

Basics
^^^^^^

PIC methods are efficient for the discretization of transport (advection) equations in high-dimensional spaces. 
They do not suffer from the curse of dimensionality and can be easily parallelized. However, a disadvantage is their slow
convergence :math:`\sim 1/\sqrt N` with the number of particles (also called markers) and the associated noise
that is present in simulations. Often times noise-reduction techniques such as the :ref:`control_var` have to be employed
to reduce simulation cost.

Low-density plasma can be described in terms of kinetic equations for its constituents (usually ions and electrons),
coupled to Maxwell's equations modeling their mean-field interaction. The solution of a kinetic equation is 
a phase space volume density :math:`f^n`, also called *distribution function*, where the superscript :math:`n` denotes the dimension 
of the phase space (position-velocity space). For three spatial coordinates and three velocity coordinates 
we have :math:`n=6`, which can be considered a quite high dimension for PDEs. Because :math:`f^n(t, q)`
is a volume density (:math:`n`-form) in phase space, with :math:`t` denoting time and :math:`q \in \mathbb R^n` denoting
phase space coordinates, it satisfies the classical conservation law

.. math::
    :label: eq:kin:n

    \partial_t f^n + \nabla_q \cdot (G(t, q) \, f^n) = 0 \,.

Here, :math:`G(t,q)` denotes the vector field in the ODE describing single particle motion in phase space,

.. math::
    :label: chars

    \dot q(t) = G(t,q(t))\qquad q(0) = q_0\,.

Conventionally, :math:`q=(x,v) \in \mathbb R^n` where :math:`x` stands for the position and :math:`v` stands for
the velocity; in this case :math:`G = (v, a)` with :math:`a` being the acceleration of the particle due to forces.
In general, :math:`q` are not Cartesian coordinates, and :math:`\nabla_q \cdot G \neq 0` such that :math:numref:`eq:kin:n`
cannot be directly written as a transport
equation. In this case :math:`f^n` is not constant along the solutions of :math:numref:`chars`. Indeed,
it can be easily shown that the kinetic equation :math:numref:`eq:kin:n` is equivalent to the following two statements:

.. math::
    :label: int
    
     \frac{\textnormal d}{\textnormal d t} \int_{q(t,\mathcal V_0)} f^n(t, q)\,\textnormal d q = 0  \qquad \Leftrightarrow\qquad \frac{\textnormal d}{\textnormal d t} \left( f^n(t, q(t,q_0)) |J(t,q_0)| \right) = 0\,,  

where :math:`\mathcal V_0` is an arbitrary phase space volume and
:math:`J(t,q_0) = \textnormal{det}\, \partial z(t,q_0)/\partial q_0` is the determinant of 
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
It follows that :math:`f^n = f^0 |J_F|`. The equation for :math:`f_0` can be derived from :math:numref:`eq:kin:n` 
by using the **Liouville theorem**,

.. math::

    \partial_t |J_F| + \nabla_q \cdot(G(t,q) |J_F|) = 0\,.

Indeed, :math:`f^0` satisfies a transport equation, since

.. math::
    :nowrap:

    \begin{align}
    &\partial_t f^n + \nabla_q \cdot (G(t, q) \, f^n) = 0 \\[1mm]
    
    \Leftrightarrow \quad &\partial_t (|J|f^0) + \nabla_q \cdot (G |J| f^0) = 0 \\[1mm]
    
    \Leftrightarrow \quad & |J|\partial_t f^0 + |J| G \cdot \nabla_q f^0 + f^0 \partial_t |J| + f^0 \nabla_q \cdot (G |J| ) = 0 \\[1mm]
    
    \Leftrightarrow \quad & \partial_t f^0 + G \cdot \nabla_q f^0 = 0 \,.
    \end{align}

The 0-form :math:`f^0` is thus constant along the characteristics :math:numref:`chars`,

.. math::

    \frac{\textnormal d}{\textnormal d t} \left( f^0(t, q(t,q_0)) \right) = 0\,.


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
We moreover assume that the corresponding :math:`n`-form or volume density :math:`s^n = s^0 |J_F|` is
a probability distribution function (PDF), i.e.

.. math::

    \int_{\mathbb R^n} s^n(t,q)\,\textnormal d q = 1 \qquad \forall t\,, \qquad \quad s^n >0\,.

The integral :math:numref:`int:1` can now be interpreted as the expectation value of the random
variable :math:`A^0\,f^0/s^0` distributed according to the PDF :math:`s^n`:

.. math::

    I = \int_{\mathbb R^n} \frac{f^0 (t,q)}{s^0 (t,q)}\, A^0(t,q) s^n(t,q)\,\textnormal d q = \mathbb E \left( \frac{f^0 }{s^0 }\, A^0 \right)_{s^n}\,.

In order to approximate the expectation value, we draw :math:`N\gg 1` markers :math:`(q_{k0})_k` according to the PDF :math:`s^n_{\textnormal{in}}`
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

These weights are implemented in :meth:`struphy.pic.particles.Particles.initialize_weights`.

The PIC algorithm can be summarized in the following steps:

1. Draw :math:`N\gg 1` markers :math:`(q_{k0})_k` according to :math:`s^n_{\textnormal{in}}`. The initial marker distribution is implemented in :meth:`struphy.pic.particles.Particles.draw_markers`. In Struphy, it is always a Gaussian in velocity space (see docstring).

2. Compute the weights :math:`(w_{k0})_k` according to equation :math:numref:`weights`. The initial condition of the kinetic equation :math:numref:`kin:f0` enters only here. Note that :math:`s^0_{\textnormal{in}} = s^n_{\textnormal{in}} / |J_F|`.

3. Solve the characteristic equations :math:numref:`chars` with initial condition :math:`q_{k0}` for each marker.

4. Whenever necessary, compute integrals according to :math:numref:`mcint`.
 

.. _binning:

Particle binning
^^^^^^^^^^^^^^^^

The aim of particle binning is to obtain a "grid-representation" of either :math:`f^0(t, q)` or :math:`f^n(t, q)`,
which are represented by markers in Struphy. The approximation is obtained by integration of the volume density :math:`f^{0/n}(t, q)`
over small volumes of phase space, called "bins". These bins are not necessarily related to the FE grid used for fluid/field variables
(but they can be). Moreover, bins do not have to be defined in all :math:`n` directions of the phase space; one could do just 1D or
2D binning, integrating over the other coordinates entirely. 
 
For example, let's say we want to do 2D binning in the :math:`(\eta_1,v_1)` subspace of the full :math:`(\eta,v) \in \mathbb R^6` phase space.
In this case we want an approximate representation of the "reduced" 2D density

.. math::

    f^{k,\textnormal{red}}(t, \eta_1, v_1) := \int_{(0, 1)^2}\int_{\mathbb R^2} f^k(t, \eta, v) \,\textnormal d \eta_2 \textnormal d \eta_3 \textnormal d v_2 \textnormal dv_3\,,

where :math:`k \in \{0, n\}` such that :math:`f^k` is either the 0-form of the volume form in logical space (here 
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
(it is 1 inside :math:`\Omega_{ij}` and zero otherwise). Depending on the value of :math:`k \in \{0, n\}`, 
there are two different integrals to approximate:

.. math::
    :nowrap:
    :label: def:Rij

    \begin{align}
    R^0_{ij}(t) &= \frac{1}{|\Omega_{ij}|}\int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, f^0(t, \eta, v)\,\textnormal d\eta\, \textnormal d v \\

    &= \frac{1}{|\Omega_{ij}|} \int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, \frac{f^0}{s^0 |\textnormal{det} J_F|}\,s^n\,\textnormal d\eta\, \textnormal d v \\
    
    &\approx \frac{1}{N|\Omega_{ij}|} \sum_{k \in \Omega_{ij}} w_{k} \frac{1}{|\textnormal{det} J_F|(\eta_k(t), v_k(t))}\,, \\[2mm] 
    
    R^n_{ij}(t) &= \frac{1}{|\Omega_{ij}|}\int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, f^n(t, \eta, v)\,\textnormal d\eta\, \textnormal d v \\

    &= \frac{1}{|\Omega_{ij}|} \int_{(0, 1)^3} \int_{\mathbb R^3} \textnormal{id}_{ij}(x_1, v_1)\, \frac{f^0}{s^0}\,s^n\,\textnormal d\eta\, \textnormal d v \\
    
    &\approx \frac{1}{N|\Omega_{ij}|} \sum_{k \in \Omega_{ij}} w_{k} \,.
    \end{align}

Here, the sum runs over all markers that are in the bin :math:`\Omega_{ij}`. 
The values :math:`R^0_{ij}` computed in :math:numref:`def:Rij` are discrete approximations of :math:`f^0`,
and values :math:`R^n_{ij}` are discrete approximations of :math:`f^n`.
This binning algorithm is implemented in :meth:`struphy.pic.particles.Particles.binning`.


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
    :label: int:1

    \begin{align}
    I &= \int_{\mathbb R^n} f^0 \, A^0 |J_F|\,\textnormal d q\,, \\[1mm]

    &= \int_{\mathbb R^n} (f^0 - \mathcal M^0) \, A^0 |J_F|\,\textnormal d q + I_{\mathcal M}\,, \\[1mm]

    &= \int_{\mathbb R^n} \frac{(f^0 - \mathcal M^0)}{s^0} \, A^0 s^n\,\textnormal d q + I_{\mathcal M}\,, \\[1mm]

    &\approx \frac{1}{N} \sum_{k=1}^N w_{k}(t)\, A^0 (t, q_k(t)) + I_{\mathcal M}\,,
    \end{align}

where we have introduced the time-dependent weights

.. math::

    w_k(t) = w_{k0} - \frac{\mathcal M^0(t, q_k(t))}{s^0_{\textnormal{in}}(q_{k0})}\,.

This algorithm is implemented in :meth:`struphy.pic.particles.Particles.update_weights`.


.. _geomFE:

Geometric finite elements
-------------------------


.. _feec_basics:

Basics
^^^^^^

In **conforming finite element (FE) methods**, the general principle of approximating a function :math:`u \in V` in a space :math:`V` is

.. math::

    u \approx u_h = \sum_{i=1}^{M} u_i \Lambda_i \quad \in V_h\,,

where :math:`\Lambda_i \in V` are linearly independent basis functions, :math:`(u_i)_i \in \mathbb R^M` are called coefficients
and :math:`M \in \mathbb N` is the dimension of the subspace :math:`V_h \subset V` spanned by the basis functions.
The thing the differentiates FE methods from spectral methods is the fact that the :math:`\Lambda_i` have local support
around some grid point :math:`x_i`. A nice feature of FE methods is that :math:`u_h \in V`, which leads
often to an elegant analysis of the method. The implementation of a FE method consists in writing down a system of equations
for the coefficients :math:`(u_i)_i \in \mathbb R^M`, leaving the basis functions fixed.

We denote by "geometric finite elements" a sequence of discrete (=finite dimensional) spaces :math:`V_h^n,\,0\leq n \leq 3` that
satisfy the **3d Derham diagram**:

.. figure:: ../pics/derham_complex.png
    :align: center

    Fig. 1: 3d Derham diagram.

The upper row is a sequence of continuous function spaces, where the last space :math:`L^2(\Omega)`
is the space of square-integrable functions on :math:`\Omega`. The other spaces are:

:math:`H^1(\Omega)`: functions :math:`f \in L^2(\Omega)` for which :math:`\nabla f \in (L^2(\Omega))^3` 

:math:`H(\textnormal{curl}, \Omega)`: functions :math:`\mathbf u \in (L^2(\Omega))^3` for which :math:`\nabla \times \mathbf u \in (L^2(\Omega))^3` 

:math:`H(\textnormal{div}, \Omega)`: functions :math:`\mathbf u \in (L^2(\Omega))^3` for which :math:`\nabla \cdot \mathbf u \in L^2(\Omega)` 

More information on these spaces can be found in many textbooks on FE methods, 
for instance in `Brezzi, Fortin, "Mixed and Hybrid Finite Element Methods" <https://link.springer.com/book/10.1007/978-1-4612-3172-1>`_.

Struphy uses a conforming" FE method, 

.. math::
    
    V_h^0 \subset H^1(\Omega),\qquad V_h^1 \subset H(\textnormal{curl}, \Omega),\qquad V_h^2 \subset H(\textnormal{div}, \Omega),\qquad V_h^3 \subset L^2(\Omega)

The operators :math:`\Pi_n,\,0\leq n \leq 3` project (:math:`\Pi_n^2 = \Pi_n`) into the subspaces :

.. math::
    
    \Pi_0: \, H^1(\Omega) \to V_h^0,\qquad \Pi_1:\, H(\textnormal{curl}, \Omega) \to V_h^1,\qquad \Pi_2:\, H(\textnormal{div}, \Omega) \to V_h^2,\qquad \Pi_3:\, L^2(\Omega) \to V_h^3

These spaces and the associated operators of the Derham diagram have been implemented in the open-source package
`Psydac <https://github.com/pyccel/psydac>`_.
Struphy interfaces to this library by means of the class :class:`struphy.psydac_api.psydac_derham.Derham`.

Both the continuous and the discrete spaces form a **complex**, which means

.. math::

    \textnormal{curl}\,\textnormal{grad}  = 0\,,\qquad \textnormal{div}\,\textnormal{curl} = 0   

holds on both levels. Moreover, the above diagram can be viewed as composed of three **commuting diagrams**, namely

.. math::
    :label: commute

    \Pi_1\textnormal{grad} = \textnormal{grad}\,\Pi_0\,,\qquad \Pi_2\textnormal{curl} = \textnormal{curl}\,\Pi_1\,,\qquad \Pi_3\textnormal{div} = \textnormal{div}\,\Pi_2

In Struphy the discrete spaces :math:`V_h^n,\,0\leq n \leq 3` are spanned by tensor-product B-spline basis functions.
Building blocks are the :ref:`uni_variate_spaces` of degree :math:`p` with basis functions :math:`N^p(\eta)` and :math:`D^{p -1}(\eta)` 
defined on the unit interval :math:`\eta \in [0, 1]`. Hence, in Struphy
the simualtion domain is always the unit cube, :math:`(\eta_1,\eta_2,\eta_3) \in \Omega = [0, 1]^3`. 
The tensor-product basis functions are denoted as follows:

.. math::
    :nowrap:

    \begin{align}
    \Lambda^0_{ijk}(\eta_1,\eta_2,\eta_3) &= N_i^{p_1}(\eta_1) N_j^{p_2}(\eta_2) N_k^{p_3}(\eta_3)\,,\qquad &&\mathbf \Lambda^0 = (\Lambda^0_{ijk})_{i=1, j=1, k=1}^{n_1, n_2, n_3} \qquad \\\\
    
    \vec{\Lambda}^1_{1,ijk}(\eta_1,\eta_2,\eta_3) &= \begin{pmatrix} D_i^{p_1 - 1}(\eta_1) N_j^{p_2}(\eta_2) N_k^{p_3}(\eta_3) \\ 0 \\ 0 \end{pmatrix}\,,\qquad &&\vec{\mathbf \Lambda}^1_1 = (\vec \Lambda^1_{1,ijk})_{i=1, j=1, k=1}^{d_1, n_2, n_3} \\\\

    \vec{\Lambda}^1_{2,ijk}(\eta_1,\eta_2,\eta_3) &= \begin{pmatrix} 0 \\ N_i^{p_1}(\eta_1) D_j^{p_2 - 1}(\eta_2) N_k^{p_3}(\eta_3) \\ 0 \end{pmatrix}\,,\qquad &&\vec{\mathbf \Lambda}^1_2 = (\vec \Lambda^1_{2,ijk})_{i=1, j=1, k=1}^{n_1, d_2, n_3} \\\\

    \vec{\Lambda}^1_{3,ijk}(\eta_1,\eta_2,\eta_3) &= \begin{pmatrix} 0 \\ 0 \\ N_i^{p_1}(\eta_1) N_j^{p_2}(\eta_2) D_k^{p_3 - 1}(\eta_3) \end{pmatrix}\,,\qquad &&\vec{\mathbf \Lambda}^1_3 = (\vec \Lambda^1_{3,ijk})_{i=1, j=1, k=1}^{n_1, n_2, d_3} \\\\

    \vec{\Lambda}^2_{1,ijk}(\eta_1,\eta_2,\eta_3) &= \begin{pmatrix} N_i^{p_1}(\eta_1) D_j^{p_2 - 1}(\eta_2) D_k^{p_3 - 1}(\eta_3) \\ 0 \\ 0 \end{pmatrix}\,,\qquad &&\vec{\mathbf \Lambda}^2_1 = (\vec \Lambda^2_{1,ijk})_{i=1, j=1, k=1}^{n_1, d_2, d_3} \\\\

    \vec{\Lambda}^2_{2,ijk}(\eta_1,\eta_2,\eta_3) &= \begin{pmatrix} 0 \\ D_i^{p_1 - 1}(\eta_1) N_j^{p_2}(\eta_2) D_k^{p_3 - 1}(\eta_3) \\ 0 \end{pmatrix}\,,\qquad &&\vec{\mathbf \Lambda}^2_2 = (\vec \Lambda^2_{2,ijk})_{i=1, j=1, k=1}^{d_1, n_2, d_3} \\\\

    \vec{\Lambda}^2_{3,ijk}(\eta_1,\eta_2,\eta_3) &= \begin{pmatrix} 0 \\ 0 \\ D_i^{p_1 - 1}(\eta_1) D_j^{p_2 - 1}(\eta_2) N_k^{p_3}(\eta_3) \end{pmatrix}\,,\qquad &&\vec{\mathbf \Lambda}^2_3 = (\vec \Lambda^2_{3,ijk})_{i=1, j=1, k=1}^{d_1, d_2, n_3} \\\\

    \Lambda^3_{ijk}(\eta_1,\eta_2,\eta_3) &= D_i^{p_1 - 1}(\eta_1) D_j^{p_2 - 1}(\eta_2) D_k^{p_3 - 1}(\eta_3)\,,\qquad &&\mathbf \Lambda^3 = (\Lambda^3_{ijk})_{i=1, j=1, k=1}^{d_1, d_2, d_3} \qquad
    \end{align}

Elements of the discrete spaces ("FE fields") are linear combinations of the respective basis functions:

.. math::
    :nowrap:

    \begin{align}
    V_h^0 &= \textnormal{span}(\mathbf \Lambda^0)\,,\qquad p_h^0(\eta_1,\eta_2,\eta_3) = \sum_{ijk} p_{ijk}\, \Lambda^0_{ijk}(\eta_1,\eta_2,\eta_3) = \mathbf p^\top \mathbf \Lambda^0  \quad \in \,V_h^0 \\

    V_h^1 &= \textnormal{span}(\vec{\mathbf \Lambda}^1_1, \vec{\mathbf \Lambda}^1_2, \vec{\mathbf \Lambda}^1_3)\,, \\
    
    &\qquad\qquad \mathbf E_h^1(\eta_1,\eta_2,\eta_3) = \sum_{\mu=1}^3 \sum_{ijk} e_{\mu, ijk}\,\vec{\Lambda}^1_{\mu, ijk}(\eta_1,\eta_2,\eta_3) = \sum_{\mu=1}^3 \mathbf e_\mu^\top \vec{\mathbf \Lambda}^1_\mu = \mathbf e^\top \vec{\mathbf \Lambda}^1 \quad \in \,V_h^1 \\

    V_h^2 &= \textnormal{span}(\vec{\mathbf \Lambda}^2_1, \vec{\mathbf \Lambda}^2_2, \vec{\mathbf \Lambda}^2_3)\,, \\
    &\qquad \qquad \mathbf B_h^2(\eta_1,\eta_2,\eta_3) = \sum_{\mu=1}^3 \sum_{ijk} b_{\mu, ijk}\,\vec{\Lambda}^2_{\mu, ijk}(\eta_1,\eta_2,\eta_3) = \sum_{\mu=1}^3 \mathbf b_\mu^\top \vec{\mathbf \Lambda}^2_\mu = \mathbf b^\top \vec{\mathbf \Lambda}^2 \quad \in \,V_h^2 \\ 

    V_h^3 &= \textnormal{span}(\mathbf \Lambda^3)\,,\qquad n_h^3(\eta_1,\eta_2,\eta_3) = \sum_{ijk}^{\phantom{3}} n_{ijk}\, \Lambda^3_{ijk}(\eta_1,\eta_2,\eta_3) = \mathbf n^\top \mathbf \Lambda^3  \quad \in \,V_h^3
    \end{align}

The discrete FE fields are represented by their **FE coefficients** :math:`\mathbf p \in \mathbb R^{N_0}`, 
:math:`\mathbf e \in \mathbb R^{N_1}`, :math:`\mathbf b \in \mathbb R^{N_2}` and :math:`\mathbf n \in \mathbb R^{N_3}`.
The class for creating such FE fields in Struphy is :class:`struphy.psydac_api.fields.Field`.
In particular, each ``Field`` object has 

1. the property ``Field.vector`` holding the FE coefficients, along with a setter method,

2. a ``__call__()`` method for evaluating the field at (an array of) points :math:`(\eta_{ijk})_{ijk}`.
 
The space dimensions are products of the uni-variate dimensions:

.. math::
    :nowrap:

    \begin{align}
    \textnormal{dim} V_h^0 &= N_0 = n_1 n_2 n_3 \\\\
    
    \textnormal{dim} V_h^1 &= N_1 = d_1 n_2 n_3 + n_1 d_2 n_3 + n_1 n_2 d_3 \\\\

    \textnormal{dim} V_h^2 &= N_2 = n_1 d_2 d_3 + d_1 n_2 d_3 + d_1 d_2 n_3 \\\\

    \textnormal{dim} V_h^3 &= N_3 = d_1 d_2 d_3
    \end{align}

The derivatives act as follows on the FE coefficients:

.. math::
    :nowrap:

    \begin{align}
    \textnormal{grad}\, p_h^0 &= (\mathbb G \mathbf p)^\top \vec{\mathbf \Lambda}^1 \quad \in \,V_h^1 \\\\

    \textnormal{curl}\, \mathbf e_h^1 &= (\mathbb C \mathbf e)^\top \vec{\mathbf \Lambda}^2 \quad \in \,V_h^2 \\\\

    \textnormal{div}\, \mathbf b_h^2 &= (\mathbb D \mathbf b)^\top \mathbf \Lambda^3 \quad \in \,V_h^3
    \end{align}

Here, we introduced the discrete linear operators :math:`\mathbb G: \mathbb R^{N_0} \to \mathbb R^{N_1}`,
:math:`\mathbb C: \mathbb R^{N_1} \to \mathbb R^{N_2}` and :math:`\mathbb D: \mathbb R^{N_2} \to \mathbb R^{N_3}`,
which satisfy the complex property

.. math::

    \mathbb C\, \mathbb G = 0\,,\qquad \mathbb D\, \mathbb C = 0

.. note::

    A struphy userguide for the operators :math:`\mathbb G`, :math:`\mathbb C` and :math:`\mathbb D` and for the projection
    operators :math:`\Pi_n` is given in `this Jupyter notebook <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/notebooks/derham_complex.ipynb>`_.

The projectors :math:`\Pi_n,\, 0\leq n \leq 3` into the discrete spaces :math:`V_h^n,\, 0\leq n \leq 3` are constructed 
such that the commuting relations :math:numref:`commute` hold. They are defined by **Degrees of freedom (DOFs)** 
:math:`\sigma^n,\, 0\leq n \leq 3` which are functionals that map continuous functions to real numbers.
The number of DOFs for each space is equal to its dimension. A common DOF is the evaluation of a function 
at a certain point in :math:`\Omega`; if we choose :math:`N_0` such distinct points and call the corresponding
DOFs :math:`(\sigma^0_{ijk})_{i=1,j=1, k=1}^{n_1,n_2,n_3}`, we can define a projector :math:`\Pi_0:H^1 \to V_h^0` by

.. math::
    :label: dofs

    \sigma^0_{ijk}(p_h^0) = \sigma^0_{ijk}(p^0)\,,\qquad p^0 \in H^1 \mapsto p_h^0 \in V_h^0

These are :math:`N_0` equations for the coefficients :math:`\mathbf p \in \mathbb R^{N_0}` of :math:`p_h^0 \in V_h^0`.
The above formula is nothing else than classical interpolation and is used for the :math:`\Pi_0`-projector in Struphy.
The other projectors are a mix of interpolation and "histopolation" (integration between two interpolation points),
as depicted in Fig. 2. For instance, to project into the space :math:`V_h^1` (1-form), the first component is "histopolated"
in :math:`\eta_1`-direction (orange lines) and interpolated in :math:`(\eta_2, \eta_3)` (blue points);
the second component is "histopolated"
in :math:`\eta_2`-direction and interpolated in :math:`(\eta_1, \eta_3)`;
the third component is "histopolated"
in :math:`\eta_3`-direction and interpolated in :math:`(\eta_1, \eta_2)`.
The DOFs for :math:`V_h^2` contain histopolation over surface elements (green) 
and for :math:`V_h^3` one has 3d histopolation over volume elements (violet). 
This ultimately leads to the commuting property.

.. figure:: ../pics/geometric_dofs.png
    :width: 500
    :align: center

    Fig. 2: Degrees of freedom (DOFs) of the commuting projectors.


.. _uni_variate_spaces:

Uni-variate spline spaces
^^^^^^^^^^^^^^^^^^^^^^^^^

Finite element spaces in Struphy are based on uni-variate B-spline spaces on the interval :math:`\eta \in [0, 1]`,
as introduced for instance in Section 2.2 of 
`Hughes, Cottrell, Bazilevs, "Isogeometric analysis: CAD, finite elements, NURBS, exact geometry and mesh refinement", Comput. Methods
Appl. Mech. Engrg. 194 (2005) <https://www.sciencedirect.com/science/article/pii/S0045782504005171>`_.
B-spline basis functions of degree :math:`p` are denoted by :math:`N_i^p(\eta)` (thus also called **N-splines** in the following). 
They are **piece-wise polynomials** with a regularity of at most :math:`p-1` (they are piece-wise constants for :math:`p=0`).
Linear combinations af these basis functions give a **spline function**

.. math::

    f_h(\eta) = \sum_{i=0}^{n-1} f_i N^p_i(\eta)\,,\qquad f_i \in \mathbb R

The dimension of the spline space is :math:`n`. In Struphy
two different boundary conditions can be chosen: **periodic** or **clamped**. For the latter,
the first and last basis functions are "interpolatory" at :math:`\eta=0` and :math:`\eta=1`, respectively,
meaning that :math:`f_h(0) = f_0` and :math:`f_h(1) = f_{n-1}`.
Assuming that the interval :math:`[0, 1]` is spit into :math:`Nel` cells (or elements), 
the dimension of the periodic space is :math:`n=Nel`, whereas the dimension of the clamped space is :math:`n=Nel + p`.
Spline basis functions of degree :math:`p=2` are potted in Fig. 3 for :math:`Nel=8` cells.

.. figure:: ../pics/splines_1d_N2.png 
    :align: center

    Fig. 3: Uni-variate spline spaces of degree :math:`p=2`.

In Fig. 3 the black crosses are the **break points** (cell interfaces) and the red dots are the **Greville points**,
which are the interpolation points used to define the DOFs for the projectors in :math:numref:`dofs`.
Moreover, we encounter the so-called **D-splines** which are related to the derivatives of N-splines via

.. math::

    \frac{\partial }{\partial \eta} N_i^p = D^{p-1}_{i-1} - D^{p-1}_i

D-splines are rescaled B-splines of degree :math:`p-1`. In the clamped case the dimension :math:`d` of the D-spline space
is thus one less than that of N-splines; in this case :math:`D^{p-1}_{-1} = D^{p-1}_{n-1} = 0`, 
where :math:`n` is the dimension of the N-spline space. 

Fig. 4 shows the uni-variate spline spaces for degree :math:`p=1`. 
Note that in this case the D-Splines are piece-wise constants and a spline function is therefore generally not continuous.

.. figure:: ../pics/splines_1d_N1.png 
    :align: center

    Fig. 4: Uni-variate spline spaces of degree :math:`p=1`.

Higher degree leads to higher smoothness in Struphy, where spline functions have the maximal regularity :math:`p-1`.
Moreover, the support of each basis function is the union of :math:`p+1` adjacent cells and thus increases with the degree.
This allows for high-order methods in Struphy.
See Fig. 5 for an example with :math:`p=4`.

.. figure:: ../pics/splines_1d_N4.png 
    :align: center

    Fig. 4: Uni-variate spline spaces of degree :math:`p=4`.


.. _polar_splines:

Polar splines
^^^^^^^^^^^^^

Coming soon !

.. image:: ../pics/polar_derham.png


.. _disc_example:

Discretization example: Vlasov-Maxwell
---------------------------------------

Coming soon !



