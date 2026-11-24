.. _gempic:

Numerics
========

Struphy discretizes plasma models with **structure-preserving** methods: the discrete equations keep the
conservation laws, constraints (such as :math:`\nabla \cdot \mathbf B = 0`) and the Hamiltonian or variational
structure of the continuous models, so that long simulations remain stable and physically consistent.
Electromagnetic fields and fluids are discretized with geometric finite elements from finite element exterior calculus
(FEEC) :footcite:p:`Arnold2006`, using spline spaces on a mapped unit cube :footcite:p:`Buffa2011`;
kinetic species are discretized with particle-in-cell (PIC) methods, coupled to the fields in the GEMPIC framework
:footcite:p:`Kraus2017`, which Struphy extends to MHD-kinetic hybrid models
:footcite:p:`Holderied2021,Possanner2023`; fluids can also be discretized with smoothed particle hydrodynamics (SPH)
derived from a Lagrangian :footcite:p:`Price2012`. The references below, and the references therein, describe the
framework.

.. footbibliography::

The following pages introduce these methods: :ref:`particle_discrete` for kinetic equations, :ref:`geomFE` for
fields and fluids, and :ref:`smoothed particle hydrodynamics <sph_method>` as a second option for fluids.
The time integrators are listed in :ref:`time_discret`, and :ref:`disc_example` shows all steps of the discretization
of the Vlasov-Maxwell system as implemented in Struphy.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   subsections/numerics-pic
   subsections/numerics-geomFE
   subsections/numerics-sph
   subsections/numerics-time-discrete
   ../markdown/vlasov-maxwell
