.. _gempic:

Numerics
========

Struphy discretization is performed according to the GEMPIC (Geometric Electro-Magnetic Particle-In-Cell) framework.
Relevant publications are (and the references therein):

1. the GEMPIC framework :footcite:p:`Kraus2017`;
2. isogeometric discrete differential forms (the finite element spaces of Struphy) :footcite:p:`Buffa2011`;
3. an MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell
   :footcite:p:`Holderied2021`;
4. smoothed particle hydrodynamics (SPH) for fluids, derived from a Lagrangian :footcite:p:`Price2012`.

In Struphy, kinetic equations are discretized with the :ref:`particle_discrete`. Equations for electromagnetic fields
and/or fluid species are discretized with :ref:`geomFE`, and fluids can also be discretized with
:ref:`smoothed particle hydrodynamics <sph_method>`. In what follows we provide a brief introduction to these methods.
In section :ref:`disc_example` we detail the discretization of the Vlasov-Maxwell system implemented in Struphy.

.. footbibliography::

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   subsections/numerics-pic
   subsections/numerics-geomFE
   subsections/numerics-sph
   subsections/numerics-time-discrete
   ../markdown/vlasov-maxwell
