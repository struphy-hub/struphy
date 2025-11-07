.. _gempic:

Numerics
========

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

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   subsections/numerics-pic
   subsections/numerics-geomFE
   subsections/numerics-sph
   subsections/numerics-time-discrete
   ../markdown/vlasov-maxwell







