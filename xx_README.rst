Welcome to STRUPHY!
###################

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` is a multi-model plasma physics package for 
the simulation energetic particles (EPs) in ambient plasma.

The package is developed at `Max Planck Institute for Plasma Physics <https://www.ipp.mpg.de/>`_ 
in the division `NMPP (Numerical Methods for Plasma Physics) <https://www.ipp.mpg.de/ippcms/de/for/bereiche/numerik>`_
and includes

    1. initial-value solvers for kinetic-fluid hybrid models (--> :ref:`models`)
    2. MHD eigenvalue solver for axis-symmetric equilibria
    3. interface to the `MHD equilibrium code GVEC <https://gitlab.mpcdf.mpg.de/gvec-group/gvec>`_
    4. interface to :file:`eqdesk` file format
    5. dispersion relation solvers for MHD, hybrid models and Vlasov-Maxwell (all in slab)

Contact:
    * Stefan Possanner `spossann@ipp.mpg.de <spossann@ipp.mpg.de>`_
    * Florian Holderied `floho@ipp.mpg.de <floho@ipp.mpg.de>`_

For a quick glance at the code: clone the repository, check out `devel` branch, install and display help::
    
    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git
    pip install .
    struphy -h


Manuals
=======

* :ref:`userguide`

* `Developer's guide on the wiki <https://gitlab.mpcdf.mpg.de/clapp/hylife/-/wikis/home>`_


Examples and Tutorials
======================

* :ref:`MHDslab`


.. include:: ../CONTRIBUTORS.rst


References
==========

If you use :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` please cite at least one of the following works:

    * | F. Holderied, S. Possanner, X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell",
      | `J. Comp. Phys. 433 (2021) 110143 <https://www.sciencedirect.com/science/article/pii/S0021999121000358?via%3Dihub>`_

    * | F. Holderied, S. Possanner, "Magneto-hydrodynamic eigenvalue solver for axisymmetric equilibria based on smooth polar splines", 
      | `IPP pinboard <https://users.euro-fusion.org/auth>`_ 



