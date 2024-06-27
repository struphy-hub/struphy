.. _mhd_equil:

MHD equilibria
--------------

Magnetized plasma equilibria are often the starting point of dynamical plasma simulations.
In Struphy, there are many options to provide MHD equilibria, for axis-symmetric (Tokamak)
as well as for non-axis-symmetric (Stellarator) configurations. In particular, there exist interfaces 
for reading `EQDSK <https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf>`_ and 
`GVEC <https://gitlab.mpcdf.mpg.de/gvec-group/gvec_to_python>`_ equilibrium data.
The following inheritance diagram shows the MHD equilibria available in Struphy:

.. inheritance-diagram:: struphy.fields_background.mhd_equil.equils
    :parts: 1

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    mhd_equils_sub





