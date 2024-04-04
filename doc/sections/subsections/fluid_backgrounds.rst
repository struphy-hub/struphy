.. _fluid_backgrounds:

Fluid backgrounds
-----------------

Two different fluid backgrounds are available in Struphy, namely ``LogicalConst``
and ``MHD``. The former has the following input structure::

    background: 
        type : LogicalConst
        LogicalConst :
            comps :
                potential_name : 1.3 # scalar-valued variable
                field_name : [.3, .15, null] # vector-valued variable

The only parameter is ``comps``; its keys are the variable names to be initialized
and its values are the constant values **in logical space**. Values can be ``int``,
``float`` or ``null``. Vector-valued variables take a list of length three.

``MHD`` has the following input structure::

    background: 
        type : MHD
        MHD :
            comps :
                0-form_name : absB0 
                1-form_name : b1
                2-form_name : b2
                3-form_name : p3
                vfield_name : bv

The only parameter is ``comps``; its keys are the variable names to be initialized
and its values must be methods (callables) of the base class 
:class:`~struphy.fields_background.mhd_equil.base.MHDequilibrium`.

Check out `this unit test <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/feec/tests/test_field_init.py?ref_type=heads>`_ for more information.