.. _change_doc:

Changing the documentation 
--------------------------

Struphy docstrings will automatically appear in the documentation through `Sphinx <https://www.sphinx-doc.org/en/master/>`_.

Moreover, there are source files (``.rst``) for the documentation in ``doc/sections/`` and
Tutorial notebooks in ``doc/tutorials/``. 

In order to build the ``.html`` file of the documentation,
`Pandoc <https://pandoc.org/>`_ needs to be installed on your system (for notebook conversion).

Changes to the documentation can be reviewed in the browser (e.g. firefox)::

    cd doc
    make html
    firefox _build/html/index.html

When making further changes, just do ``make html`` and refresh the window in your browser.