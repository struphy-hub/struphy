# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

with open("../src/struphy/console/main.py") as f:
    exec(f.read())

# -- Project information -----------------------------------------------------

project = "struphy"
copyright = "2019-2025 (c) Struphy dev team | Max Planck Institute for Plasma Physics"
author = "Struphy dev team | Max Planck Institute for Plasma Physics"
version = __version__

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.graphviz",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

nbsphinx_execute = "auto"

napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_attr_annotations = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'classic'
# html_theme = 'press'
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "sidebarwidth": 270,
    "show_nav_level": 3,
    "show_toc_level": 1,
    "navigation_depth": 4,
    "header_links_before_dropdown": 8,
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "external_links": [
        {"name": "Struphy repo", "url": "https://github.com/struphy-hub/struphy"},
        {"name": "Struphy LinkedIn", "url": "https://www.linkedin.com/company/struphy/"},
        {
            "name": "Struphy MatrixChat",
            "url": "https://matrix.to/#/!wqjcJpsUvAbTPOUXen:mpg.de?via=mpg.de&via=academiccloud.de",
        },
    ],
}

# html_theme_options = {
#     "rightsidebar": "false",
#     "stickysidebar": "true",
#     "footerbgcolor": "Coral",
#     "externalrefs": "true",
#     #"body_min_width": 800,
# }

html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "searchbox.html"],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

# other
highlight_language = "none"

autodoc_member_order = "bysource"

html_logo = "dog-cartoon-struphy.jpg"
# html_theme_options = {
#     'display_version': True,
#     'style_external_links': True,
# }

# inheritance diagrams
inheritance_graph_attrs = dict(rankdir="LR", ratio="auto", size='"4.0, 20.0"', fontsize="8", resolution=300.0)

inheritance_node_attrs = dict(shape="ellipse", fontsize="8", height=0.25, color="maroon4", style="filled")

# markdown parsing
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_dmath_allow_labels = True
