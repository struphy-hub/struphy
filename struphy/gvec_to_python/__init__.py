# Specify __all__ variable to enable `from gvec_to_python import *`.
# Source: https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
__all__ = ['base', 'hmap', 'struphy', 'reader', 'util', 'writer', 'GVEC_functions', 'GVEC', 'Form', 'Variable']

# Add `gvec_to_python` to path to allow imports from within `struphy. repository.
import os, sys
basedir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(basedir, '..'))

# Because __name__ at package root is always '__main__', relative imports are not allowed.
# => Use ONLY absolute import.
# Source: https://docs.python.org/3/tutorial/modules.html#intra-package-references
from gvec_to_python import base
from gvec_to_python import hmap
from gvec_to_python import hylife
from gvec_to_python import reader
from gvec_to_python import util
from gvec_to_python import writer
from gvec_to_python import GVEC_functions
from gvec_to_python.GVEC_functions import GVEC, Form, Variable
