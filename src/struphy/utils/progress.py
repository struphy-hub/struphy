"""Progress bars that write to stdout.

``tqdm`` writes to ``sys.stderr`` by default, which pollutes the ``.err`` files of
batch jobs on clusters with normal progress output. Import ``tqdm`` from here instead
of from the ``tqdm`` package to send the bars to stdout.
"""

import functools
import sys

from tqdm import tqdm as _tqdm

tqdm = functools.partial(_tqdm, file=sys.stdout)
