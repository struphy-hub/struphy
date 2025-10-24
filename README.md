[![Testing](https://github.com/struphy-hub/struphy/actions/workflows/testing.yml/badge.svg)](https://github.com/struphy-hub/struphy/actions/workflows/testing.yml)
[![Testing](https://github.com/struphy-hub/struphy/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/struphy-hub/struphy/actions/workflows/static_analysis.yml)
![PyPI](https://img.shields.io/pypi/v/struphy?label=pypi%20package)
[![PyPI Downloads](https://img.shields.io/pypi/dm/struphy.svg?label=PyPI%20downloads)](
https://pypi.org/project/struphy/)
![Release](https://img.shields.io/github/v/release/struphy-hub/struphy)
![License](https://img.shields.io/badge/License-MIT-violet)

# Welcome!

Struphy is a Python package for plasma physics PDEs.

Join the [Struphy mailing list](https://listserv.gwdg.de/mailman/listinfo/struphy) and stay informed on updates.

## Documentation
See the [Struphy pages](https://struphy.pages.mpcdf.de/struphy/index.html) for details regarding installation, tutorials, use, and development.

## Quick install

Use a virtual environment:

    python3 -m pip install --upgrade virtualenv
    python3 -m venv struphy_env
    source struphy_env/bin/activate

Install latest release:

    pip install --no-cache-dir --upgrade struphy

Compile kernels:

    struphy compile

Quick help:

    struphy -h

In case of problems visit [Trouble shooting](https://struphy.pages.mpcdf.de/struphy/sections/install.html#trouble-shooting).

## Run tests from the command-line

Run available verification tests for [Struphy models](https://struphy.pages.mpcdf.de/struphy/sections/models.html):

    struphy test models --verification --fast --show-plots 

The corresponding parameter files are in [struphy/io/inp/verification/](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/tree/devel/src/struphy/io/inp/verification).
The corresponding diagnostics functions are in [struphy/models/tests/verification.py](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/models/tests/verification.py). You can repeat the verification run of a single `<model_name>` by typing

    struphy test <model_name> --verification --fast --show-plots 

## Tutorial notebooks

Struphy tutorials are available in the form of [Jupyter notebooks](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/tree/devel/doc/tutorials).  

## Reference paper

* S. Possanner, F. Holderied, Y. Li, B.-K. Na, D. Bell, S. Hadjout and Y. Güçlü, [**High-Order Structure-Preserving Algorithms for Plasma Hybrid Models**](https://link.springer.com/chapter/10.1007/978-3-031-38299-4_28), International Conference on Geometric Science of Information 2023, 263-271, Springer Nature Switzerland.

## Contact

* Stefan Possanner [stefan.possanner@ipp.mpg.de](mailto:spossann@ipp.mpg.de)
* Eric Sonnendrücker [eric.sonnendruecker@ipp.mpg.de](mailto:eric.sonnendruecker@ipp.mpg.de)
* Xin Wang [xin.wang@ipp.mpg.de](mailto:xin.wang@ipp.mpg.de)
