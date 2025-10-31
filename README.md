<h1 align="center">
<img src="https://raw.githubusercontent.com/struphy-hub/.github/refs/heads/main/profile/struphy_header_with_subs.png">
</h1><br>

[![Ubuntu latest](https://github.com/struphy-hub/struphy/actions/workflows/ubuntu-latest.yml/badge.svg)](https://github.com/struphy-hub/struphy/actions/workflows/ubuntu-latest.yml)
[![MacOS latest](https://github.com/struphy-hub/struphy/actions/workflows/macos-latest.yml/badge.svg)](https://github.com/struphy-hub/struphy/actions/workflows/macos-latest.yml)
[![isort and ruff](https://github.com/struphy-hub/struphy/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/struphy-hub/struphy/actions/workflows/static_analysis.yml)
[![PyPI](https://img.shields.io/pypi/v/struphy?label=PyPI)](https://pypi.org/project/struphy/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/struphy.svg?label=PyPI%20downloads)](
https://pypi.org/project/struphy/)
[![Release](https://img.shields.io/github/v/release/struphy-hub/struphy?label=Release)](https://github.com/struphy-hub/struphy/releases)
[![License](https://img.shields.io/badge/License-MIT-violet)](https://github.com/struphy-hub/struphy/blob/devel/LICENSE)

# Welcome!

**This is a Python package for solving partial differential equations (PDEs) mainly - but not exclusively - for plasma physics.**

**STRUPHY** stands for "**STRU**cture in **PHY**sics" or "**STRU**cture-**P**reserving **HY**brid codes". The package provides off-the-shelf physics models for plasma physics problems, such as

* Maxwell's equations
* Magneto-hydrodynamics (MHD)
* Vlasov-Poisson and Vlasov-Maxwell kinetic models
* Drift-kinetic models for strongly magnetized plasma
* MHD-kinetic hybrid models 

All models can be run on multiple cores through MPI (distributed memory) and OpenMP (shared memory). The compute-intensive parts of the code are translated and compiled ("transpiled") using [pyccel](https://github.com/pyccel/pyccel), giving you the speed of Fortran or C while working within the familiar Python environment. 

The code is freely available under an [MIT license](https://github.com/struphy-hub/struphy/blob/devel/LICENSE) - Copyright (c) 2019-2025, Struphy developers, Max Planck Institute for Plasma Physics.

<h1 align="center">
<img src="https://raw.githubusercontent.com/struphy-hub/.github/refs/heads/main/profile/MPI_PP_Logo_Vertical_E_green_rgb.png" width="200">
</h1>

## Tutorials

Get familiar with Struphy right away on [mybinder](https://mybinder.org/v2/gh/struphy-hub/struphy-tutorials/main) - no installation needed.


## Quick install

Quick install on your computer (using a virtual environment):

```
python -m venv struphy_env
source struphy_env/bin/activate
pip install -U pip
pip install -U struphy
struphy compile --status
struphy compile
```

In case you face troubles with install/compile:

1. check the [requirements](https://struphy-hub.github.io/struphy/sections/install.html#requirements)
2. visit [trouble shooting](https://struphy-hub.github.io/struphy/sections/install.html#trouble-shooting)


## Quick run

As an example, let's say we want to solve Maxwell's equations. We can use the CLI and generate a default launch file via

```
struphy params Maxwell
```
Hit yes when prompted - this will create the file `params_Maxwell.py` in your current working directory (cwd). You can open the file and - if you feel like it already - change some parameters, then run

```
python params_Maxwell.py
```

The default output is in `sim/` in your cwd. You can change the output path via the class `EnvironmentOptions` in the parameter file.

Parallel simulations are run for example with

```
pip install -U mpi4py
mpirun -n 4 python params_Maxwell.py
```

You can also put the run command in a batch script.


## Documentation

The doc is on [Github pages](https://struphy-hub.github.io/struphy/index.html), we recommend in particular to visit:

* [Install](https://struphy-hub.github.io/struphy/sections/install.html)
* [Userguide](https://struphy-hub.github.io/struphy/sections/userguide.html)
* [Available models](https://struphy-hub.github.io/struphy/sections/models.html)
* [Numerical methods](https://struphy-hub.github.io/struphy/sections/numerics.html)


## Get in touch

* [Issues](https://github.com/struphy-hub/struphy/issues)
* [Discussions](https://github.com/struphy-hub/struphy/discussions)
* @spossann [stefan.possanner@ipp.mpg.de](mailto:spossann@ipp.mpg.de) (Maintainer)
* @max-models [Max.Lindqvist@ipp.mpg.de](mailto:Max.Lindqvist@ipp.mpg.de) (Maintainer)
* [LinkedIn profile](https://www.linkedin.com/company/struphy/)


## Citing Struphy

* S. Possanner, F. Holderied, Y. Li, B.-K. Na, D. Bell, S. Hadjout and Y. Güçlü, [**High-Order Structure-Preserving Algorithms for Plasma Hybrid Models**](https://link.springer.com/chapter/10.1007/978-3-031-38299-4_28), International Conference on Geometric Science of Information 2023, 263-271, Springer Nature Switzerland.

