#!/bin/bash -e

SO_EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

cd ..
struphy_path=$(python3 -c "import struphy as _; print(_.__path__[0])")

BTS=$struphy_path/eigenvalue_solvers/kernels_projectors_global_mhd

if [[ ! -f $BTS$SO_EXT ]] ; then
    echo 'File' $BTS$SO_EXT 'is not there, aborting.'
    exit 1
elif [[ -f $BTS$SO_EXT ]] ; then
    echo 'Compilation successful.'
fi
