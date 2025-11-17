#!/bin/bash -e

SO_EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

cd ..
struphy_path=$(python3 -c "import struphy as _; print(_.__path__[0])")
