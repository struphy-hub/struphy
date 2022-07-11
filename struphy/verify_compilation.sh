#!/bin/bash -e

SO_EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Struphy path
STRUPHY=
li=$(pip show struphy)
#echo $li
take=false
for i in $li; do

    if [ "$take" = "true" ]; then
        #echo $take
        STRUPHY=$i/struphy
        break
    fi
    #echo "$i"
    if [ "$i" = "Location:" ]; then
        #echo "$i"
        take=true
    fi
done

BTS=$STRUPHY/psydac_api/banded_to_stencil_kernels

if [[ ! -f $BTS$SO_EXT ]] ; then
    echo 'File' $BTS$SO_EXT 'is not there, aborting.'
    exit 1
elif [[ -f $BTS$SO_EXT ]] ; then
    echo 'Compilation successful.'
fi
