#!/bin/bash

# Default values
MACHINE="unknown"
MODULES="unknown"

# Detect system
case "$CLUSTER" in
    TOK)
        MACHINE="tok"
        MODULES="gcc/14 openmpi/5.0 python-waterboa/2025.06 cmake/4.0 netcdf-serial/4.9.2 mkl/2025.1 hdf5-serial/2.0.0"
        ;;
    RAVEN)
        MACHINE="raven"
        MODULES="gcc/14 openmpi/5.0 python-waterboa/2025.06 cmake/4.0 netcdf-serial/4.9.2 mkl/2025.3 hdf5-serial/2.0.0"
        ;;
    VIPER)
        MACHINE="viper"
        MODULES="gcc/14 openmpi/5.0 python-waterboa/2025.06 cmake/4.0 netcdf-serial/4.9.2 mkl/2025.3 hdf5-serial/2.0.0"
        ;;
esac

if [[ "$HPC_SYSTEM" == *"pitagora"* ]]; then
    MACHINE="pitagora"
    MODULES="intel-oneapi-compilers/2024.1.0 intel-oneapi-mpi/2021.12.1"
fi

# Handle arguments
ACTION="${1:-display}"  # Default to 'display' if no argument is given

case "$ACTION" in
    load)
        echo "Loading modules for $MACHINE, MODULES=$MODULES"
        #module purge
        module load $MODULES
	module list
	echo 
        ;;
    display)
        #echo "MACHINE=$MACHINE"
        echo "module load $MODULES"
        ;;
    *)
        echo "Usage: $0 {load|display}"
        exit 1
        ;;
esac
