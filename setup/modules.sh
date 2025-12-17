#!/bin/bash

# Default values
MACHINE="unknown"
MODULES="unknown"

# Detect system
case "$CLUSTER" in
    TOK)
        MACHINE="tok"
        ;;
    RAVEN)
        MACHINE="raven"
        ;;
    VIPER)
        MACHINE="viper"
        ;;
esac

if [[ "$HPC_SYSTEM" == *"pitagora"* ]]; then
    MACHINE="pitagora"
fi

# Handle arguments
ACTION="${1:-display}"  # Default to 'display' if no argument is given

source setup/modules.${MACHINE}.sh

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
