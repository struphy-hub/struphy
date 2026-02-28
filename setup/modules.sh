# ------------------------------------------------------------------------------
# modules.sh
#
# Purpose:
#   Detects the current HPC system and compiler family, then loads or displays
#   the appropriate environment modules for building and running Struphy.
#
# Usage:
#   ./modules.sh [load|display]
#     load    - Loads the required modules for the detected system/compiler
#     display - Prints the module load command for the detected system/compiler
#
# Environment variables:
#   CLUSTER, HPC_SYSTEM, COMPILER_FAMILY
#     Used to detect the machine and compiler type.
#
# Module files:
#   setup/modules.<machine>.sh
#     Should define MODULES_INTEL and MODULES_GCC for each supported system.
# ------------------------------------------------------------------------------
#!/bin/bash

#set -euo pipefail

# Default values
MACHINE="unknown"
MODULES="unknown"

# Detect system
CLUSTER="${CLUSTER:-}"      # Provide default empty if unset
HPC_SYSTEM="${HPC_SYSTEM:-}" # Provide default empty if unset
COMPILER_FAMILY="${COMPILER_FAMILY:-gcc}" # Use gcc compiler by default

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


MODULE_FILE="setup/modules.${MACHINE}.sh"
if [[ -f "$MODULE_FILE" ]]; then
    source "$MODULE_FILE"
else
    echo "Warning: module file $MODULE_FILE not found"
    MODULES_INTEL=""
    MODULES_GCC=""
    exit 1
fi

# Set MODULES to the intel/gcc modules
case "$COMPILER_FAMILY" in
    intel)
        MODULES=$MODULES_INTEL
        ;;
    gcc)
        MODULES=$MODULES_GCC
        ;;
    *)
        echo "Usage: $0 {load|display}"
        exit 1
        ;;
esac

echo $ACTION

case "$ACTION" in
    load)
        echo "Loading modules for $MACHINE, MODULES=$MODULES"
        module purge
        module load $MODULES
	module list 
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
