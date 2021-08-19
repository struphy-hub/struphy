#!/bin/bash

# set path to struphy simulations folder
path_root=$(pwd)
all_sim=$path_root/my_struphy_sims
diagn=my_diagnostics

# select which diagnostics to run
file=MHD_spectra_slab.py

# add as many simulations as needed:
sims=()
sims+="$all_sim/sim_1 "
#sims+="$all_sim/sim_2 "
#sims+="$all_sim/sim_3 "
#sims+="$all_sim/sim_4 "

# run diagnostics
python3 $diagn/$file $sims