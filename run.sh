#!/bin/bash


path_root=$(pwd)
all_sim=simulations    
run_dir=example_analytical

cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

#############################
##### grid construction #####
#############################

# mesh generation on logical domain
Nel : [16, 16, 2] 

# boundary conditions (True: periodic, False: else)
bc : [True, True, True]

# spline degrees
p : [2, 2, 1] 

# number of quadrature points per element
nq_el : [6, 6, 2]

# number of quadrature points per histopolation cell
nq_pr : [6, 6, 2]

# do time integration?
time_int : True

# time step
dt : .05

# simulation time
Tend : 0.1

# maximum runtime of program in minutes
max_time : 3600

# add non-Hamiltonian terms to simulation?
add_pressure : False

# geometry 
# 1: slab
# 2: hollow cylinder
# 3: colella
kind_map : 1

# parameters for mapping 
params_map : [20., 1., 1.]        
        
# adiabatic exponent
gamma : 1.6666666666666666666666666666                 

###############################
##### particle parameters #####
###############################

# add kinetic terms to simulation?
add_PIC : True     

# total number of particles
Np : 125000              

# control variate? 
control : False       

# shift of Maxwellian 
v0 : [2.5, 0., 0.]

# hot ion thermal velocity
vth : 1.

# particle loading
loading : pseudo-random

# Is this run a restart?
restart : False

# number of restart files
num_restart : 0

# Create restart files at the end of the simulation? 
create_restart : True

# initial conditions 
ic_from_params : False
EOF

echo "Your hylife repository is here:" $path_root
echo "Your simulations are here:     " $path_root/$all_sim
echo "Your current run is here:      " $path_root/$all_sim/$run_dir

# interface
cp hylife/interface_original_analytical.py hylife/interface_analytical.py

var0="s|sed_replace_path_root|"
var1="s|sed_replace_all_sim|"
var2="s|sed_replace_run_dir|"
var3="|g" 

sed -i $var1$all_sim$var3 hylife/interface_analytical.py
sed -i $var2$run_dir$var3 hylife/interface_analytical.py

# makefile
cp Makefile $all_sim/$run_dir/Makefile

make all_sim=$all_sim run_dir=$run_dir

# main code
cp STRUPHY_original.py $all_sim/$run_dir/STRUPHY.py

sed -i $var0$path_root$var3 $all_sim/$run_dir/STRUPHY.py
sed -i $var2$run_dir$var3 $all_sim/$run_dir/STRUPHY.py


# run the code
cd $all_sim/$run_dir
python3 STRUPHY.py
