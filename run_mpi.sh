#!/bin/bash

# set simulation folders
path_root=$(pwd)
all_sim=$HOME/ptmp_link/simulations   
run_dir=example_node_2_np_128000000

#TODO: remove results.hdf5 file
rm $all_sim/$run_dir/results_$run_dir.hdf5

export PYTHONPATH="${PYTHONPATH}:$all_sim/$run_dir"
export PYTHONPATH="${PYTHONPATH}:$path_root"

echo $PYTHONPATH

# set parameters
cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

#############################
##### grid construction #####
#############################

# mesh generation on logical domain
Nel : [16, 2, 2] 

# boundary conditions (True: periodic, False: else)
bc : [True, True, True]

# spline degrees
p : [2, 1, 1] 

# number of quadrature points per element
nq_el : [6, 2, 2]

# number of quadrature points per histopolation cell
nq_pr : [6, 2, 2]

# do time integration?
time_int : True

# time step
dt : .05

# simulation time
Tend : 10.

# maximum runtime of program in minutes
max_time : 1000.

# add non-Hamiltonian terms to simulation?
add_pressure : False

# geometry 
# 1: slab
# 2: hollow cylinder
# 3: colella
kind_map : 1

# parameters for mapping 
params_map : [1., 1., 1.]        
        
# adiabatic exponent
gamma : 1.6666666666666666666666666666                 

###############################
##### particle parameters #####
###############################

# add kinetic terms to simulation?
add_PIC : True     

# total number of particles
Np : 1280000000             

# control variate? 
control : False       

# shift of Maxwellian 
v0 : [2.5, 0., 0.]

# hot ion thermal velocity
vth : 1.

# particle loading
loading : pseudo-random

###############################
##### restart function ########
###############################

# Is this run a restart?
restart : False

# number of restart files
num_restart : 0

# Create restart files at the end of the simulation? 
create_restart : False

EOF

# print location of simulation
echo "Your hylife repository is here:" $path_root
echo "Your simulations are here:     " $all_sim
echo "Your current run is here:      " $all_sim/$run_dir

var1="s|sed_replace_run_dir|"
var2="|g" 


# copy subroutines and replace import paths to current simulation
SDIR=$all_sim/$run_dir/source_run

mkdir $SDIR

cp hylife/utilitis_FEEC/kernels_control_variate.py $SDIR/kernels_control_variate.py
cp hylife/utilitis_FEEC/projectors/kernels_projectors_local_eva_ana.py $SDIR/kernels_projectors_local_eva_ana.py
cp hylife/utilitis_PIC/fields.py $SDIR/fields.py
cp hylife/utilitis_PIC/sampling.py $SDIR/sampling.py

cp hylife/utilitis_FEEC/control_variate.py $SDIR/control_variate.py
cp hylife/utilitis_FEEC/projectors/projectors_local.py $SDIR/projectors_local.py
cp hylife/utilitis_FEEC/projectors/projectors_local_mhd.py $SDIR/projectors_local_mhd.py

#sed -i $var1$all_sim$var3 $all_sim/$run_dir/source_run/kernels_control_variate.py
#sed -i $var2$run_dir$var3 $all_sim/$run_dir/source_run/kernels_control_variate.py

#sed -i $var1$all_sim$var3 $all_sim/$run_dir/source_run/kernels_projectors_local_eva_ana.py
#sed -i $var2$run_dir$var3 $all_sim/$run_dir/source_run/kernels_projectors_local_eva_ana.py

#sed -i $var1$all_sim$var3 $all_sim/$run_dir/source_run/fields.py
#sed -i $var2$run_dir$var3 $all_sim/$run_dir/source_run/fields.py

#sed -i $var1$all_sim$var3 $all_sim/$run_dir/source_run/sampling.py
#sed -i $var2$run_dir$var3 $all_sim/$run_dir/source_run/sampling.py

# run Makefile
#make all_sim=$all_sim run_dir=$run_dir

# copy main code and adjust to current simulation
cp STRUPHY_mpi_original.py $all_sim/$run_dir/STRUPHY_mpi.py
cp batch_draco_mpi.sh $all_sim/$run_dir/.

sed -i $var1$run_dir$var2 $all_sim/$run_dir/STRUPHY_mpi.py

# run the code
cd $all_sim/$run_dir

sbatch batch_draco_mpi.sh

#export OMP_NUM_THREADS=2
#export OMP_PLACES=cores 
#srun -n 4 python3 STRUPHY_mpi.py
#python3 STRUPHY_mpi.py

#make clean all_sim=$all_sim run_dir=$run_dir