#!/bin/bash

# ============== set simulation folders ===========
path_root=$(pwd)
all_sim=$HOME/ptmp_link/simulations   
run_dir=example_node_1_np_6400000
# =================================================

#TODO: remove results.hdf5 file
rm $all_sim/$run_dir/results_$run_dir.hdf5



# ============ add paths to python ================
export PYTHONPATH="${PYTHONPATH}:$path_root"
export PYTHONPATH="${PYTHONPATH}:$all_sim/$run_dir"

echo $PYTHONPATH
# =================================================


# ====== set parameters and create .yml file ======
cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

#############################
##### grid construction #####
#############################

# mesh generation on logical domain
Nel : [16, 16, 16] 

# boundary conditions (True: periodic, False: else)
bc : [True, True, True]

# spline degrees
p : [3, 3, 3] 

# number of quadrature points per element
nq_el : [6, 6, 6]

# number of quadrature points per histopolation cell
nq_pr : [6, 6, 6]

# do time integration?
time_int : True

# time step
dt : .05

# simulation time
Tend : 1.

# maximum runtime of program in minutes
max_time : 1000.

# add non-Hamiltonian terms to simulation?
add_pressure : False

# geometry (1: slab, 2: hollow cylinder, 3: colella)
kind_map : 3

# parameters for mapping 
params_map : [7.853981634, 7.853981634, 0.1, 7.853981634]        
        
# adiabatic exponent
gamma : 1.6666666666666666666666666666                 

###############################
##### particle parameters #####
###############################

# add kinetic terms to simulation?
add_PIC : True     

# total number of particles
Np : 5120000             

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
create_restart : True

EOF
# =================================================


# == print location of repository and simulation == 
echo "Your hylife repository is here:" $path_root
echo "Your simulations are here:     " $all_sim
echo "Your current run is here:      " $all_sim/$run_dir
# =================================================


# == create source_run folder and copy subrouotines into it
SDIR=$all_sim/$run_dir/source_run

mkdir $SDIR

cp hylife/utilitis_FEEC/kernels_control_variate.py $SDIR/kernels_control_variate.py
cp hylife/utilitis_FEEC/projectors/kernels_projectors_local_eva_ana.py $SDIR/kernels_projectors_local_eva_ana.py
cp hylife/utilitis_PIC/fields.py $SDIR/fields.py
cp hylife/utilitis_PIC/sampling.py $SDIR/sampling.py

cp hylife/utilitis_FEEC/control_variate.py $SDIR/control_variate.py
cp hylife/utilitis_FEEC/projectors/projectors_local.py $SDIR/projectors_local.py
cp hylife/utilitis_FEEC/projectors/projectors_local_mhd.py $SDIR/projectors_local_mhd.py
# =================================================


# ================= run Makefile ==================
#make all_sim=$all_sim run_dir=$run_dir
# =================================================



# copy main code and adjust to current simulation =
var1="s|sed_replace_run_dir|"
var2="|g"

cp STRUPHY_mpi_original.py $all_sim/$run_dir/STRUPHY_mpi.py
#cp batch_draco_mpi.sh $all_sim/$run_dir/.
cp batch_draco_mpi_openmp.sh $all_sim/$run_dir/.

sed -i $var1$run_dir$var2 $all_sim/$run_dir/STRUPHY_mpi.py
# =================================================


# ================== run the code =================
cd $all_sim/$run_dir

# job submission
#sbatch batch_draco_mpi.sh
sbatch batch_draco_mpi_openmp.sh

# interactive run
#export OMP_NUM_THREADS=2
#export OMP_PLACES=cores 
#srun -n 1 python3 STRUPHY_mpi.py
#python3 STRUPHY_mpi.py
# =================================================