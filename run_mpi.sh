#!/bin/bash

# ============== set simulation folders ===========
path_root=$(pwd)
all_sim=$HOME/Schreibtisch/PHD/02_Projekte/simulations_hylife   
run_dir=sim_2020_08_19_1
# =================================================

# ============== if you want to use OpenMp ========
#flag_openmp=
flag_openmp=--openmp
# =================================================


# == print location of repository and simulation == 
echo "Your hylife repository is here:" $path_root
echo "Your simulations are here:     " $all_sim
echo "Your current run is here:      " $all_sim/$run_dir
# =================================================

# ============ add paths to python ================
export PYTHONPATH="${PYTHONPATH}:$path_root"
export PYTHONPATH="${PYTHONPATH}:$all_sim/$run_dir"

echo $PYTHONPATH
# =================================================

# ========== clean simulation folder ==============
rm $all_sim/$run_dir/STRUPHY.py
#rm $all_sim/$run_dir/*.hdf5
rm $all_sim/$run_dir/sim*.*
rm $all_sim/$run_dir/batch*.*
# =================================================


# ====== set parameters and create .yml file ======
cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

#############################
##### grid construction #####
#############################

# number of elements, boundary conditions and spline degrees
Nel : [16, 16, 2] 
bc  : [True, True, True]
p   : [2, 2, 1] 


# number of quadrature points per element and histopolation cell
nq_el : [6, 6, 2]
nq_pr : [6, 6, 2]


# do time integration?, time step, simulation time and maximum runtime of program (in minutes)
time_int : True
dt       : .05
Tend     : 1.
max_time : 1000.


# add non-Hamiltonian terms to simulation?
add_pressure : False

# geometry (1: slab, 2: hollow cylinder, 3: colella) and parameters for geometry
kind_map   : 3
params_map : [7.853981634, 7.853981634, 0.1, 1.]        
        
# adiabatic exponent
gamma : 1.6666666666666666666666666666                 

###############################
##### particle parameters #####
###############################

# add kinetic terms to simulation?
add_PIC : True     

# total number of particles
Np : 512000           

# control variate? 
control : True       

# shift of Maxwellian 
v0 : [2.5, 0., 0.]

# hot ion thermal velocity
vth : 1.

# particle loading
loading : pseudo-random

# seed for random number generator
seed : 1234

# directory of particles if loaded externally
dir_particles : /home/florian/Schreibtisch/PHD/02_Projekte/hylife/hylife_florian/simulations/reference_colella/results_reference_colella.hdf5

###############################
##### restart function ########
###############################

# Is this run a restart?
restart : True

# number of restart files
num_restart : 0

# Create restart files at the end of the simulation? 
create_restart : True

EOF
# =================================================


# == create source_run folder and copy subroutines into it
SDIR=$all_sim/$run_dir/source_run

mkdir $SDIR

cp hylife/utilitis_FEEC/kernels_control_variate.py $SDIR/kernels_control_variate.py
cp hylife/utilitis_FEEC/projectors/kernels_projectors_local_eva_ana.py $SDIR/kernels_projectors_local_eva_ana.py
#cp hylife/utilitis_PIC/fields.py $SDIR/fields.py

cp hylife/utilitis_PIC/pusher.py $SDIR/pusher.py
cp hylife/utilitis_PIC/accumulation_kernels.py $SDIR/accumulation_kernels.py
cp hylife/utilitis_PIC/sampling.py $SDIR/sampling.py

cp hylife/utilitis_FEEC/control_variate.py $SDIR/control_variate.py
cp hylife/utilitis_FEEC/projectors/projectors_local.py $SDIR/projectors_local.py
cp hylife/utilitis_FEEC/projectors/projectors_local_mhd.py $SDIR/projectors_local_mhd.py
# =================================================


# ================= run Makefile ==================
#make all_sim=$all_sim run_dir=$run_dir flags_openmp=$flag_openmp
# =================================================


# copy main code and adjust to current simulation =
var1="s|sed_replace_run_dir|"
var2="|g"

cp STRUPHY_mpi_original.py $all_sim/$run_dir/STRUPHY.py
#cp batch_draco_mpi.sh $all_sim/$run_dir/.
#cp batch_draco_mpi_openmp.sh $all_sim/$run_dir/.

sed -i $var1$run_dir$var2 $all_sim/$run_dir/STRUPHY.py
# =================================================


# ================== run the code =================
cd $all_sim/$run_dir

# job submission
#sbatch batch_draco_mpi.sh
#sbatch batch_draco_mpi_openmp.sh

# interactive run
#export OMP_NUM_THREADS=2
#export OMP_PLACES=cores 
#srun -n 1 python3 STRUPHY_mpi.py
#python3 STRUPHY_mpi.py

mpirun -n 4 python3 STRUPHY.py
# =================================================