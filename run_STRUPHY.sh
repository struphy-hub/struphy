#!/bin/bash

# ============== set simulation folders ===========
path_root=$(pwd)
all_sim=$HOME/STRUPHY_simulations
run_dir=tests
# =================================================

# ============== if you want to use OpenMp ========
flag_openmp_mhd=--openmp
flag_openmp_pic=--openmp
# =================================================

# ========== analytical or discrete mapping ? =====
mapping=analytical
#mapping=discrete
# =================================================

# == print location of repository and simulation == 
echo "Your hylife repository is here:" $path_root
echo "Your simulations are here:     " $all_sim
echo "Your current run is here:      " $all_sim/$run_dir
# =================================================

# ============ add paths to python ================
export PYTHONPATH="${PYTHONPATH}:$path_root"
export PYTHONPATH="${PYTHONPATH}:$all_sim/$run_dir"
# =================================================

# ========== clean simulation folder ==============
rm -f $all_sim/$run_dir/STRUPHY.py
rm -f $all_sim/$run_dir/*.hdf5
rm -f $all_sim/$run_dir/sim*.*
rm -f $all_sim/$run_dir/batch*.*
# =================================================


# ====== set parameters and create .yml file ======
cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

#############################
##### grid construction #####
#############################

# number of elements, boundary conditions and spline degrees (finite elements)
Nel : [16, 16, 2] 
bc  : [True, True, True]
p   : [2, 2, 1] 

# number of quadrature points per element and histopolation cell
nq_el : [6, 6, 2]
nq_pr : [6, 6, 2]

# analytical (0) or spline mapping (1)?
mapping : 0

# ----> for analytical geometry: kind of mapping (1: slab, 2: hollow cylinder, 3: colella) and parameters
kind_map   : 1
params_map : [7.853981634, 7.853981634, 1.]        

# ----> for spline geometry: number of elements, boundary conditions and spline degrees
Nel_F : [16, 16, 2] 
bc_F  : [False, False, False]
p_F   : [2, 2, 1] 


#############################
##### time integration ######
#############################

# do time integration?, time step, simulation time and maximum runtime of program (in minutes)
time_int : True
dt       : 0.05
Tend     : 1.
max_time : 1000.


###############################
##### linear solvers ##########
###############################

# ILUs (default: drop_tol=1e-4, fill_fac=10.)
drop_tol_S2 : 0.0001
fill_fac_S2 : 10.

drop_tol_A  : 0.0001
fill_fac_A  : 10.

drop_tol_M0 : 0.0001
fill_fac_M0 : 10.

# tolerances for iterative solvers (default: tol=1e-8)
tol1        : 0.00000001
tol2        : 0.00000001
tol3        : 0.00000001
tol6        : 0.00000001


###############################
####### MHD parameters ########
###############################

# add non-Hamiltonian terms to simulation?
add_pressure : False

# adiabatic exponent
gamma : 1.6666666666666666666666666666

###############################
##### kinetic parameters ######
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

# seed for random number generator
seed : 1234

# directory of particles if loaded externally
dir_particles : path_to_particles

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



# ========= set parameters for batch script =======
cat >$all_sim/$run_dir/batch_$run_dir.sh <<'EOF'
#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./sim.out
#SBATCH -e ./sim.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J test_struphy
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
# for OpenMP:
#SBATCH --cpus-per-task=16
#
#SBATCH --mail-type=none
#SBATCH --mail-user=<userid>@rzg.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:20:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
#export OMP_PLACES=cores 

# Run the program:
srun python3 STRUPHY.py > test.out
EOF
# =================================================



# == create source_run folder and copy subroutines into it
SDIR=$all_sim/$run_dir/source_run

mkdir -p $SDIR

cp hylife/utilitis_FEEC/projectors/projectors_local.py $SDIR/projectors_local.py
cp hylife/utilitis_FEEC/projectors/projectors_local_mhd.py $SDIR/projectors_local_mhd.py
cp hylife/utilitis_FEEC/control_variates/control_variate.py $SDIR/control_variate.py

if [ "$mapping" = "analytical" ]
then
cp hylife/utilitis_FEEC/control_variates/kernels_cv_analytical.py $SDIR/kernels_control_variate.py
cp hylife/utilitis_FEEC/projectors/kernels_projectors_local_eva_ana.py $SDIR/kernels_projectors_local_eva.py

cp hylife/utilitis_PIC/pusher.py $SDIR/pusher.py
cp hylife/utilitis_PIC/accumulation_kernels.py $SDIR/accumulation_kernels.py
cp hylife/utilitis_PIC/sampling.py $SDIR/sampling.py


elif [ "$mapping" = "discrete" ]
then
cp hylife/utilitis_FEEC/control_variates/kernels_cv_discrete.py $SDIR/kernels_control_variate.py
cp hylife/utilitis_FEEC/projectors/kernels_projectors_local_eva_dis.py $SDIR/kernels_projectors_local_eva.py

cp hylife/utilitis_PIC/discrete_mapping/pusher.py $SDIR/pusher.py
cp hylife/utilitis_PIC/discrete_mapping/accumulation_kernels.py $SDIR/accumulation_kernels.py
cp hylife/utilitis_PIC/discrete_mapping/sampling.py $SDIR/sampling.py

fi
# =================================================


# ============== run Makefile =====================
make all_sim=$all_sim run_dir=$run_dir flags_openmp_mhd=$flag_openmp_mhd flags_openmp_pic=$flag_openmp_pic
# =================================================


# copy main code and adjust to current simulation =
var1="s|sed_replace_run_dir|"
var2="|g"

cp STRUPHY_original.py $all_sim/$run_dir/STRUPHY.py

sed -i -e $var1$run_dir$var2 $all_sim/$run_dir/STRUPHY.py
# =================================================


# ================== run the code =================
cd $all_sim/$run_dir

# job submission via SLURM
#sbatch batch_$run_dir.sh

# interactive run on an interactive node on e.g. Draco or Cobra (indicate number of MPI processes after -n)
#export OMP_NUM_THREADS=4
#export OMP_PLACES=cores
#srun -n 1 python3 STRUPHY.py

# for run on a local machine (indicate number of MPI processes after -n)
#mpirun -n 4 python3 STRUPHY.py
export OMP_NUM_THREADS=1
python3 STRUPHY.py
# =================================================
