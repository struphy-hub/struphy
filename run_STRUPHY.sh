#!/bin/bash

# ============== set simulation folders ===========
path_root=$(pwd)
#all_sim=/home/florian/Schreibtisch/PHD/02_Projekte/simulations_hylife/particle_pusher_2020_12
all_sim=/home/florian/Schreibtisch/PHD/02_Projekte/simulations_hylife
run_dir=sim_2021_03_03_2
# =================================================

# ======= name of main code =======================
#code_name=STRUPHY_original.py
code_name=STRUPHY_original_new.py
# =================================================

# ============== if you want to use OpenMp ========
flag_openmp_mhd=
flag_openmp_pic=--openmp
# =================================================

# ======= if you want to run the makefile =========
make=false
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
rm $all_sim/$run_dir/*.hdf5
rm $all_sim/$run_dir/sim*.*
rm $all_sim/$run_dir/batch*.*
# =================================================


# ====== set parameters and create .yml file ======
cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

#############################
##### grid construction #####
#############################

# number of elements, clamped (False) or periodic (True) spline and spline degrees (finite elements)
Nel : [2, 2, 80] 
bc  : [True, True, True]
p   : [1, 1, 3]

# boundary conditions for u1 and b1 at eta1 = 0 and eta1 = 1 (homogeneous Dirichlet = 'dirichlet', free boundary = 'free')
bc_u1 : [free, free]
bc_b1 : [free, free]  

# projector (global vs. local)
use_projector : global

# tolerance for approximation of inverse interpolation/histopolation matrices
tol_approx_reduced : 0.1

# number of quadrature points per element (nq_el) and histopolation cell (nq_pr)
nq_el : [4, 4, 4]
nq_pr : [4, 4, 4]

# polar splines in poloidal plane
polar : False

# basis for bulk velocity
basis_u : 2

# ----> for analytical geometry: kind of mapping (1: slab, 2: hollow cylinder, 3: colella) and parameters
#kind_map   :  0
#params_map : [0.5, 3.141592654, 10.36725576]
#params_map : [7.853981634, 7.853981634, 1.]
#params_map  : [1., 1., 1.]

kind_map   : cuboid
#params_map : [7.853981634, 1., 1.]
params_map : [0.5, 3.141592654, 10.36725576]
#params_map : [10.36725576, 10.36725576, 1.]

#kind_map   : spline
#params_map : [0., 0.5, 10.36725576]  
#params_map : [7.853981634, 1., 1.]

# ----> for spline geometry: number of elements, boundary conditions and spline degrees
Nel_MAP : [32, 33, 8] 
bc_MAP  : [False, True, False]
p_MAP   : [2, 2, 3] 


#############################
##### time integration ######
#############################

# do time integration?, time step, simulation time and maximum runtime of program (in minutes)
time_int : True
dt       : 0.01
Tend     : 40.
max_time : 1000.


###############################
##### linear solvers ##########
###############################

# ILUs (default: drop_tol=1e-4, fill_fac=10.)
# From scipy: "To improve the better approximation to the inverse, you may need to increase fill_factor AND decrease drop_tol."
drop_tol_S2 : 0.0001
fill_fac_S2 : 10.

drop_tol_A  : 0.0001
fill_fac_A  : 10.

drop_tol_S6 : 0.0001
fill_fac_S6 : 10.

# tolerances for iterative solvers (default: tol=1e-8)
tol1        : 0.00000001
tol2        : 0.00000001
tol3        : 0.00000001
tol6        : 0.00000001


###############################
####### MHD parameters ########
###############################

# add non-Hamiltonian terms to simulation?
add_pressure : True
add_jeq_step2 : True

# adiabatic exponent
gamma : 1.6666666666666666666666666666

###############################
##### kinetic parameters ######
###############################

# add kinetic terms to simulation?
add_PIC : False     

# total number of particles
Np : 10 

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

mkdir $SDIR

cp hylife/utilitis_FEEC/control_variates/kernels_control_variate.py $SDIR/kernels_control_variate.py
cp hylife/utilitis_FEEC/projectors/kernels_projectors_evaluation.py $SDIR/kernels_projectors_evaluation.py
cp hylife/utilitis_PIC/sampling.py $SDIR/sampling.py
# =================================================


# ============== run Makefile =====================
if [ "$make" = true ]
then
make all_sim=$all_sim run_dir=$run_dir flags_openmp_mhd=$flag_openmp_mhd flags_openmp_pic=$flag_openmp_pic
fi
# =================================================


# copy main code and adjust to current simulation =
var1="s|sed_replace_run_dir|"
var2="|g"

cp $code_name $all_sim/$run_dir/STRUPHY.py

sed -i $var1$run_dir$var2 $all_sim/$run_dir/STRUPHY.py
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
#mpirun -n 1 python3 STRUPHY.py
export OMP_NUM_THREADS=1
python3 STRUPHY.py
# =================================================