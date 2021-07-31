#!/bin/bash

# ============== set simulation folders ===========
path_root=$(pwd)
all_sim=$path_root/my_struphy_sims
run_dir=sim_1
# =================================================

# ======= name of main code =======================
code_name=STRUPHY_cc_lin_6D.py
# =================================================

# ============== if you want to use OpenMP ========
flag_openmp_mhd=
flag_openmp_pic=--openmp
# =================================================

# ======= if you want to run the makefile =========
make=true
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
rm -f $all_sim/$run_dir/*.hdf5  # IMPORTANT: if this run is a restart, comment this line!
rm -f $all_sim/$run_dir/sim*.*
rm -f $all_sim/$run_dir/batch*.*
# =================================================


# ====== set parameters and create .yml file ======
cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

# ---------------- mesh parameters -----------------------
# number of elements, clamped (False) or periodic (True) splines and spline degrees
Nel      : [2, 2, 16] 
spl_kind : [True, True, True]
p        : [1, 1, 3]

# boundary conditions for U2_1 (or Uv_1) and B2_1 at eta_1 = 0 and eta_1 = 1 (homogeneous Dirichlet = d, free boundary = f)
bc : [f, f]

# number of quadrature points per element (nq_el) and histopolation cell (nq_pr)
nq_el : [6, 6, 6]
nq_pr : [6, 6, 6]

# polar splines in eta_1-eta_2 (usually poloidal) plane?
polar : False

# representation of MHD bulk velocity (0 : vector field with 0-form basis for each component, 2 : 2-form)
basis_u : 0

# geometry (cuboid, colella, spline cylinder, etc., see mappings_3d.py) and parameters
geometry   : cuboid
params_map : [1., 1., 7.853981634]

#geometry   : colella
#params_map : [1., 1., 0.05, 7.853981634]

#geometry   : spline cylinder
#params_map : []  

#geometry   : spline torus
#params_map : []
# --------------------------------------------------------


# --------------- MHD parameters -------------------------
# which equilibrium file (currently slab and circular available)
eq_type : slab

# <<< parameters for simulations in slab geometry >>>
B0x    : 0.
B0y    : 0.
B0z    : 1.
rho0   : 1.
beta_s : 0.

# <<< parameters for simulations in circular geometry (cylinder or circular torus) >>>
# minor and major radius in m
a  : 1.
R0 : 10.

# on-axis toroidal magnetic field in T
B0 : 1.

# safety factor profile
q0    : 1.1
q1    : 1.85
q_add : 0
rl    : 1.

# current profile
bmp0 : 0.
cg0  : 0.2
wg0  : 0.3
bmp1 : 0.
cg1  : 0.2
wg1  : 0.3
bmp2 : 0.
cg2  : 0.2
wg2  : 0.3

# add order-eps Shafranov shift?
shafranov : 0

# density profile
r1    : 4.
r2    : 3.
rho_a : 0.

# pressure profile
beta : 0.2
p1   : 0.95
p2   : 0.05
# ---------------------------------------------------------


# ---------- time integration -----------------------------
# run mode (0 : MHD eigenvalue solver, 1: initial-value solver with MHD eigenfunction, 2 : initial-value solver with input functions, 3 : initial-value solver with random noise)
run_mode : 2

# --> parameters for run mode 0 (tor. mode number, projection of eq. profiles?, directory to solve spectrum)
n_tor    : -1
profiles : False
dir_eig  : eigenstates.npy

# --> parameters for run mode 1 (tor. mode number, projection of eq. profiles?, real (11) or imag (12) part, squared eigenfreq.)
n_tor    : -1
profiles : False
eig_kind : 11
eig_freq : 0.

# --> parameters for run mode 3 (plane for noise: xy, yz or xz)
plane : yz

# for initial-value-solver: do time integration?, time step, simulation time and maximum runtime of program in minutes
time_int : True
dt       : 0.1
Tend     : 10.
max_time : 1000.

# location of j_eq X B term (either step_2 or step_6) 
loc_j_eq : step_6
# ----------------------------------------------------------


# -------------- linear solvers ----------------------------
# pre-conditioner for linear systems (ILU or FFT)
PRE : FFT 

# for ILUs: set tolerance for approximation of inverse interpolation/histopolation matrices (values < tol_inv are set to zero)
tol_inv : 0.1

# for ILUs: set drop_tol and fill_fac (default: drop_tol=1e-4, fill_fac=10.)
# From scipy: "To improve the better approximation to the inverse, you may need to increase fill_factor AND decrease drop_tol."
drop_tol_A  : 0.0001
fill_fac_A  : 10.

drop_tol_S2 : 0.0001
fill_fac_S2 : 10.

drop_tol_S6 : 0.0001
fill_fac_S6 : 10.

# parameters for iterative solvers (default for tolerances: tol=1e-8)
solver_type_2 : cg
solver_type_3 : cg

tol1 : 0.00000001
tol2 : 0.00000001
tol3 : 0.00000001
tol6 : 0.00000001

# maximum number of iterations
maxiter1 : 1000
maxiter2 : 1000
maxiter3 : 1000
maxiter6 : 1000
# -----------------------------------------------------------


# ---------------- kinetic parameters -----------------------
# add kinetic terms to simulation?
add_PIC : True    

# total number of particles
Np : 128000

# control variate (delta-f)? 
control : True  

# shift of Maxwellian 
v0 : [0., 0., 2.5]

# hot ion thermal velocity
vth : 1.

# particle loading
loading : pseudo-random

# seed for random number generator
seed : 1234

# directory of particles if loaded externally
dir_particles : path_to_particles
# -----------------------------------------------------------


# --------------- restart function --------------------------
# Is this run a restart?
restart : False

# If yes, number of restart files
num_restart : 0

# Create restart files at the end of the simulation? 
create_restart : True
# -----------------------------------------------------------

# enable plotting
enable_plotting : False

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

cp hylife/utilitis_FEEC/control_variates/kernels_control_variate.py $SDIR/kernels_control_variate.py
cp hylife/utilitis_PIC/sampling.py $SDIR/sampling.py
# =================================================


# for use of pyccel version >= v0.10.1
export SYMPY_USE_CACHE=no


# ============== run Makefile =====================
if [ "$make" = true ]
then
echo "Pre-compilation:"
make all_sim=$all_sim run_dir=$run_dir flags_openmp_mhd=$flag_openmp_mhd flags_openmp_pic=$flag_openmp_pic
fi
# =================================================


# copy main code and adjust to current simulation =
var1="s|sed_replace_run_dir|"
var2="|g"

cp $code_name $all_sim/$run_dir/STRUPHY.py

sed -i -e $var1$run_dir$var2 $all_sim/$run_dir/STRUPHY.py
# =================================================


# ================== run the code =================
cd $all_sim/$run_dir

echo "Start of STRUPHY:"

# option 1: job submission via SLURM
#sbatch batch_$run_dir.sh

# option 2 : interactive run on an interactive node on e.g. Cobra (either pure MPI or pure OpenMP)
#export OMP_NUM_THREADS=1
#srun -n 4 -p interactive python3 STRUPHY.py > STRUPHY.out

#export OMP_NUM_THREADS=4
#python3 STRUPHY.py

# option 3: run on a local machine e.g. laptop (either pure MPI or OpenMP)
#export OMP_NUM_THREADS=1
#mpirun -n 4 python3 STRUPHY.py

#export OMP_NUM_THREADS=1
#python3 STRUPHY.py
# =================================================
