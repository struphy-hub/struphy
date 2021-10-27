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
rm -f $all_sim/$run_dir/*.yml
rm -f $all_sim/$run_dir/sim*.*
rm -f $all_sim/$run_dir/batch*.*
# =================================================


# ====== set parameters and create .yml file ======
cat >$all_sim/$run_dir/parameters_$run_dir.yml <<'EOF'

# ---------------- geometry ------------------------------
# geometry (cuboid, colella, spline cylinder, spline torus etc., see mappings_3d.py) and parameters
geometry   : cuboid
params_map : [1., 1., 7.853981634]

# kind of angular coordinate (only relevant for spline torus: 'equal arc' or 'straight')
chi : equal arc
# --------------------------------------------------------


# ---------------- mesh parameters -----------------------
# number of elements, clamped (False) or periodic (True) splines and spline degrees
Nel      : [2, 2, 16] 
spl_kind : [True, True, True]
p        : [1, 1, 3]

# boundary conditions for U2_1 (or Uv_1) and B2_1 at eta_1 = 0 and eta_1 = 1 (homogeneous Dirichlet = d, free boundary = f)
bc : [f, f]

# number of quadrature points per element (nq_el) and histopolation cell of projectors (nq_pr)
nq_el : [6, 6, 6]
nq_pr : [6, 6, 6]

# C^1 polar splines in eta_1-eta_2-plane (usually poloidal)?
polar : False

# representation of MHD bulk velocity (0 : vector field with 0-form basis for each component, 2 : 2-form)
basis_u : 0



# --------------------------------------------------------


# --------------- MHD parameters -------------------------
# which equilibrium file (currently slab and circular available)
eq_type : slab

# mass of bulk plasma (in units of proton mass, i.e. Hydrogen : 1, Deuterium: 2, etc.)
Ab : 1.

# <<<<<<<<<<<<<<<<<<<< parameters for simulations in slab geometry >>>>>>>>>>>>>>>>>>>

# magnetic field components in T, normalized bulk density and plasma beta = 2*mu0*p/|B|^2 in %
B0_slab   : [0., 0., 1.]
rho0_slab : 1.
beta_slab : 0.

# <<< parameters for simulations in circular geometry (cylinder or circular torus) >>>

# minor and major radius in m (for cylinder: length = 2*pi*R0)
a  : 1.
R0 : 10.

# on-axis magnetic field in T
B0 : 1.

# safety factor profile: q(r) = q0*(1 + (r/(a*rp))^(2*rl))^(1/rl), rp = ((q1/q0)^rl - 1)^(-1/(2*rl))
q0 : 1.1
q1 : 1.85
rl : 1.

# toroidal correction: q(r) = q_add*q(r)*sqrt(1 - (r/R0)^2)
q_add : 0

# bulk density profile: rho(r) = (1 - rho_a)*(1 - (r/a)^r1)^r2 + rho_a)
r1    : 4.
r2    : 3.
rho_a : 0.

# bulk pressure profile: p_kind = 0 : exact cylindrical equilibrium, p_kind = 1 : ad hoc profile, p(r) = B0^2*beta/200*(1 - p1*(r/a)^2 - p2*(r/a)^4), beta = on-axis plasma beta in %
p_kind : 1
beta   : 0.2      
p1     : 0.95
p2     : 0.05
# ---------------------------------------------------------


# ---------- time integration -----------------------------
# Available run modes of STRUPHY:
# 0 : MHD eigenvalue solver (1d or 2d)
# 1 : initial-value solver with MHD eigenfunction
# 2 : initial-value solver with input functions
# 3 : initial-value solver with random noise)
run_mode : 2

# --> parameters for run mode 0 and 1 (n_tor : toroidal mode number, dir_eig : directory to solve or load spectrum)
n_tor     : -1
dir_eig   : eigenstates.npy
load_spec : False
eig_freq  : 0.
eig_kind  : r

# --> parameters for run mode 2 (initial B(t=0,r) = Bi*sin(k*r))
Bi : [0., 0.0001, 0.]
k  : [0., 0., 0.8]

# --> parameters for run mode 3 (plane for noise: xy, yz or xz)
plane : yz

# for initial-value-solver: time step and simulation time (in units of Alfvén time L_0/v_A0 = B_0/sqrt(mu_0*rho_0) with L_0 = 1 m, B0 = 1 T and mu_0 = 1.25663706212⋅10^(-6) H/m)
dt   : 0.1
Tend : 10.

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
solver_type_2 : CG
solver_type_3 : CG

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
# ratio hot/bulk number densities (nuh<=0.0 is a run without particles)
nuh : 0.05

# charge and mass of hot ion species in units of elementary charge and proton mass
Zh : 1.
Ah : 1.

# coupling parameter alpha = Omega_{cp0}*L0/v_{A0} with L0 = 1 m
alpha : 1.

# total number of particles
Np : 32000

# control variate (delta-f)? 
control : True  

# shift of Maxwellian 
v0 : [0., 0., 2.5]

# hot ion thermal velocity
vth : 1.

# particle loading
loading : pseudo_random

# seed for random number generator
seed : 1234

# directory of particles if loaded externally
dir_particles : path_to_particles
# -----------------------------------------------------------


# ----- programme related parameters ------------------------
# maximum run time of programme in minutes
max_time : 1000.

# enable plotting?
enable_plotting : True

# Is this run a restart?
restart : False

# if yes: how many restarts have there been before?
num_restart : 0

# Create restart files at the end of the simulation? 
create_restart : True
# -----------------------------------------------------------

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
#SBATCH -J STRUPHY_CC_lin_6D
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
# for OpenMP:
#SBATCH --cpus-per-task=20
#
# mem=40000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=<userid>@rzg.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:20:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
#export OMP_PLACES=cores 

# Request larger stacksize (in MB)
#export OMP_STACKSIZE=512m

# Run the program:
srun python3 STRUPHY.py > test.out
EOF
# =================================================


# for use of pyccel version >= v0.10.1
export SYMPY_USE_CACHE=no


# ============== run Makefile =====================
if [ "$make" = true ]
then
echo
echo "Pre-compilation:"
make all_sim=$all_sim run_dir=$run_dir flags_openmp_mhd=$flag_openmp_mhd flags_openmp_pic=$flag_openmp_pic
fi
# =================================================


# copy main code and adjust to current simulation =
var1="s|sed_replace_run_dir|"
var2="|g"

#cp $code_name $all_sim/$run_dir/STRUPHY.py
cp simulations/$code_name $all_sim/$run_dir/STRUPHY.py

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
