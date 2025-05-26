#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_gpu
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=12:00:00

module purge
module load nvhpcsdk/25 cuda/12.6-nvhpcsdk_25 openmpi/5.0 openmpi_gpu/5.0 gcc/13 likwid/5.3

LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)
export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib:$LD_LIBRARY_PATH

source ~/virtual_envs/env_struphy/bin/activate

python src/struphy/gpu/test_cupy_timings.py
python src/struphy/gpu/test_pyccel_timings.py

# Run with CPU

struphy run Vlasov       --time-trace -o sim_cpu # CPU
struphy run Vlasov --gpu --time-trace -o sim_gpu # GPU

# post processing

struphy pproc -d sim_cpu --time-trace
struphy pproc -d sim_gpu --time-trace
