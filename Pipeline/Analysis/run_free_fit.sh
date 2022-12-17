#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --job-name=gaia-free-fit
#SBATCH --mail-type=NONE
#SBATCH --array=1601-3385


module load python/intel/3.8.6
python free_fit.py ${SLURM_ARRAY_TASK_ID}
