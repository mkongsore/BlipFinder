#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --job-name=gaia-free-multinest
#SBATCH --mail-type=NONE

### There are 3386 total data files, i.e. the array should go between 0 and 3385
#SBATCH --array=1601-3385

module load python/intel/3.8.6
python free_multinest.py ${SLURM_ARRAY_TASK_ID}