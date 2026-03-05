#!/bin/bash


#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=atmo-retrieval-run1-260246
#SBATCH --account=PAS2489

cd $SLURM_SUBMIT_DIR
module load intel
mpicc -O2 hello.c -o hello
srun ./hello > hello_results
