#!/bin/bash

#SBATCH --job-name=encounters_mcmc_mars_only
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/submit/bvl/encounters/output/%x_%j.out
#SBATCH --error=/work/submit/bvl/encounters/output/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bvl@mit.edu

# Load MPI module or any other necessary modules
# module load mpi

# Run the script
mpirun -np $SLURM_NTASKS \
    python run_mcmc.py \
        --case mars_only \
        --n-steps 10000 \
        --output /work/submit/bvl/encounters/chains/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.h5
