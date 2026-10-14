#!/bin/bash
#SBATCH --job-name=ifs_interp
#SBATCH --array=0-16%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --partition=priority-predict
#SBATCH --output=/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/nohup_out_files/260910_ifs_interp_to_imerg_%A_%a.out
#SBATCH --error=/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/nohup_out_files/260910_ifs_interp_to_imerg_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matthew.wright@physics.ox.ac.uk

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

VARIABLES=(
    mucape
    tclw
    tciw
    tcrw
    sp
    tcw
    tcwv
    lsp
    cp
    u10
    v10
    t2m
    ssr
    mcc
    tp
    u700
    v700
)

VAR="${VARIABLES[$SLURM_ARRAY_TASK_ID]}"

echo "======================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Variable:  $VAR"
echo "Started:   $(date)"
echo "======================================"

python -u /home/m/matthewwright/radar-ensemble-cgan/dsrnngan/0000_regrid_ifs_hind_to_radar.py --variable "$VAR"

echo "Finished: $(date)"