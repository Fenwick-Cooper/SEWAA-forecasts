#!/bin/bash
#SBATCH --job-name=ifs_interp
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --partition=priority-predict
#SBATCH --output=/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/nohup_out_files/ifs_interp_%A_%a.out
#SBATCH --error=/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/nohup_out_files/ifs_interp_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matthew.wright@physics.ox.ac.uk

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

YEARS=(
    2018
    2019
    # 2020
    # 2021
    # 2022
    # 2023
    # 2024
    # 2025
    # 2026
)

YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Year:      $YEAR"
echo "Started:   $(date)"
echo "======================================"

python -u 0000_regrid_ifs_fcst_to_radar.py --year "$YEAR"

echo "Finished: $(date)"