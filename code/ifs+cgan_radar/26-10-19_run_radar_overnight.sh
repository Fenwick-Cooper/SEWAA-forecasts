#!/bin/bash
set -euo pipefail

# Initialise Conda
source "/Users/matthewwright/miniconda3/etc/profile.d/conda.sh"
conda activate radar_processing

# Run processing
python -u \
    /Users/matthewwright/git/SEWAA-forecasts-Rwanda_radar/SEWAA-forecasts/code/ifs+cgan_radar/0000_process_old_radar_samples.py

echo "Processing finished. Starting transfer..."

# Transfer both 6h and 24h products
rsync -avhP \
    -e "ssh -o BatchMode=yes" \
    /Volumes/Rain/processed_rwandan_radar/ \
    server1:/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/rwanda_radar_data/full_radar_data/

echo "All transfers complete."