#!/bin/bash
# Script to run the compare_four_approaches experiment

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate appfl

# Run the experiment in background
nohup python examples/compare_four_approaches.py > output04.log 2>&1 &

echo "Experiment started in background. Monitor with: tail -f output04.log"
echo "Check running process with: ps aux | grep compare_four_approaches.py"
