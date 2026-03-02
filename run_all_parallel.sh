#!/bin/bash

# Run all three models in parallel on different GPUs
# Usage: bash run_all_parallel.sh

echo "=========================================="
echo "Training all models in parallel"
echo "=========================================="
echo ""
echo "GPU allocation:"
echo "  GPU 0: EAC model"
echo "  GPU 1: ScaleShift model"
echo "  GPU 2: VariationalScaleShift model"
echo ""
echo "=========================================="
echo ""

# Run EAC on GPU 0 in background
echo "Starting EAC on GPU 0..."
nohup python main.py --method EAC --logname eac --seed 42 --gpuid 0 --train 1 --data_process 0 > log_eac.txt 2>&1 &
PID_EAC=$!
echo "  PID: $PID_EAC"

# Run ScaleShift on GPU 1 in background
echo "Starting ScaleShift on GPU 1..."
nohup python main.py --method ScaleShift --logname scaleshift --seed 42 --gpuid 1 --train 1 --data_process 0 > log_scaleshift.txt 2>&1 &
PID_SS=$!
echo "  PID: $PID_SS"

# Run VariationalScaleShift on GPU 2 in background
echo "Starting VariationalScaleShift on GPU 2..."
nohup python main.py --method VariationalScaleShift --logname var_scaleshift --seed 42 --gpuid 2 --train 1 --data_process 0 > log_var_scaleshift.txt 2>&1 &
PID_VSS=$!
echo "  PID: $PID_VSS"

echo ""
echo "=========================================="
echo "All models started!"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  EAC: $PID_EAC"
echo "  ScaleShift: $PID_SS"
echo "  VariationalScaleShift: $PID_VSS"
echo ""
echo "Monitor progress:"
echo "  tail -f log_eac.txt"
echo "  tail -f log_scaleshift.txt"
echo "  tail -f log_var_scaleshift.txt"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Kill all processes:"
echo "  kill $PID_EAC $PID_SS $PID_VSS"
echo "=========================================="

