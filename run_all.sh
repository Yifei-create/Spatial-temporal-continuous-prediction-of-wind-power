#!/bin/bash

echo "=========================================="
echo "Training EAC model on GPU 0..."
echo "=========================================="
python main.py --method EAC --logname eac --seed 42 --gpuid 0 --train 1 --data_process 0
echo ""
echo "=========================================="
echo "EAC training completed!"
echo "=========================================="
echo ""

echo "=========================================="
echo "Training ScaleShift model on GPU 0..."
echo "=========================================="
python main.py --method ScaleShift --logname scaleshift --seed 42 --gpuid 0 --train 1 --data_process 0
echo ""
echo "=========================================="
echo "ScaleShift training completed!"
echo "=========================================="
echo ""

echo "=========================================="
echo "Training VariationalScaleShift model on GPU 0..."
echo "=========================================="
python main.py --method VariationalScaleShift --logname var_scaleshift --seed 42 --gpuid 0 --train 1 --data_process 0
echo ""
echo "=========================================="
echo "VariationalScaleShift training completed!"
echo "=========================================="
echo ""

echo "=========================================="
echo "All models training completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - log/eac-42/"
echo "  - log/scaleshift-42/"
echo "  - log/var_scaleshift-42/"
echo "=========================================="
