# STCWPF: Spatio-Temporal Continual Wind Power Forecasting

Continual learning framework for wind power forecasting with incremental node addition.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models (sequential, see progress in terminal)
bash run_all.sh

# Or train single model
python main.py --method EAC --gpuid 0
```

## Models

- **EAC**: Low-rank adaptation
- **ScaleShift**: Deterministic affine transformation
- **VariationalScaleShift**: Variational inference

## Configuration

Key parameters in `config/config.py`:
- Input/output length: 12 timesteps
- Features: 16 (including Patv)
- Periods: 4 (44→66→99→134 turbines)

## Results

Training results saved in:
```
log/{model}-{seed}/
├── {model}.log          # Training log
├── 0/                   # Period 0 models
│   └── {loss}.pkl
├── 1/                   # Period 1 models
├── 2/                   # Period 2 models
└── 3/                   # Period 3 models
```

Metrics (MAE, RMSE, MAPE) are printed in terminal and saved in log files.

## Requirements

```
torch>=1.10.0
torch-geometric>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
networkx>=2.6.0
tqdm>=4.62.0
```
