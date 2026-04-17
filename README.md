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

Run artifacts are saved in:
```
results/{dataset}/{experiment_dir}/
├── run.log
├── config.json
├── metrics.json
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── predictions/
    └── streaming_predictions.npz
```

Preprocessed caches and stage graphs remain in:
```
data/processed/{dataset}/{preprocess_dir}/
data/graph/{dataset}/{preprocess_dir}/
```

## Requirements

```
torch>=1.10.0
torch-geometric>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
networkx>=2.6.0
tqdm>=4.62.0
```
