from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.dataset_registry import DATASET_REGISTRY
from data.adapters import load_dataset_bundle


DEFAULT_DATASETS = ["sdwpf", "penmanshiel", "norrekaer_enge"]


def _safe_ratio(valid_count: np.ndarray | float, total_count: int) -> np.ndarray | float:
    if total_count <= 0:
        raise ValueError(f"total_count must be positive. Received {total_count}.")
    return valid_count / float(total_count)


def _dataset_report(dataset: str) -> tuple[pd.DataFrame, dict]:
    cfg = DATASET_REGISTRY[dataset]
    bundle = load_dataset_bundle(dataset, cfg["raw_data_path"], cfg["location_path"])

    feature_cols = list(bundle.feature_cols)
    observed = bundle.feature_observed_mask.astype(bool)
    patv_mask = bundle.patv_mask.astype(bool)
    turbine_ids = list(bundle.turbine_ids.tolist())
    timestamps = pd.to_datetime(bundle.timestamps)

    num_timesteps, num_turbines, num_features = observed.shape
    val_end_idx = int(num_timesteps * 0.3)
    if val_end_idx <= 0:
        raise ValueError(f"Dataset {dataset} has invalid val_end_idx={val_end_idx}.")

    graph_valid_mask = None
    if "wind_valid_mask" in bundle.graph_context:
        graph_valid_mask = np.asarray(bundle.graph_context["wind_valid_mask"], dtype=bool)
        if graph_valid_mask.shape != (num_timesteps, num_turbines):
            raise ValueError(
                f"Dataset {dataset} graph_valid_mask shape mismatch: "
                f"{graph_valid_mask.shape} vs {(num_timesteps, num_turbines)}."
            )

    rows: list[dict] = []
    for turbine_idx, turbine_id in enumerate(turbine_ids):
        row = {
            "dataset": dataset,
            "turbine": turbine_id,
            "num_timesteps": int(num_timesteps),
            "val_end_idx": int(val_end_idx),
            "timestamp_start": str(timestamps[0]),
            "timestamp_end": str(timestamps[-1]),
            "overall_missing_ratio": float(1.0 - observed[:, turbine_idx, :].mean()),
            "patv_invalid_ratio": float(1.0 - patv_mask[:, turbine_idx].mean()),
        }

        for feature_idx, feature_name in enumerate(feature_cols):
            feature_valid_total = int(observed[:, turbine_idx, feature_idx].sum())
            feature_valid_pre30 = int(observed[:val_end_idx, turbine_idx, feature_idx].sum())
            row[f"{feature_name}_missing_ratio"] = float(
                1.0 - _safe_ratio(feature_valid_total, num_timesteps)
            )
            row[f"{feature_name}_missing_ratio_pre30"] = float(
                1.0 - _safe_ratio(feature_valid_pre30, val_end_idx)
            )
            row[f"{feature_name}_valid_count"] = feature_valid_total
            row[f"{feature_name}_valid_count_pre30"] = feature_valid_pre30

        if graph_valid_mask is not None:
            graph_valid_total = int(graph_valid_mask[:, turbine_idx].sum())
            graph_valid_pre30 = int(graph_valid_mask[:val_end_idx, turbine_idx].sum())
            row["graph_valid_ratio"] = float(_safe_ratio(graph_valid_total, num_timesteps))
            row["graph_missing_ratio"] = float(1.0 - row["graph_valid_ratio"])
            row["graph_valid_count"] = graph_valid_total
            row["graph_valid_ratio_pre30"] = float(_safe_ratio(graph_valid_pre30, val_end_idx))
            row["graph_missing_ratio_pre30"] = float(1.0 - row["graph_valid_ratio_pre30"])
            row["graph_valid_count_pre30"] = graph_valid_pre30
            row["graph_zero_pre30"] = bool(graph_valid_pre30 == 0)
        else:
            row["graph_valid_ratio"] = None
            row["graph_missing_ratio"] = None
            row["graph_valid_count"] = None
            row["graph_valid_ratio_pre30"] = None
            row["graph_missing_ratio_pre30"] = None
            row["graph_valid_count_pre30"] = None
            row["graph_zero_pre30"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    zero_graph_pre30 = []
    if graph_valid_mask is not None:
        zero_graph_pre30 = df.loc[df["graph_zero_pre30"], "turbine"].tolist()

    summary = {
        "dataset": dataset,
        "shape": [int(num_timesteps), int(num_turbines), int(num_features)],
        "feature_cols": feature_cols,
        "timestamp_start": str(timestamps[0]),
        "timestamp_end": str(timestamps[-1]),
        "val_end_idx": int(val_end_idx),
        "num_turbines": int(num_turbines),
        "zero_graph_valid_pre30_turbines": zero_graph_pre30,
        "worst_overall_missing": df.sort_values("overall_missing_ratio", ascending=False)
        .head(10)[["turbine", "overall_missing_ratio", "patv_invalid_ratio"]]
        .to_dict(orient="records"),
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Inspect per-turbine missingness for the formal datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Datasets to inspect.",
    )
    parser.add_argument(
        "--outdir",
        default="results/data_missingness_audit",
        help="Directory used to store CSV and JSON reports.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_summary: dict[str, dict] = {}
    for dataset in args.datasets:
        print(f"[inspect] loading dataset={dataset}", flush=True)
        df, summary = _dataset_report(dataset)
        csv_path = outdir / f"{dataset}_turbine_missingness.csv"
        df.to_csv(csv_path, index=False)
        all_summary[dataset] = summary
        print(
            f"[inspect] dataset={dataset} shape={summary['shape']} "
            f"zero_graph_valid_pre30={summary['zero_graph_valid_pre30_turbines']}",
            flush=True,
        )
        del df
        gc.collect()

    summary_path = outdir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"[inspect] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
