import argparse
import json
import os.path as osp

import numpy as np
import torch

from config.config import Config, SUPPORTED_DATASETS, WARMUP_METHODS
from config.dataset_registry import DATASET_REGISTRY
from data.data_processing import process_unified_dataset
from data.graph_generation import SUPPORTED_GRAPH_VARIANTS
from model import EAC_Model, PatchTST_Model, ScaleShift_Model, VariationalScaleShift_Model
from trainer import mkdirs, pretrain, streaming_test
from util.logger import get_logger
from util.training_utils import seed_anything


def _format_float_token(value):
    text = format(float(value), ".10g")
    return text.replace(".", "p")


def _expansion_token(args):
    return "auto" if args.num_expansions is None else str(int(args.num_expansions))


def _warmup_token(args, warmup_enabled=None):
    enabled = bool(args.use_warmup) if warmup_enabled is None else bool(warmup_enabled)
    return "warmup-on" if enabled else "warmup-off"


def _build_experiment_dir_name(args, warmup_enabled=None):
    return (
        f"{args.method}"
        f"__graph-{args.graph_variant}"
        f"__seed-{int(args.seed)}"
        f"__x{int(args.x_len)}_y{int(args.y_len)}"
        f"__bs{int(args.batch_size)}"
        f"__lr{_format_float_token(args.lr)}"
        f"__drop{_format_float_token(args.dropout)}"
        f"__{_warmup_token(args, warmup_enabled=warmup_enabled)}"
        f"__exp-{_expansion_token(args)}"
    )


def _build_preprocess_dir_name(args):
    return (
        f"preprocess"
        f"__graph-{args.graph_variant}"
        f"__seed-{int(args.seed)}"
        f"__x{int(args.x_len)}_y{int(args.y_len)}"
        f"__exp-{_expansion_token(args)}"
    )


def _configure_preprocess_cache_paths(args):
    preprocess_dir_name = _build_preprocess_dir_name(args)
    args.preprocess_dir_name = preprocess_dir_name
    args.save_data_path = osp.join("data", "processed", args.dataset, preprocess_dir_name)
    args.graph_path = osp.join("data", "graph", args.dataset, preprocess_dir_name)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default)


def _build_config_payload(args):
    return {
        "dataset": args.dataset,
        "method": args.method,
        "graph_variant": args.graph_variant,
        "seed": int(args.seed),
        "logname": args.logname,
        "x_len": int(args.x_len),
        "y_len": int(args.y_len),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "dropout": float(args.dropout),
        "num_expansions": None if args.num_expansions is None else int(args.num_expansions),
        "epoch": int(args.epoch),
        "no_warmup": bool(getattr(args, "no_warmup", False)),
        "use_warmup": bool(args.use_warmup),
        "device": str(args.device),
        "run_dir": args.path,
        "save_data_path": args.save_data_path,
        "graph_path": args.graph_path,
        "raw_data_path": args.raw_data_path,
        "location_path": args.location_path,
    }


def _write_config_snapshot(args):
    _write_json(osp.join(args.path, "config.json"), _build_config_payload(args))


def _write_preprocess_artifacts(args):
    stage_adj_files = []
    for stage_idx in range(int(args.num_expansions) + 1 if args.num_expansions is not None else len(DATASET_REGISTRY[args.dataset]["default_expansion_groups"]) + 1):
        stage_path = osp.join(args.graph_path, f"stage_{stage_idx}_adj.npz")
        if osp.exists(stage_path):
            stage_adj_files.append(stage_path)
    payload = {
        "dataset": args.dataset,
        "graph_variant": args.graph_variant,
        "unified_data_path": osp.join(args.save_data_path, "unified_data.npz"),
        "graph_dir": args.graph_path,
        "stage_adj_files": stage_adj_files,
    }
    _write_json(osp.join(args.path, "artifacts.json"), payload)


def _write_metrics_snapshot(args):
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "graph_variant": args.graph_variant,
        "seed": int(args.seed),
        "hyperparams": {
            "x_len": int(args.x_len),
            "y_len": int(args.y_len),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "dropout": float(args.dropout),
            "num_expansions": None if args.num_expansions is None else int(args.num_expansions),
            "epoch": int(args.epoch),
            "no_warmup": bool(getattr(args, "no_warmup", False)),
            "use_warmup": bool(args.use_warmup),
        },
        "best_val_metric": getattr(args, "best_val_metric", None),
        "streaming": getattr(args, "result", {}).get("streaming"),
        "paths": {
            "checkpoint_best": getattr(args, "best_checkpoint_path", None),
            "checkpoint_last": getattr(args, "last_checkpoint_path", None),
            "predictions": getattr(args, "predictions_path", None),
        },
    }
    _write_json(osp.join(args.path, "metrics.json"), payload)


def _build_turbine_schedule(data):
    keys = data["turbine_schedule_keys"].tolist()
    offsets = data["turbine_schedule_t_offsets"].tolist()
    new_cols = data["turbine_schedule_new_cols"]
    return {int(key): (int(offset), cols.tolist()) for key, offset, cols in zip(keys, offsets, new_cols)}


def _require_key(data, key):
    if key not in data.files:
        raise KeyError(f"Missing key '{key}' in unified_data.npz.")


def _validate_loaded_dataset(data, args):
    for key in [
        "dataset_name",
        "adj_type",
        "feature_cols",
        "static_feature_names",
        "raw_data",
        "feature_observed_mask",
        "patv_mask",
        "static_data",
        "static_mean",
        "static_std",
        "raw_timestamps",
    ]:
        _require_key(data, key)

    dataset_name = str(data["dataset_name"])
    graph_variant = str(data["adj_type"])
    feature_cols = data["feature_cols"].tolist()
    static_feature_names = data["static_feature_names"].tolist()

    if dataset_name != args.dataset:
        raise ValueError(f"Cached dataset is {dataset_name}, but current request is {args.dataset}.")
    if args.method != "PatchTST" and graph_variant != args.graph_variant:
        raise ValueError(f"Cached graph variant is {graph_variant}, but current runtime requires {args.graph_variant}.")
    if feature_cols != DATASET_REGISTRY[args.dataset]["feature_cols"]:
        raise ValueError(f"Cached feature_cols do not match the current registry for dataset={args.dataset}.")
    if static_feature_names != ["x", "y"]:
        raise ValueError(f"Static features must be exactly ['x', 'y']. Received {static_feature_names}.")

    if args.streaming_freq_mode == "dynamic":
        _require_key(data, "supported_frequency_minutes")
        supported_frequency_minutes = data["supported_frequency_minutes"].astype(np.int64).tolist()
        expected = DATASET_REGISTRY[args.dataset]["supported_frequency_minutes"]
        if supported_frequency_minutes != expected:
            raise ValueError(
                f"Cached supported_frequency_minutes={supported_frequency_minutes}, but registry expects {expected}."
            )
    else:
        _require_key(data, "frequency_minutes")
        cached_frequency = int(data["frequency_minutes"])
        expected = int(DATASET_REGISTRY[args.dataset]["frequency_minutes"])
        if cached_frequency != expected:
            raise ValueError(f"Cached frequency_minutes={cached_frequency}, but registry expects {expected}.")


def main(args):
    args.logger.info("params : %s", vars(args))
    mkdirs(args.save_data_path)
    _write_config_snapshot(args)

    unified_path = osp.join(args.save_data_path, "unified_data.npz")
    should_process_data = bool(args.data_process) or not osp.exists(unified_path)
    if should_process_data:
        if args.data_process:
            args.logger.info("[*] --data_process=1 received; regenerating unified dataset cache.")
        else:
            args.logger.info("[*] Missing unified dataset cache at %s; generating it first.", unified_path)
        args.logger.info("[*] Processing raw %s data into unified_data.npz ...", args.dataset)
        process_unified_dataset(
            raw_csv_path=args.raw_data_path,
            location_csv_path=args.location_path,
            save_dir=args.save_data_path,
            graph_dir=args.graph_path,
            x_len=args.x_len,
            y_len=args.y_len,
            num_expansions=args.num_expansions,
            args=args,
            dataset=args.dataset,
        )
        args.logger.info("[*] Data processing completed.")
        if args.data_process and not args.train:
            _write_preprocess_artifacts(args)
            return

    args.logger.info("[*] Loading unified dataset from %s", unified_path)
    data = np.load(unified_path, allow_pickle=True)
    _validate_loaded_dataset(data, args)

    args.feature_cols = data["feature_cols"].tolist()
    args.num_features = len(args.feature_cols)
    args.raw_data = data["raw_data"]
    args.feature_observed_mask = data["feature_observed_mask"]
    args.patv_mask = data["patv_mask"]
    args.static_data = data["static_data"]
    args.static_dim = args.static_data.shape[1]
    if args.static_dim != 2:
        raise ValueError(f"static_data must have exactly 2 columns [x, y]. Received shape {args.static_data.shape}.")
    args.static_mean = data["static_mean"].astype(np.float32)
    args.static_std = data["static_std"].astype(np.float32)
    if args.static_mean.shape != (args.static_dim,) or args.static_std.shape != (args.static_dim,):
        raise ValueError(
            f"static_mean/static_std must have shape ({args.static_dim},). "
            f"Received mean={args.static_mean.shape}, std={args.static_std.shape}."
        )
    args.x_mean = data["x_mean"].astype(np.float32)
    args.x_std = data["x_std"].astype(np.float32)
    if args.x_mean.shape != (args.num_features,) or args.x_std.shape != (args.num_features,):
        raise ValueError(
            f"x_mean/x_std must have shape ({args.num_features},). "
            f"Received mean={args.x_mean.shape}, std={args.x_std.shape}."
        )
    args.y_mean = float(data["y_mean"])
    args.y_std = float(data["y_std"])
    args.pretrain_end_idx = int(data["pretrain_end_idx"])
    args.val_end_idx = int(data["val_end_idx"])
    args.initial_cols = data["initial_cols"].tolist()
    args.base_node_size = int(data["initial_n"])
    args.raw_timestamps = data["raw_timestamps"]
    args.turbine_schedule = _build_turbine_schedule(data)
    args.streaming_freq_mode = str(data["streaming_freq_mode"])

    if args.streaming_freq_mode == "dynamic":
        args.supported_frequency_minutes = data["supported_frequency_minutes"].astype(np.int64).tolist()
        args.frequency_minutes = None
    else:
        args.supported_frequency_minutes = []
        args.frequency_minutes = int(data["frequency_minutes"])
    if args.use_freq_embedding:
        args.freq_num_embeddings = len(args.supported_frequency_minutes)
        args.freq_to_id = {freq: idx for idx, freq in enumerate(args.supported_frequency_minutes)}
    else:
        args.freq_num_embeddings = 1
        args.freq_to_id = {}

    if args.method == "PatchTST":
        args.use_static_embedding = False
        args.use_freq_embedding = False
        args.use_warmup = False
        args.freq_num_embeddings = 1
        args.freq_to_id = {}

    if args.train:
        args.logger.info("[*] Starting pretrain ...")
        pretrain(
            raw_data=args.raw_data,
            feature_observed_mask=args.feature_observed_mask,
            patv_mask=args.patv_mask,
            initial_cols=args.initial_cols,
            pretrain_end_idx=args.pretrain_end_idx,
            val_end_idx=args.val_end_idx,
            args=args,
        )

    args.logger.info("[*] Starting streaming test ...")
    streaming_test(
        raw_data=args.raw_data[args.val_end_idx :],
        feature_observed_mask=args.feature_observed_mask[args.val_end_idx :],
        patv_mask=args.patv_mask[args.val_end_idx :],
        turbine_schedule=args.turbine_schedule,
        initial_cols=args.initial_cols,
        timestamps=args.raw_timestamps[args.val_end_idx :],
        args=args,
    )

    if hasattr(args, "result"):
        for phase, metrics in args.result.items():
            for horizon, values in metrics.items():
                args.logger.info(
                    "[%s][%s] MAE %.4f / RMSE %.4f / MAPE %.4f",
                    phase,
                    horizon,
                    values["MAE"],
                    values["RMSE"],
                    values["MAPE"],
                )
    _write_metrics_snapshot(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type=str, default=SUPPORTED_DATASETS[0], choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--method", type=str, default="ScaleShift", choices=["ScaleShift", "VariationalScaleShift", "EAC", "PatchTST"])
    parser.add_argument("--graph_variant", type=str, default=SUPPORTED_GRAPH_VARIANTS[0], choices=list(SUPPORTED_GRAPH_VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--logname", type=str, default="run")
    parser.add_argument("--data_process", type=int, default=0)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--num_expansions", type=int, default=None)
    parser.add_argument("--no_warmup", type=int, default=0)
    args = parser.parse_args()

    config = Config(
        method=args.method,
        logname=args.logname,
        seed=args.seed,
        gpuid=args.gpuid,
        dataset=args.dataset,
        graph_variant=args.graph_variant,
    )
    for key, value in vars(config).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.num_expansions is not None:
        args.num_expansions = int(args.num_expansions)
    args.no_warmup = bool(args.no_warmup)
    args.warmup_capable = args.method in WARMUP_METHODS
    if args.no_warmup:
        args.use_warmup = False

    args.device = torch.device(f"cuda:{args.gpuid}") if torch.cuda.is_available() and args.gpuid != -1 else torch.device("cpu")
    args.methods = {
        "ScaleShift": ScaleShift_Model,
        "VariationalScaleShift": VariationalScaleShift_Model,
        "EAC": EAC_Model,
        "PatchTST": PatchTST_Model,
    }
    seed_anything(args.seed)

    _configure_preprocess_cache_paths(args)

    run_dir_name = args.preprocess_dir_name if args.data_process and not args.train else _build_experiment_dir_name(args)
    args.path = osp.join(args.model_path, args.dataset, run_dir_name)
    if args.no_warmup and args.warmup_capable:
        args.checkpoint_fallback_path = osp.join(
            args.model_path,
            args.dataset,
            _build_experiment_dir_name(args, warmup_enabled=True),
        )
    else:
        args.checkpoint_fallback_path = None
    mkdirs(args.path)
    args.logger = get_logger("STCWPF", log_dir=args.path, log_file="run.log")
    main(args)
