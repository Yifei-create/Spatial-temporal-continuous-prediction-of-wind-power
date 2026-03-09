import argparse
import torch
import numpy as np
import os.path as osp

from config.config import Config
from model import EAC_Model, ScaleShift_Model, VariationalScaleShift_Model
from data.data_processing import process_unified_dataset
from trainer import pretrain, streaming_test, mkdirs
from util.logger import get_logger
from util.training_utils import seed_anything


def _build_turbine_schedule(data):
    """
    Reconstruct turbine_schedule dict from the flat arrays stored in unified_data.npz.

    Returns:
        dict {exp_idx: (t_offset, [col_indices])}
    """
    keys      = data["turbine_schedule_keys"].tolist()
    t_offsets = data["turbine_schedule_t_offsets"].tolist()
    new_cols  = data["turbine_schedule_new_cols"]   # object array of arrays

    schedule = {}
    for k, t_off, cols in zip(keys, t_offsets, new_cols):
        schedule[int(k)] = (int(t_off), cols.tolist())
    return schedule


def main(args):
    args.logger.info("params : %s", vars(args))
    mkdirs(args.save_data_path)

    # ------------------------------------------------------------------ #
    # 1. Data processing                                                   #
    # ------------------------------------------------------------------ #
    if args.data_process:
        args.logger.info("[*] Processing raw data into unified_data.npz ...")
        process_unified_dataset(
            raw_csv_path="data/raw/sdwpf_2001_2112_full.csv",
            location_csv_path="data/raw/sdwpf_turb_location_elevation.csv",
            save_dir=args.save_data_path,
            graph_dir=args.graph_path,
            x_len=args.x_len,
            y_len=args.y_len,
            num_expansions=args.num_expansions,
        )
        args.logger.info("[*] Data processing done. Set --data_process 0 and re-run.")
        return

    # ------------------------------------------------------------------ #
    # 2. Load unified dataset                                              #
    # ------------------------------------------------------------------ #
    unified_path = osp.join(args.save_data_path, "unified_data.npz")
    args.logger.info(f"[*] Loading unified dataset from {unified_path}")
    data = np.load(unified_path, allow_pickle=True)

    raw_data         = data["raw_data"]                        # (T, N_all, D)
    vars(args)["x_mean"] = float(data["x_mean"])
    vars(args)["x_std"]  = float(data["x_std"])
    vars(args)["y_mean"] = float(data["y_mean"])
    vars(args)["y_std"]  = float(data["y_std"])

    pretrain_end_idx = int(data["pretrain_end_idx"])
    val_end_idx      = int(data["val_end_idx"])
    initial_cols     = data["initial_cols"].tolist()
    initial_n        = int(data["initial_n"])

    turbine_schedule = _build_turbine_schedule(data)

    args.logger.info(
        f"[*] Dataset: T={raw_data.shape[0]}, N_all={raw_data.shape[1]}, D={raw_data.shape[2]}"
    )
    args.logger.info(
        f"[*] Split: pretrain=[0,{pretrain_end_idx}), val=[{pretrain_end_idx},{val_end_idx}), "
        f"test=[{val_end_idx},{raw_data.shape[0]})"
    )
    args.logger.info(f"[*] Initial turbines: {initial_n}, expansions: {len(turbine_schedule)}")

    # Update base_node_size to match actual initial turbine count
    vars(args)["base_node_size"] = initial_n

    # ------------------------------------------------------------------ #
    # 3. Pretrain                                                          #
    # ------------------------------------------------------------------ #
    if args.train:
        args.logger.info("[*] Starting pretrain ...")
        pretrain(
            raw_data=raw_data,
            initial_cols=initial_cols,
            pretrain_end_idx=pretrain_end_idx,
            val_end_idx=val_end_idx,
            args=args,
        )

    # ------------------------------------------------------------------ #
    # 4. Streaming test                                                    #
    # ------------------------------------------------------------------ #
    args.logger.info("[*] Starting streaming test ...")
    test_raw = raw_data[val_end_idx:]   # (T_test, N_all, D)

    streaming_test(
        raw_data=test_raw,
        turbine_schedule=turbine_schedule,
        initial_cols=initial_cols,
        args=args,
    )

    # ------------------------------------------------------------------ #
    # 5. Summary                                                           #
    # ------------------------------------------------------------------ #
    if hasattr(args, "result") and "streaming" in args.result:
        r = args.result["streaming"]
        args.logger.info(
            "Final: MAE {:.4f} / RMSE {:.4f} / MAPE {:.4f}".format(
                r["MAE"], r["RMSE"], r["MAPE"]
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--method", type=str, default="EAC",
                        help="Model method: EAC, ScaleShift, VariationalScaleShift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--logname", type=str, default="eac")
    parser.add_argument("--data_process", type=int, default=0,
                        help="1: process raw data, 0: load processed data")
    parser.add_argument("--train", type=int, default=1,
                        help="1: run pretrain then streaming test, 0: streaming test only")
    parser.add_argument("--num_expansions", type=int, default=None,
                        help="Override num_expansions in config")

    args = parser.parse_args()

    config = Config(method=args.method, logname=args.logname, seed=args.seed, gpuid=args.gpuid)

    # Merge config into args (config values fill in missing fields)
    for key, value in vars(config).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # CLI overrides
    if args.num_expansions is not None:
        setattr(args, "num_expansions", args.num_expansions)

    vars(args)["device"] = (
        torch.device(f"cuda:{args.gpuid}")
        if torch.cuda.is_available() and args.gpuid != -1
        else torch.device("cpu")
    )

    vars(args)["methods"] = {
        "EAC": EAC_Model,
        "ScaleShift": ScaleShift_Model,
        "VariationalScaleShift": VariationalScaleShift_Model,
    }

    seed_anything(args.seed)

    vars(args)["path"] = osp.join(args.model_path, args.logname + "-" + str(args.seed))
    mkdirs(args.path)
    logger = get_logger("STCWPF", log_dir=args.path, log_file=args.logname + ".log")
    vars(args)["logger"] = logger

    main(args)
