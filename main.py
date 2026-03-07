import sys
import argparse
import random
import torch
import numpy as np
import os.path as osp
import networkx as nx

from torch_geometric.loader import DataLoader

from config.config import Config
from model import EAC_Model, ScaleShift_Model, VariationalScaleShift_Model
from data.dataset import SpatioTemporalDataset
from data.data_processing import generate_samples, process_and_save_all_periods
from trainer import train, test_model, load_test_best_model, mkdirs
from util.logger import get_logger
from util.training_utils import seed_anything


def main(args):
    args.logger.info("params : %s", vars(args))

    # store metrics by period
    args.result = {}  # {period: {"MAE":..., "RMSE":..., "MAPE":..., "total_time":...}}

    mkdirs(args.save_data_path)
    vars(args)["graph_size_list"] = []

    if args.data_process:
        args.logger.info("[*] Processing raw data...")
        process_and_save_all_periods(
            raw_csv_path="data/raw/sdwpf_2001_2112_full.csv",
            location_csv_path="data/raw/sdwpf_turb_location_elevation.csv",
            save_dir=args.save_data_path,
            graph_dir=args.graph_path,
            x_len=args.x_len,
            y_len=args.y_len
        )
        args.logger.info("[*] Data processing completed! Please set data_process=False and run again.")
        return

    for period in range(args.begin_period, args.end_period + 1):
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(period) + "_adj.npz"))["x"])

        if period == args.begin_period:
            vars(args)["init_graph_size"] = graph.number_of_nodes()
            vars(args)["subgraph"] = graph
            vars(args)["base_node_size"] = graph.number_of_nodes()
        else:
            vars(args)["init_graph_size"] = args.graph_size

        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["period"] = period
        args.graph_size_list.append(graph.number_of_nodes())

        loaded_data = np.load(osp.join(args.save_data_path, str(period) + ".npz"), allow_pickle=True)

        if 'train_x' in loaded_data:
            inputs = loaded_data
        else:
            args.logger.info("[*] Generating samples from raw data...")
            inputs = generate_samples(args.x_len, args.y_len, "", loaded_data["x"], graph, val_test_mix=False)

        args.logger.info("[*] Period {} load from {}.npz".format(args.period, osp.join(args.save_data_path, str(period))))

        adj = np.load(osp.join(args.graph_path, str(args.period) + "_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        if period == args.begin_period and args.load_first_period and not args.train:
            model, _ = load_test_best_model(args)
            test_loader = DataLoader(
                SpatioTemporalDataset(inputs, "test"),
                batch_size=args.batch_size, shuffle=False,
                pin_memory=True, num_workers=4
            )
            test_model(model, args, test_loader, pin_memory=True)
            continue

        vars(args)["node_list"] = list()

        if period > args.begin_period and args.strategy == "incremental":
            args.logger.warning("[*] Incremental learning not implemented yet, using retrain strategy.")

        if args.train:
            train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_test_best_model(args)
                test_loader = DataLoader(
                    SpatioTemporalDataset(inputs, "test"),
                    batch_size=args.batch_size, shuffle=False,
                    pin_memory=True, num_workers=4
                )
                test_model(model, args, test_loader, pin_memory=True)

    args.logger.info("\n\n=== Final Metrics by Period ===")
    mae_list, rmse_list, mape_list = [], [], []

    for period in range(args.begin_period, args.end_period + 1):
        if period in args.result and all(k in args.result[period] for k in ["MAE", "RMSE", "MAPE"]):
            mae = args.result[period]["MAE"]
            rmse = args.result[period]["RMSE"]
            mape = args.result[period]["MAPE"]

            args.logger.info("Period {}: MAE {:.4f} / RMSE {:.4f} / MAPE {:.4f}".format(period, mae, rmse, mape))

            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
        else:
            args.logger.info("Period {}: metrics not found (training/testing may have failed).".format(period))

    if len(mae_list) > 0:
        args.logger.info(
            "Avg({}~{}): MAE {:.4f} / RMSE {:.4f} / MAPE {:.4f}".format(
                args.begin_period, args.end_period,
                float(np.mean(mae_list)),
                float(np.mean(rmse_list)),
                float(np.mean(mape_list)),
            )
        )

    total_time = 0.0
    args.logger.info("\n=== Time Summary ===")
    for period in range(args.begin_period, args.end_period + 1):
        if period in args.result and all(k in args.result[period] for k in ["total_time", "average_time", "epoch_num"]):
            info = "period\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(
                period,
                args.result[period]["total_time"],
                args.result[period]["average_time"],
                args.result[period]["epoch_num"]
            )
            total_time += float(args.result[period]["total_time"])
            args.logger.info(info)

    args.logger.info("total time: {:.4f}".format(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--method", type=str, default="EAC",
                        help="Model method: EAC, ScaleShift, VariationalScaleShift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--logname", type=str, default="eac")
    parser.add_argument("--data_process", type=int, default=0,
                        help="1: process raw data first time, 0: load processed data")
    parser.add_argument("--train", type=int, default=1, help="1: train, 0: test only")

    # allow overriding period range from CLI
    parser.add_argument("--begin_period", type=int, default=None, help="Override begin_period in config")
    parser.add_argument("--end_period", type=int, default=None, help="Override end_period in config")

    args = parser.parse_args()

    config = Config(method=args.method, logname=args.logname, seed=args.seed, gpuid=args.gpuid)

    # merge config into args if args doesn't already have the field
    for key, value in vars(config).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # CLI overrides config
    if args.begin_period is not None:
        setattr(args, "begin_period", args.begin_period)
    if args.end_period is not None:
        setattr(args, "end_period", args.end_period)

    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) \
        if torch.cuda.is_available() and args.gpuid != -1 else torch.device("cpu")

    vars(args)["methods"] = {
        'EAC': EAC_Model,
        'ScaleShift': ScaleShift_Model,
        'VariationalScaleShift': VariationalScaleShift_Model
    }

    seed_anything(args.seed)

    vars(args)["path"] = osp.join(args.model_path, args.logname + "-" + str(args.seed))
    mkdirs(args.path)
    logger = get_logger("STCWPF", log_dir=args.path, log_file=args.logname + ".log")
    vars(args)["logger"] = logger

    main(args)