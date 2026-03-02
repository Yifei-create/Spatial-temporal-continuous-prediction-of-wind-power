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
    args.result = {
        "3": {" MAE": {}, "MAPE": {}, "RMSE": {}}, 
        "6": {" MAE": {}, "MAPE": {}, "RMSE": {}}, 
        "12": {" MAE": {}, "MAPE": {}, "RMSE": {}}, 
        "Avg": {" MAE": {}, "MAPE": {}, "RMSE": {}}
    }
    mkdirs(args.save_data_path)
    vars(args)["graph_size_list"] = []
    
    if args.data_process:
        args.logger.info("[*] Processing raw data...")
        process_and_save_all_periods(
            raw_csv_path="data/raw/sdwpf_2001_2112_full.csv",
            location_csv_path="data/raw/sdwpf_turb_location_elevation.csv",
            save_dir=args.save_data_path,
            graph_dir=args.graph_path
        )
        args.logger.info("[*] Data processing completed! Please set data_process=False and run again.")
        return
    
    for year in range(args.begin_year, args.end_year + 1):
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        
        if year == args.begin_year:
            vars(args)["init_graph_size"] = graph.number_of_nodes()
            vars(args)["subgraph"] = graph
            vars(args)["base_node_size"] = graph.number_of_nodes()
        else:
            vars(args)["init_graph_size"] = args.graph_size
        
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        args.graph_size_list.append(graph.number_of_nodes())
        
        loaded_data = np.load(osp.join(args.save_data_path, str(year)+".npz"), allow_pickle=True)
        
        if 'train_x' in loaded_data:
            inputs = loaded_data
        else:
            args.logger.info("[*] Generating samples from raw data...")
            inputs = generate_samples(12, "", loaded_data["x"], graph, val_test_mix=False)
        
        args.logger.info("[*] Year {} load from {}.npz".format(args.year, osp.join(args.save_data_path, str(year))))
        
        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        
        if year == args.begin_year and args.load_first_year and not args.train:
            model, _ = load_test_best_model(args)
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), 
                                    batch_size=args.batch_size, shuffle=False, 
                                    pin_memory=True, num_workers=4)
            test_model(model, args, test_loader, pin_memory=True)
            continue
        
        vars(args)["node_list"] = list()
        
        if year > args.begin_year and args.strategy == "incremental":
            args.logger.warning("[*] Incremental learning not implemented yet, using retrain strategy.")
        
        if args.train:
            train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_test_best_model(args)
                test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), 
                                        batch_size=args.batch_size, shuffle=False, 
                                        pin_memory=True, num_workers=4)
                test_model(model, args, test_loader, pin_memory=True)
    
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info = ""
            info_list = []
            for year in range(args.begin_year, args.end_year+1):
                if i in args.result:
                    if j in args.result[i]:
                        if year in args.result[i][j]:
                            info += "{:>10.2f}\t".format(args.result[i][j][year])
                            info_list.append(args.result[i][j][year])
            if len(info_list) > 0:
                args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(info_list)))
    
    total_time = 0
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            info = "year\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(
                year, args.result[year]["total_time"], 
                args.result[year]["average_time"], 
                args.result[year]['epoch_num']
            )
            total_time += args.result[year]["total_time"]
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
    
    args = parser.parse_args()
    
    config = Config(method=args.method, logname=args.logname, seed=args.seed, gpuid=args.gpuid)
    
    for key, value in vars(config).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else torch.device("cpu")
    
    vars(args)["methods"] = {
        'EAC': EAC_Model,
        'ScaleShift': ScaleShift_Model,
        'VariationalScaleShift': VariationalScaleShift_Model
    }
    
    seed_anything(args.seed)
    
    vars(args)["path"] = osp.join(args.model_path, args.logname+"-"+str(args.seed))
    mkdirs(args.path)
    logger = get_logger("STCWPF", log_dir=args.path, log_file=args.logname+".log")
    vars(args)["logger"] = logger
    
    main(args)
