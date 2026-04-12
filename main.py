import argparse
import os.path as osp

import numpy as np
import torch

from config.config import Config
from config.dataset_registry import DATASET_REGISTRY
from data.data_processing import process_unified_dataset
from model import EAC_Model, PatchTST_Model, ScaleShift_Model, VariationalScaleShift_Model
from trainer import mkdirs, pretrain, streaming_test
from util.logger import get_logger
from util.training_utils import seed_anything


def _build_turbine_schedule(data):
    keys = data['turbine_schedule_keys'].tolist()
    t_offsets = data['turbine_schedule_t_offsets'].tolist()
    new_cols = data['turbine_schedule_new_cols']
    schedule = {}
    for k, t_off, cols in zip(keys, t_offsets, new_cols):
        schedule[int(k)] = (int(t_off), cols.tolist())
    return schedule


def _validate_loaded_dataset(data, args):
    dataset_name = str(data['dataset_name'])
    processing_tag = str(data['processing_tag'])
    adj_type = str(data['adj_type'])
    feature_cols = data['feature_cols'].tolist()
    static_feature_names = data['static_feature_names'].tolist()
    if dataset_name != args.dataset:
        raise ValueError(f"Cached dataset is {dataset_name}, but current request is {args.dataset}. Please reprocess the data.")
    if processing_tag != args.processing_tag:
        raise ValueError(f"Cached processing_tag is {processing_tag}, but current request is {args.processing_tag}. Please reprocess the data.")
    if adj_type != args.adj_type:
        raise ValueError(f"Cached adj_type is {adj_type}, but current request is {args.adj_type}. Please reprocess the data.")
    if feature_cols != DATASET_REGISTRY[args.dataset]['feature_cols']:
        raise ValueError("Cached feature_cols do not match the current registry. Please reprocess the data.")
    if static_feature_names != ['x', 'y']:
        raise ValueError("Cached static features do not match current framework requirements. Please reprocess the data.")


def main(args):
    args.logger.info('params : %s', vars(args))
    mkdirs(args.save_data_path)

    if args.data_process:
        args.logger.info('[*] Processing raw data into unified_data.npz ...')
        process_unified_dataset(
            raw_csv_path=args.raw_data_path,
            location_csv_path=args.location_path,
            save_dir=args.save_data_path,
            graph_dir=args.graph_path,
            x_len=args.x_len,
            y_len=args.y_len,
            num_expansions=args.num_expansions,
            adj_type=args.adj_type,
            wind_top_k=args.wind_top_k,
            dataset=args.dataset,
            args=args,
        )
        args.logger.info('[*] Data processing done. Set --data_process 0 and re-run.')
        return

    unified_path = osp.join(args.save_data_path, 'unified_data.npz')
    args.logger.info(f'[*] Loading unified dataset from {unified_path}')
    data = np.load(unified_path, allow_pickle=True)
    _validate_loaded_dataset(data, args)

    raw_data = data['raw_data']
    patv_mask = data['patv_mask']
    static_data = data['static_data']
    raw_timestamps = data['raw_timestamps']
    vars(args)['feature_cols'] = data['feature_cols'].tolist()
    vars(args)['num_features'] = len(args.feature_cols)
    vars(args)['static_dim'] = static_data.shape[1]
    if args.static_dim != 2:
        raise ValueError('Current framework requires static feature dimension to be exactly 2.')
    vars(args)['x_mean'] = float(data['x_mean'])
    vars(args)['x_std'] = float(data['x_std'])
    vars(args)['y_mean'] = float(data['y_mean'])
    vars(args)['y_std'] = float(data['y_std'])
    pretrain_end_idx = int(data['pretrain_end_idx'])
    val_end_idx = int(data['val_end_idx'])
    initial_cols = data['initial_cols'].tolist()
    initial_n = int(data['initial_n'])
    turbine_schedule = _build_turbine_schedule(data)
    vars(args)['base_node_size'] = initial_n
    vars(args)['static_data'] = static_data
    vars(args)['static_x_bin'] = data['static_x_bin'].astype(np.int64)
    vars(args)['static_y_bin'] = data['static_y_bin'].astype(np.int64)
    vars(args)['raw_timestamps'] = raw_timestamps
    vars(args)['streaming_freq_mode'] = str(data['streaming_freq_mode'])
    if args.streaming_freq_mode == 'dynamic':
        vars(args)['raw_data_base'] = data['raw_data_base']
        vars(args)['patv_mask_base'] = data['patv_mask_base']
        vars(args)['pretrain_freq_minutes'] = data['pretrain_freq_minutes'].astype(np.int64).tolist()
        vars(args)['base_resolution_minutes'] = int(data['base_resolution_minutes'])
        vars(args)['freq_num_embeddings'] = len(args.pretrain_freq_minutes)
    else:
        vars(args)['frequency_minutes'] = int(data['frequency_minutes'])
        vars(args)['freq_num_embeddings'] = 1

    if args.train:
        args.logger.info('[*] Starting pretrain ...')
        pretrain(
            raw_data=raw_data,
            patv_mask=patv_mask,
            initial_cols=initial_cols,
            pretrain_end_idx=pretrain_end_idx,
            val_end_idx=val_end_idx,
            args=args,
        )

    vars(args)['no_warmup'] = bool(args.no_warmup)
    args.logger.info('[*] Starting streaming test ...')
    streaming_test(
        raw_data=raw_data[val_end_idx:],
        patv_mask=patv_mask[val_end_idx:],
        turbine_schedule=turbine_schedule,
        initial_cols=initial_cols,
        timestamps=raw_timestamps[val_end_idx:],
        args=args,
    )

    if hasattr(args, 'result') and args.result:
        for key, r in args.result.items():
            for horizon, metrics in r.items():
                args.logger.info(f"[{key}][{horizon}] MAE {metrics['MAE']:.4f} / RMSE {metrics['RMSE']:.4f} / MAPE {metrics['MAPE']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', type=str, default='sdwpf', choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument('--method', type=str, default='EAC')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--logname', type=str, default='eac')
    parser.add_argument('--data_process', type=int, default=0)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--num_expansions', type=int, default=None)
    parser.add_argument('--no_warmup', type=int, default=0)
    parser.add_argument('--adj_type', type=str, default='distance')
    parser.add_argument('--wind_top_k', type=int, default=16)
    args = parser.parse_args()

    config = Config(method=args.method, logname=args.logname, seed=args.seed, gpuid=args.gpuid, dataset=args.dataset)
    for key, value in vars(config).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.num_expansions is not None:
        setattr(args, 'num_expansions', args.num_expansions)
    setattr(args, 'adj_type', args.adj_type)
    setattr(args, 'wind_top_k', args.wind_top_k)
    setattr(args, 'save_data_path', osp.join('data', 'processed', args.dataset, args.processing_tag, args.adj_type))
    setattr(args, 'graph_path', osp.join('data', 'graph', args.dataset, args.processing_tag, args.adj_type))
    vars(args)['device'] = torch.device(f'cuda:{args.gpuid}') if torch.cuda.is_available() and args.gpuid != -1 else torch.device('cpu')
    vars(args)['methods'] = {
        'EAC': EAC_Model,
        'ScaleShift': ScaleShift_Model,
        'VariationalScaleShift': VariationalScaleShift_Model,
        'PatchTST': PatchTST_Model,
    }
    seed_anything(args.seed)
    vars(args)['path'] = osp.join(args.model_path, f"{args.dataset}-{args.processing_tag}-{args.adj_type}-{args.logname}-{args.seed}")
    mkdirs(args.path)
    logger = get_logger('STCWPF', log_dir=args.path, log_file=args.logname + '.log')
    vars(args)['logger'] = logger
    main(args)
