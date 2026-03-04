import random
import numpy as np
import torch

def seed_anything(seed=42):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mask_np(array, null_val):
    """Generate mask matrix"""
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    """Calculate masked MAE"""
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    """Calculate masked MSE"""
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    """Calculate masked MAPE"""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def cal_metric(ground_truth, prediction, args):
    """
    Calculate and log metrics for the whole prediction horizon.

    ground_truth / prediction: (num_samples, num_nodes, y_len)
    Only valid points (y_true != 0) are counted (masked metrics).
    """
    period = args.period
    y_len = getattr(args, "y_len", ground_truth.shape[-1])

    # Use full horizon up to y_len
    gt = ground_truth[:, :, :y_len]
    pr = prediction[:, :, :y_len]

    mae = masked_mae_np(gt, pr, 0)
    rmse = masked_mse_np(gt, pr, 0) ** 0.5
    mape = masked_mape_np(gt, pr, 0)

    # Store in args.result by period
    if not hasattr(args, "result") or args.result is None:
        args.result = {}
    if period not in args.result:
        args.result[period] = {}

    args.result[period]["MAE"] = float(mae)
    args.result[period]["RMSE"] = float(rmse)
    args.result[period]["MAPE"] = float(mape)

    args.logger.info("[*] period {}, testing".format(period))
    args.logger.info("Period {}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(period, mae, rmse, mape))