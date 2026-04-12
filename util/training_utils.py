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


def _normalize_binary_mask(mask):
    return (mask == 1).astype(np.float32)


def masked_mae_np_with_mask(y_true, y_pred, mask, eps=1e-6):
    valid_mask = _normalize_binary_mask(mask)
    err = np.abs(y_true - y_pred)
    return float(np.sum(err * valid_mask) / (np.sum(valid_mask) + eps))


def masked_mse_np_with_mask(y_true, y_pred, mask, eps=1e-6):
    valid_mask = _normalize_binary_mask(mask)
    err = (y_true - y_pred) ** 2
    return float(np.sum(err * valid_mask) / (np.sum(valid_mask) + eps))


def masked_mape_np_with_mask(y_true, y_pred, mask, eps=1e-6):
    valid_mask = _normalize_binary_mask(mask)
    value_mask = (np.abs(y_true) > eps).astype(np.float32)
    final_mask = valid_mask * value_mask
    err = np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))
    return float(np.sum(err * final_mask) / (np.sum(final_mask) + eps) * 100.0)
