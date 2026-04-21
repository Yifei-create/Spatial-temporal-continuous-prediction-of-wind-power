import os
import os.path as osp

import numpy as np
import torch
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from data.dataset import SingleTurbineDataset, SpatioTemporalDataset
from data.graph_generation import STAGE_ADJ_NORMALIZATION, STAGE_ADJ_STORAGE_LAYOUT
from data.streaming_plan import build_streaming_plan, constant_frequency_segments, resolve_allowed_frequency_minutes
from util.training_utils import masked_mae_np_with_mask, masked_mape_np_with_mask, masked_mse_np_with_mask


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_checkpoints_dir(args):
    path = osp.join(args.path, "checkpoints")
    mkdirs(path)
    return path


def get_best_checkpoint_path(args):
    return osp.join(get_checkpoints_dir(args), "best.pt")


def get_last_checkpoint_path(args):
    return osp.join(get_checkpoints_dir(args), "last.pt")


def get_predictions_dir(args):
    path = osp.join(args.path, "predictions")
    mkdirs(path)
    return path


def get_predictions_path(args):
    return osp.join(get_predictions_dir(args), "streaming_predictions.npz")


def normalize_x(raw_x, args, eps=1e-6):
    x_mean = np.asarray(args.x_mean, dtype=np.float32)
    x_std = np.asarray(args.x_std, dtype=np.float32)
    return (raw_x - x_mean) / (x_std + eps)


def normalize_y(raw_y, args, eps=1e-6):
    return (raw_y - args.y_mean) / (args.y_std + eps)


def denormalize_y(norm_y, args, eps=1e-6):
    return norm_y * (args.y_std + eps) + args.y_mean


def masked_mse_torch(pred, target, mask, eps=1e-6):
    err = (pred - target) ** 2
    return torch.sum(err * mask) / (torch.sum(mask) + eps)


def rebuild_adj(stage_idx, args):
    adj_path = osp.join(args.graph_path, f"stage_{stage_idx}_adj.npz")
    stage_adj = np.load(adj_path, allow_pickle=False)
    for key in ["x", "graph_variant", "storage_layout", "normalization"]:
        if key not in stage_adj.files:
            raise KeyError(f"Missing key '{key}' in stage adjacency file {adj_path}.")
    graph_variant = str(stage_adj["graph_variant"])
    storage_layout = str(stage_adj["storage_layout"])
    normalization = str(stage_adj["normalization"])
    if graph_variant != args.graph_variant:
        raise ValueError(f"Stage adjacency variant is {graph_variant}, but runtime requires {args.graph_variant}.")
    if storage_layout != STAGE_ADJ_STORAGE_LAYOUT:
        raise ValueError(
            f"Stage adjacency storage_layout is {storage_layout}, but runtime expects {STAGE_ADJ_STORAGE_LAYOUT}."
        )
    if normalization != STAGE_ADJ_NORMALIZATION:
        raise ValueError(
            f"Stage adjacency normalization is {normalization}, but runtime expects {STAGE_ADJ_NORMALIZATION}."
        )
    adj = stage_adj["x"]
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Stage adjacency must be a square matrix. Received shape {adj.shape} from {adj_path}.")
    return torch.from_numpy(adj).float().to(args.device)


def freeze_backbone(model):
    model.freeze_backbone()


NODE_ADAPTIVE_PARAMS = {
    "U",
    "scale",
    "shift",
    "mu_scale",
    "mu_shift",
    "log_var_scale",
    "log_var_shift",
    "input_bias",
    "stage_residual",
}
GLOBAL_ADAPTIVE_PARAMS = {
    "V",
    "stage_bias",
}


def _is_named_param(name, target_names):
    return any(name == param_name or name.endswith("." + param_name) for param_name in target_names)


def _is_adaptive_param(name):
    return _is_named_param(name, NODE_ADAPTIVE_PARAMS | GLOBAL_ADAPTIVE_PARAMS)


def _is_node_adaptive_param(name):
    return _is_named_param(name, NODE_ADAPTIVE_PARAMS)


def _is_global_adaptive_param(name):
    return _is_named_param(name, GLOBAL_ADAPTIVE_PARAMS)


def _set_adaptive_requires_grad(model, value):
    for name, param in model.named_parameters():
        if _is_adaptive_param(name):
            param.requires_grad = value


def _window_to_flat(x_win):
    return x_win.transpose(0, 2, 1, 3).reshape(x_win.shape[0], x_win.shape[2], x_win.shape[1] * x_win.shape[3]).transpose(0, 2, 1)


def _apply_feature_mask_after_normalization(x_norm, feature_mask):
    return x_norm * feature_mask.astype(np.float32)


def _empty_inputs(split_name, input_dim):
    return {
        f"{split_name}_x": np.empty((0, input_dim, 0), dtype=np.float32),
        f"{split_name}_y": np.empty((0, 0, 0), dtype=np.float32),
        f"{split_name}_y_mask": np.empty((0, 0, 0), dtype=np.float32),
        f"{split_name}_static_data": np.empty((0, 0, 0), dtype=np.float32),
        f"{split_name}_freq_id": np.empty((0,), dtype=np.int64),
        f"{split_name}_stage_idx": np.empty((0,), dtype=np.int64),
    }


def _empty_single_turbine_inputs(split_name, input_dim, y_len):
    return {
        f"{split_name}_x": np.empty((0, input_dim), dtype=np.float32),
        f"{split_name}_y": np.empty((0, y_len), dtype=np.float32),
        f"{split_name}_y_mask": np.empty((0, y_len), dtype=np.float32),
    }


def _allowed_frequency_minutes(args):
    return resolve_allowed_frequency_minutes(
        args.streaming_freq_mode,
        supported_frequency_minutes=getattr(args, "supported_frequency_minutes", None),
        frequency_minutes=getattr(args, "frequency_minutes", None),
    )


def _constant_frequency_segments(timestamps, args):
    return constant_frequency_segments(timestamps, _allowed_frequency_minutes(args))


def _resolve_freq_id(args, freq_minutes):
    if not args.use_freq_embedding:
        return 0
    if freq_minutes not in args.freq_to_id:
        raise ValueError(
            f"Encountered stream frequency {freq_minutes} minutes, which is not in supported_frequency_minutes={args.supported_frequency_minutes}."
        )
    return args.freq_to_id[freq_minutes]


def _build_window_inputs(
    segment,
    feature_segment_mask,
    segment_mask,
    timestamps,
    static_data,
    stage_idx,
    args,
    split_name,
):
    x_len = args.x_len
    y_len = args.y_len
    num_features = segment.shape[2]
    patv_col = num_features - 1

    xs, feature_masks, ys, ys_mask, static_features, freq_ids, stage_idxs = [], [], [], [], [], [], []
    for segment_start, segment_end, freq_minutes in _constant_frequency_segments(timestamps, args):
        if segment_end - segment_start < x_len + y_len:
            continue
        freq_id = _resolve_freq_id(args, freq_minutes)
        max_start = segment_end - x_len - y_len
        for start_idx in range(segment_start, max_start + 1):
            x_window = segment[start_idx:start_idx + x_len]
            if np.all(x_window == 0.0):
                continue
            y_window_mask = segment_mask[start_idx + x_len:start_idx + x_len + y_len]
            if not np.any(y_window_mask > 0.0):
                continue

            xs.append(x_window)
            feature_masks.append(feature_segment_mask[start_idx:start_idx + x_len])
            ys.append(segment[start_idx + x_len:start_idx + x_len + y_len, :, patv_col])
            ys_mask.append(y_window_mask)
            static_features.append(static_data)
            freq_ids.append(freq_id)
            stage_idxs.append(stage_idx)

    if not xs:
        return _empty_inputs(split_name, args.gcn["in_channel"])

    xs = np.asarray(xs, dtype=np.float32)
    feature_masks = np.asarray(feature_masks, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    ys_mask = np.asarray(ys_mask, dtype=np.float32)
    xs = np.nan_to_num(normalize_x(xs, args), nan=0.0, posinf=0.0, neginf=0.0)
    xs = _apply_feature_mask_after_normalization(xs, feature_masks)
    xs = _window_to_flat(xs)
    ys = np.nan_to_num(normalize_y(ys, args), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        f"{split_name}_x": xs,
        f"{split_name}_y": ys,
        f"{split_name}_y_mask": ys_mask,
        f"{split_name}_static_data": np.asarray(static_features, dtype=np.float32),
        f"{split_name}_freq_id": np.asarray(freq_ids, dtype=np.int64),
        f"{split_name}_stage_idx": np.asarray(stage_idxs, dtype=np.int64),
    }


def _build_standard_inputs(
    raw_data,
    feature_observed_mask,
    patv_mask,
    timestamps,
    initial_cols,
    start_idx,
    end_idx,
    args,
    split_name,
):
    segment = raw_data[start_idx:end_idx][:, initial_cols, :]
    feature_segment_mask = feature_observed_mask[start_idx:end_idx][:, initial_cols, :]
    segment_mask = patv_mask[start_idx:end_idx][:, initial_cols]
    segment_timestamps = timestamps[start_idx:end_idx]
    return _build_window_inputs(
        segment,
        feature_segment_mask,
        segment_mask,
        segment_timestamps,
        args.static_data[initial_cols],
        0,
        args=args,
        split_name=split_name,
    )


def _build_pretrain_inputs(raw_data, feature_observed_mask, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args):
    train_inputs = _build_standard_inputs(
        raw_data,
        feature_observed_mask,
        patv_mask,
        args.raw_timestamps,
        initial_cols,
        0,
        pretrain_end_idx,
        args,
        "train",
    )
    val_inputs = _build_standard_inputs(
        raw_data,
        feature_observed_mask,
        patv_mask,
        args.raw_timestamps,
        initial_cols,
        pretrain_end_idx,
        val_end_idx,
        args,
        "val",
    )
    if train_inputs["train_x"].shape[0] == 0 or val_inputs["val_x"].shape[0] == 0:
        raise ValueError("Pretrain or validation split has no samples. Check the sdwpf preprocessing outputs.")
    return {**train_inputs, **val_inputs}


def _build_patchtst_window_inputs(segment, feature_segment_mask, segment_mask, timestamps, args, split_name):
    x_len = args.x_len
    y_len = args.y_len
    num_features = segment.shape[2]
    patv_col = num_features - 1

    xs, ys, ys_mask = [], [], []
    for segment_start, segment_end, _ in _constant_frequency_segments(timestamps, args):
        if segment_end - segment_start < x_len + y_len:
            continue
        max_start = segment_end - x_len - y_len
        for start_idx in range(segment_start, max_start + 1):
            x_window = segment[start_idx:start_idx + x_len]
            x_feature_mask = feature_segment_mask[start_idx:start_idx + x_len]
            y_window = segment[start_idx + x_len:start_idx + x_len + y_len, :, patv_col]
            y_window_mask = segment_mask[start_idx + x_len:start_idx + x_len + y_len]

            x_window = np.nan_to_num(normalize_x(x_window, args), nan=0.0, posinf=0.0, neginf=0.0)
            x_window = _apply_feature_mask_after_normalization(x_window, x_feature_mask)

            num_nodes = x_window.shape[1]
            for node_idx in range(num_nodes):
                node_x = x_window[:, node_idx, :]
                node_y = y_window[:, node_idx]
                node_y_mask = y_window_mask[:, node_idx].astype(np.float32)
                if np.all(node_x == 0.0):
                    continue
                if not np.any(node_y_mask > 0.0):
                    continue

                xs.append(node_x.reshape(-1))
                ys.append(np.nan_to_num(normalize_y(node_y, args), nan=0.0, posinf=0.0, neginf=0.0))
                ys_mask.append(node_y_mask)

    if not xs:
        return _empty_single_turbine_inputs(split_name, args.gcn["in_channel"], y_len)

    return {
        f"{split_name}_x": np.asarray(xs, dtype=np.float32),
        f"{split_name}_y": np.asarray(ys, dtype=np.float32),
        f"{split_name}_y_mask": np.asarray(ys_mask, dtype=np.float32),
    }


def _build_patchtst_standard_inputs(
    raw_data,
    feature_observed_mask,
    patv_mask,
    timestamps,
    initial_cols,
    start_idx,
    end_idx,
    args,
    split_name,
):
    segment = raw_data[start_idx:end_idx][:, initial_cols, :]
    feature_segment_mask = feature_observed_mask[start_idx:end_idx][:, initial_cols, :]
    segment_mask = patv_mask[start_idx:end_idx][:, initial_cols]
    segment_timestamps = timestamps[start_idx:end_idx]
    return _build_patchtst_window_inputs(
        segment,
        feature_segment_mask,
        segment_mask,
        segment_timestamps,
        args=args,
        split_name=split_name,
    )


def _build_patchtst_pretrain_inputs(raw_data, feature_observed_mask, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args):
    train_inputs = _build_patchtst_standard_inputs(
        raw_data,
        feature_observed_mask,
        patv_mask,
        args.raw_timestamps,
        initial_cols,
        0,
        pretrain_end_idx,
        args,
        "train",
    )
    val_inputs = _build_patchtst_standard_inputs(
        raw_data,
        feature_observed_mask,
        patv_mask,
        args.raw_timestamps,
        initial_cols,
        pretrain_end_idx,
        val_end_idx,
        args,
        "val",
    )
    if train_inputs["train_x"].shape[0] == 0 or val_inputs["val_x"].shape[0] == 0:
        raise ValueError("PatchTST pretrain or validation split has no samples after filtering.")
    return {**train_inputs, **val_inputs}


def _build_loader(inputs, split_name, args, shuffle):
    dataset = SpatioTemporalDataset(inputs, split_name)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def _build_patchtst_loader(inputs, split_name, args, shuffle):
    dataset = SingleTurbineDataset(inputs, split_name)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def _forward_and_loss(model, data, adj, args):
    out = model(data, adj)
    pred, kl_term = out if isinstance(out, tuple) else (out, None)
    pred_dense, _ = to_dense_batch(pred, batch=data.batch)
    y_dense, _ = to_dense_batch(data.y, batch=data.batch)
    mask_dense, _ = to_dense_batch(data.y_mask, batch=data.batch)
    loss = masked_mse_torch(pred_dense, y_dense, mask_dense)
    if kl_term is not None:
        loss = loss + args.kl_weight * kl_term
    return pred_dense, y_dense, mask_dense, loss


def _patchtst_forward_and_loss(model, data, args):
    pred = model(data)
    y = data.y.squeeze(1)
    y_mask = data.y_mask.squeeze(1)
    loss = masked_mse_torch(pred, y, y_mask)
    return pred, y, y_mask, loss


def pretrain(raw_data, feature_observed_mask, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args):
    if args.method == "PatchTST":
        return pretrain_patchtst(raw_data, feature_observed_mask, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args)

    get_checkpoints_dir(args)
    inputs = _build_pretrain_inputs(
        raw_data,
        feature_observed_mask,
        patv_mask,
        initial_cols,
        pretrain_end_idx,
        val_end_idx,
        args,
    )
    train_loader = _build_loader(inputs, "train", args, shuffle=True)
    val_loader = _build_loader(inputs, "val", args, shuffle=False)

    args.graph_size = len(initial_cols)
    model = args.methods[args.method](args).to(args.device)
    model.expand_adaptive_params(len(initial_cols))
    args.adj = rebuild_adj(0, args)
    args.current_stage_idx = 0
    model.count_parameters()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    patience = 5
    stale_epochs = 0

    for epoch in range(args.epoch):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, _, _, loss = _forward_and_loss(model, batch, args.adj, args)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device, non_blocking=True)
                pred_dense, y_dense, mask_dense, _ = _forward_and_loss(model, batch, args.adj, args)
                val_losses.append(
                    masked_mae_np_with_mask(y_dense.cpu().numpy(), pred_dense.cpu().numpy(), mask_dense.cpu().numpy())
                )

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        args.logger.info(f"[*] Pretrain epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, get_best_checkpoint_path(args))
        else:
            stale_epochs += 1
            if stale_epochs > patience:
                break
        torch.save({"model_state_dict": model.state_dict()}, get_last_checkpoint_path(args))

    args.best_val_metric = float(best_val)
    args.best_checkpoint_path = get_best_checkpoint_path(args)
    args.last_checkpoint_path = get_last_checkpoint_path(args)


def pretrain_patchtst(raw_data, feature_observed_mask, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args):
    get_checkpoints_dir(args)
    inputs = _build_patchtst_pretrain_inputs(
        raw_data,
        feature_observed_mask,
        patv_mask,
        initial_cols,
        pretrain_end_idx,
        val_end_idx,
        args,
    )
    train_loader = _build_patchtst_loader(inputs, "train", args, shuffle=True)
    val_loader = _build_patchtst_loader(inputs, "val", args, shuffle=False)

    model = args.methods[args.method](args).to(args.device)
    model.count_parameters()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    patience = 5
    stale_epochs = 0

    for epoch in range(args.epoch):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, _, _, loss = _patchtst_forward_and_loss(model, batch, args)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device, non_blocking=True)
                pred, y, y_mask, _ = _patchtst_forward_and_loss(model, batch, args)
                val_losses.append(masked_mae_np_with_mask(y.cpu().numpy(), pred.cpu().numpy(), y_mask.cpu().numpy()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        args.logger.info(f"[*] PatchTST pretrain epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, get_best_checkpoint_path(args))
        else:
            stale_epochs += 1
            if stale_epochs > patience:
                break
        torch.save({"model_state_dict": model.state_dict()}, get_last_checkpoint_path(args))

    args.best_val_metric = float(best_val)
    args.best_checkpoint_path = get_best_checkpoint_path(args)
    args.last_checkpoint_path = get_last_checkpoint_path(args)


def load_pretrained_model(args, n_turbines):
    best_path = get_best_checkpoint_path(args)
    fallback_dir = getattr(args, "checkpoint_fallback_path", None)
    if not osp.exists(best_path) and fallback_dir:
        fallback_best = osp.join(fallback_dir, "checkpoints", "best.pt")
        if osp.exists(fallback_best):
            best_path = fallback_best
    if not osp.exists(best_path):
        raise FileNotFoundError(f"No pretrained checkpoint was found at {best_path}.")
    model = args.methods[args.method](args).to(args.device)
    model.expand_adaptive_params(n_turbines)
    checkpoint = torch.load(best_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    args.best_checkpoint_path = best_path
    if best_path == get_best_checkpoint_path(args):
        args.last_checkpoint_path = get_last_checkpoint_path(args)
    else:
        args.last_checkpoint_path = osp.join(osp.dirname(best_path), "last.pt")
    return model


def _build_patchtst_streaming_step(raw_data, feature_observed_mask, patv_mask, current_cols, start_idx, args):
    x_len = args.x_len
    y_len = args.y_len
    patv_col = raw_data.shape[2] - 1

    x_raw = raw_data[start_idx:start_idx + x_len][:, current_cols, :]
    x_feature_mask = feature_observed_mask[start_idx:start_idx + x_len][:, current_cols, :]
    y_raw = raw_data[start_idx + x_len:start_idx + x_len + y_len][:, current_cols, patv_col]
    y_mask = patv_mask[start_idx + x_len:start_idx + x_len + y_len][:, current_cols].astype(np.float32)
    if x_raw.shape[0] != x_len or y_raw.shape[0] != y_len:
        raise ValueError(
            f"Invalid PatchTST streaming sample at start_idx={start_idx}. x_raw.shape={x_raw.shape}, y_raw.shape={y_raw.shape}."
        )

    x_norm = np.nan_to_num(normalize_x(x_raw, args), nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = _apply_feature_mask_after_normalization(x_norm, x_feature_mask)
    y_norm = np.nan_to_num(normalize_y(y_raw, args), nan=0.0, posinf=0.0, neginf=0.0)

    xs, ys, ys_mask = [], [], []
    for node_idx in range(len(current_cols)):
        node_x = x_norm[:, node_idx, :]
        node_y_mask = y_mask[:, node_idx]
        xs.append(node_x.reshape(-1))
        ys.append(y_norm[:, node_idx])
        ys_mask.append(node_y_mask)

    batch = Data(
        x=torch.from_numpy(np.asarray(xs, dtype=np.float32)).float(),
        y=torch.from_numpy(np.asarray(ys, dtype=np.float32)).float(),
        y_mask=torch.from_numpy(np.asarray(ys_mask, dtype=np.float32)).float(),
    )
    return batch


def _build_graph_sample(raw_data, feature_observed_mask, patv_mask, current_cols, start_idx, freq_id, args):
    x_len = args.x_len
    y_len = args.y_len
    patv_col = raw_data.shape[2] - 1

    x_raw = raw_data[start_idx:start_idx + x_len][:, current_cols, :]
    x_feature_mask = feature_observed_mask[start_idx:start_idx + x_len][:, current_cols, :]
    y_raw = raw_data[start_idx + x_len:start_idx + x_len + y_len][:, current_cols, patv_col]
    y_mask = patv_mask[start_idx + x_len:start_idx + x_len + y_len][:, current_cols]
    if x_raw.shape[0] != x_len or y_raw.shape[0] != y_len:
        raise ValueError(
            f"Invalid streaming sample at start_idx={start_idx}. x_raw.shape={x_raw.shape}, y_raw.shape={y_raw.shape}."
        )

    x_norm = np.nan_to_num(normalize_x(x_raw, args), nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = _apply_feature_mask_after_normalization(x_norm, x_feature_mask)
    y_norm = np.nan_to_num(normalize_y(y_raw, args), nan=0.0, posinf=0.0, neginf=0.0)
    x_flat = _window_to_flat(x_norm[np.newaxis, ...])[0]
    return Data(
        x=torch.from_numpy(x_flat.T).float(),
        y=torch.from_numpy(y_norm.T).float(),
        y_mask=torch.from_numpy(y_mask.T.astype(np.float32)).float(),
        static_data=torch.from_numpy(args.static_data[current_cols]).float(),
        freq_id=torch.tensor([int(freq_id)], dtype=torch.long),
        stage_idx=torch.tensor([int(getattr(args, "current_stage_idx", 0))], dtype=torch.long),
    )


def _build_streaming_plan(timestamps, args):
    return build_streaming_plan(timestamps, args.x_len, args.y_len, _allowed_frequency_minutes(args))


def _register_node_mask_hook(param, new_local_idx, args):
    mask = torch.zeros(param.shape[0], dtype=torch.float32, device=args.device)
    for idx in new_local_idx:
        if idx < param.shape[0]:
            mask[idx] = 1.0
    if param.dim() > 1:
        mask = mask.unsqueeze(1).expand_as(param)

    def hook(grad):
        return grad * mask

    return param.register_hook(hook)


def warmup_update(model, samples, new_local_idx, adj, args):
    if not args.use_warmup or not samples:
        return
    if not any(_is_adaptive_param(name) for name, _ in model.named_parameters()):
        return

    for _, param in model.named_parameters():
        param.requires_grad = False

    hooks = []
    for name, param in model.named_parameters():
        if _is_global_adaptive_param(name):
            param.requires_grad = True
        elif _is_node_adaptive_param(name):
            param.requires_grad = True
            hooks.append(_register_node_mask_hook(param, new_local_idx, args))

    optimizer = optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.warmup_lr)
    model.train()
    for _ in range(args.warmup_gradient_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=args.device)
        for sample in samples:
            sample = sample.to(args.device)
            out = model(sample, adj)
            pred, kl_term = out if isinstance(out, tuple) else (out, None)
            loss = masked_mse_torch(pred, sample.y, sample.y_mask)
            if kl_term is not None:
                loss = loss + args.kl_weight * kl_term
            total_loss = total_loss + loss
        (total_loss / len(samples)).backward()
        optimizer.step()

    for hook in hooks:
        hook.remove()
    freeze_backbone(model)
    _set_adaptive_requires_grad(model, True)
    model.eval()


def streaming_test(raw_data, feature_observed_mask, patv_mask, turbine_schedule, initial_cols, timestamps, args):
    if args.method == "PatchTST":
        return streaming_test_patchtst(raw_data, feature_observed_mask, patv_mask, turbine_schedule, initial_cols, timestamps, args)

    if raw_data.shape[0] != len(timestamps):
        raise ValueError(
            f"raw_data and timestamps must align. raw_data.shape[0]={raw_data.shape[0]}, len(timestamps)={len(timestamps)}."
        )

    model = load_pretrained_model(args, len(initial_cols))
    freeze_backbone(model)
    _set_adaptive_requires_grad(model, True)
    model.eval()

    current_stage = 0
    current_cols = list(initial_cols)
    args.adj = rebuild_adj(current_stage, args)
    args.current_stage_idx = current_stage
    expansion_lookup = {int(offset): list(new_cols) for _, (offset, new_cols) in turbine_schedule.items()}
    stream_plan = _build_streaming_plan(timestamps, args)
    if not stream_plan:
        raise ValueError("The streaming test split has no valid constant-frequency windows.")
    valid_stream_starts = {start_idx for start_idx, _ in stream_plan}
    missing_expansion_offsets = sorted(offset for offset in expansion_lookup if offset not in valid_stream_starts)
    if missing_expansion_offsets:
        raise ValueError(
            f"Turbine expansion offsets do not align with valid streaming prediction starts: {missing_expansion_offsets}."
        )

    warmup_state = None
    pending_sample = None
    all_preds, all_truths, all_masks = [], [], []

    for t, freq_minutes in stream_plan:
        decision_time = timestamps[t + args.x_len - 1]

        if t in expansion_lookup:
            new_global_cols = expansion_lookup[t]
            old_n = len(current_cols)
            current_cols = sorted(current_cols + new_global_cols)
            current_stage += 1
            model.expand_adaptive_params(len(current_cols))
            freeze_backbone(model)
            _set_adaptive_requires_grad(model, True)
            args.adj = rebuild_adj(current_stage, args)
            args.current_stage_idx = current_stage
            warmup_state = None
            pending_sample = None
            if args.use_warmup and args.method != "PatchTST":
                warmup_state = {
                    "new_local_idx": list(range(old_n, len(current_cols))),
                    "deadline": decision_time + np.timedelta64(args.warmup_days, "D"),
                }

        if warmup_state is not None and pending_sample is not None:
            if decision_time <= warmup_state["deadline"]:
                warmup_update(model, [pending_sample], warmup_state["new_local_idx"], args.adj, args)
            else:
                warmup_state = None

        freq_id = _resolve_freq_id(args, freq_minutes)
        sample = _build_graph_sample(raw_data, feature_observed_mask, patv_mask, current_cols, t, freq_id, args)
        with torch.no_grad():
            out = model(sample.to(args.device), args.adj)
        pred = out[0] if isinstance(out, tuple) else out

        pred_np = pred.detach().cpu().numpy()
        truth_np = sample.y.detach().cpu().numpy()
        mask_np = sample.y_mask.detach().cpu().numpy()
        all_preds.append(pred_np)
        all_truths.append(truth_np)
        all_masks.append(mask_np)

        pending_sample = sample

    preds_raw = [np.clip(denormalize_y(pred, args), 0.0, None) for pred in all_preds]
    truths_raw = [denormalize_y(truth, args) for truth in all_truths]

    horizon_steps = [3, 6, 12]
    horizon_results = {}
    for horizon in horizon_steps:
        mae_values, rmse_values, mape_values = [], [], []
        for pred, truth, mask in zip(preds_raw, truths_raw, all_masks):
            mae_values.append(masked_mae_np_with_mask(truth[:, :horizon], pred[:, :horizon], mask[:, :horizon]))
            rmse_values.append(masked_mse_np_with_mask(truth[:, :horizon], pred[:, :horizon], mask[:, :horizon]) ** 0.5)
            mape_values.append(masked_mape_np_with_mask(truth[:, :horizon], pred[:, :horizon], mask[:, :horizon]))
        horizon_results[horizon] = {
            "MAE": float(np.mean(mae_values)),
            "RMSE": float(np.mean(rmse_values)),
            "MAPE": float(np.mean(mape_values)),
        }
        args.logger.info(
            f"[*] T={horizon:2d} MAE {horizon_results[horizon]['MAE']:.4f} / RMSE {horizon_results[horizon]['RMSE']:.4f} / MAPE {horizon_results[horizon]['MAPE']:.4f}"
        )

    avg_mae = float(np.mean([masked_mae_np_with_mask(truth, pred, mask) for pred, truth, mask in zip(preds_raw, truths_raw, all_masks)]))
    avg_rmse = float(np.mean([masked_mse_np_with_mask(truth, pred, mask) ** 0.5 for pred, truth, mask in zip(preds_raw, truths_raw, all_masks)]))
    avg_mape = float(np.mean([masked_mape_np_with_mask(truth, pred, mask) for pred, truth, mask in zip(preds_raw, truths_raw, all_masks)]))

    args.result = {
        "streaming": {
            "T3": horizon_results[3],
            "T6": horizon_results[6],
            "T12": horizon_results[12],
            "Avg": {"MAE": avg_mae, "RMSE": avg_rmse, "MAPE": avg_mape},
        }
    }

    predictions_path = get_predictions_path(args)
    np.savez(
        predictions_path,
        preds=np.array(preds_raw, dtype=object),
        truths=np.array(truths_raw, dtype=object),
        masks=np.array(all_masks, dtype=object),
    )
    args.predictions_path = predictions_path


def streaming_test_patchtst(raw_data, feature_observed_mask, patv_mask, turbine_schedule, initial_cols, timestamps, args):
    if raw_data.shape[0] != len(timestamps):
        raise ValueError(
            f"raw_data and timestamps must align. raw_data.shape[0]={raw_data.shape[0]}, len(timestamps)={len(timestamps)}."
        )

    model = load_pretrained_model(args, len(initial_cols))
    model.eval()

    current_cols = list(initial_cols)
    expansion_lookup = {int(offset): list(new_cols) for _, (offset, new_cols) in turbine_schedule.items()}
    stream_plan = _build_streaming_plan(timestamps, args)
    if not stream_plan:
        raise ValueError("The PatchTST streaming test split has no valid constant-frequency windows.")
    valid_stream_starts = {start_idx for start_idx, _ in stream_plan}
    missing_expansion_offsets = sorted(offset for offset in expansion_lookup if offset not in valid_stream_starts)
    if missing_expansion_offsets:
        raise ValueError(
            f"Turbine expansion offsets do not align with valid PatchTST streaming prediction starts: {missing_expansion_offsets}."
        )

    all_preds, all_truths, all_masks = [], [], []

    for t, _ in stream_plan:
        if t in expansion_lookup:
            current_cols = sorted(current_cols + expansion_lookup[t])

        batch = _build_patchtst_streaming_step(raw_data, feature_observed_mask, patv_mask, current_cols, t, args)

        with torch.no_grad():
            pred = model(batch.to(args.device))

        pred_np = pred.detach().cpu().numpy()
        truth_np = batch.y.detach().cpu().numpy()
        mask_np = batch.y_mask.detach().cpu().numpy()
        all_preds.append(pred_np)
        all_truths.append(truth_np)
        all_masks.append(mask_np)

    preds_raw = [np.clip(denormalize_y(pred, args), 0.0, None) for pred in all_preds]
    truths_raw = [denormalize_y(truth, args) for truth in all_truths]

    horizon_steps = [3, 6, 12]
    horizon_results = {}
    for horizon in horizon_steps:
        mae_values, rmse_values, mape_values = [], [], []
        for pred, truth, mask in zip(preds_raw, truths_raw, all_masks):
            mae_values.append(masked_mae_np_with_mask(truth[:, :horizon], pred[:, :horizon], mask[:, :horizon]))
            rmse_values.append(masked_mse_np_with_mask(truth[:, :horizon], pred[:, :horizon], mask[:, :horizon]) ** 0.5)
            mape_values.append(masked_mape_np_with_mask(truth[:, :horizon], pred[:, :horizon], mask[:, :horizon]))
        horizon_results[horizon] = {
            "MAE": float(np.mean(mae_values)),
            "RMSE": float(np.mean(rmse_values)),
            "MAPE": float(np.mean(mape_values)),
        }
        args.logger.info(
            f"[*] PatchTST T={horizon:2d} MAE {horizon_results[horizon]['MAE']:.4f} / RMSE {horizon_results[horizon]['RMSE']:.4f} / MAPE {horizon_results[horizon]['MAPE']:.4f}"
        )

    avg_mae = float(np.mean([masked_mae_np_with_mask(truth, pred, mask) for pred, truth, mask in zip(preds_raw, truths_raw, all_masks)]))
    avg_rmse = float(np.mean([masked_mse_np_with_mask(truth, pred, mask) ** 0.5 for pred, truth, mask in zip(preds_raw, truths_raw, all_masks)]))
    avg_mape = float(np.mean([masked_mape_np_with_mask(truth, pred, mask) for pred, truth, mask in zip(preds_raw, truths_raw, all_masks)]))

    args.result = {
        "streaming": {
            "T3": horizon_results[3],
            "T6": horizon_results[6],
            "T12": horizon_results[12],
            "Avg": {"MAE": avg_mae, "RMSE": avg_rmse, "MAPE": avg_mape},
        }
    }

    predictions_path = get_predictions_path(args)
    np.savez(
        predictions_path,
        preds=np.array(preds_raw, dtype=object),
        truths=np.array(truths_raw, dtype=object),
        masks=np.array(all_masks, dtype=object),
    )
    args.predictions_path = predictions_path
