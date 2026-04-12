import os
import os.path as osp

import numpy as np
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from data.dataset import SpatioTemporalDataset
from util.training_utils import masked_mae_np_with_mask, masked_mse_np_with_mask, masked_mape_np_with_mask


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_x(raw_x, args, eps=1e-6):
    return (raw_x - args.x_mean) / (args.x_std + eps)


def normalize_y(raw_y, args, eps=1e-6):
    return (raw_y - args.y_mean) / (args.y_std + eps)


def denormalize_y(norm_y, args, eps=1e-6):
    return norm_y * (args.y_std + eps) + args.y_mean


def masked_mse_torch(pred, target, mask, eps=1e-6):
    err = (pred - target) ** 2
    return torch.sum(err * mask) / (torch.sum(mask) + eps)


def rebuild_adj(stage_idx, args):
    adj_path = osp.join(args.graph_path, f'stage_{stage_idx}_adj.npz')
    adj = np.load(adj_path)['x']
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)


def freeze_backbone(model):
    model.freeze_backbone()


def _is_adaptive_param(name):
    adaptive_keys = ('U', 'V', 'scale', 'shift', 'mu_scale', 'mu_shift', 'log_var_scale', 'log_var_shift')
    return any(name == k or name.endswith('.' + k) for k in adaptive_keys)


def _set_adaptive_requires_grad(model, value):
    for name, param in model.named_parameters():
        if _is_adaptive_param(name):
            param.requires_grad = value


def _window_to_flat(x_win):
    return x_win.transpose(0, 2, 1, 3).reshape(x_win.shape[0], x_win.shape[2], x_win.shape[1] * x_win.shape[3]).transpose(0, 2, 1)


def _empty_inputs(split_name):
    empty_long = np.empty((0, 0), dtype=np.int64)
    return {
        f'{split_name}_x': np.empty((0, 0, 0), dtype=np.float32),
        f'{split_name}_y': np.empty((0, 0, 0), dtype=np.float32),
        f'{split_name}_y_mask': np.empty((0, 0, 0), dtype=np.float32),
        f'{split_name}_static_x_bin': empty_long,
        f'{split_name}_static_y_bin': empty_long,
        f'{split_name}_freq_id': np.empty((0,), dtype=np.int64),
    }


def _build_standard_inputs(raw_data, patv_mask, initial_cols, start_idx, end_idx, args, split_name='train'):
    x_len = args.x_len
    y_len = args.y_len
    D = raw_data.shape[2]
    patv_col = D - 1
    segment = raw_data[start_idx:end_idx, :, :][:, initial_cols, :]
    seg_mask = patv_mask[start_idx:end_idx, :][:, initial_cols]
    static_x_bin = args.static_x_bin[initial_cols]
    static_y_bin = args.static_y_bin[initial_cols]
    xs, ys, ys_mask, x_bins, y_bins, freq_ids = [], [], [], [], [], []
    for t in range(segment.shape[0] - x_len - y_len + 1):
        xs.append(segment[t:t + x_len, :, :])
        ys.append(segment[t + x_len:t + x_len + y_len, :, patv_col])
        ys_mask.append(seg_mask[t + x_len:t + x_len + y_len, :])
        x_bins.append(static_x_bin)
        y_bins.append(static_y_bin)
        freq_ids.append(1)
    if not xs:
        return _empty_inputs(split_name)
    xs = _window_to_flat(np.array(xs, dtype=np.float32))
    ys = np.array(ys, dtype=np.float32)
    ys_mask = np.array(ys_mask, dtype=np.float32)
    xs = np.nan_to_num(normalize_x(xs, args), nan=0.0, posinf=0.0, neginf=0.0)
    ys = np.nan_to_num(normalize_y(ys, args), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        f'{split_name}_x': xs,
        f'{split_name}_y': ys,
        f'{split_name}_y_mask': ys_mask,
        f'{split_name}_static_x_bin': np.array(x_bins, dtype=np.int64),
        f'{split_name}_static_y_bin': np.array(y_bins, dtype=np.int64),
        f'{split_name}_freq_id': np.array(freq_ids, dtype=np.int64),
    }


def _build_multifreq_inputs(raw_data_base, patv_mask_base, initial_cols, args, start_ratio, end_ratio, split_name):
    x_len = args.x_len
    y_len = args.y_len
    D = raw_data_base.shape[2]
    patv_col = D - 1
    T = raw_data_base.shape[0]
    start_idx = int(T * start_ratio)
    end_idx = int(T * end_ratio)
    xs, ys, ys_mask, x_bins, y_bins, freq_ids = [], [], [], [], [], []
    static_x_bin = args.static_x_bin[initial_cols]
    static_y_bin = args.static_y_bin[initial_cols]
    freq_map = {5: 0, 10: 1, 15: 2}
    for freq in args.pretrain_freq_minutes:
        step = freq // args.base_resolution_minutes
        segment = raw_data_base[start_idx:end_idx:step, :, :][:, initial_cols, :]
        seg_mask = patv_mask_base[start_idx:end_idx:step, :][:, initial_cols]
        for t in range(segment.shape[0] - x_len - y_len + 1):
            xs.append(segment[t:t + x_len, :, :])
            ys.append(segment[t + x_len:t + x_len + y_len, :, patv_col])
            ys_mask.append(seg_mask[t + x_len:t + x_len + y_len, :])
            x_bins.append(static_x_bin)
            y_bins.append(static_y_bin)
            freq_ids.append(freq_map[freq])
    if not xs:
        return _empty_inputs(split_name)
    xs = _window_to_flat(np.array(xs, dtype=np.float32))
    ys = np.array(ys, dtype=np.float32)
    ys_mask = np.array(ys_mask, dtype=np.float32)
    xs = np.nan_to_num(normalize_x(xs, args), nan=0.0, posinf=0.0, neginf=0.0)
    ys = np.nan_to_num(normalize_y(ys, args), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        f'{split_name}_x': xs,
        f'{split_name}_y': ys,
        f'{split_name}_y_mask': ys_mask,
        f'{split_name}_static_x_bin': np.array(x_bins, dtype=np.int64),
        f'{split_name}_static_y_bin': np.array(y_bins, dtype=np.int64),
        f'{split_name}_freq_id': np.array(freq_ids, dtype=np.int64),
    }


def _build_pretrain_inputs(raw_data, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args):
    if args.streaming_freq_mode == 'dynamic' and len(args.pretrain_freq_minutes) > 0:
        train_inputs = _build_multifreq_inputs(args.raw_data_base, args.patv_mask_base, initial_cols, args, 0.0, 0.2, 'train')
        val_inputs = _build_multifreq_inputs(args.raw_data_base, args.patv_mask_base, initial_cols, args, 0.2, 0.3, 'val')
    else:
        train_inputs = _build_standard_inputs(raw_data, patv_mask, initial_cols, 0, pretrain_end_idx, args, 'train')
        val_inputs = _build_standard_inputs(raw_data, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args, 'val')
    if train_inputs['train_x'].shape[0] == 0 or val_inputs['val_x'].shape[0] == 0:
        raise ValueError('Pretrain or validation split has no samples. Please check data length and split settings.')
    return {**train_inputs, **val_inputs}


def pretrain(raw_data, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args):
    path = osp.join(args.path, 'pretrain')
    mkdirs(path)
    inputs = _build_pretrain_inputs(raw_data, patv_mask, initial_cols, pretrain_end_idx, val_end_idx, args)
    train_loader = DataLoader(SpatioTemporalDataset(inputs, 'train'), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(SpatioTemporalDataset(inputs, 'val'), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    vars(args)['graph_size'] = len(initial_cols)
    model = args.methods[args.method](args).to(args.device)
    model.expand_adaptive_params(len(initial_cols))
    vars(args)['adj'] = rebuild_adj(0, args)
    model.count_parameters()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lowest_val_loss = 1e9
    counter = 0
    patience = 10
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        cn = 0
        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            out = model(data, args.adj)
            pred, kl_term = out if isinstance(out, tuple) else (out, None)
            pred_dense, _ = to_dense_batch(pred, batch=data.batch)
            y_dense, _ = to_dense_batch(data.y, batch=data.batch)
            m_dense, _ = to_dense_batch(data.y_mask, batch=data.batch)
            loss = masked_mse_torch(pred_dense, y_dense, m_dense)
            if kl_term is not None:
                loss = loss + args.kl_weight * kl_term
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
            cn += 1
        train_loss /= cn
        model.eval()
        val_loss = 0.0
        cn = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(args.device, non_blocking=True)
                out = model(data, args.adj)
                pred = out[0] if isinstance(out, tuple) else out
                pred_dense, _ = to_dense_batch(pred, batch=data.batch)
                y_dense, _ = to_dense_batch(data.y, batch=data.batch)
                m_dense, _ = to_dense_batch(data.y_mask, batch=data.batch)
                val_loss += masked_mae_np_with_mask(y_dense.cpu().numpy(), pred_dense.cpu().numpy(), m_dense.cpu().numpy())
                cn += 1
        val_loss /= cn
        args.logger.info(f'[*] Pretrain: epoch {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        if val_loss < lowest_val_loss:
            counter = 0
            lowest_val_loss = round(val_loss, 4)
            torch.save({'model_state_dict': model.state_dict()}, osp.join(path, f'{lowest_val_loss}.pkl'))
        else:
            counter += 1
            if counter > patience:
                break


def load_pretrained_model(args, n_turbines):
    pretrain_dir = osp.join(args.path, 'pretrain')
    ckpt_files = [f for f in os.listdir(pretrain_dir) if f.endswith('.pkl')]
    best_file = min(ckpt_files, key=lambda f: float(osp.splitext(f)[0]))
    ckpt_path = osp.join(pretrain_dir, best_file)
    vars(args)['graph_size'] = n_turbines
    model = args.methods[args.method](args).to(args.device)
    model.expand_adaptive_params(n_turbines)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def _infer_dynamic_window_freq_minutes(timestamps, t, x_len):
    end = min(t + x_len, len(timestamps))
    if end - t < 2:
        return 10
    diffs = np.diff(timestamps[t:end]).astype('timedelta64[m]').astype(np.int64)
    return int(np.median(diffs))


def warmup_update(model, warmup_raw, warmup_mask, static_x_bin, static_y_bin, current_cols, new_cols_local_idx, adj, args, freq_id):
    has_adaptive = any(_is_adaptive_param(n) for n, _ in model.named_parameters())
    if not has_adaptive:
        return
    x_len = args.x_len
    y_len = args.y_len
    D = warmup_raw.shape[2]
    patv_col = D - 1
    xs, ys, ys_mask = [], [], []
    for t in range(warmup_raw.shape[0] - x_len - y_len + 1):
        xs.append(warmup_raw[t:t + x_len, :, :])
        ys.append(warmup_raw[t + x_len:t + x_len + y_len, :, patv_col])
        ys_mask.append(warmup_mask[t + x_len:t + x_len + y_len, :])
    if not xs:
        return
    xs = _window_to_flat(np.array(xs, dtype=np.float32))
    ys = np.array(ys, dtype=np.float32)
    ys_mask = np.array(ys_mask, dtype=np.float32)
    xs = np.nan_to_num(normalize_x(xs, args), nan=0.0, posinf=0.0, neginf=0.0)
    ys = np.nan_to_num(normalize_y(ys, args), nan=0.0, posinf=0.0, neginf=0.0)
    warmup_inputs = {
        'warmup_x': xs,
        'warmup_y': ys,
        'warmup_y_mask': ys_mask,
        'warmup_static_x_bin': np.repeat(static_x_bin[np.newaxis, :], xs.shape[0], axis=0).astype(np.int64),
        'warmup_static_y_bin': np.repeat(static_y_bin[np.newaxis, :], xs.shape[0], axis=0).astype(np.int64),
        'warmup_freq_id': np.full((xs.shape[0],), freq_id, dtype=np.int64),
    }
    warmup_dataset = SpatioTemporalDataset(warmup_inputs, 'warmup')
    for p in model.parameters():
        p.requires_grad = False
    hooks = []
    new_idx_set = set(new_cols_local_idx)

    def _make_mask_hook(param_shape):
        mask = torch.zeros(param_shape[0], dtype=torch.float32, device=args.device)
        for i in new_idx_set:
            if i < param_shape[0]:
                mask[i] = 1.0
        if len(param_shape) > 1:
            mask = mask.unsqueeze(1).expand(param_shape)

        def hook(grad):
            return grad * mask

        return hook

    for name, param in model.named_parameters():
        if _is_adaptive_param(name):
            param.requires_grad = True
            hooks.append(param.register_hook(_make_mask_hook(param.shape)))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.warmup_lr)
    batch_size = min(32, len(warmup_dataset))
    model.train()
    for _ in range(10):
        perm = torch.randperm(len(warmup_dataset))
        for start in range(0, len(warmup_dataset), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=args.device)
            for i in idx.tolist():
                data = warmup_dataset[i].to(args.device)
                if data.x.shape[0] != adj.shape[0] or data.x.shape[1] != args.gcn['in_channel']:
                    raise ValueError(
                        f'Warmup sample shape mismatch: x={tuple(data.x.shape)}, adj={tuple(adj.shape)}, '
                        f'expected node_dim={args.gcn["in_channel"]}'
                )
                out = model(data, adj)
                pred = out[0] if isinstance(out, tuple) else out
                kl_term = out[1] if isinstance(out, tuple) else None
                loss = masked_mse_torch(pred, data.y, data.y_mask)
                if kl_term is not None:
                    loss = loss + args.kl_weight * kl_term
                total_loss = total_loss + loss
            (total_loss / len(idx)).backward()
            optimizer.step()
    for h in hooks:
        h.remove()
    _set_adaptive_requires_grad(model, True)
    model.eval()


def streaming_test(raw_data, patv_mask, turbine_schedule, initial_cols, timestamps, args):
    no_warmup = getattr(args, 'no_warmup', False)
    x_len = args.x_len
    y_len = args.y_len
    D = raw_data.shape[2]
    patv_col = D - 1
    T_test = raw_data.shape[0]
    t_to_expansion = {t_off: (exp_idx, new_cols) for exp_idx, (t_off, new_cols) in turbine_schedule.items()}
    n_current = len(initial_cols)
    model = load_pretrained_model(args, n_current)
    freeze_backbone(model)
    _set_adaptive_requires_grad(model, True)
    model.eval()
    current_stage = 0
    vars(args)['adj'] = rebuild_adj(current_stage, args)
    current_cols = list(initial_cols)
    all_preds, all_truths, all_masks = [], [], []
    freq_map = {5: 0, 10: 1, 15: 2}
    fixed_freq_id = 0
    t = 0
    while t + x_len + y_len <= T_test:
        current_freq_minutes = _infer_dynamic_window_freq_minutes(timestamps, t, x_len) if args.streaming_freq_mode == 'dynamic' else args.frequency_minutes
        current_freq_id = freq_map[current_freq_minutes] if args.streaming_freq_mode == 'dynamic' else fixed_freq_id
        if t in t_to_expansion:
            _, new_global_cols = t_to_expansion[t]
            current_stage += 1
            old_n_current = len(current_cols)
            current_cols = sorted(dict.fromkeys(current_cols + list(new_global_cols)))
            n_current = len(current_cols)
            new_local_idx = list(range(old_n_current, n_current))
            model.expand_adaptive_params(n_current)
            freeze_backbone(model)
            _set_adaptive_requires_grad(model, True)
            vars(args)['adj'] = rebuild_adj(current_stage, args)
            steps_per_day = int((24 * 60) / current_freq_minutes)
            warmup_steps = args.warmup_days * steps_per_day
            warmup_end = min(t + warmup_steps, T_test)
            if not no_warmup:
                warmup_update(
                    model,
                    raw_data[t:warmup_end, :, :][:, current_cols, :],
                    patv_mask[t:warmup_end, :][:, current_cols],
                    args.static_x_bin[current_cols],
                    args.static_y_bin[current_cols],
                    current_cols,
                    new_local_idx,
                    args.adj,
                    args,
                    current_freq_id,
                )
                has_adaptive = any(_is_adaptive_param(n) for n, _ in model.named_parameters())
                if has_adaptive:
                    t += (warmup_end - t)
                    t = (t // y_len) * y_len
                    continue
        x_raw = raw_data[t:t + x_len, :, :][:, current_cols, :]
        x_norm = np.nan_to_num(normalize_x(x_raw, args), nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = x_norm.transpose(1, 0, 2).reshape(len(current_cols), x_len * D).T
        y_raw = raw_data[t + x_len:t + x_len + y_len, :, patv_col][:, current_cols]
        y_mask = patv_mask[t + x_len:t + x_len + y_len, :][:, current_cols]
        y_norm = np.nan_to_num(normalize_y(y_raw, args), nan=0.0, posinf=0.0, neginf=0.0)
        from torch_geometric.data import Data
        data = Data(
            x=torch.from_numpy(x_flat).float().to(args.device).T,
            static_x_bin=torch.from_numpy(args.static_x_bin[current_cols]).long().to(args.device),
            static_y_bin=torch.from_numpy(args.static_y_bin[current_cols]).long().to(args.device),
            freq_id=torch.tensor([current_freq_id], dtype=torch.long, device=args.device),
        )
        with torch.no_grad():
            out = model(data, args.adj)
        pred = out[0] if isinstance(out, tuple) else out
        all_preds.append(pred.cpu().numpy())
        all_truths.append(y_norm.T)
        all_masks.append(y_mask.T)
        t += y_len
    preds_raw_list = [np.clip(denormalize_y(p, args), 0, None) for p in all_preds]
    truths_raw_list = [denormalize_y(tr, args) for tr in all_truths]
    masks_list = all_masks
    horizon_steps = [3, 6, 12]
    horizon_results = {}
    for h in horizon_steps:
        mae_list, rmse_list, mape_list = [], [], []
        for p, tr, m in zip(preds_raw_list, truths_raw_list, masks_list):
            mae_list.append(masked_mae_np_with_mask(tr[:, :h], p[:, :h], m[:, :h]))
            rmse_list.append(masked_mse_np_with_mask(tr[:, :h], p[:, :h], m[:, :h]) ** 0.5)
            mape_list.append(masked_mape_np_with_mask(tr[:, :h], p[:, :h], m[:, :h]))
        horizon_results[h] = {'MAE': float(np.mean(mae_list)), 'RMSE': float(np.mean(rmse_list)), 'MAPE': float(np.mean(mape_list))}
        args.logger.info(f'[*] T={h:2d}  MAE {horizon_results[h]["MAE"]:.4f} / RMSE {horizon_results[h]["RMSE"]:.4f} / MAPE {horizon_results[h]["MAPE"]:.4f}')
    avg_mae = float(np.mean([masked_mae_np_with_mask(tr, p, m) for p, tr, m in zip(preds_raw_list, truths_raw_list, masks_list)]))
    avg_rmse = float(np.mean([masked_mse_np_with_mask(tr, p, m) ** 0.5 for p, tr, m in zip(preds_raw_list, truths_raw_list, masks_list)]))
    avg_mape = float(np.mean([masked_mape_np_with_mask(tr, p, m) for p, tr, m in zip(preds_raw_list, truths_raw_list, masks_list)]))
    if not hasattr(args, 'result') or args.result is None:
        args.result = {}
    key = 'ablation_no_warmup' if no_warmup else 'streaming'
    args.result[key] = {'T3': horizon_results[3], 'T6': horizon_results[6], 'T12': horizon_results[12], 'Avg': {'MAE': avg_mae, 'RMSE': avg_rmse, 'MAPE': avg_mape}}
    results_dir = osp.join(args.path, 'results')
    mkdirs(results_dir)
    save_name = 'ablation_no_warmup_predictions.npz' if no_warmup else 'streaming_predictions.npz'
    np.savez(
        osp.join(results_dir, save_name),
        predictions=np.array(preds_raw_list, dtype=object),
        ground_truth=np.array(truths_raw_list, dtype=object),
        masks=np.array(masks_list, dtype=object),
        mae_avg=np.array(avg_mae),
        rmse_avg=np.array(avg_rmse),
        mape_avg=np.array(avg_mape),
    )
