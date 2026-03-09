import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from data.dataset import SpatioTemporalDataset
from util.training_utils import masked_mae_np


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _tensor_stat(logger, name, t):
    nan = torch.isnan(t).any().item()
    inf = torch.isinf(t).any().item()
    td = t.detach()
    logger.info(
        f"[STAT] {name}: shape={tuple(td.shape)} nan={nan} inf={inf} "
        f"min={td.min().item():.6g} max={td.max().item():.6g} "
        f"mean={td.mean().item():.6g} std={td.std().item():.6g}"
    )
    return bool(nan or inf)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_x(raw_x, args, eps=1e-6):
    """Z-score normalise X using pretrain stats. raw_x: numpy array."""
    return (raw_x - args.x_mean) / (args.x_std + eps)


def normalize_y(raw_y, args, eps=1e-6):
    """Z-score normalise Y (Patv) using pretrain stats. raw_y: numpy array."""
    return (raw_y - args.y_mean) / (args.y_std + eps)


def denormalize_y(norm_y, args, eps=1e-6):
    """Reverse Y normalisation."""
    return norm_y * (args.y_std + eps) + args.y_mean


# ---------------------------------------------------------------------------
# Adjacency matrix helpers
# ---------------------------------------------------------------------------

def rebuild_adj(stage_idx, args):
    """Load pre-built adjacency matrix for the given expansion stage."""
    adj_path = osp.join(args.graph_path, f'stage_{stage_idx}_adj.npz')
    adj = np.load(adj_path)["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)


# ---------------------------------------------------------------------------
# Backbone freeze / unfreeze
# ---------------------------------------------------------------------------

BACKBONE_NAMES = ("gcn1", "tcn1", "gcn2", "fc")


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if any(b in name for b in BACKBONE_NAMES):
            param.requires_grad = False


def _is_adaptive_param(name):
    adaptive_keys = ("U", "V", "scale", "shift",
                     "mu_scale", "mu_shift", "log_var_scale", "log_var_shift")
    return any(name == k or name.endswith("." + k) for k in adaptive_keys)


def _set_adaptive_requires_grad(model, value):
    for name, param in model.named_parameters():
        if _is_adaptive_param(name):
            param.requires_grad = value


# ---------------------------------------------------------------------------
# Pretrain data helpers
# ---------------------------------------------------------------------------

def _build_pretrain_inputs(raw_data, initial_cols, pretrain_end_idx, val_end_idx, args):
    """
    Build normalised train/val sample dicts for pretraining.

    raw_data: (T, N_all, D)  un-normalised
    Returns dict with train_x, train_y, val_x, val_y  (no test split needed here)
    """
    x_len = args.x_len
    y_len = args.y_len
    D = raw_data.shape[2]
    patv_col = D - 1  # Patv is last feature

    def _make_samples(segment):
        T_seg = segment.shape[0]
        xs, ys = [], []
        for t in range(T_seg - x_len - y_len + 1):
            x_win = segment[t:t + x_len, :, :]          # (x_len, N, D)
            y_win = segment[t + x_len:t + x_len + y_len, :, patv_col]  # (y_len, N)
            xs.append(x_win)
            ys.append(y_win)
        if not xs:
            return None, None
        xs = np.array(xs, dtype=np.float32)   # (S, x_len, N, D)
        ys = np.array(ys, dtype=np.float32)   # (S, y_len, N)
        # (S, x_len, N, D) -> (S, N, x_len*D) -> (S, x_len*D, N)
        xs = xs.transpose(0, 2, 1, 3).reshape(xs.shape[0], xs.shape[2], x_len * D).transpose(0, 2, 1)
        return xs, ys

    train_seg = raw_data[:pretrain_end_idx, :, :][:, initial_cols, :]
    val_seg   = raw_data[pretrain_end_idx:val_end_idx, :, :][:, initial_cols, :]

    train_x_raw, train_y_raw = _make_samples(train_seg)
    val_x_raw,   val_y_raw   = _make_samples(val_seg)

    # Normalise X
    train_x = normalize_x(train_x_raw, args)
    val_x   = normalize_x(val_x_raw,   args)

    # Normalise Y
    train_y = normalize_y(train_y_raw, args)
    val_y   = normalize_y(val_y_raw,   args)

    # Clean NaN/Inf
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
    val_x   = np.nan_to_num(val_x,   nan=0.0, posinf=0.0, neginf=0.0)
    train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)
    val_y   = np.nan_to_num(val_y,   nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'train_x': train_x, 'train_y': train_y,
        'val_x':   val_x,   'val_y':   val_y,
    }


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------

def pretrain(raw_data, initial_cols, pretrain_end_idx, val_end_idx, args):
    """
    Pretrain model on the first 20% of data (initial turbines only).
    Backbone + adaptive params all trained. Early stopping on val loss.
    Saves best checkpoint to args.path/pretrain/
    """
    path = osp.join(args.path, "pretrain")
    mkdirs(path)

    args.logger.info(f"[*] Pretrain on device: {args.device}")

    inputs = _build_pretrain_inputs(raw_data, initial_cols, pretrain_end_idx, val_end_idx, args)
    args.logger.info(f"[*] Pretrain samples: train={inputs['train_x'].shape[0]}, val={inputs['val_x'].shape[0]}")

    train_loader = DataLoader(
        SpatioTemporalDataset(inputs, "train"),
        batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4
    )
    val_loader = DataLoader(
        SpatioTemporalDataset(inputs, "val"),
        batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4
    )

    # Build model with initial node count
    vars(args)["graph_size"] = len(initial_cols)
    model = args.methods[args.method](args).to(args.device)
    model.expand_adaptive_params(len(initial_cols))

    # Load adj for stage 0 (initial turbines)
    vars(args)["adj"] = rebuild_adj(0, args)

    model.count_parameters()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    lowest_val_loss = 1e9
    counter = 0
    patience = 10
    best_ckpt_path = None

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        cn = 0
        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            out = model(data, args.adj)
            if isinstance(out, tuple):
                pred, kl_term = out
            else:
                pred, kl_term = out, None

            pred_dense, _ = to_dense_batch(pred, batch=data.batch)   # (B, N, y_len)
            y_dense, _    = to_dense_batch(data.y, batch=data.batch) # (B, N, y_len)

            loss = F.mse_loss(pred_dense, y_dense)
            if kl_term is not None:
                loss = loss + args.kl_weight * kl_term
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
            cn += 1
        train_loss /= cn

        # Validation
        model.eval()
        val_loss = 0.0
        cn = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(args.device, non_blocking=True)
                out = model(data, args.adj)
                pred = out[0] if isinstance(out, tuple) else out
                pred_dense, _ = to_dense_batch(pred, batch=data.batch)
                y_dense, _    = to_dense_batch(data.y, batch=data.batch)
                loss = masked_mae_np(
                    y_dense.cpu().numpy(), pred_dense.cpu().numpy(), 0
                )
                val_loss += float(loss)
                cn += 1
        val_loss /= cn

        args.logger.info(f"[*] Pretrain: epoch {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < lowest_val_loss:
            counter = 0
            lowest_val_loss = round(val_loss, 4)
            best_ckpt_path = osp.join(path, f"{lowest_val_loss}.pkl")
            torch.save({'model_state_dict': model.state_dict()}, best_ckpt_path)
        else:
            counter += 1
            if counter > patience:
                args.logger.info(f"[*] Early stopping at epoch {epoch}")
                break

    args.logger.info(f"[*] Pretrain done. Best val_loss={lowest_val_loss}, ckpt={best_ckpt_path}")
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Load pretrained model
# ---------------------------------------------------------------------------

def load_pretrained_model(args, n_turbines):
    """Load best pretrain checkpoint and expand to n_turbines."""
    pretrain_dir = osp.join(args.path, "pretrain")
    ckpt_files = [f for f in os.listdir(pretrain_dir) if f.endswith(".pkl")]
    if not ckpt_files:
        raise FileNotFoundError(f"No pretrain checkpoint found in {pretrain_dir}")
    best_file = sorted(ckpt_files)[0]
    ckpt_path = osp.join(pretrain_dir, best_file)
    args.logger.info(f"[*] Loading pretrained model from {ckpt_path}")

    vars(args)["graph_size"] = n_turbines
    model = args.methods[args.method](args).to(args.device)
    model.expand_adaptive_params(n_turbines)

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Online update (single sample, adaptive params only)
# ---------------------------------------------------------------------------

def online_update(model, x_batch, y_batch, adj, args):
    """
    Single-sample online update of adaptive parameters only.

    x_batch: (1, x_len*D, N) tensor  (already normalised)
    y_batch: (1, N, y_len)   tensor  (already normalised)
    adj:     (N, N) tensor
    """
    # Only adaptive params should have requires_grad=True at this point
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.online_lr
    )
    optimizer.zero_grad()
    model.train()

    # Build a minimal Data object matching SpatioTemporalDataset.__getitem__
    from torch_geometric.data import Data
    # x_batch: (1, x_len*D, N) -> (N, x_len*D)
    x_node = x_batch[0].T.to(args.device)   # (N, D_in)
    y_node = y_batch[0].to(args.device)     # (N, y_len)
    data = Data(x=x_node, y=y_node)

    out = model(data, adj)
    pred = out[0] if isinstance(out, tuple) else out
    kl_term = out[1] if isinstance(out, tuple) else None

    loss = F.mse_loss(pred, y_node)
    if kl_term is not None:
        loss = loss + args.kl_weight * kl_term
    loss.backward()
    optimizer.step()
    model.eval()


# ---------------------------------------------------------------------------
# Warmup update for new turbines
# ---------------------------------------------------------------------------

def warmup_update(model, warmup_raw, current_cols, new_cols_local_idx, adj, args):
    """
    Update only the adaptive parameters of newly added turbines using warmup data.

    warmup_raw: (T_warm, N_current, D)  un-normalised raw data for current turbines
    current_cols: list of column indices (length = N_current)
    new_cols_local_idx: indices within current_cols that are new turbines
    adj: (N_current, N_current) tensor
    """
    x_len = args.x_len
    y_len = args.y_len
    D = warmup_raw.shape[2]
    patv_col = D - 1
    T_warm = warmup_raw.shape[0]

    # Build samples
    xs, ys = [], []
    for t in range(T_warm - x_len - y_len + 1):
        x_win = warmup_raw[t:t + x_len, :, :]
        y_win = warmup_raw[t + x_len:t + x_len + y_len, :, patv_col]
        xs.append(x_win)
        ys.append(y_win)

    if not xs:
        args.logger.info("[*] Warmup: not enough data, skipping")
        return

    xs = np.array(xs, dtype=np.float32)   # (S, x_len, N, D)
    ys = np.array(ys, dtype=np.float32)   # (S, y_len, N)
    xs = xs.transpose(0, 2, 1, 3).reshape(xs.shape[0], xs.shape[2], x_len * D).transpose(0, 2, 1)
    # Normalise
    xs = np.nan_to_num(normalize_x(xs, args), nan=0.0, posinf=0.0, neginf=0.0)
    ys = np.nan_to_num(normalize_y(ys, args), nan=0.0, posinf=0.0, neginf=0.0)

    xs_t = torch.from_numpy(xs).float().to(args.device)  # (S, x_len*D, N)
    ys_t = torch.from_numpy(ys).float().to(args.device)  # (S, y_len, N)

    # Freeze all params, then set gradient mask hook on adaptive params
    # so only new_cols_local_idx rows get updated
    for p in model.parameters():
        p.requires_grad = False

    hooks = []
    new_idx_set = set(new_cols_local_idx)

    def _make_mask_hook(param_shape):
        mask = torch.zeros(param_shape[0], dtype=torch.float32, device=args.device)
        for i in new_idx_set:
            if i < param_shape[0]:
                mask[i] = 1.0
        # Reshape for broadcasting: (N, 1) or (N, rank)
        if len(param_shape) == 1:
            mask = mask
        else:
            mask = mask.unsqueeze(1).expand(param_shape)
        def hook(grad):
            return grad * mask
        return hook

    for name, param in model.named_parameters():
        if _is_adaptive_param(name):
            param.requires_grad = True
            h = param.register_hook(_make_mask_hook(param.shape))
            hooks.append(h)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.online_lr
    )

    warmup_epochs = 5
    batch_size = min(32, len(xs_t))
    model.train()
    from torch_geometric.data import Data
    for ep in range(warmup_epochs):
        perm = torch.randperm(len(xs_t))
        for start in range(0, len(xs_t), batch_size):
            idx = perm[start:start + batch_size]
            xb = xs_t[idx]   # (B, x_len*D, N)
            yb = ys_t[idx]   # (B, y_len, N)
            # Process each sample individually (model expects single-graph Data)
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=args.device)
            for i in range(xb.shape[0]):
                x_node = xb[i].T   # (N, D_in)
                y_node = yb[i].T   # (N, y_len)
                data = Data(x=x_node, y=y_node)
                out = model(data, adj)
                pred = out[0] if isinstance(out, tuple) else out
                kl_term = out[1] if isinstance(out, tuple) else None
                loss = F.mse_loss(pred, y_node)
                if kl_term is not None:
                    loss = loss + args.kl_weight * kl_term
                total_loss = total_loss + loss
            (total_loss / xb.shape[0]).backward()
            optimizer.step()

    # Remove hooks and restore adaptive params to trainable
    for h in hooks:
        h.remove()
    _set_adaptive_requires_grad(model, True)
    model.eval()
    args.logger.info(f"[*] Warmup update done ({len(xs)} samples, {warmup_epochs} epochs)")


# ---------------------------------------------------------------------------
# Streaming test
# ---------------------------------------------------------------------------

def streaming_test(raw_data, turbine_schedule, initial_cols, args):
    """
    Streaming test loop.

    raw_data: (T_test, N_all, D)  un-normalised, starting from val_end_idx
    turbine_schedule: dict {exp_idx: (t_offset, new_global_cols)}
    initial_cols: list of column indices for initial turbines
    """
    x_len = args.x_len
    y_len = args.y_len
    D = raw_data.shape[2]
    patv_col = D - 1
    T_test = raw_data.shape[0]

    # Build t_offset -> (exp_idx, new_cols) lookup
    t_to_expansion = {}
    for exp_idx, (t_off, new_cols) in turbine_schedule.items():
        t_to_expansion[t_off] = (exp_idx, new_cols)

    # Load pretrained model with initial turbines
    n_current = len(initial_cols)
    model = load_pretrained_model(args, n_current)
    freeze_backbone(model)
    _set_adaptive_requires_grad(model, True)
    model.eval()

    current_stage = 0
    vars(args)["adj"] = rebuild_adj(current_stage, args)
    current_cols = list(initial_cols)

    all_preds   = []   # list of (N_current, y_len) numpy arrays
    all_truths  = []   # list of (N_current, y_len) numpy arrays

    args.logger.info(f"[*] Streaming test started, T_test={T_test} steps")

    t = 0
    while t + x_len + y_len <= T_test:

        # ---- check for expansion event ----
        if t in t_to_expansion:
            exp_idx, new_global_cols = t_to_expansion[t]
            current_stage += 1

            # Determine local indices of new turbines within the updated current_cols
            new_local_idx = list(range(len(current_cols), len(current_cols) + len(new_global_cols)))
            current_cols = sorted(current_cols + list(new_global_cols))
            n_current = len(current_cols)

            # Expand adaptive params
            model.expand_adaptive_params(n_current)
            freeze_backbone(model)   # re-freeze after expand (expand creates new params)
            _set_adaptive_requires_grad(model, True)

            # Rebuild adj
            vars(args)["adj"] = rebuild_adj(current_stage, args)

            # Warmup: use warmup_days of data starting at t
            steps_per_day = 144   # 10-min intervals -> 144 per day
            warmup_steps = args.warmup_days * steps_per_day
            warmup_end = min(t + warmup_steps, T_test)
            warmup_raw = raw_data[t:warmup_end, :, :][:, current_cols, :]

            args.logger.info(
                f"[*] Expansion {exp_idx} at t={t}: {len(new_global_cols)} new turbines added "
                f"(total {n_current}), warmup steps={warmup_end - t}"
            )
            warmup_update(model, warmup_raw, current_cols, new_local_idx, args.adj, args)

            # Skip warmup window from evaluation
            t += (warmup_end - t)
            # Snap to y_len boundary
            t = (t // y_len) * y_len
            continue

        # ---- prediction step ----
        x_raw = raw_data[t:t + x_len, :, :][:, current_cols, :]   # (x_len, N, D)
        # (x_len, N, D) -> (N, x_len*D) -> (1, x_len*D, N)
        x_norm = normalize_x(x_raw, args)
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = x_norm.transpose(1, 0, 2).reshape(n_current, x_len * D).T  # (x_len*D, N)
        x_tensor = torch.from_numpy(x_flat).float().unsqueeze(0).to(args.device)  # (1, x_len*D, N)

        y_raw = raw_data[t + x_len:t + x_len + y_len, :, patv_col][:, current_cols]  # (y_len, N)
        y_norm = normalize_y(y_raw, args)
        y_norm = np.nan_to_num(y_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass
        from torch_geometric.data import Data
        x_node = x_tensor[0].T   # (N, D_in)
        with torch.no_grad():
            out = model(Data(x=x_node), args.adj)
        pred = out[0] if isinstance(out, tuple) else out
        pred_np = pred.cpu().numpy()   # (N, y_len)

        all_preds.append(pred_np)
        all_truths.append(y_norm.T)    # (N, y_len)

        # Online update with this new (x, y) sample
        # y_norm: (y_len, N) -> .T -> (N, y_len) -> unsqueeze(0) -> (1, N, y_len)
        online_update(
            model,
            x_tensor,                                                         # (1, x_len*D, N)
            torch.from_numpy(y_norm.T).float().unsqueeze(0).to(args.device),  # (1, N, y_len)
            args.adj,
            args
        )

        t += y_len

    # ---- aggregate and evaluate ----
    if not all_preds:
        args.logger.info("[*] No predictions collected.")
        return

    # Stack: (num_windows, N_max, y_len) — but N varies across windows due to expansions
    # Evaluate only on windows with the same N (or pad — simplest: evaluate all together
    # by treating each window independently and averaging)
    preds_raw_list  = [denormalize_y(p, args) for p in all_preds]
    truths_raw_list = [denormalize_y(tr, args) for tr in all_truths]

    # Clip negatives (Patv >= 0)
    preds_raw_list  = [np.clip(p, 0, None) for p in preds_raw_list]

    # Compute metrics per window then average
    from util.training_utils import masked_mae_np, masked_mse_np, masked_mape_np
    mae_list, rmse_list, mape_list = [], [], []
    for p, tr in zip(preds_raw_list, truths_raw_list):
        mae_list.append(masked_mae_np(tr, p, 0))
        rmse_list.append(masked_mse_np(tr, p, 0) ** 0.5)
        mape_list.append(masked_mape_np(tr, p, 0))

    mae  = float(np.mean(mae_list))
    rmse = float(np.mean(rmse_list))
    mape = float(np.mean(mape_list))

    args.logger.info(f"[*] Final metrics: MAE {mae:.4f} / RMSE {rmse:.4f} / MAPE {mape:.4f}")

    # Store in args.result under key "streaming"
    if not hasattr(args, "result") or args.result is None:
        args.result = {}
    args.result["streaming"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    # Save predictions
    results_dir = osp.join(args.path, "results")
    mkdirs(results_dir)
    save_path = osp.join(results_dir, "streaming_predictions.npz")
    np.savez(save_path,
             predictions=np.array(preds_raw_list, dtype=object),
             ground_truth=np.array(truths_raw_list, dtype=object))
    args.logger.info(f"[*] Saved streaming predictions to {save_path}")
