import numpy as np
import pandas as pd
import os
import os.path as osp
from config.node_schedule import get_expansion_groups, get_initial_turbines, get_all_turbines, turbid_to_index


def load_raw_data(raw_data_path):
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} rows")
    return df


def load_location_data(location_path):
    return pd.read_csv(location_path)


def z_score(data, mean=None, std=None, eps=1e-6):
    if mean is None:
        mean = np.nanmean(data)
    if std is None:
        std = np.nanstd(data)
    if not np.isfinite(std) or std == 0:
        std = eps
    out = (data - mean) / (std + eps)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_sdwpf_patv_rules(vals, col_idx):
    """Apply SDWPF paper rules to Patv only."""
    wspd = vals[col_idx['Wspd']]
    wdir = vals[col_idx['Wdir']]
    ndir = vals[col_idx['Ndir']]
    pab1 = vals[col_idx['Pab1']]
    pab2 = vals[col_idx['Pab2']]
    pab3 = vals[col_idx['Pab3']]
    patv = vals[col_idx['Patv']]

    if not np.isfinite(patv):
        vals[col_idx['Patv']] = 0.0
        return vals

    wspd_ok = float(wspd) if np.isfinite(wspd) else -1e9
    wdir_ok = float(wdir) if np.isfinite(wdir) else 0.0
    ndir_ok = float(ndir) if np.isfinite(ndir) else 0.0
    pab1_ok = float(pab1) if np.isfinite(pab1) else -1e9
    pab2_ok = float(pab2) if np.isfinite(pab2) else -1e9
    pab3_ok = float(pab3) if np.isfinite(pab3) else -1e9
    patv_raw = float(patv)

    abnormal = (wdir_ok < -180.0) or (wdir_ok > 180.0) or (ndir_ok < -720.0) or (ndir_ok > 720.0)
    if abnormal:
        vals[col_idx['Patv']] = 0.0
        return vals

    unknown = ((patv_raw <= 0.0) and (wspd_ok > 2.5)) or (pab1_ok > 89.0) or (pab2_ok > 89.0) or (pab3_ok > 89.0)
    if unknown:
        vals[col_idx['Patv']] = 0.0
        return vals

    vals[col_idx['Patv']] = max(0.0, patv_raw)
    return vals


def _build_raw_matrix(df_data, all_turbines, timestamps, feature_cols, col_idx):
    """
    Build (T, N_all, D) raw data matrix for all turbines and all timestamps.
    Missing entries remain 0.
    """
    T = len(timestamps)
    N = len(all_turbines)
    D = len(feature_cols)
    tid_to_idx = turbid_to_index(all_turbines)
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    data_matrix = np.zeros((T, N, D), dtype=np.float32)

    for _, row in df_data.iterrows():
        tid = row['TurbID']
        ts = row['Tmstamp']
        if tid not in tid_to_idx or ts not in ts_to_idx:
            continue
        t_idx = ts_to_idx[ts]
        n_idx = tid_to_idx[tid]
        vals = row[feature_cols].values.astype(np.float32)
        vals = _apply_sdwpf_patv_rules(vals, col_idx)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        data_matrix[t_idx, n_idx, :] = vals

    return data_matrix


def process_unified_dataset(raw_csv_path, location_csv_path, save_dir, graph_dir,
                             x_len=12, y_len=12, num_expansions=3):
    """
    Process full dataset into unified_data.npz for pretrain + streaming test framework.

    Output unified_data.npz contains:
      raw_data        : (T, N_all, D)  cleaned, un-normalised
      x_mean, x_std   : X normalisation stats (from pretrain train split)
      y_mean, y_std   : Patv normalisation stats (from pretrain train split)
      pretrain_end_idx: time index where pretrain split ends
      val_end_idx     : time index where validation split ends
      turbine_schedule: pickled dict {exp_idx: (time_idx, [turbine_col_indices])}
      initial_n       : number of initial turbines
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3',
                    'Prtv', 'T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp', 'Patv']
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    patv_col = col_idx['Patv']

    print("Loading raw data...")
    df_data = load_raw_data(raw_csv_path)
    df_location = load_location_data(location_csv_path)

    df_data['Tmstamp'] = pd.to_datetime(df_data['Tmstamp'])
    all_turbines = sorted(df_data['TurbID'].unique().tolist())
    timestamps = sorted(df_data['Tmstamp'].unique().tolist())
    T = len(timestamps)
    N_all = len(all_turbines)
    print(f"Total: {T} timesteps, {N_all} turbines")

    print("Building raw data matrix (this may take a while)...")
    raw_data = _build_raw_matrix(df_data, all_turbines, timestamps, feature_cols, col_idx)
    print(f"raw_data shape: {raw_data.shape}")

    # ---- time split: 20% pretrain / 10% val / 70% streaming test ----
    pretrain_end_idx = int(0.20 * T)
    val_end_idx = int(0.30 * T)   # pretrain 20% + val 10%

    # ---- initial turbines (first base_n columns by sorted TurbID) ----
    initial_turbines = get_initial_turbines()
    # Map TurbIDs to column indices in all_turbines array
    tid_to_col = turbid_to_index(all_turbines)
    initial_cols = sorted([tid_to_col[t] for t in initial_turbines if t in tid_to_col])
    initial_n = len(initial_cols)
    print(f"Initial turbines: {initial_n}")

    # ---- compute normalisation stats from pretrain split (initial turbines only) ----
    pretrain_raw = raw_data[:pretrain_end_idx, initial_cols, :]   # (T_pre, N_init, D)

    # X stats: all features, all pretrain samples
    x_mean = float(np.nanmean(pretrain_raw))
    x_std  = float(np.nanstd(pretrain_raw))

    # Y stats: Patv column only
    patv_pretrain = pretrain_raw[:, :, patv_col]
    y_mean = float(np.nanmean(patv_pretrain))
    y_std  = float(np.nanstd(patv_pretrain))
    print(f"X stats: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"Y stats: mean={y_mean:.4f}, std={y_std:.4f}")

    # ---- build turbine schedule for streaming test ----
    # Expansion events are uniformly distributed over the streaming test window
    expansion_groups = get_expansion_groups()
    n_exp = min(num_expansions, len(expansion_groups))
    T_test = T - val_end_idx
    # Place expansion events at 1/(n_exp+1), 2/(n_exp+1), ... fractions of T_test
    turbine_schedule = {}
    for i in range(n_exp):
        frac = (i + 1) / (n_exp + 1)
        t_offset = int(frac * T_test)
        # Snap to a y_len boundary
        t_offset = (t_offset // y_len) * y_len
        new_tids = expansion_groups[i]
        new_cols = sorted([tid_to_col[t] for t in new_tids if t in tid_to_col])
        turbine_schedule[i] = (t_offset, new_cols)
        print(f"Expansion {i}: t_offset={t_offset}, {len(new_cols)} new turbines")

    # ---- generate and save adjacency matrices for each expansion stage ----
    from data.graph_generation import generate_adjacency_matrix
    df_loc = df_location.copy()
    df_loc = df_loc[df_loc['TurbID'].isin(all_turbines)].sort_values('TurbID')

    # Build cumulative turbine column lists per stage
    stage_cols = [initial_cols]
    cumulative = list(initial_cols)
    for i in range(n_exp):
        _, new_cols = turbine_schedule[i]
        cumulative = sorted(cumulative + new_cols)
        stage_cols.append(list(cumulative))

    for stage_idx, cols in enumerate(stage_cols):
        tids_in_stage = [all_turbines[c] for c in cols]
        df_loc_stage = df_loc[df_loc['TurbID'].isin(tids_in_stage)].sort_values('TurbID')
        coords = df_loc_stage[['x', 'y']].values
        adj = generate_adjacency_matrix(coords, sigma_km=30.0, top_k=16)
        np.savez(osp.join(graph_dir, f'stage_{stage_idx}_adj.npz'), x=adj)
        print(f"Saved adj for stage {stage_idx}: {len(cols)} turbines")

    # ---- save unified_data.npz ----
    save_path = osp.join(save_dir, 'unified_data.npz')
    np.savez(
        save_path,
        raw_data=raw_data,
        x_mean=np.array(x_mean),
        x_std=np.array(x_std),
        y_mean=np.array(y_mean),
        y_std=np.array(y_std),
        pretrain_end_idx=np.array(pretrain_end_idx),
        val_end_idx=np.array(val_end_idx),
        initial_cols=np.array(initial_cols),
        initial_n=np.array(initial_n),
        # turbine_schedule stored as flat arrays; reconstructed in main.py
        turbine_schedule_keys=np.array(list(turbine_schedule.keys())),
        turbine_schedule_t_offsets=np.array([v[0] for v in turbine_schedule.values()]),
        # variable-length new_cols: store as object array
        turbine_schedule_new_cols=np.array([np.array(v[1]) for v in turbine_schedule.values()], dtype=object),
    )
    print(f"Saved unified_data.npz to {save_path}")
    print(f"  raw_data: {raw_data.shape}")
    print(f"  pretrain_end_idx={pretrain_end_idx}, val_end_idx={val_end_idx}")
    print(f"  initial_n={initial_n}, num_expansions={n_exp}")


# ---- legacy helpers kept for backward compatibility ----

def generate_samples(x_len, y_len, save_path, data, graph, val_test_mix=False):
    """
    Generate train/val/test samples with z-score normalization (train stats only).
    Kept for backward compatibility; not used in unified streaming framework.
    """
    T, N, D = data.shape
    samples_x, samples_y = [], []
    for t in range(T - x_len - y_len + 1):
        x = data[t:t + x_len, :, :]
        y = data[t + x_len:t + x_len + y_len, :, -1]
        samples_x.append(x)
        samples_y.append(y)

    samples_x = np.array(samples_x, dtype=np.float32)
    samples_y = np.array(samples_y, dtype=np.float32)

    samples_x = samples_x.transpose(0, 2, 1, 3)
    samples_x = samples_x.reshape(samples_x.shape[0], samples_x.shape[1], x_len * D)
    samples_x = samples_x.transpose(0, 2, 1)

    num_samples = len(samples_x)
    train_size = int(0.7 * num_samples)
    val_size = int(0.2 * num_samples)

    train_x = samples_x[:train_size]
    train_y = samples_y[:train_size]
    val_x = samples_x[train_size:train_size + val_size]
    val_y = samples_y[train_size:train_size + val_size]
    test_x = samples_x[train_size + val_size:]
    test_y = samples_y[train_size + val_size:]

    train_mean = np.nanmean(train_x)
    train_std = np.nanstd(train_x)
    train_x = z_score(train_x, mean=train_mean, std=train_std)
    val_x = z_score(val_x, mean=train_mean, std=train_std)
    test_x = z_score(test_x, mean=train_mean, std=train_std)

    train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)
    val_y = np.nan_to_num(val_y, nan=0.0, posinf=0.0, neginf=0.0)
    test_y = np.nan_to_num(test_y, nan=0.0, posinf=0.0, neginf=0.0)

    result = {
        'train_x': train_x, 'train_y': train_y,
        'val_x': val_x, 'val_y': val_y,
        'test_x': test_x, 'test_y': test_y,
    }
    if save_path:
        np.savez(save_path + '.npz', **result)
        print(f"  Saved to {save_path}.npz")
    return result
