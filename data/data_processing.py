import numpy as np
import pandas as pd
import os.path as osp
from config.node_schedule import get_node_schedule, get_period_info


def load_raw_data(raw_data_path):
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} rows")
    return df


def load_location_data(location_path):
    df = pd.read_csv(location_path)
    return df


def get_time_range_for_period(period):
    time_ranges = {
        0: ('2020-01-01', '2020-06-30'),
        1: ('2020-07-01', '2020-12-31'),
        2: ('2021-01-01', '2021-06-30'),
        3: ('2021-07-01', '2021-12-31')
    }
    return time_ranges.get(period, (None, None))


def z_score(data, mean=None, std=None, eps=1e-6):
    """
    Z-score normalization.

    IMPORTANT:
    - To avoid leakage, mean/std should be computed on TRAIN split only,
      then reused for val/test.
    - For stability, use nan-safe stats and finally clean nan/inf.
    """
    if mean is None:
        mean = np.nanmean(data)
    if std is None:
        std = np.nanstd(data)

    # Avoid division by zero / invalid std
    if not np.isfinite(std) or std == 0:
        std = eps

    out = (data - mean) / (std + eps)

    # Clean any remaining nan/inf
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _apply_sdwpf_patv_rules(vals, col_idx):
    """
    Apply SDWPF paper recommended handling on ONE row (one turbine at one timestamp).

    Rules (paper):
    - Negative Patv: set to 0
    - Missing Patv: set to 0 (will be masked in evaluation by null_val=0)
    - Unknown Patv: set to 0
        (Patv <= 0 and Wspd > 2.5) OR (Pab1/2/3 > 89)
    - Abnormal value row: set Patv to 0 if
        Wdir not in [-180, 180] OR Ndir not in [-720, 720]

    NOTE:
    - This function ONLY changes Patv.
    - Do NOT zero-out Patv just because other feature fields are NaN.
      (Other features can be safely nan_to_num later without destroying Patv.)
    """
    wspd = vals[col_idx['Wspd']]
    wdir = vals[col_idx['Wdir']]
    ndir = vals[col_idx['Ndir']]
    pab1 = vals[col_idx['Pab1']]
    pab2 = vals[col_idx['Pab2']]
    pab3 = vals[col_idx['Pab3']]
    patv = vals[col_idx['Patv']]

    # Missing/invalid Patv itself -> 0
    if not np.isfinite(patv):
        vals[col_idx['Patv']] = 0.0
        return vals

    # For rule checks, use safe versions of feature fields:
    # If a feature is nan/inf, treat it as "cannot trigger the rule" rather than forcing Patv=0.
    # (We will nan_to_num features later anyway.)
    wspd_ok = float(wspd) if np.isfinite(wspd) else -1e9
    wdir_ok = float(wdir) if np.isfinite(wdir) else 0.0
    ndir_ok = float(ndir) if np.isfinite(ndir) else 0.0
    pab1_ok = float(pab1) if np.isfinite(pab1) else -1e9
    pab2_ok = float(pab2) if np.isfinite(pab2) else -1e9
    pab3_ok = float(pab3) if np.isfinite(pab3) else -1e9

    patv_raw = float(patv)

    # Abnormal row (depends on Wdir/Ndir ranges)
    abnormal = (wdir_ok < -180.0) or (wdir_ok > 180.0) or (ndir_ok < -720.0) or (ndir_ok > 720.0)
    if abnormal:
        vals[col_idx['Patv']] = 0.0
        return vals

    # Unknown row (use raw Patv sign/zero + conditions)
    unknown = ((patv_raw <= 0.0) and (wspd_ok > 2.5)) or (pab1_ok > 89.0) or (pab2_ok > 89.0) or (pab3_ok > 89.0)
    if unknown:
        vals[col_idx['Patv']] = 0.0
        return vals

    # Negative Patv -> 0 (clip)
    if patv_raw < 0.0:
        vals[col_idx['Patv']] = 0.0
    else:
        vals[col_idx['Patv']] = patv_raw

    return vals


def generate_samples(x_len, y_len, save_path, data, graph, val_test_mix=False):
    """
    Generate train/val/test samples with z-score normalization (train stats only)

    Args:
        x_len: input window length (history steps)
        y_len: prediction horizon length (future steps)
        save_path: path prefix for saving .npz
        data: (T, N, D)
    """
    T, N, D = data.shape

    samples_x = []
    samples_y = []

    for t in range(T - x_len - y_len + 1):
        # x: (x_len, N, D)
        x = data[t:t + x_len, :, :]
        # y: (y_len, N)  target = Patv (last feature)
        y = data[t + x_len:t + x_len + y_len, :, -1]
        samples_x.append(x)
        samples_y.append(y)

    # samples_x: (num_samples, x_len, N, D)
    # samples_y: (num_samples, y_len, N)
    samples_x = np.array(samples_x, dtype=np.float32)
    samples_y = np.array(samples_y, dtype=np.float32)

    # (B, x_len, N, D) -> (B, N, x_len, D)
    samples_x = samples_x.transpose(0, 2, 1, 3)
    # (B, N, x_len, D) -> (B, N, x_len*D)
    samples_x = samples_x.reshape(samples_x.shape[0], samples_x.shape[1], x_len * D)
    # (B, N, x_len*D) -> (B, x_len*D, N)  (保持你 Dataset 的约定： (T, D, N))
    samples_x = samples_x.transpose(0, 2, 1)

    num_samples = len(samples_x)
    train_size = int(0.7 * num_samples)  # 70% train
    val_size = int(0.2 * num_samples)    # 20% val
    # Remaining 10% is test

    # Split data
    train_x = samples_x[:train_size]
    train_y = samples_y[:train_size]
    val_x = samples_x[train_size:train_size + val_size]
    val_y = samples_y[train_size:train_size + val_size]
    test_x = samples_x[train_size + val_size:]
    test_y = samples_y[train_size + val_size:]

    # Z-score normalization on X only (train stats only)
    train_mean = np.nanmean(train_x)
    train_std = np.nanstd(train_x)

    train_x = z_score(train_x, mean=train_mean, std=train_std)
    val_x = z_score(val_x, mean=train_mean, std=train_std)
    test_x = z_score(test_x, mean=train_mean, std=train_std)

    # Keep Y in ORIGINAL SCALE.
    # Important: Y uses 0 as "masked/invalid" according to your metric functions (null_val=0).
    train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)
    val_y = np.nan_to_num(val_y, nan=0.0, posinf=0.0, neginf=0.0)
    test_y = np.nan_to_num(test_y, nan=0.0, posinf=0.0, neginf=0.0)

    result = {
        'train_x': train_x,  # (num_samples, x_len*D, N)
        'train_y': train_y,  # (num_samples, y_len, N)
        'val_x': val_x,
        'val_y': val_y,
        'test_x': test_x,
        'test_y': test_y
    }

    if save_path:
        np.savez(save_path + '.npz', **result)
        print(f"  Saved to {save_path}.npz")

    return result


def process_and_save_all_periods(raw_csv_path, location_csv_path, save_dir, graph_dir,
                                 x_len=12, y_len=12, use_parallel=True, n_jobs=None):
    import os
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    print("=" * 80)
    print("Processing data with node schedule")
    print("=" * 80)

    df_data = load_raw_data(raw_csv_path)
    df_location = load_location_data(location_csv_path)

    df_data['Tmstamp'] = pd.to_datetime(df_data['Tmstamp'])

    feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3',
                    'Prtv', 'T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp', 'Patv']

    # Build column index for rule checking
    col_idx = {c: i for i, c in enumerate(feature_cols)}

    node_schedule = get_node_schedule()
    period_info = get_period_info()

    from data.graph_generation import generate_adjacency_matrix

    for period in sorted(node_schedule.keys()):
        print(f"\nPeriod {period}:")

        active_turbines = sorted(node_schedule[period])
        N = len(active_turbines)
        D = len(feature_cols)

        start_date, end_date = get_time_range_for_period(period)

        df_period = df_data[
            (df_data['Tmstamp'] >= start_date) &
            (df_data['Tmstamp'] <= end_date) &
            (df_data['TurbID'].isin(active_turbines))
        ].copy()

        timestamps = sorted(df_period['Tmstamp'].unique())
        T = len(timestamps)

        data_matrix = np.zeros((T, N, D), dtype=np.float32)

        turb_to_idx = {tid: idx for idx, tid in enumerate(active_turbines)}

        print(f"  Filling {T} timesteps × {N} turbines...")
        for t_idx, ts in enumerate(timestamps):
            if t_idx % 1000 == 0:
                print(f"    Progress: {t_idx}/{T} ({100 * t_idx / T:.1f}%)")
            df_ts = df_period[df_period['Tmstamp'] == ts]
            for _, row in df_ts.iterrows():
                turb_id = row['TurbID']
                if turb_id in turb_to_idx:
                    turb_idx = turb_to_idx[turb_id]

                    vals = row[feature_cols].values.astype(np.float32)

                    # Apply SDWPF paper rules ONLY to Patv (target)
                    vals = _apply_sdwpf_patv_rules(vals, col_idx)

                    # Now make features numerically safe (keep Patv as possibly 0 from rules)
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

                    data_matrix[t_idx, turb_idx, :] = vals

        # Defensive clean (should be safe already)
        data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  Generating training samples...")
        generate_samples(x_len, y_len, osp.join(save_dir, str(period)), data_matrix, None, val_test_mix=False)

        df_loc_period = df_location[df_location['TurbID'].isin(active_turbines)].copy()
        df_loc_period = df_loc_period.sort_values('TurbID')
        coords = df_loc_period[['x', 'y']].values

        print(f"  Generating adjacency matrix...")
        adj = generate_adjacency_matrix(coords, sigma_km=30.0, top_k=16)
        np.savez(osp.join(graph_dir, f'{period}_adj.npz'), x=adj)

        print(f"  ✓ Period {period} completed: {N} turbines")

    print("\n" + "=" * 80)
    print("✓ Data processing completed!")
    print("=" * 80)
    print(f"Features: {feature_cols}")
    print(f"Split: train:val:test = 2:1:7")
    for period in sorted(period_info.keys()):
        info = period_info[period]
        print(f"  Period {period}: {info['total_nodes']} turbines")
    print("=" * 80)