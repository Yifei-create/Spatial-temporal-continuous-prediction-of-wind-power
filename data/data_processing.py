import os
import os.path as osp
from typing import Any

import numpy as np
import pandas as pd

from config.dataset_registry import DATASET_REGISTRY
from data.adapters import CanonicalBundle, load_dataset_bundle
from data.graph_generation import (
    GRAPH_VARIANT_BASELINE,
    GRAPH_VARIANT_LOCAL_UPSTREAM,
    compute_local_upstream_probability,
    generate_baseline_adjacency,
    generate_local_upstream_adjacency,
    save_stage_adjacency,
)
from data.streaming_plan import build_streaming_plan, resolve_allowed_frequency_minutes, select_expansion_offsets


def _validate_bundle(bundle: CanonicalBundle, dataset_cfg: dict[str, Any]):
    if bundle.feature_cols != list(dataset_cfg['feature_cols']):
        raise ValueError(
            f"Adapter feature_cols={bundle.feature_cols}, but registry expects {dataset_cfg['feature_cols']}."
        )
    if bundle.raw_data.ndim != 3:
        raise ValueError(f'raw_data must have shape [T, N, F]. Received {bundle.raw_data.shape}.')
    if bundle.feature_observed_mask.shape != bundle.raw_data.shape:
        raise ValueError(
            f'feature_observed_mask must match raw_data. Received {bundle.feature_observed_mask.shape} vs {bundle.raw_data.shape}.'
        )
    if bundle.patv_mask.shape != bundle.raw_data.shape[:2]:
        raise ValueError(f'patv_mask must have shape [T, N]. Received {bundle.patv_mask.shape}.')
    if bundle.static_data.shape != (bundle.raw_data.shape[1], 2):
        raise ValueError(
            f'static_data must have shape ({bundle.raw_data.shape[1]}, 2). Received {bundle.static_data.shape}.'
        )
    if len(bundle.timestamps) != bundle.raw_data.shape[0]:
        raise ValueError(
            f'timestamps length must match raw_data time dimension. Received {len(bundle.timestamps)} vs {bundle.raw_data.shape[0]}.'
        )
    if len(bundle.turbine_ids) != bundle.raw_data.shape[1]:
        raise ValueError(
            f'turbine_ids length must match raw_data node dimension. Received {len(bundle.turbine_ids)} vs {bundle.raw_data.shape[1]}.'
        )
    ts = pd.to_datetime(bundle.timestamps)
    if not ts.is_monotonic_increasing:
        raise ValueError('timestamps must be sorted in nondecreasing order.')

    configured_available = dataset_cfg.get('available_turbines')
    if configured_available is not None:
        actual = list(bundle.turbine_ids.tolist())
        expected = list(configured_available)
        if actual != expected:
            raise ValueError(
                f"Configured available_turbines do not match source data. "
                f"expected={expected}, actual={actual}"
            )

        expected_set = set(expected)
        initial = list(dataset_cfg['default_initial_turbines'])
        missing_initial = [tid for tid in initial if tid not in expected_set]
        if missing_initial:
            raise ValueError(f'default_initial_turbines contains unavailable turbines: {missing_initial}')

        assigned = set(initial)
        for group_idx, group in enumerate(dataset_cfg['default_expansion_groups']):
            missing_group = [tid for tid in group if tid not in expected_set]
            if missing_group:
                raise ValueError(
                    f'default_expansion_groups[{group_idx}] contains unavailable turbines: {missing_group}'
                )
            overlap = [tid for tid in group if tid in assigned]
            if overlap:
                raise ValueError(
                    f'default_expansion_groups[{group_idx}] reuses turbines already assigned earlier: {overlap}'
                )
            assigned.update(group)

        unassigned = [tid for tid in expected if tid not in assigned]
        if unassigned:
            raise ValueError(f'available_turbines contains unassigned turbines: {unassigned}')



def _build_static_stats(static_data: np.ndarray):
    static_mean = static_data.mean(axis=0).astype(np.float32)
    static_std = static_data.std(axis=0).astype(np.float32)
    if not np.all(np.isfinite(static_mean)) or not np.all(np.isfinite(static_std)) or np.any(static_std <= 0.0):
        raise ValueError(f'Invalid static normalization statistics: mean={static_mean}, std={static_std}')
    return static_mean, static_std



def _compress_expansion_groups(all_turbines, initial_turbines, expansion_groups):
    available = set(all_turbines)
    active = set(initial_turbines)
    cleaned_groups = []
    for group in expansion_groups:
        new_group = [tid for tid in group if tid in available and tid not in active]
        if not new_group:
            continue
        cleaned_groups.append(new_group)
        active.update(new_group)
    return cleaned_groups



def _build_schedule_and_stages(
    all_turbines,
    initial_turbines,
    expansion_groups,
    test_timestamps,
    x_len,
    y_len,
    allowed_frequency_minutes,
):
    tid_to_col = {tid: idx for idx, tid in enumerate(all_turbines.tolist())}
    initial_cols = [tid_to_col[tid] for tid in initial_turbines]
    if not initial_cols:
        raise ValueError('The initial turbine set is empty after filtering against available turbines.')

    stream_plan = build_streaming_plan(test_timestamps, x_len, y_len, allowed_frequency_minutes)
    valid_stream_starts = [start_idx for start_idx, _ in stream_plan]
    if not valid_stream_starts:
        raise ValueError('The test split has no valid streaming prediction starts for expansion scheduling.')

    turbine_schedule = {}
    stage_cols = [list(initial_cols)]
    cumulative = list(initial_cols)
    num_events = len(expansion_groups)
    expansion_offsets = select_expansion_offsets(valid_stream_starts, num_events)
    for event_idx, group in enumerate(expansion_groups):
        t_offset = expansion_offsets[event_idx]
        new_cols = [tid_to_col[tid] for tid in group]
        turbine_schedule[event_idx] = (int(t_offset), new_cols)
        cumulative = sorted(cumulative + new_cols)
        stage_cols.append(list(cumulative))
    return initial_cols, turbine_schedule, stage_cols



def _compute_train_stats(raw_data, feature_observed_mask, patv_mask, initial_cols, pretrain_end_idx, patv_idx):
    observed_train = feature_observed_mask[:pretrain_end_idx][:, initial_cols, :].astype(bool)
    if not observed_train.any():
        raise ValueError('No observed training inputs are available in the pretrain split.')

    raw_train = raw_data[:pretrain_end_idx][:, initial_cols, :]
    num_features = raw_train.shape[2]
    x_mean = np.zeros(num_features, dtype=np.float32)
    x_std = np.zeros(num_features, dtype=np.float32)
    for feature_idx in range(num_features):
        feature_values = raw_train[:, :, feature_idx][observed_train[:, :, feature_idx]]
        if feature_values.size == 0:
            raise ValueError(f'No observed values are available for feature index {feature_idx} in the pretrain split.')
        feature_mean = float(np.mean(feature_values))
        feature_std = float(np.std(feature_values))
        if not np.isfinite(feature_std) or feature_std <= 0.0:
            raise ValueError(
                f'Invalid x_std computed from training data for feature index {feature_idx}: {feature_std}'
            )
        x_mean[feature_idx] = feature_mean
        x_std[feature_idx] = feature_std

    patv_train = raw_train[:, :, patv_idx]
    patv_train_mask = patv_mask[:pretrain_end_idx][:, initial_cols].astype(bool)
    if not patv_train_mask.any():
        raise ValueError('No valid Patv targets are available in the pretrain split.')

    y_values = patv_train[patv_train_mask]
    y_mean = float(np.mean(y_values))
    y_std = float(np.std(y_values))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError(f'Invalid y_std computed from training data: {y_std}')
    return x_mean, x_std, y_mean, y_std



def _graph_context_for_variant(bundle: CanonicalBundle, dataset: str, graph_variant: str, val_end_idx: int):
    if graph_variant == GRAPH_VARIANT_BASELINE:
        return None
    if graph_variant != GRAPH_VARIANT_LOCAL_UPSTREAM:
        raise ValueError(
            f"Unsupported graph_variant='{graph_variant}'. Expected '{GRAPH_VARIANT_BASELINE}' or '{GRAPH_VARIANT_LOCAL_UPSTREAM}'."
        )
    for key in ['wind_from_deg', 'wind_valid_mask']:
        if key not in bundle.graph_context:
            raise KeyError(
                f"Missing graph_context['{key}'] required by local_upstream graph generation for dataset='{dataset}'."
            )
    return compute_local_upstream_probability(
        bundle.static_data,
        bundle.graph_context['wind_from_deg'][:val_end_idx, :],
        bundle.graph_context['wind_valid_mask'][:val_end_idx, :],
    )



def _save_stage_graphs(static_data, stage_cols, graph_variant, graph_dir, args, upstream_probability_all=None):
    for stage_idx, cols in enumerate(stage_cols):
        stage_cols_idx = np.asarray(cols, dtype=np.int64)
        coords = static_data[stage_cols_idx]
        if graph_variant == GRAPH_VARIANT_BASELINE:
            adj = generate_baseline_adjacency(
                coords,
                weight_threshold=args.baseline_weight_threshold,
                self_loop=False,
            )
        else:
            if upstream_probability_all is None:
                raise ValueError('upstream_probability_all is required for local_upstream graph generation.')
            stage_probability = upstream_probability_all[np.ix_(stage_cols_idx, stage_cols_idx)]
            adj = generate_local_upstream_adjacency(
                coords,
                source_to_target_probability=stage_probability,
                top_k=args.local_upstream_top_k,
            )
        save_stage_adjacency(osp.join(graph_dir, f'stage_{stage_idx}_adj.npz'), adj, graph_variant)



def process_unified_dataset(raw_csv_path, location_csv_path, save_dir, graph_dir, x_len=12, y_len=12, num_expansions=None, args=None, dataset='sdwpf'):
    if args is None:
        raise ValueError('process_unified_dataset requires the runtime args object.')
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f'Unknown dataset={dataset}. Available datasets: {sorted(DATASET_REGISTRY)}')

    dataset_cfg = DATASET_REGISTRY[dataset]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    bundle = load_dataset_bundle(dataset, raw_csv_path, location_csv_path)
    _validate_bundle(bundle, dataset_cfg)

    timestamps = np.asarray(bundle.timestamps, dtype='datetime64[ns]')
    raw_data = bundle.raw_data.astype(np.float32)
    feature_observed_mask = bundle.feature_observed_mask.astype(np.float32)
    patv_mask = bundle.patv_mask.astype(np.float32)
    static_data = bundle.static_data.astype(np.float32)
    all_turbines = np.asarray(bundle.turbine_ids, dtype=object)

    total_timesteps = len(timestamps)
    pretrain_end_idx = int(total_timesteps * 0.2)
    val_end_idx = int(total_timesteps * 0.3)
    if pretrain_end_idx <= 0 or val_end_idx <= pretrain_end_idx:
        raise ValueError(
            f'Invalid split. pretrain_end_idx={pretrain_end_idx}, val_end_idx={val_end_idx}, total={total_timesteps}.'
        )

    available = set(all_turbines.tolist())
    initial_turbines = [tid for tid in dataset_cfg['default_initial_turbines'] if tid in available]
    if not initial_turbines:
        raise ValueError(f'No configured initial turbines are present in dataset={dataset}.')

    default_groups = dataset_cfg['default_expansion_groups']
    if num_expansions is not None:
        default_groups = default_groups[:num_expansions]
    expansion_groups = _compress_expansion_groups(all_turbines.tolist(), initial_turbines, default_groups)
    test_timestamps = timestamps[val_end_idx:]
    allowed_frequency_minutes = resolve_allowed_frequency_minutes(
        dataset_cfg['streaming_freq_mode'],
        supported_frequency_minutes=dataset_cfg.get('supported_frequency_minutes'),
        frequency_minutes=dataset_cfg.get('frequency_minutes'),
    )
    initial_cols, turbine_schedule, stage_cols = _build_schedule_and_stages(
        all_turbines,
        initial_turbines,
        expansion_groups,
        test_timestamps,
        x_len,
        y_len,
        allowed_frequency_minutes,
    )

    static_mean, static_std = _build_static_stats(static_data)
    patv_idx = bundle.feature_cols.index('Patv')
    x_mean, x_std, y_mean, y_std = _compute_train_stats(
        raw_data,
        feature_observed_mask,
        patv_mask,
        initial_cols,
        pretrain_end_idx,
        patv_idx,
    )

    upstream_probability_all = _graph_context_for_variant(bundle, dataset, args.graph_variant, val_end_idx)
    _save_stage_graphs(static_data, stage_cols, args.graph_variant, graph_dir, args, upstream_probability_all)

    save_payload = {
        'dataset_name': np.array(dataset, dtype='U'),
        'adj_type': np.array(args.graph_variant, dtype='U'),
        'feature_cols': np.array(bundle.feature_cols, dtype=object),
        'static_feature_names': np.array(dataset_cfg['static_feature_names'], dtype=object),
        'raw_data': raw_data,
        'feature_observed_mask': feature_observed_mask,
        'patv_mask': patv_mask,
        'static_data': static_data,
        'static_mean': static_mean,
        'static_std': static_std,
        'x_mean': np.asarray(x_mean, dtype=np.float32),
        'x_std': np.asarray(x_std, dtype=np.float32),
        'y_mean': np.array(y_mean, dtype=np.float32),
        'y_std': np.array(y_std, dtype=np.float32),
        'pretrain_end_idx': np.array(pretrain_end_idx, dtype=np.int64),
        'val_end_idx': np.array(val_end_idx, dtype=np.int64),
        'initial_cols': np.asarray(initial_cols, dtype=np.int64),
        'initial_n': np.array(len(initial_cols), dtype=np.int64),
        'initial_turbines': np.asarray(initial_turbines, dtype=object),
        'expansion_groups': np.array([np.asarray(group, dtype=object) for group in expansion_groups], dtype=object),
        'turbine_schedule_keys': np.asarray(list(turbine_schedule.keys()), dtype=np.int64),
        'turbine_schedule_t_offsets': np.asarray([item[0] for item in turbine_schedule.values()], dtype=np.int64),
        'turbine_schedule_new_cols': np.array([np.asarray(item[1], dtype=np.int64) for item in turbine_schedule.values()], dtype=object),
        'raw_timestamps': timestamps.astype('datetime64[ns]'),
        'streaming_freq_mode': np.array(dataset_cfg['streaming_freq_mode'], dtype='U'),
    }
    if dataset_cfg['streaming_freq_mode'] == 'dynamic':
        save_payload['supported_frequency_minutes'] = np.asarray(dataset_cfg['supported_frequency_minutes'], dtype=np.int64)
    else:
        save_payload['frequency_minutes'] = np.array(int(dataset_cfg['frequency_minutes']), dtype=np.int64)

    np.savez(osp.join(save_dir, 'unified_data.npz'), **save_payload)
