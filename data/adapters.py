from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from config.dataset_registry import DATASET_REGISTRY


@dataclass(frozen=True)
class CanonicalBundle:
    dataset_name: str
    feature_cols: list[str]
    turbine_ids: np.ndarray
    timestamps: np.ndarray
    raw_data: np.ndarray
    feature_observed_mask: np.ndarray
    patv_mask: np.ndarray
    static_data: np.ndarray
    graph_context: dict[str, Any]


class DatasetAdapter:
    dataset_name: str

    def __init__(self, dataset_name: str, dataset_cfg: dict[str, Any], raw_path: str, location_path: str):
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.raw_path = raw_path
        self.location_path = location_path

    def load_bundle(self) -> CanonicalBundle:
        raise NotImplementedError


def _build_wind_graph_context_from_wdir(dense: CanonicalBundle) -> dict[str, Any]:
    if 'Wdir' not in dense.feature_cols:
        raise KeyError("Feature 'Wdir' is required to build local_upstream graph context.")

    wdir_idx = dense.feature_cols.index('Wdir')
    wind_from_deg = dense.raw_data[:, :, wdir_idx].astype(np.float32)
    wind_valid_mask = dense.feature_observed_mask[:, :, wdir_idx].astype(bool)
    return {
        'wind_from_deg': wind_from_deg,
        'wind_valid_mask': wind_valid_mask,
    }


def _subset_graph_context(graph_context: dict[str, Any], node_indices: np.ndarray, num_nodes: int) -> dict[str, Any]:
    subset = {}
    for key, value in graph_context.items():
        if isinstance(value, np.ndarray):
            if value.ndim >= 2 and value.shape[1] == num_nodes:
                subset[key] = np.take(value, node_indices, axis=1)
            elif value.ndim == 1 and value.shape[0] == num_nodes:
                subset[key] = np.take(value, node_indices, axis=0)
            else:
                subset[key] = value
        else:
            subset[key] = value
    return subset


def _subset_bundle_to_available_turbines(bundle: CanonicalBundle, available_turbines: list[Any]) -> CanonicalBundle:
    actual_turbines = list(bundle.turbine_ids.tolist())
    if actual_turbines == list(available_turbines):
        return bundle

    turbine_to_idx = {tid: idx for idx, tid in enumerate(actual_turbines)}
    missing = [tid for tid in available_turbines if tid not in turbine_to_idx]
    if missing:
        raise ValueError(
            f"Configured available_turbines contain turbines absent from the source bundle: {missing}"
        )

    node_indices = np.asarray([turbine_to_idx[tid] for tid in available_turbines], dtype=np.int64)
    graph_context = _subset_graph_context(bundle.graph_context, node_indices, len(actual_turbines))
    return CanonicalBundle(
        dataset_name=bundle.dataset_name,
        feature_cols=list(bundle.feature_cols),
        turbine_ids=np.asarray(available_turbines, dtype=object),
        timestamps=np.asarray(bundle.timestamps, dtype='datetime64[ns]'),
        raw_data=bundle.raw_data[:, node_indices, :],
        feature_observed_mask=bundle.feature_observed_mask[:, node_indices, :],
        patv_mask=bundle.patv_mask[:, node_indices],
        static_data=bundle.static_data[node_indices, :],
        graph_context=graph_context,
    )


def latlon_to_xy(lat, lon):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat - lat0) * 111320.0
    return x.astype(np.float32), y.astype(np.float32)


def _validate_feature_cols(feature_cols: list[str]):
    if not feature_cols:
        raise ValueError('feature_cols cannot be empty.')
    if feature_cols[-1] != 'Patv':
        raise ValueError(f"The last feature must be 'Patv'. Received {feature_cols}.")


def _build_static_data(df_location: pd.DataFrame, turbine_ids: np.ndarray) -> np.ndarray:
    loc = df_location.set_index('TurbID').reindex(turbine_ids)
    if loc.isna().any().any():
        missing = loc[loc.isna().any(axis=1)].index.tolist()
        raise ValueError(f'Missing static coordinates for turbines: {missing[:10]}')
    static_data = np.stack(
        [
            loc['x'].to_numpy(dtype=np.float32),
            loc['y'].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    if static_data.ndim != 2 or static_data.shape[1] != 2:
        raise ValueError(f'static_data must have shape (N, 2). Received {static_data.shape}.')
    return static_data


def _build_dense_tensor_from_dataframe(
    df_data: pd.DataFrame,
    feature_cols: list[str],
    *,
    turbine_ids: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    patv_valid_mask: np.ndarray | None = None,
    extra_graph_context: dict[str, Any] | None = None,
) -> CanonicalBundle:
    _validate_feature_cols(feature_cols)
    if 'TurbID' not in df_data.columns or 'Tmstamp' not in df_data.columns:
        raise ValueError("df_data must contain 'TurbID' and 'Tmstamp'.")

    working = df_data.copy()
    working['Tmstamp'] = pd.to_datetime(working['Tmstamp'])
    working = working.sort_values(['Tmstamp', 'TurbID']).drop_duplicates(['Tmstamp', 'TurbID'], keep='last')

    if turbine_ids is None:
        turbine_ids = np.asarray(sorted(working['TurbID'].unique().tolist()), dtype=object)
    else:
        turbine_ids = np.asarray(turbine_ids, dtype=object)
    if timestamps is None:
        timestamps = np.asarray(sorted(working['Tmstamp'].unique()), dtype='datetime64[ns]')
    else:
        timestamps = np.asarray(pd.to_datetime(timestamps), dtype='datetime64[ns]')

    turbine_index = pd.Index(turbine_ids)
    timestamp_index = pd.Index(pd.to_datetime(timestamps))
    row_idx = timestamp_index.get_indexer(working['Tmstamp'])
    col_idx = turbine_index.get_indexer(working['TurbID'])
    if (row_idx < 0).any() or (col_idx < 0).any():
        raise ValueError('Failed to index rows into the dense tensor layout.')

    raw_feature_values = working[feature_cols].to_numpy(dtype=np.float32)
    feature_observed_mask = np.isfinite(raw_feature_values).astype(np.float32)
    values = np.nan_to_num(raw_feature_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    patv_col = feature_cols.index('Patv')
    if patv_valid_mask is None:
        patv_valid_mask = np.isfinite(raw_feature_values[:, patv_col])
    patv_valid_mask = np.asarray(patv_valid_mask, dtype=bool)
    if patv_valid_mask.shape != (len(working),):
        raise ValueError(
            f'patv_valid_mask must have shape ({len(working)},). Received {patv_valid_mask.shape}.'
        )

    patv_raw = working['Patv'].to_numpy(dtype=np.float32)
    values[:, patv_col] = np.where(patv_valid_mask, np.maximum(patv_raw, 0.0), 0.0).astype(np.float32)

    num_times = len(timestamps)
    num_nodes = len(turbine_ids)
    num_features = len(feature_cols)
    raw_data = np.zeros((num_times, num_nodes, num_features), dtype=np.float32)
    observed = np.zeros((num_times, num_nodes, num_features), dtype=np.float32)
    patv_mask = np.zeros((num_times, num_nodes), dtype=np.float32)

    raw_data[row_idx, col_idx, :] = values
    observed[row_idx, col_idx, :] = feature_observed_mask
    patv_mask[row_idx, col_idx] = patv_valid_mask.astype(np.float32)

    graph_context = {} if extra_graph_context is None else dict(extra_graph_context)
    return CanonicalBundle(
        dataset_name='',
        feature_cols=list(feature_cols),
        turbine_ids=turbine_ids,
        timestamps=np.asarray(timestamps, dtype='datetime64[ns]'),
        raw_data=raw_data,
        feature_observed_mask=observed,
        patv_mask=patv_mask,
        static_data=np.empty((0, 2), dtype=np.float32),
        graph_context=graph_context,
    )


class SdwpfAdapter(DatasetAdapter):
    @staticmethod
    def _patv_valid_mask(df_data: pd.DataFrame) -> np.ndarray:
        patv = df_data['Patv'].to_numpy(dtype=np.float32)
        wspd = df_data['Wspd'].to_numpy(dtype=np.float32)
        wdir = df_data['Wdir'].to_numpy(dtype=np.float32)
        ndir = df_data['Ndir'].to_numpy(dtype=np.float32)
        pab1 = df_data['Pab1'].to_numpy(dtype=np.float32)
        pab2 = df_data['Pab2'].to_numpy(dtype=np.float32)
        pab3 = df_data['Pab3'].to_numpy(dtype=np.float32)

        valid_patv = np.isfinite(patv)
        wspd_ok = np.where(np.isfinite(wspd), wspd, -1e9)
        wdir_ok = np.where(np.isfinite(wdir), wdir, 0.0)
        ndir_ok = np.where(np.isfinite(ndir), ndir, 0.0)
        pab1_ok = np.where(np.isfinite(pab1), pab1, -1e9)
        pab2_ok = np.where(np.isfinite(pab2), pab2, -1e9)
        pab3_ok = np.where(np.isfinite(pab3), pab3, -1e9)

        abnormal = (wdir_ok < -180.0) | (wdir_ok > 180.0) | (ndir_ok < -720.0) | (ndir_ok > 720.0)
        unknown = ((patv <= 0.0) & (wspd_ok > 2.5)) | (pab1_ok > 89.0) | (pab2_ok > 89.0) | (pab3_ok > 89.0)
        return valid_patv & (~abnormal) & (~unknown)

    @staticmethod
    def _direction_valid_mask(df_data: pd.DataFrame) -> np.ndarray:
        wdir = df_data['Wdir'].to_numpy(dtype=np.float32)
        ndir = df_data['Ndir'].to_numpy(dtype=np.float32)
        finite = np.isfinite(wdir) & np.isfinite(ndir)
        range_ok = (wdir >= -180.0) & (wdir <= 180.0) & (ndir >= -720.0) & (ndir <= 720.0)
        return finite & range_ok

    def load_bundle(self) -> CanonicalBundle:
        feature_cols = list(self.dataset_cfg['feature_cols'])
        _validate_feature_cols(feature_cols)

        usecols = ['TurbID', 'Tmstamp'] + feature_cols
        dtype_map = {'TurbID': np.int16}
        for col in feature_cols:
            dtype_map[col] = np.float32
        df_data = pd.read_csv(self.raw_path, usecols=usecols, dtype=dtype_map)
        df_data['Tmstamp'] = pd.to_datetime(df_data['Tmstamp'])
        df_data = df_data.sort_values(['Tmstamp', 'TurbID']).drop_duplicates(['Tmstamp', 'TurbID'], keep='last')

        df_location = pd.read_csv(
            self.location_path,
            usecols=['TurbID', 'x', 'y'],
            dtype={'TurbID': np.int16, 'x': np.float32, 'y': np.float32},
        ).sort_values('TurbID').drop_duplicates(['TurbID'], keep='last')

        direction_valid = self._direction_valid_mask(df_data)
        wind_from_deg = np.mod(
            df_data['Ndir'].to_numpy(dtype=np.float32) + df_data['Wdir'].to_numpy(dtype=np.float32),
            360.0,
        ).astype(np.float32)

        dense = _build_dense_tensor_from_dataframe(
            df_data,
            feature_cols,
            patv_valid_mask=self._patv_valid_mask(df_data),
        )
        static_data = _build_static_data(df_location, dense.turbine_ids)

        turbine_index = pd.Index(dense.turbine_ids)
        timestamp_index = pd.Index(pd.to_datetime(dense.timestamps))
        row_idx = timestamp_index.get_indexer(df_data['Tmstamp'])
        col_idx = turbine_index.get_indexer(df_data['TurbID'])
        wind_from = np.zeros((len(dense.timestamps), len(dense.turbine_ids)), dtype=np.float32)
        wind_valid = np.zeros((len(dense.timestamps), len(dense.turbine_ids)), dtype=bool)
        wind_from[row_idx[direction_valid], col_idx[direction_valid]] = wind_from_deg[direction_valid]
        wind_valid[row_idx[direction_valid], col_idx[direction_valid]] = True

        return CanonicalBundle(
            dataset_name=self.dataset_name,
            feature_cols=feature_cols,
            turbine_ids=np.asarray(dense.turbine_ids),
            timestamps=np.asarray(dense.timestamps),
            raw_data=dense.raw_data,
            feature_observed_mask=dense.feature_observed_mask,
            patv_mask=dense.patv_mask,
            static_data=static_data,
            graph_context={'wind_from_deg': wind_from, 'wind_valid_mask': wind_valid},
        )


class PenmanshielAdapter(DatasetAdapter):
    def load_bundle(self) -> CanonicalBundle:
        feature_cols = list(self.dataset_cfg['feature_cols'])
        mapping = {
            '# Date and time': 'Tmstamp',
            'Wind speed (m/s)': 'Wspd',
            'Wind direction (°)': 'Wdir',
            'Nacelle ambient temperature (°C)': 'Etmp',
            'Nacelle temperature (°C)': 'Itmp',
            'Nacelle position (°)': 'Ndir',
            'Blade angle (pitch position) A (°)': 'Pab1',
            'Blade angle (pitch position) B (°)': 'Pab2',
            'Blade angle (pitch position) C (°)': 'Pab3',
            'Reactive power (kvar)': 'Prtv',
            'Power (kW)': 'Patv',
        }
        frames = []
        for path in sorted(Path(self.raw_path).glob('**/Turbine_Data_*.csv')):
            turbine_id = int(path.name.split('_')[3])
            df = pd.read_csv(path, skiprows=9, usecols=list(mapping.keys()))
            missing_required = [col for col in mapping if col not in df.columns]
            if missing_required:
                raise ValueError(f'Penmanshiel file is missing required columns {missing_required}: {path}')
            df = df.rename(columns=mapping)
            df['TurbID'] = turbine_id
            df['Tmstamp'] = pd.to_datetime(df['Tmstamp'])
            frames.append(df[['TurbID', 'Tmstamp'] + feature_cols])
        if not frames:
            raise FileNotFoundError(f'No Penmanshiel turbine data files were found under {self.raw_path}.')
        df_data = pd.concat(frames, ignore_index=True)

        df_loc_raw = pd.read_csv(self.location_path).dropna(subset=['Title'])
        df_loc_raw['TurbID'] = df_loc_raw['Alternative Title'].str.replace('T', '', regex=False).astype(int)
        x, y = latlon_to_xy(df_loc_raw['Latitude'], df_loc_raw['Longitude'])
        df_location = pd.DataFrame({'TurbID': df_loc_raw['TurbID'].values, 'x': x, 'y': y})

        dense = _build_dense_tensor_from_dataframe(df_data, feature_cols)
        return CanonicalBundle(
            dataset_name=self.dataset_name,
            feature_cols=feature_cols,
            turbine_ids=np.asarray(dense.turbine_ids),
            timestamps=np.asarray(dense.timestamps),
            raw_data=dense.raw_data,
            feature_observed_mask=dense.feature_observed_mask,
            patv_mask=dense.patv_mask,
            static_data=_build_static_data(df_location, dense.turbine_ids),
            graph_context=_build_wind_graph_context_from_wdir(dense),
        )


class HlrsAdapter(DatasetAdapter):
    @staticmethod
    def _load_layout(location_path: str) -> pd.DataFrame:
        with open(location_path, 'r', encoding='utf-8') as f:
            text = f.read()
        lines = [line for line in text.splitlines() if not line.strip().startswith('turbines: !include')]
        farm = yaml.safe_load('\n'.join(lines))
        coords = farm['layouts'][0]['coordinates']
        x = np.asarray(coords['x'], dtype=np.float32)
        y = np.asarray(coords['y'], dtype=np.float32)
        turbids = list(range(1, len(x) + 1))
        return pd.DataFrame({'TurbID': turbids, 'x': x, 'y': y})

    def load_bundle(self) -> CanonicalBundle:
        import netCDF4 as nc

        feature_cols = list(self.dataset_cfg['feature_cols'])
        raw_dir = Path(self.raw_path)
        obs = nc.Dataset(raw_dir / 'observedpower.nc')
        era = nc.Dataset(raw_dir / 'era5_wind_timeseries_flow_testpark_v2.nc')

        obs_times = pd.to_datetime([str(v) for v in obs.variables['time'][:]]).round('h')
        power = np.asarray(obs.variables['power'][:], dtype=np.float32)
        ewspd = np.asarray(obs.variables['effective_wind_speed'][:], dtype=np.float32)
        num_turbines = power.shape[1]
        turbids = list(range(1, num_turbines + 1))

        era_times = pd.to_datetime(era.variables['time'][:], unit='h', origin=pd.Timestamp('1900-01-01'))
        heights = np.asarray(era.variables['height'][:])
        target_height = self.dataset_cfg['era5_height_m']
        match = np.where(heights == target_height)[0]
        if match.size != 1:
            raise ValueError(f'Expected exactly one ERA5 height match for {target_height}. Found {match.tolist()}')
        height_idx = int(match[0])
        era_df = pd.DataFrame(
            {
                'Tmstamp': era_times,
                'Wdir': np.asarray(era.variables['WD'][:, height_idx], dtype=np.float32),
                'T2m': np.asarray(era.variables['T2'][:], dtype=np.float32),
                'Wspd_w': np.asarray(era.variables['WS'][:, height_idx], dtype=np.float32),
            }
        ).drop_duplicates(subset=['Tmstamp']).set_index('Tmstamp')
        era_df = era_df.reindex(obs_times)
        if era_df.isna().any().any():
            raise ValueError('HLRS ERA5 time axis cannot be aligned exactly with observedpower.')

        frames = []
        for col_idx, tid in enumerate(turbids):
            frames.append(
                pd.DataFrame(
                    {
                        'TurbID': tid,
                        'Tmstamp': obs_times,
                        'Wspd': ewspd[:, col_idx],
                        'Wdir': era_df['Wdir'].values,
                        'T2m': era_df['T2m'].values,
                        'Wspd_w': era_df['Wspd_w'].values,
                        'Patv': power[:, col_idx],
                    }
                )
            )
        df_data = pd.concat(frames, ignore_index=True)
        dense = _build_dense_tensor_from_dataframe(df_data, feature_cols)
        df_location = self._load_layout(self.location_path)
        return CanonicalBundle(
            dataset_name=self.dataset_name,
            feature_cols=feature_cols,
            turbine_ids=np.asarray(dense.turbine_ids),
            timestamps=np.asarray(dense.timestamps),
            raw_data=dense.raw_data,
            feature_observed_mask=dense.feature_observed_mask,
            patv_mask=dense.patv_mask,
            static_data=_build_static_data(df_location, dense.turbine_ids),
            graph_context={},
        )


class NorrekaerEngeAdapter(DatasetAdapter):
    def load_bundle(self) -> CanonicalBundle:
        import netCDF4 as nc

        feature_cols = list(self.dataset_cfg['feature_cols'])
        ds = nc.Dataset(self.raw_path)
        time_var = ds.variables['time']
        times = pd.to_datetime(nc.num2date(time_var[:], units=getattr(time_var, 'units'), only_use_cftime_datetimes=False))

        location_df = pd.read_csv(self.location_path)
        required_loc_cols = ['TurbID', 'x', 'y']
        if list(location_df.columns) != required_loc_cols:
            raise ValueError('Norre location file columns must be exactly: TurbID,x,y.')

        all_vars = set(ds.variables.keys())
        turbine_ids = sorted({name.rsplit('_', 1)[0] for name in all_vars if name.endswith('_wsn')})
        kept_turbines = [tid for tid in turbine_ids if f'{tid}_pow' in all_vars and f'{tid}_ym' in all_vars]
        if len(kept_turbines) != 40:
            raise ValueError(f'Norre should contain exactly 40 predictable turbines, but got {len(kept_turbines)}.')

        missing_locations = sorted(set(kept_turbines) - set(location_df['TurbID'].tolist()))
        if missing_locations:
            raise ValueError(f'Norre location file is missing turbines: {missing_locations}')

        wind_dir_signal = self.dataset_cfg['wind_direction_signal']
        wdir = np.asarray(ds.variables[wind_dir_signal][:], dtype=np.float32)
        frames = []
        for tid in kept_turbines:
            frames.append(
                pd.DataFrame(
                    {
                        'TurbID': tid,
                        'Tmstamp': times,
                        'Wspd': np.asarray(ds.variables[f'{tid}_wsn'][:], dtype=np.float32),
                        'Wdir': wdir,
                        'Ndir': np.asarray(ds.variables[f'{tid}_ym'][:], dtype=np.float32),
                        'Patv': np.asarray(ds.variables[f'{tid}_pow'][:], dtype=np.float32),
                    }
                )
            )
        df_data = pd.concat(frames, ignore_index=True)
        dense = _build_dense_tensor_from_dataframe(df_data, feature_cols)
        return CanonicalBundle(
            dataset_name=self.dataset_name,
            feature_cols=feature_cols,
            turbine_ids=np.asarray(dense.turbine_ids),
            timestamps=np.asarray(dense.timestamps),
            raw_data=dense.raw_data,
            feature_observed_mask=dense.feature_observed_mask,
            patv_mask=dense.patv_mask,
            static_data=_build_static_data(location_df, dense.turbine_ids),
            graph_context=_build_wind_graph_context_from_wdir(dense),
        )


ADAPTERS = {
    'sdwpf': SdwpfAdapter,
    'penmanshiel': PenmanshielAdapter,
    'hlrs': HlrsAdapter,
    'norrekaer_enge': NorrekaerEngeAdapter,
}


def load_dataset_bundle(dataset_name: str, raw_path: str, location_path: str) -> CanonicalBundle:
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f'Unknown dataset={dataset_name}. Available datasets: {sorted(DATASET_REGISTRY)}')
    if dataset_name not in ADAPTERS:
        raise KeyError(f'No adapter is registered for dataset={dataset_name}.')
    adapter = ADAPTERS[dataset_name](dataset_name, DATASET_REGISTRY[dataset_name], raw_path, location_path)
    bundle = adapter.load_bundle()
    if bundle.dataset_name != dataset_name:
        raise ValueError(f'Adapter returned dataset_name={bundle.dataset_name}, expected {dataset_name}.')
    available_turbines = DATASET_REGISTRY[dataset_name].get('available_turbines')
    if available_turbines is not None:
        bundle = _subset_bundle_to_available_turbines(bundle, list(available_turbines))
    return bundle
