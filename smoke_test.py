"""
Smoke test: adapter loading + pipeline processing for all 4 datasets.
Run from repo root: python smoke_test.py
"""
import sys
import traceback
import tempfile
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config.dataset_registry import DATASET_REGISTRY
from data.adapters import load_dataset_bundle, CanonicalBundle
from data.graph_generation import GRAPH_VARIANT_BASELINE

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []


def check(name, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append((name, "PASS", None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [{FAIL}] {name}")
        print(f"         {type(e).__name__}: {e}")
        results.append((name, "FAIL", tb))


def warn(name, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append((name, "PASS", None))
    except Exception as e:
        print(f"  [{WARN}] {name} (non-fatal)")
        print(f"         {type(e).__name__}: {e}")
        results.append((name, "WARN", str(e)))


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def validate_bundle(bundle: CanonicalBundle, dataset_name: str):
    assert bundle.dataset_name == dataset_name, f"dataset_name mismatch: {bundle.dataset_name}"
    T, N, F = bundle.raw_data.shape
    assert T > 0 and N > 0 and F > 0, f"raw_data shape invalid: {bundle.raw_data.shape}"
    assert bundle.feature_observed_mask.shape == (T, N, F), \
        f"feature_observed_mask shape {bundle.feature_observed_mask.shape} != {(T, N, F)}"
    assert bundle.patv_mask.shape == (T, N), \
        f"patv_mask shape {bundle.patv_mask.shape} != {(T, N)}"
    assert bundle.static_data.shape == (N, 2), \
        f"static_data shape {bundle.static_data.shape} != {(N, 2)}"
    assert len(bundle.turbine_ids) == N, \
        f"turbine_ids length {len(bundle.turbine_ids)} != N={N}"
    assert len(bundle.timestamps) == T, \
        f"timestamps length {len(bundle.timestamps)} != T={T}"
    assert bundle.feature_cols[-1] == "Patv", \
        f"Last feature must be Patv, got {bundle.feature_cols[-1]}"

    import pandas as pd
    ts = pd.to_datetime(bundle.timestamps)
    assert ts.is_monotonic_increasing, "timestamps not monotonically increasing"

    obs_rate = float(bundle.feature_observed_mask.mean())
    patv_rate = float(bundle.patv_mask.mean())
    print(f"         T={T}, N={N}, F={F}  obs_rate={obs_rate:.3f}  patv_valid_rate={patv_rate:.3f}")
    print(f"         turbine_ids sample: {list(bundle.turbine_ids[:5])}")
    print(f"         time range: {bundle.timestamps[0]} → {bundle.timestamps[-1]}")
    print(f"         feature_cols: {bundle.feature_cols}")


def make_mock_args(graph_variant=GRAPH_VARIANT_BASELINE):
    class Args:
        pass
    a = Args()
    a.graph_variant = graph_variant
    a.baseline_weight_threshold = 0.95
    a.local_upstream_top_k = 16
    return a


# ─────────────────────────────────────────────
# 1. Adapter tests
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("1. ADAPTER TESTS")
print("="*60)

# ── SDWPF ──
print("\n[sdwpf]")
sdwpf_bundle = None

def _load_sdwpf():
    global sdwpf_bundle
    cfg = DATASET_REGISTRY["sdwpf"]
    sdwpf_bundle = load_dataset_bundle("sdwpf", cfg["raw_data_path"], cfg["location_path"])
    validate_bundle(sdwpf_bundle, "sdwpf")
    assert "wind_from_deg" in sdwpf_bundle.graph_context, "Missing wind_from_deg in graph_context"
    assert "wind_valid_mask" in sdwpf_bundle.graph_context, "Missing wind_valid_mask in graph_context"
    wf = sdwpf_bundle.graph_context["wind_from_deg"]
    wv = sdwpf_bundle.graph_context["wind_valid_mask"]
    T, N, _ = sdwpf_bundle.raw_data.shape
    assert wf.shape == (T, N), f"wind_from_deg shape {wf.shape} != {(T, N)}"
    assert wv.shape == (T, N), f"wind_valid_mask shape {wv.shape} != {(T, N)}"
    valid_frac = float(wv.mean())
    print(f"         wind_valid_frac={valid_frac:.3f}")

check("sdwpf: load_bundle + shape validation", _load_sdwpf)

def _sdwpf_patv_range():
    assert sdwpf_bundle is not None
    patv_idx = sdwpf_bundle.feature_cols.index("Patv")
    patv = sdwpf_bundle.raw_data[:, :, patv_idx]
    mask = sdwpf_bundle.patv_mask.astype(bool)
    valid_patv = patv[mask]
    assert (valid_patv >= 0).all(), f"Negative Patv after masking: min={valid_patv.min()}"
    print(f"         Patv range (valid): [{valid_patv.min():.1f}, {valid_patv.max():.1f}]  mean={valid_patv.mean():.1f}")

check("sdwpf: Patv >= 0 after masking", _sdwpf_patv_range)

def _sdwpf_static():
    assert sdwpf_bundle is not None
    sd = sdwpf_bundle.static_data
    assert np.all(np.isfinite(sd)), "static_data contains non-finite values"
    std = sd.std(axis=0)
    assert np.all(std > 0), f"static_data has zero std: {std}"
    print(f"         static_data x range: [{sd[:,0].min():.1f}, {sd[:,0].max():.1f}]  y range: [{sd[:,1].min():.1f}, {sd[:,1].max():.1f}]")

check("sdwpf: static_data finite + nonzero std", _sdwpf_static)

def _sdwpf_schedule_order():
    cfg = DATASET_REGISTRY["sdwpf"]
    initial = cfg["default_initial_turbines"]
    groups = cfg["default_expansion_groups"]
    assert initial == list(range(1, 61)), f"Expected sdwpf initial turbines to be 1..60, got {initial}"
    expected_groups = [
        list(range(61, 76)),
        list(range(76, 94)),
        list(range(94, 109)),
        list(range(109, 135)),
    ]
    assert groups == expected_groups, f"Unexpected sdwpf expansion groups: {groups}"
    print(f"         initial={initial[0]}..{initial[-1]}  expansions={[f'{g[0]}..{g[-1]}' for g in groups]}")

check("sdwpf: schedule order is ascending", _sdwpf_schedule_order)

# ── Penmanshiel ──
print("\n[penmanshiel]")
penman_bundle = None

def _load_penmanshiel():
    global penman_bundle
    cfg = DATASET_REGISTRY["penmanshiel"]
    penman_bundle = load_dataset_bundle("penmanshiel", cfg["raw_data_path"], cfg["location_path"])
    validate_bundle(penman_bundle, "penmanshiel")
    assert penman_bundle.graph_context == {}, f"Expected empty graph_context, got {penman_bundle.graph_context.keys()}"

check("penmanshiel: load_bundle + shape validation", _load_penmanshiel)

def _penmanshiel_turbine_ids():
    assert penman_bundle is not None
    ids = penman_bundle.turbine_ids.tolist()
    assert all(isinstance(i, (int, np.integer)) for i in ids), f"Expected int turbine IDs, got {ids[:5]}"
    print(f"         turbine_ids: {sorted(ids)}")
    cfg = DATASET_REGISTRY["penmanshiel"]
    expected_initial = cfg["default_initial_turbines"]
    available = set(ids)
    missing = [t for t in expected_initial if t not in available]
    if missing:
        print(f"         WARN: initial turbines missing from data: {missing}")

check("penmanshiel: turbine IDs are integers", _penmanshiel_turbine_ids)

def _penmanshiel_time_coverage():
    assert penman_bundle is not None
    import pandas as pd
    ts = pd.to_datetime(penman_bundle.timestamps)
    diffs = ts.diff().dropna()
    freq_counts = diffs.value_counts()
    print(f"         time diff distribution (top 5): {freq_counts.head().to_dict()}")
    dominant_minutes = int(diffs.mode()[0].total_seconds() / 60)
    assert dominant_minutes == 10, f"Expected dominant 10min frequency, got {dominant_minutes}min"

check("penmanshiel: dominant frequency is 10min", _penmanshiel_time_coverage)

# ── HLRS ──
print("\n[hlrs]")
hlrs_bundle = None

def _load_hlrs():
    global hlrs_bundle
    cfg = DATASET_REGISTRY["hlrs"]
    hlrs_bundle = load_dataset_bundle("hlrs", cfg["raw_data_path"], cfg["location_path"])
    validate_bundle(hlrs_bundle, "hlrs")
    assert hlrs_bundle.graph_context == {}, f"Expected empty graph_context, got {hlrs_bundle.graph_context.keys()}"

check("hlrs: load_bundle + shape validation", _load_hlrs)

def _hlrs_turbine_count():
    assert hlrs_bundle is not None
    N = hlrs_bundle.raw_data.shape[1]
    assert N == 200, f"Expected 200 turbines, got {N}"
    print(f"         N={N} turbines confirmed")

check("hlrs: exactly 200 turbines", _hlrs_turbine_count)

def _hlrs_time_frequency():
    assert hlrs_bundle is not None
    import pandas as pd
    ts = pd.to_datetime(hlrs_bundle.timestamps)
    diffs = ts.diff().dropna()
    dominant_minutes = int(diffs.mode()[0].total_seconds() / 60)
    assert dominant_minutes == 60, f"Expected dominant 60min frequency, got {dominant_minutes}min"
    print(f"         dominant freq={dominant_minutes}min  T={len(ts)}")

check("hlrs: dominant frequency is 60min", _hlrs_time_frequency)

def _hlrs_era5_alignment():
    assert hlrs_bundle is not None
    # ERA5 fields should be non-NaN in raw_data (they're broadcast to all turbines)
    cfg = DATASET_REGISTRY["hlrs"]
    feat = hlrs_bundle.feature_cols
    wdir_idx = feat.index("Wdir")
    wdir_data = hlrs_bundle.raw_data[:, 0, wdir_idx]  # turbine 0, Wdir from ERA5
    obs = hlrs_bundle.feature_observed_mask[:, 0, wdir_idx].astype(bool)
    assert obs.mean() > 0.95, f"ERA5 Wdir observed rate too low: {obs.mean():.3f}"
    print(f"         ERA5 Wdir observed rate (turbine 0): {obs.mean():.4f}")

check("hlrs: ERA5 Wdir alignment coverage > 95%", _hlrs_era5_alignment)

# ── Norrekaer Enge ──
print("\n[norrekaer_enge]")
norre_bundle = None

def _load_norre():
    global norre_bundle
    cfg = DATASET_REGISTRY["norrekaer_enge"]
    norre_bundle = load_dataset_bundle("norrekaer_enge", cfg["raw_data_path"], cfg["location_path"])
    validate_bundle(norre_bundle, "norrekaer_enge")
    assert norre_bundle.graph_context == {}, f"Expected empty graph_context, got {norre_bundle.graph_context.keys()}"

check("norrekaer_enge: load_bundle + shape validation", _load_norre)

def _norre_turbine_ids():
    assert norre_bundle is not None
    ids = norre_bundle.turbine_ids.tolist()
    assert all(isinstance(i, str) for i in ids), f"Expected string turbine IDs, got {ids[:5]}"
    assert len(ids) == 40, f"Expected 40 turbines, got {len(ids)}"
    print(f"         turbine_ids sample: {sorted(ids)[:8]}")

check("norrekaer_enge: 40 string turbine IDs", _norre_turbine_ids)

def _norre_time_axis():
    assert norre_bundle is not None
    import pandas as pd
    ts = pd.to_datetime(norre_bundle.timestamps)
    diffs = ts.diff().dropna()
    dominant_minutes = int(diffs.mode()[0].total_seconds() / 60)
    assert dominant_minutes == 10, f"Expected dominant 10min frequency, got {dominant_minutes}min"
    minute_phase = sorted(pd.Series(ts.minute).unique().tolist())
    assert any(minute % 10 != 0 for minute in minute_phase), (
        f"Expected Norre to preserve its native off-grid phase, but minutes were {minute_phase}."
    )
    print(f"         dominant freq={dominant_minutes}min  native minute phases={minute_phase[:12]}  T={len(ts)}")

check("norrekaer_enge: preserve native 10min time axis", _norre_time_axis)

def _norre_initial_turbines_present():
    assert norre_bundle is not None
    cfg = DATASET_REGISTRY["norrekaer_enge"]
    available = set(norre_bundle.turbine_ids.tolist())
    initial = cfg["default_initial_turbines"]
    missing = [t for t in initial if t not in available]
    assert not missing, f"Initial turbines missing from data: {missing}"
    print(f"         All {len(initial)} initial turbines present in data")

check("norrekaer_enge: all default_initial_turbines present", _norre_initial_turbines_present)


# ─────────────────────────────────────────────
# 2. Pipeline tests (process_unified_dataset)
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("2. PIPELINE TESTS (process_unified_dataset)")
print("="*60)

from data.data_processing import process_unified_dataset

REQUIRED_KEYS_FIXED = [
    "dataset_name", "adj_type", "feature_cols", "static_feature_names",
    "raw_data", "feature_observed_mask", "patv_mask", "static_data",
    "static_mean", "static_std", "x_mean", "x_std", "y_mean", "y_std",
    "pretrain_end_idx", "val_end_idx", "initial_cols", "initial_n", "initial_turbines",
    "expansion_groups", "turbine_schedule_keys", "turbine_schedule_t_offsets",
    "turbine_schedule_new_cols", "raw_timestamps", "streaming_freq_mode", "frequency_minutes",
]
REQUIRED_KEYS_DYNAMIC = [k for k in REQUIRED_KEYS_FIXED if k != "frequency_minutes"] + ["supported_frequency_minutes"]


def run_pipeline(dataset_name):
    cfg = DATASET_REGISTRY[dataset_name]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "processed")
        graph_dir = os.path.join(tmpdir, "graph")
        args = make_mock_args(GRAPH_VARIANT_BASELINE)
        process_unified_dataset(
            raw_csv_path=cfg["raw_data_path"],
            location_csv_path=cfg["location_path"],
            save_dir=save_dir,
            graph_dir=graph_dir,
            x_len=12,
            y_len=12,
            args=args,
            dataset=dataset_name,
        )
        npz_path = os.path.join(save_dir, "unified_data.npz")
        assert os.path.exists(npz_path), f"unified_data.npz not created at {npz_path}"
        data = np.load(npz_path, allow_pickle=True)

        freq_mode = cfg["streaming_freq_mode"]
        required = REQUIRED_KEYS_DYNAMIC if freq_mode == "dynamic" else REQUIRED_KEYS_FIXED
        missing_keys = [k for k in required if k not in data.files]
        assert not missing_keys, f"Missing keys in unified_data.npz: {missing_keys}"

        raw_data = data["raw_data"]
        T, N, F = raw_data.shape
        pretrain_end = int(data["pretrain_end_idx"])
        val_end = int(data["val_end_idx"])
        assert 0 < pretrain_end < val_end < T, \
            f"Invalid split: pretrain={pretrain_end}, val={val_end}, T={T}"

        x_mean = data["x_mean"]
        x_std = data["x_std"]
        assert x_mean.shape == (F,), f"x_mean shape {x_mean.shape} != ({F},)"
        assert x_std.shape == (F,), f"x_std shape {x_std.shape} != ({F},)"
        assert np.all(x_std > 0), f"x_std has zero entries: {x_std}"

        static_data = data["static_data"]
        assert static_data.shape == (N, 2), f"static_data shape {static_data.shape} != ({N}, 2)"

        initial_cols = data["initial_cols"].tolist()
        assert len(initial_cols) > 0, "initial_cols is empty"

        # Check stage graph files
        n_stages = len(data["turbine_schedule_keys"]) + 1
        for stage_idx in range(n_stages):
            adj_path = os.path.join(graph_dir, f"stage_{stage_idx}_adj.npz")
            assert os.path.exists(adj_path), f"Missing stage graph: {adj_path}"
            adj_data = np.load(adj_path)
            adj = adj_data["x"]
            assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], \
                f"stage_{stage_idx} adj not square: {adj.shape}"

        print(f"         T={T}, N={N}, F={F}  split={pretrain_end}/{val_end}/{T}")
        print(f"         initial_cols={len(initial_cols)}  n_stages={n_stages}")
        print(f"         x_std range: [{x_std.min():.4f}, {x_std.max():.4f}]")


for ds in ["sdwpf", "penmanshiel", "hlrs", "norrekaer_enge"]:
    print(f"\n[{ds}]")
    check(f"{ds}: process_unified_dataset → unified_data.npz + stage graphs", lambda d=ds: run_pipeline(d))


# ─────────────────────────────────────────────
# 3. Registry consistency checks
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("3. REGISTRY CONSISTENCY CHECKS")
print("="*60)

def _registry_all_datasets():
    for ds_name, cfg in DATASET_REGISTRY.items():
        assert "feature_cols" in cfg, f"{ds_name}: missing feature_cols"
        assert cfg["feature_cols"][-1] == "Patv", f"{ds_name}: last feature must be Patv"
        assert "streaming_freq_mode" in cfg, f"{ds_name}: missing streaming_freq_mode"
        mode = cfg["streaming_freq_mode"]
        if mode == "dynamic":
            assert "supported_frequency_minutes" in cfg, f"{ds_name}: dynamic mode needs supported_frequency_minutes"
        elif mode == "fixed":
            assert "frequency_minutes" in cfg, f"{ds_name}: fixed mode needs frequency_minutes"
        else:
            raise ValueError(f"{ds_name}: unknown streaming_freq_mode={mode}")
        assert "default_initial_turbines" in cfg, f"{ds_name}: missing default_initial_turbines"
        assert "default_expansion_groups" in cfg, f"{ds_name}: missing default_expansion_groups"
        assert len(cfg["default_initial_turbines"]) > 0, f"{ds_name}: empty default_initial_turbines"
    print(f"         All {len(DATASET_REGISTRY)} datasets pass registry schema check")

check("registry: all datasets have required fields", _registry_all_datasets)

def _norre_expansion_turbine_overlap():
    cfg = DATASET_REGISTRY["norrekaer_enge"]
    initial = set(cfg["default_initial_turbines"])
    for i, group in enumerate(cfg["default_expansion_groups"]):
        overlap = initial & set(group)
        assert not overlap, f"norre expansion group {i} overlaps with initial: {overlap}"
    print("         No overlap between initial turbines and expansion groups")

check("norrekaer_enge: no overlap between initial and expansion groups", _norre_expansion_turbine_overlap)

def _norre_schedule_order():
    cfg = DATASET_REGISTRY["norrekaer_enge"]
    expected_initial = [
        "b4", "a5", "b5", "f3", "d6", "d5", "e4", "b6", "a6", "e6", "d2", "e2",
        "d4", "c6", "b7", "f1", "f2", "d7", "c1", "d3", "d1", "e5", "f4", "e3",
    ]
    expected_groups = [
        ["c7", "c4", "f7", "e7", "a4", "a2"],
        ["f5", "e1", "a7", "b1", "b3", "c2"],
        ["c3", "c5", "a3", "b2"],
    ]
    assert cfg["default_initial_turbines"] == expected_initial, cfg["default_initial_turbines"]
    assert cfg["default_expansion_groups"] == expected_groups, cfg["default_expansion_groups"]
    print("         fixed random split matches registry (seeded protocol)")

check("norrekaer_enge: schedule order matches fixed random protocol", _norre_schedule_order)


# ─────────────────────────────────────────────
# 4. Downstream schema validation (main.py logic)
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("4. DOWNSTREAM SCHEMA VALIDATION")
print("="*60)

def run_schema_validation(dataset_name):
    """Simulate what main.py does when loading unified_data.npz."""
    cfg = DATASET_REGISTRY[dataset_name]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "processed")
        graph_dir = os.path.join(tmpdir, "graph")
        args = make_mock_args(GRAPH_VARIANT_BASELINE)
        process_unified_dataset(
            raw_csv_path=cfg["raw_data_path"],
            location_csv_path=cfg["location_path"],
            save_dir=save_dir,
            graph_dir=graph_dir,
            x_len=12,
            y_len=12,
            args=args,
            dataset=dataset_name,
        )
        npz_path = os.path.join(save_dir, "unified_data.npz")
        data = np.load(npz_path, allow_pickle=True)

        # Replicate main.py _validate_loaded_dataset logic
        class MockArgs:
            pass
        mock = MockArgs()
        mock.dataset = dataset_name
        mock.graph_variant = GRAPH_VARIANT_BASELINE
        mock.streaming_freq_mode = cfg["streaming_freq_mode"]

        assert str(data["dataset_name"]) == mock.dataset
        assert str(data["adj_type"]) == mock.graph_variant
        assert data["feature_cols"].tolist() == cfg["feature_cols"]
        assert data["static_feature_names"].tolist() == ["x", "y"]

        if mock.streaming_freq_mode == "dynamic":
            assert "supported_frequency_minutes" in data.files
            assert data["supported_frequency_minutes"].tolist() == cfg["supported_frequency_minutes"]
        else:
            assert "frequency_minutes" in data.files
            assert int(data["frequency_minutes"]) == int(cfg["frequency_minutes"])

        # Simulate args population
        raw_data = data["raw_data"]
        F = raw_data.shape[2]
        x_mean = data["x_mean"].astype(np.float32)
        x_std = data["x_std"].astype(np.float32)
        assert x_mean.shape == (F,), f"x_mean shape {x_mean.shape}"
        assert x_std.shape == (F,), f"x_std shape {x_std.shape}"

        static_data = data["static_data"]
        assert static_data.shape[1] == 2

        print(f"         Schema validation passed for {dataset_name}")


for ds in ["sdwpf", "penmanshiel", "hlrs", "norrekaer_enge"]:
    print(f"\n[{ds}]")
    check(f"{ds}: downstream schema validation (main.py compatible)", lambda d=ds: run_schema_validation(d))


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
passed = sum(1 for _, s, _ in results if s == "PASS")
warned = sum(1 for _, s, _ in results if s == "WARN")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"  PASS: {passed}  WARN: {warned}  FAIL: {failed}  TOTAL: {len(results)}")

if failed > 0:
    print("\nFailed tests:")
    for name, status, tb in results:
        if status == "FAIL":
            print(f"\n  [{FAIL}] {name}")
            if tb:
                print(tb)

sys.exit(0 if failed == 0 else 1)
