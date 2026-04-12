from pathlib import Path


def _p(*parts):
    return str(Path(*parts))


DATASET_REGISTRY = {
    "sdwpf": {
        "dataset_key": "sdwpf",
        "raw_data_path": _p("data", "raw", "sdwpf", "sdwpf_2001_2112_full.csv"),
        "location_path": _p("data", "raw", "sdwpf", "sdwpf_turb_location_elevation.csv"),
        "streaming_freq_mode": "dynamic",
        "pretrain_freq_minutes": [5, 10, 15],
        "base_resolution_minutes": 5,
        "rated_power": 3200.0,
        "feature_cols": [
            "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3",
            "Prtv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp", "Patv"
        ],
        "static_feature_names": ["x", "y"],
        "processing_tag": "final_xy_16feat_multifreq_maskv1",
        "time_scope_tag": "fullrange",
        "mask_rule_tag": "sdwpf_mask_v1",
        "default_initial_turbines": list(range(75, 135)),
        "default_expansion_groups": [
            list(range(81, 91)),
            list(range(69, 81)),
            list(range(51, 69)),
            list(range(36, 51)),
            list(range(1, 36)),
        ],
    },
    "penmanshiel": {
        "dataset_key": "penmanshiel",
        "raw_data_path": _p("data", "raw", "penmanshiel"),
        "location_path": _p("data", "raw", "penmanshiel", "Penmanshiel_WT_static.csv"),
        "streaming_freq_mode": "fixed",
        "frequency_minutes": 10,
        "rated_power": 2050.0,
        "feature_cols": [
            "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"
        ],
        "static_feature_names": ["x", "y"],
        "processing_tag": "final_xy_10feat_10min_maskv1",
        "time_scope_tag": "fullrange",
        "mask_rule_tag": "penmanshiel_mask_v1",
        "default_initial_turbines": list(range(1, 9)),
        "default_expansion_groups": [[9, 10], [11, 12], [13], [14, 15]],
    },
    "hlrs": {
        "dataset_key": "hlrs",
        "raw_data_path": _p("data", "raw", "hlrs"),
        "location_path": _p("data", "raw", "hlrs", "farm.yaml"),
        "streaming_freq_mode": "fixed",
        "frequency_minutes": 60,
        "rated_power": 1650.0,
        "feature_cols": ["Wspd", "Wdir", "T2m", "Wspd_w", "Patv"],
        "static_feature_names": ["x", "y"],
        "processing_tag": "final_xy_5feat_hourly_maskv1",
        "time_scope_tag": "20100901_20110831",
        "mask_rule_tag": "hlrs_mask_v1",
        "default_initial_turbines": list(range(1, 81)),
        "default_expansion_groups": [
            list(range(81, 111)),
            list(range(111, 141)),
            list(range(141, 171)),
            list(range(171, 201)),
        ],
        "era5_height_m": 100,
    },
    "norrekaer_enge": {
        "dataset_key": "norrekaer_enge",
        "raw_data_path": _p("data", "raw", "norrekaer_enge", "norre_m2_all.nc"),
        "location_path": _p("data", "raw", "norrekaer_enge", "norre_m2_turbine_locations.csv"),
        "streaming_freq_mode": "fixed",
        "frequency_minutes": 10,
        "rated_power": 330.0,
        "feature_cols": ["Wspd", "Wdir", "Ndir", "Patv"],
        "static_feature_names": ["x", "y"],
        "processing_tag": "final_xy_4feat_fullrange10min_maskv1",
        "time_scope_tag": "19911222_19930627",
        "mask_rule_tag": "norre_mask_v1",
        "default_initial_turbines": [
            "b2", "b4", "c2", "c3", "c4", "c5", "d2", "d3", "d4", "d5", "e2", "e3",
            "e4", "e5", "f2", "f3", "f4", "f5", "a3", "a5", "b7", "d7", "e7", "f7"
        ],
        "default_expansion_groups": [
            ["b3", "b5", "c1", "d1", "e1", "f1"],
            ["a2", "a4", "b1", "c6", "d6", "e6"],
            ["a6", "a7", "b6", "c7", "d7", "e7"],
        ],
        "wind_direction_signal": "d31_1",
    },
}
