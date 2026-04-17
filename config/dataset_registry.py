from pathlib import Path


def _p(*parts):
    return str(Path(*parts))


DATASET_REGISTRY = {
    "sdwpf": {
        "dataset_key": "sdwpf",
        "raw_data_path": _p("data", "raw", "sdwpf", "sdwpf_2001_2112_full.csv"),
        "location_path": _p("data", "raw", "sdwpf", "sdwpf_turb_location_elevation.csv"),
        "streaming_freq_mode": "dynamic",
        "supported_frequency_minutes": [10, 15],
        "rated_power": 3200.0,
        "feature_cols": [
            "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3",
            "Prtv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp", "Patv"
        ],
        "static_feature_names": ["x", "y"],
        "available_turbines": list(range(1, 135)),
        "default_initial_turbines": list(range(1, 61)),
        "default_expansion_groups": [
            list(range(61, 76)),
            list(range(76, 94)),
            list(range(94, 109)),
            list(range(109, 135)),
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
        "available_turbines": [1, 2, 4, 5, 6, 7, 8, 9, 10],
        "default_initial_turbines": [1, 2, 4, 5, 6, 7],
        "default_expansion_groups": [[8], [9], [10]],
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
        "available_turbines": [
            "a2", "a3", "a4", "a5", "a6", "a7", "b1", "b2", "b3", "b4",
            "b5", "b6", "b7", "c1", "c2", "c3", "c4", "c5", "c6", "c7",
            "d1", "d2", "d3", "d4", "d5", "d6", "d7", "e1", "e2", "e3",
            "e4", "e5", "e6", "e7", "f1", "f2", "f3", "f4", "f5", "f7",
        ],
        "default_initial_turbines": [
            "b4", "a5", "b5", "f3", "d6", "d5", "e4", "b6", "a6", "e6", "d2", "e2",
            "d4", "c6", "b7", "f1", "f2", "d7", "c1", "d3", "d1", "e5", "f4", "e3",
        ],
        "default_expansion_groups": [
            ["c7", "c4", "f7", "e7", "a4", "a2"],
            ["f5", "e1", "a7", "b1", "b3", "c2"],
            ["c3", "c5", "a3", "b2"],
        ],
        "wind_direction_signal": "d31_1",
    },
}
