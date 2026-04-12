from config.dataset_registry import DATASET_REGISTRY


"""
Node Scheduling Strategy for STCWPF
Defines turbine groupings for streaming test expansions.
"""


def _filter_available(items, available):
    if available is None:
        return list(items)
    available_set = set(available)
    return [x for x in items if x in available_set]


def get_initial_turbines(dataset="sdwpf", available_turbines=None):
    items = DATASET_REGISTRY[dataset]["default_initial_turbines"]
    return _filter_available(items, available_turbines)


def get_expansion_groups(dataset="sdwpf", available_turbines=None):
    groups = DATASET_REGISTRY[dataset]["default_expansion_groups"]
    if available_turbines is None:
        return [list(g) for g in groups]
    return [_filter_available(g, available_turbines) for g in groups if _filter_available(g, available_turbines)]


def get_all_turbines(dataset="sdwpf", available_turbines=None):
    if available_turbines is not None:
        return sorted(available_turbines)
    initial = get_initial_turbines(dataset)
    groups = get_expansion_groups(dataset)
    all_turbines = set(initial)
    for g in groups:
        all_turbines.update(g)
    return sorted(all_turbines)


def turbid_to_index(all_turbines=None):
    if all_turbines is None:
        all_turbines = get_all_turbines("sdwpf")
    return {tid: idx for idx, tid in enumerate(sorted(all_turbines))}
