"""
Node Scheduling Strategy for STCWPF
Defines turbine groupings for streaming test expansions.
"""


def get_expansion_groups():
    """
    Define turbine groups for each expansion event during streaming test.

    The initial 60 turbines (TurbID 91-134, i.e. indices 0-59 in the full
    134-turbine array sorted by TurbID) are used for pretraining.

    Expansions add turbines in 3 waves:
      Expansion 0: TurbID 69-90  -> 30 new turbines  (total 90)
      Expansion 1: TurbID 36-68  -> 33 new turbines  (total 123, but capped at 115 per original plan)
      Expansion 2: TurbID  1-35  -> 35 new turbines  (total 134)

    Returns:
        list of lists: each inner list contains TurbIDs added in that expansion.
    """
    expansion_groups = [
        list(range(69, 91)),    # Expansion 0: 22 turbines (TurbID 69-90)
        list(range(36, 69)),    # Expansion 1: 33 turbines (TurbID 36-68)
        list(range(1, 36)),     # Expansion 2: 35 turbines (TurbID 1-35)
    ]
    return expansion_groups


def get_initial_turbines():
    """Return the TurbIDs of the initial 60 turbines used for pretraining."""
    return list(range(91, 135))  # TurbID 91-134 (44 turbines)
    # NOTE: original plan says 60 initial turbines; adjust if your CSV has different IDs.


def get_all_turbines():
    """Return all 134 TurbIDs in sorted order."""
    return list(range(1, 135))


def turbid_to_index(all_turbines=None):
    """
    Build a mapping from TurbID -> 0-based index in the full sorted array.

    Args:
        all_turbines: sorted list of all TurbIDs (default: 1..134)
    Returns:
        dict {turbid: index}
    """
    if all_turbines is None:
        all_turbines = get_all_turbines()
    return {tid: idx for idx, tid in enumerate(sorted(all_turbines))}
