"""
Node Scheduling Strategy for STCWPF
Simulating wind farm expansion in 4 periods
"""

def get_node_schedule():
    """
    Define which turbines are active in each period
    Simulating wind farm construction in 4 phases
    
    Returns:
        dict: {period: list of active TurbIDs}
    """
    # Strategy: Expand from west to east based on x-coordinate
    # This simulates realistic wind farm expansion
    
    node_schedule = {
        0: list(range(113, 135)) + list(range(91, 113)),  # Period 0: 60 turbines (TurbID 91-134)
        1: list(range(69, 135)),                            # Period 1: 90 turbines (add 69-90, total 69-134)
        2: list(range(36, 135)),                            # Period 2: 115 turbines (add 36-68, total 36-134)
        3: list(range(1, 135))                              # Period 3: 134 turbines (add 1-35, all turbines)
    }
    
    return node_schedule

def get_period_info():
    """
    Get detailed information about each period
    
    Returns:
        dict: Period information including node counts and new nodes
    """
    schedule = get_node_schedule()
    
    info = {}
    for period in sorted(schedule.keys()):
        nodes = schedule[period]
        if period == 0:
            new_nodes = nodes
        else:
            prev_nodes = schedule[period - 1]
            new_nodes = [n for n in nodes if n not in prev_nodes]
        
        info[period] = {
            'total_nodes': len(nodes),
            'new_nodes': len(new_nodes),
            'node_list': sorted(nodes),
            'new_node_list': sorted(new_nodes),
            'time_range': get_time_range(period)
        }
    
    return info

def get_time_range(period):
    """Get time range for each period"""
    time_ranges = {
        0: "2020-01 to 2020-06 (Q1-Q2)",
        1: "2020-07 to 2020-12 (Q3-Q4)",
        2: "2021-01 to 2021-06 (Q1-Q2)",
        3: "2021-07 to 2021-12 (Q3-Q4)"
    }
    return time_ranges.get(period, "Unknown")

def print_schedule_summary():
    """Print a summary of the node schedule"""
    info = get_period_info()
    
    print("=" * 80)
    print("STCWPF Node Schedule - Wind Farm Expansion Simulation")
    print("=" * 80)
    print()
    
    for period in sorted(info.keys()):
        p_info = info[period]
        print(f"Period {period}: {p_info['time_range']}")
        print(f"  Total Turbines: {p_info['total_nodes']}")
        print(f"  New Turbines:   {p_info['new_nodes']}")
        if period == 0:
            print(f"  Action: Initial construction (train from scratch)")
        else:
            print(f"  Action: Expansion (freeze backbone, train adaptive params)")
        new_list = p_info['new_node_list']
        if len(new_list) <= 10:
            print(f"  New TurbIDs: {new_list}")
        else:
            print(f"  New TurbIDs: {new_list[:10]}... (total {len(new_list)})")
        print()
    
    print("=" * 80)
    print("Model Update Summary:")
    print("  - Period 0: Train all parameters (backbone + adaptive)")
    print("  - Period 1: Update 1 - Freeze backbone, expand & train adaptive params")
    print("  - Period 2: Update 2 - Freeze backbone, expand & train adaptive params")
    print("  - Period 3: Update 3 - Freeze backbone, expand & train adaptive params")
    print("=" * 80)
    print()
    print("Total Updates: 3 times (Period 1, 2, 3)")
    print("Update Stage: Training phase (not validation)")
    print("=" * 80)

if __name__ == "__main__":
    print_schedule_summary()
