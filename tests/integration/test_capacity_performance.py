"""Performance and integration benchmarks for Phase 4 Capacity-Aware MRP."""

import time
import pytest

from src.core.explosion import BOMExplosionEngine
from src.capacity.rccp import RCCPEngine
from src.capacity.integration import CapacityAwareMRPPipeline


def test_performance_500_item_bom_capacity_aware_mrp_under_10_seconds():
    """Verify Phase 4 Exit Criteria: 500-item BOM with capacity check & leveling completes in < 10 seconds."""
    num_fg = 10
    num_sa = 50
    num_rm = 440

    items_list = []
    bom_list = []
    routings_list = []

    for i in range(1, num_fg + 1):
        items_list.append({
            "item_id": f"P-FG-{i:03d}",
            "type": "finished_good",
            "lead_time_periods": 1,
            "safety_stock": 10.0,
            "lot_sizing_rule": "L4L",
        })
        routings_list.append({
            "item_id": f"P-FG-{i:03d}",
            "work_center_id": "WC-300",
            "operation_sequence": 10,
            "setup_time_hours": 2.0,
            "run_time_hours_per_unit": 0.5,
        })

    for i in range(1, num_sa + 1):
        items_list.append({
            "item_id": f"P-SA-{i:03d}",
            "type": "subassembly",
            "lead_time_periods": 1,
            "safety_stock": 20.0,
            "lot_sizing_rule": "L4L",
        })
        routings_list.append({
            "item_id": f"P-SA-{i:03d}",
            "work_center_id": "WC-200",
            "operation_sequence": 10,
            "setup_time_hours": 1.0,
            "run_time_hours_per_unit": 0.2,
        })

    for i in range(1, num_rm + 1):
        items_list.append({
            "item_id": f"P-RM-{i:03d}",
            "type": "raw_material",
            "lead_time_periods": 2,
            "safety_stock": 50.0,
            "lot_sizing_rule": "L4L",
        })

    edge_id = 1
    for fg_idx in range(1, num_fg + 1):
        for sa_offset in range(5):
            sa_idx = ((fg_idx * 5 + sa_offset) % num_sa) + 1
            bom_list.append({
                "bom_id": f"B-{edge_id}",
                "parent_item_id": f"P-FG-{fg_idx:03d}",
                "component_item_id": f"P-SA-{sa_idx:03d}",
                "quantity_per": 2.0,
                "scrap_factor": 0.0,
            })
            edge_id += 1

    for sa_idx in range(1, num_sa + 1):
        for rm_offset in range(8):
            rm_idx = ((sa_idx * 8 + rm_offset) % num_rm) + 1
            bom_list.append({
                "bom_id": f"B-{edge_id}",
                "parent_item_id": f"P-SA-{sa_idx:03d}",
                "component_item_id": f"P-RM-{rm_idx:03d}",
                "quantity_per": 3.0,
                "scrap_factor": 0.01,
            })
            edge_id += 1

    work_centers_list = [
        {"work_center_id": "WC-200", "capacity_hours_per_week": 1000.0, "efficiency_factor": 1.0},
        {"work_center_id": "WC-300", "capacity_hours_per_week": 500.0, "efficiency_factor": 1.0},
    ]

    periods = list(range(1, 13))
    mps_dict = {f"P-FG-{i:03d}": [50.0 + (i * 2.0)] * 12 for i in range(1, num_fg + 1)}

    engine = BOMExplosionEngine(bom_list, items_list)
    rccp = RCCPEngine(routings_list, work_centers_list)
    pipeline = CapacityAwareMRPPipeline(engine, rccp)

    t0 = time.perf_counter()
    res = pipeline.run_capacity_aware_mrp(mps_dict, periods, auto_level=True)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"Capacity MRP took {elapsed:.2f}s, exceeding 10s budget!"
    assert len(res.final_planned_orders) > 0

