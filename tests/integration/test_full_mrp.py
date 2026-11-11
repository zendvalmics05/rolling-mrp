"""Integration and Performance Tests for Full Multi-Level MRP Run (Phase 1)."""

import time
import pytest
import pandas as pd
from src.integration.data_generator import generate_synthetic_data
from src.integration.load import clean_and_validate_dataset, initialize_inventory_status
from src.core.netting import aggregate_gross_requirements
from src.core.explosion import BOMExplosionEngine


def test_full_factory_25_item_mrp_run():
    """Verify end-to-end multi-level MRP run across all 25 items and 12 planning periods."""
    raw_data = generate_synthetic_data(seed=42)
    cleaned_data, _ = clean_and_validate_dataset(raw_data)
    inv_df = initialize_inventory_status(cleaned_data["inventory"], cleaned_data["items"])

    periods = list(range(1, 13))

    # Aggregate Finished Goods demand from MPS
    mps_dict = aggregate_gross_requirements(cleaned_data["mps"], periods)

    # Format starting inventory dict
    inv_dict = {}
    for _, row in inv_df.iterrows():
        inv_dict[str(row["item_id"])] = {
            "on_hand": float(row["quantity_on_hand"]),
            "allocated": float(row["quantity_allocated"]),
            "safety_stock": float(row["safety_stock"]),
            "lead_time_days": int(row["lead_time_days"]),
        }

    # Initialize BOM Engine
    engine = BOMExplosionEngine(cleaned_data["bom"], cleaned_data["items"])

    # Run multi-level MRP
    start_time = time.perf_counter()
    results, pegging_tracker = engine.run_multi_level_mrp(
        mps_gross_requirements=mps_dict,
        periods=periods,
        inventory_status=inv_dict,
    )
    elapsed = time.perf_counter() - start_time

    # 1. Coverage: all 25 items processed
    assert len(results) == 25
    assert set(results.keys()) == set(cleaned_data["items"]["item_id"])

    # 2. Validity: No negative PAB without corresponding planned orders
    for item_id, res in results.items():
        for p_idx, pab in enumerate(res.projected_available_balance):
            if pab < res.safety_stock:
                # If below safety stock, net requirement or orders must have been triggered
                assert res.net_requirements[p_idx] > 0 or len(res.planned_orders) > 0

    # 3. Planned Orders Generated across all tiers
    total_orders = sum(len(res.planned_orders) for res in results.values())
    assert total_orders > 0

    # 4. Pegging Traceability
    pegging_report = pegging_tracker.generate_pegging_report()
    assert not pegging_report.empty
    # Must trace from raw materials up to MPS
    rm_items = set(cleaned_data["items"][cleaned_data["items"]["type"] == "raw_material"]["item_id"])
    pegged_children = set(pegging_report["child_item_id"])
    assert len(rm_items.intersection(pegged_children)) > 0

    # 5. Speed: 25-item MRP run executes in under 0.5s
    assert elapsed < 0.5


def test_performance_500_item_bom_mrp_under_5_seconds():
    """Verify Master Plan & Implementation Plan §1 Exit Criteria: 500-item BOM MRP run completes in < 5 seconds."""
    # Construct synthetic 500-item tree: 10 finished goods, 50 subassemblies, 440 components
    num_fg = 10
    num_sa = 50
    num_rm = 440
    total_items = num_fg + num_sa + num_rm

    items_list = []
    bom_list = []

    for i in range(1, num_fg + 1):
        items_list.append({"item_id": f"P-FG-{i:03d}", "type": "finished_good", "lead_time_periods": 1, "safety_stock": 10.0, "lot_sizing_rule": "L4L"})
    for i in range(1, num_sa + 1):
        items_list.append({"item_id": f"P-SA-{i:03d}", "type": "subassembly", "lead_time_periods": 1, "safety_stock": 20.0, "lot_sizing_rule": "L4L"})
    for i in range(1, num_rm + 1):
        items_list.append({"item_id": f"P-RM-{i:03d}", "type": "raw_material", "lead_time_periods": 2, "safety_stock": 50.0, "lot_sizing_rule": "L4L"})

    # Wire BOM connections: each FG uses 5 SAs; each SA uses 8-9 RMs
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

    periods = list(range(1, 13))
    mps_dict = {f"P-FG-{i:03d}": [50.0 + (i * 2.0)] * 12 for i in range(1, num_fg + 1)}

    engine = BOMExplosionEngine(bom_list, items_list)

    start = time.perf_counter()
    results, pegging = engine.run_multi_level_mrp(mps_dict, periods)
    elapsed = time.perf_counter() - start

    assert len(results) == total_items
    assert elapsed < 5.0, f"500-item MRP run took {elapsed:.2f}s (target < 5.0s)"

