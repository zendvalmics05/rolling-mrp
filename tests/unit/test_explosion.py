"""Unit tests for BOM Explosion Engine and Multi-Level Netting (src/core/explosion.py)."""

import pytest
import pandas as pd
from src.core.explosion import BOMExplosionEngine


def test_low_level_code_calculation():
    items = [
        {"item_id": "FG", "type": "finished_good"},
        {"item_id": "SA", "type": "subassembly"},
        {"item_id": "RM", "type": "raw_material"},
    ]
    bom = [
        {"parent_item_id": "FG", "component_item_id": "SA", "quantity_per": 1.0, "scrap_factor": 0.0},
        {"parent_item_id": "SA", "component_item_id": "RM", "quantity_per": 2.0, "scrap_factor": 0.0},
        # Also direct edge from FG to RM (commonality across tiers)
        {"parent_item_id": "FG", "component_item_id": "RM", "quantity_per": 1.0, "scrap_factor": 0.0},
    ]
    engine = BOMExplosionEngine(bom, items)
    # FG is level 0, SA is level 1, RM must have LLC = 2 (max depth)
    assert engine.low_level_codes["FG"] == 0
    assert engine.low_level_codes["SA"] == 1
    assert engine.low_level_codes["RM"] == 2


def test_single_level_explosion_with_scrap():
    items = [
        {"item_id": "P", "type": "finished_good"},
        {"item_id": "C1", "type": "component"},
        {"item_id": "C2", "type": "component"},
    ]
    bom = [
        {"parent_item_id": "P", "component_item_id": "C1", "quantity_per": 2.0, "scrap_factor": 0.0},
        {"parent_item_id": "P", "component_item_id": "C2", "quantity_per": 1.0, "scrap_factor": 0.2}, # 20% scrap
    ]
    engine = BOMExplosionEngine(bom, items)
    res = engine.explode_single_level("P", parent_quantity=10.0)

    c1 = [c for c in res if c["component_item_id"] == "C1"][0]
    assert c1["gross_quantity"] == 20.0

    c2 = [c for c in res if c["component_item_id"] == "C2"][0]
    # 10 * 1.0 / (1 - 0.2) = 10 / 0.8 = 12.5
    assert c2["gross_quantity"] == 12.5


def test_multi_level_memoized_explosion_and_phantoms():
    items = [
        {"item_id": "TOP", "type": "finished_good"},
        {"item_id": "PHANTOM-KIT", "type": "phantom"},
        {"item_id": "SUB-PART-A", "type": "raw_material"},
        {"item_id": "SUB-PART-B", "type": "raw_material"},
    ]
    bom = [
        {"parent_item_id": "TOP", "component_item_id": "PHANTOM-KIT", "quantity_per": 1.0, "scrap_factor": 0.0, "is_phantom": True},
        {"parent_item_id": "PHANTOM-KIT", "component_item_id": "SUB-PART-A", "quantity_per": 3.0, "scrap_factor": 0.0},
        {"parent_item_id": "PHANTOM-KIT", "component_item_id": "SUB-PART-B", "quantity_per": 5.0, "scrap_factor": 0.0},
    ]
    engine = BOMExplosionEngine(bom, items)
    exploded = engine.explode_multi_level("TOP", parent_quantity=2.0)

    # Phantom should be passed through; only sub-parts appear
    comp_ids = {e["component_item_id"] for e in exploded}
    assert "PHANTOM-KIT" not in comp_ids
    assert comp_ids == {"SUB-PART-A", "SUB-PART-B"}

    part_a = [e for e in exploded if e["component_item_id"] == "SUB-PART-A"][0]
    assert part_a["gross_quantity"] == 6.0  # 2 * 3.0

    # Second call should use memoized cache
    exploded_cached = engine.explode_multi_level("TOP", parent_quantity=2.0)
    assert len(exploded_cached) == len(exploded)


def test_tc002_multi_level_mrp_exact_match():
    """Verify exact match against Hand-Solved Benchmark TC-002."""
    items = [
        {"item_id": "FG-1", "type": "finished_good", "lead_time_periods": 1, "safety_stock": 0.0, "lot_sizing_rule": "L4L"},
        {"item_id": "SA-1", "type": "subassembly", "lead_time_periods": 1, "safety_stock": 10.0, "lot_sizing_rule": "L4L"},
        {"item_id": "SA-2", "type": "subassembly", "lead_time_periods": 1, "safety_stock": 0.0, "lot_sizing_rule": "L4L"},
        {"item_id": "RM-1", "type": "raw_material", "lead_time_periods": 2, "safety_stock": 20.0, "lot_sizing_rule": "L4L"},
        {"item_id": "RM-2", "type": "raw_material", "lead_time_periods": 1, "safety_stock": 10.0, "lot_sizing_rule": "L4L"},
    ]
    bom = [
        {"parent_item_id": "FG-1", "component_item_id": "SA-1", "quantity_per": 1.0, "scrap_factor": 0.0},
        {"parent_item_id": "FG-1", "component_item_id": "SA-2", "quantity_per": 2.0, "scrap_factor": 0.0},
        {"parent_item_id": "SA-1", "component_item_id": "RM-1", "quantity_per": 2.0, "scrap_factor": 0.0},
        {"parent_item_id": "SA-2", "component_item_id": "RM-1", "quantity_per": 1.0, "scrap_factor": 0.0},
        {"parent_item_id": "SA-2", "component_item_id": "RM-2", "quantity_per": 3.0, "scrap_factor": 0.0},
    ]
    inventory = {
        "FG-1": {"on_hand": 0.0, "allocated": 0.0},
        "SA-1": {"on_hand": 15.0, "allocated": 0.0},
        "SA-2": {"on_hand": 20.0, "allocated": 0.0},
        "RM-1": {"on_hand": 50.0, "allocated": 0.0},
        "RM-2": {"on_hand": 30.0, "allocated": 0.0},
    }
    mps = {
        "FG-1": [0.0, 20.0, 0.0, 40.0, 0.0, 50.0],
    }
    periods = [1, 2, 3, 4, 5, 6]

    engine = BOMExplosionEngine(bom, items)
    results, pegging = engine.run_multi_level_mrp(mps, periods, inventory_status=inventory)

    # 1. Check FG-1 Netting
    fg1 = results["FG-1"]
    assert fg1.net_requirements == [0.0, 20.0, 0.0, 40.0, 0.0, 50.0]
    fg1_releases = {po.due_period: po.release_period for po in fg1.planned_orders}
    assert fg1_releases == {2: 1, 4: 3, 6: 5}

    # 2. Check SA-1 (qty 1 from FG-1 PORel, on_hand 15, SS 10)
    sa1 = results["SA-1"]
    # Gross requirements at periods 1, 3, 5
    assert sa1.gross_requirements == [20.0, 0.0, 40.0, 0.0, 50.0, 0.0]
    assert sa1.net_requirements == [15.0, 0.0, 40.0, 0.0, 50.0, 0.0]

    # 3. Check SA-2 (qty 2 from FG-1 PORel, on_hand 20, SS 0)
    sa2 = results["SA-2"]
    assert sa2.gross_requirements == [40.0, 0.0, 80.0, 0.0, 100.0, 0.0]
    assert sa2.net_requirements == [20.0, 0.0, 80.0, 0.0, 100.0, 0.0]

    # 4. Check Common Component RM-1 (used in SA-1 qty 2, SA-2 qty 1)
    # Aggregated gross requirements at P2: (2*40) + (1*80) = 160; P4: (2*50) + (1*100) = 200
    rm1 = results["RM-1"]
    assert rm1.gross_requirements[1] == 160.0  # Period 2
    assert rm1.gross_requirements[3] == 200.0  # Period 4
    # Net requirements with on_hand=50, SS=20: P2 net is 130; P4 net is 200
    assert rm1.net_requirements[1] == 130.0
    assert rm1.net_requirements[3] == 200.0

    # 5. Check RM-2 (used in SA-2 qty 3): P2: 3*80=240, P4: 3*100=300
    rm2 = results["RM-2"]
    assert rm2.gross_requirements[1] == 240.0
    assert rm2.gross_requirements[3] == 300.0
    # Net requirements with on_hand=30, SS=10: P2 net is 220; P4 net is 300
    assert rm2.net_requirements[1] == 220.0
    assert rm2.net_requirements[3] == 300.0

    # 6. Verify Pegging Traceability
    rm1_trace = pegging.trace_up("RM-1")
    assert len(rm1_trace) > 0
    # Pegging report
    df_rep = pegging.generate_pegging_report()
    assert not df_rep.empty
    assert "RM-1" in df_rep["child_item_id"].values

