"""Unit and integration tests for Phase 4.3 & 4.4: Leveling Heuristics & MRP Integration."""

import json
from pathlib import Path
import pytest

from src.capacity.rccp import RCCPEngine, OperationRouting, WorkCenterProfile
from src.capacity.bottleneck import BottleneckDetector
from src.capacity.leveling import CapacityLevelingEngine, LevelingStrategy
from src.capacity.integration import CapacityAwareMRPPipeline
from src.core.explosion import BOMExplosionEngine
from src.integration.data_generator import generate_synthetic_data
from src.integration.load import clean_and_validate_dataset, initialize_inventory_status


VALIDATION_DIR = Path(__file__).resolve().parent.parent.parent / "validation"


def test_tc004_capacity_leveling_exact_match():
    """Verify that CapacityLevelingEngine pre-build leveling exactly reproduces TC-004 leveled benchmark."""
    tc_file = VALIDATION_DIR / "TC-004" / "expected_output.json"
    with open(tc_file, "r", encoding="utf-8") as f:
        tc_data = json.load(f)

    routings = [
        OperationRouting("FG-X", "WC-A", 10, setup_time_hours=4.0, run_time_hours_per_unit=0.50),
        OperationRouting("FG-X", "WC-B", 20, setup_time_hours=2.0, run_time_hours_per_unit=0.40),
        OperationRouting("FG-Y", "WC-A", 10, setup_time_hours=5.0, run_time_hours_per_unit=0.60),
        OperationRouting("FG-Y", "WC-B", 20, setup_time_hours=3.0, run_time_hours_per_unit=0.50),
    ]

    work_centers = [
        WorkCenterProfile("WC-A", capacity_hours_per_week=40.0),
        WorkCenterProfile("WC-B", capacity_hours_per_week=30.0),
    ]

    engine = RCCPEngine(routings, work_centers)
    leveler = CapacityLevelingEngine(
        engine,
        holding_cost_rate=1.0,
        item_holding_costs={"FG-X": 0.5, "FG-Y": 1.0},  # FG-X lower unit value
    )

    # TC-004 Unconstrained Production
    orders = [
        {"item_id": "FG-X", "period": 1, "quantity": 20.0},
        {"item_id": "FG-Y", "period": 1, "quantity": 15.0},
        {"item_id": "FG-X", "period": 2, "quantity": 30.0},
        {"item_id": "FG-Y", "period": 2, "quantity": 25.0},
        {"item_id": "FG-X", "period": 3, "quantity": 40.0},
        {"item_id": "FG-Y", "period": 3, "quantity": 35.0},
        {"item_id": "FG-X", "period": 4, "quantity": 10.0},
        {"item_id": "FG-Y", "period": 4, "quantity": 10.0},
    ]

    res = leveler.level_capacity_plan(
        planned_orders=orders,
        periods=[1, 2, 3, 4],
        timing_key="period",
        allow_overtime=False,
    )

    # 1. Leveling must be completely feasible
    assert res.is_feasible is True
    assert res.remaining_overload_hours <= 1e-2

    # 2. Check leveled production quantities match TC-004 expected output:
    # Leveled FG-X: [40, 30, 20, 10]
    # Leveled FG-Y: [15, 26, 34, 10]
    exp_leveled_x = tc_data["leveled_plan"]["production"]["FG-X"]
    exp_leveled_y = tc_data["leveled_plan"]["production"]["FG-Y"]

    act_x = [0.0] * 4
    act_y = [0.0] * 4
    for o in res.leveled_orders:
        p_idx = int(o["period"]) - 1
        if o["item_id"] == "FG-X":
            act_x[p_idx] += float(o["quantity"])
        elif o["item_id"] == "FG-Y":
            act_y[p_idx] += float(o["quantity"])

    for p in range(4):
        assert pytest.approx(act_x[p], abs=0.5) == exp_leveled_x[p]
        assert pytest.approx(act_y[p], abs=0.5) == exp_leveled_y[p]

    # 3. Check leveled loads do not exceed capacities (WC-A <= 40, WC-B <= 30)
    final_loads = engine.calculate_plan_load(res.leveled_orders, periods=[1, 2, 3, 4], timing_key="period")
    for p in [1, 2, 3, 4]:
        assert final_loads["WC-A"][p].load_hours <= 40.0 + 1e-2
        assert final_loads["WC-B"][p].load_hours <= 30.0 + 1e-2


def test_capacity_aware_mrp_pipeline_integration():
    """Verify closed-loop pipeline from MRP explosion to RCCP check and leveling."""
    raw_data = generate_synthetic_data(seed=42)
    cleaned_data, _ = clean_and_validate_dataset(raw_data)
    inv_df = initialize_inventory_status(cleaned_data["inventory"], cleaned_data["items"])

    explosion_engine = BOMExplosionEngine(cleaned_data["bom"], cleaned_data["items"])
    rccp_engine = RCCPEngine(cleaned_data["routings"], cleaned_data["work_centers"])
    pipeline = CapacityAwareMRPPipeline(explosion_engine, rccp_engine)

    fg_items = cleaned_data["items"][cleaned_data["items"]["type"] == "finished_good"]["item_id"].tolist()
    periods = list(range(1, 13))

    # Demand with peak in period 6
    mps_gross = {}
    for fg in fg_items:
        mps_gross[fg] = [50.0] * 12
        mps_gross[fg][5] = 200.0  # Big demand shock in week 6

    inv_dict = {}
    for _, row in inv_df.iterrows():
        inv_dict[str(row["item_id"])] = {
            "on_hand": float(row["quantity_on_hand"]),
            "allocated": float(row["quantity_allocated"]),
            "safety_stock": float(row["safety_stock"]),
            "lead_time_days": int(row["lead_time_days"]),
        }

    # Run closed loop with auto-leveling
    result = pipeline.run_capacity_aware_mrp(
        mps_gross_requirements=mps_gross,
        periods=periods,
        inventory_status=inv_dict,
        auto_level=True,
    )

    # Verify pipeline components execute cleanly
    assert len(result.initial_planned_orders) > 0
    assert len(result.final_planned_orders) > 0
    assert len(result.initial_load_matrix) > 0
    assert len(result.final_load_matrix) > 0
    # Overload exceptions generated
    if len(result.initial_bottlenecks) > 0:
        assert len(result.capacity_exceptions) > 0
        assert any(e.severity in ["CRITICAL", "HIGH"] for e in result.capacity_exceptions)
