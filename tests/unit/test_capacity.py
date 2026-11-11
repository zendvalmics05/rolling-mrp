"""Unit tests for Phase 4.1 & 4.2: Rough-Cut Capacity Planning (RCCP) and Bottleneck Detection."""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.capacity.rccp import RCCPEngine, OperationRouting, WorkCenterProfile
from src.capacity.bottleneck import BottleneckDetector


VALIDATION_DIR = Path(__file__).resolve().parent.parent.parent / "validation"


def test_tc004_rccp_unconstrained_load_exact_match():
    """Verify that unconstrained RCCP load calculations match hand-solved TC-004 benchmark exactly."""
    tc_file = VALIDATION_DIR / "TC-004" / "expected_output.json"
    with open(tc_file, "r", encoding="utf-8") as f:
        tc_data = json.load(f)

    # Setup routings from TC-004 spec:
    # FG-X: WC-A (setup: 4.0, run: 0.50), WC-B (setup: 2.0, run: 0.40)
    # FG-Y: WC-A (setup: 5.0, run: 0.60), WC-B (setup: 3.0, run: 0.50)
    routings = [
        OperationRouting("FG-X", "WC-A", 10, setup_time_hours=4.0, run_time_hours_per_unit=0.50),
        OperationRouting("FG-X", "WC-B", 20, setup_time_hours=2.0, run_time_hours_per_unit=0.40),
        OperationRouting("FG-Y", "WC-A", 10, setup_time_hours=5.0, run_time_hours_per_unit=0.60),
        OperationRouting("FG-Y", "WC-B", 20, setup_time_hours=3.0, run_time_hours_per_unit=0.50),
    ]

    # Setup work centers: WC-A: 40h, WC-B: 30h
    work_centers = [
        WorkCenterProfile("WC-A", capacity_hours_per_week=40.0),
        WorkCenterProfile("WC-B", capacity_hours_per_week=30.0),
    ]

    engine = RCCPEngine(routings, work_centers)

    # TC-004 Unconstrained Production:
    # Period 1: FG-X = 20, FG-Y = 15
    # Period 2: FG-X = 30, FG-Y = 25
    # Period 3: FG-X = 40, FG-Y = 35
    # Period 4: FG-X = 10, FG-Y = 10
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

    load_matrix = engine.calculate_plan_load(orders, periods=[1, 2, 3, 4], timing_key="period")

    # Expected values from TC-004:
    # WC-A loads: [28.0, 39.0, 50.0, 20.0], util: [70.0, 97.5, 125.0, 50.0]
    # WC-B loads: [20.5, 29.5, 38.5, 14.0], util: [68.3, 98.3, 128.3, 46.7]
    exp_wca_loads = tc_data["unconstrained_plan"]["load_hours"]["WC-A"]
    exp_wcb_loads = tc_data["unconstrained_plan"]["load_hours"]["WC-B"]
    exp_wca_util = tc_data["unconstrained_plan"]["utilization_pct"]["WC-A"]
    exp_wcb_util = tc_data["unconstrained_plan"]["utilization_pct"]["WC-B"]

    for i, p in enumerate([1, 2, 3, 4]):
        # WC-A checks
        assert pytest.approx(load_matrix["WC-A"][p].load_hours, abs=0.05) == exp_wca_loads[i]
        assert pytest.approx(load_matrix["WC-A"][p].utilization_pct, abs=0.1) == exp_wca_util[i]
        # WC-B checks
        assert pytest.approx(load_matrix["WC-B"][p].load_hours, abs=0.05) == exp_wcb_loads[i]
        assert pytest.approx(load_matrix["WC-B"][p].utilization_pct, abs=0.1) == exp_wcb_util[i]

    # Verify Period 3 overloads
    assert load_matrix["WC-A"][3].overload_hours == 10.0
    assert load_matrix["WC-B"][3].overload_hours == 8.5


def test_bottleneck_detection_and_ranking():
    """Verify bottleneck identification, severity ranking, and root cause tracing."""
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

    orders = [
        {"item_id": "FG-X", "period": 3, "quantity": 40.0},
        {"item_id": "FG-Y", "period": 3, "quantity": 35.0},
    ]

    load_matrix = engine.calculate_plan_load(orders, periods=[3], timing_key="period")
    detector = BottleneckDetector(utilization_threshold_pct=100.0)

    bottlenecks = detector.detect_bottlenecks(load_matrix)

    # Both WC-A and WC-B are overloaded in period 3
    assert len(bottlenecks) == 2
    # Verify rankings assigned
    assert bottlenecks[0].rank == 1
    assert bottlenecks[1].rank == 2

    # Top constrained work center check
    wc_ids = {b.work_center_id for b in bottlenecks}
    assert wc_ids == {"WC-A", "WC-B"}

    # Root cause tracing check for WC-B
    wcb = [b for b in bottlenecks if b.work_center_id == "WC-B"][0]
    incident_p3 = wcb.incidents[0]
    assert incident_p3.period == 3
    assert len(incident_p3.top_drivers) == 2
    # FG-Y drives 3 + 35*0.5 = 20.5 hrs; FG-X drives 2 + 40*0.4 = 18.0 hrs
    assert incident_p3.top_drivers[0]["item_id"] == "FG-Y"
    assert incident_p3.top_drivers[0]["hours"] == 20.5
    assert incident_p3.top_drivers[1]["item_id"] == "FG-X"
    assert incident_p3.top_drivers[1]["hours"] == 18.0


def test_calendar_aware_capacity_adjustment():
    """Verify capacity adjusts proportionally for holidays / short weeks."""
    wc = WorkCenterProfile("WC-100", capacity_hours_per_week=40.0, efficiency_factor=0.90)

    # Full 5-day week: 40 * 0.9 = 36.0 hours
    assert wc.get_effective_capacity(working_days=5.0) == 36.0

    # 4-day holiday week: 40 * 0.9 * (4/5) = 28.8 hours
    assert pytest.approx(wc.get_effective_capacity(working_days=4.0), abs=1e-4) == 28.8

