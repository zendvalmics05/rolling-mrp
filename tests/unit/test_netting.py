"""Unit tests for Gross Requirements and Netting Calculation (src/core/netting.py)."""

import pytest
import pandas as pd
from src.core.netting import (
    aggregate_gross_requirements,
    calculate_net_requirements,
    NettingResult,
)


def test_aggregate_gross_requirements():
    mps_records = [
        {"item_id": "FG-1", "period_number": 1, "quantity": 10.0, "demand_type": "actual"},
        {"item_id": "FG-1", "period_number": 2, "quantity": 25.0, "demand_type": "actual"},
        {"item_id": "FG-1", "period_number": 4, "quantity": 30.0, "demand_type": "forecast"},
        {"item_id": "FG-2", "period_number": 2, "quantity": 50.0, "demand_type": "actual"},
    ]
    periods = [1, 2, 3, 4]

    # Aggregate all demand types
    aggregated = aggregate_gross_requirements(mps_records, periods)
    assert aggregated["FG-1"] == [10.0, 25.0, 0.0, 30.0]
    assert aggregated["FG-2"] == [0.0, 50.0, 0.0, 0.0]

    # Filter only actual demand
    actual_only = aggregate_gross_requirements(mps_records, periods, demand_types=["actual"])
    assert actual_only["FG-1"] == [10.0, 25.0, 0.0, 0.0]


def test_netting_tc001_exact_match():
    """Verify exact numerical match against Hand-Solved Benchmark TC-001."""
    periods = [1, 2, 3, 4]

    # --- 1. Finished Good FG-A ---
    fg_result = calculate_net_requirements(
        item_id="FG-A",
        periods=periods,
        gross_requirements=[15.0, 25.0, 20.0, 30.0],
        on_hand=20.0,
        allocated=0.0,
        safety_stock=10.0,
        lead_time=1,
        scheduled_receipts=[0.0, 0.0, 0.0, 0.0],
    )
    assert fg_result.initial_pab == 20.0
    assert fg_result.net_requirements == [5.0, 25.0, 20.0, 30.0]
    assert fg_result.planned_order_receipts == [5.0, 25.0, 20.0, 30.0]
    assert fg_result.projected_available_balance == [10.0, 10.0, 10.0, 10.0]

    orders = fg_result.planned_orders
    assert len(orders) == 4
    # Period 1 order with LT=1 releases in Period 0 (past due expedite)
    assert orders[0].release_period == 0
    assert orders[0].past_due is True
    assert orders[1].release_period == 1
    assert orders[1].past_due is False
    assert len(fg_result.exceptions) == 1

    # --- 2. Component COMP-B (Scheduled receipt in P1 = 20) ---
    comp_b_result = calculate_net_requirements(
        item_id="COMP-B",
        periods=periods,
        gross_requirements=[50.0, 40.0, 60.0, 0.0],
        on_hand=30.0,
        allocated=0.0,
        safety_stock=15.0,
        lead_time=1,
        scheduled_receipts=[20.0, 0.0, 0.0, 0.0],
    )
    assert comp_b_result.initial_pab == 30.0
    assert comp_b_result.net_requirements == [15.0, 40.0, 60.0, 0.0]
    assert comp_b_result.planned_order_receipts == [15.0, 40.0, 60.0, 0.0]
    assert comp_b_result.projected_available_balance == [15.0, 15.0, 15.0, 15.0]
    assert len(comp_b_result.planned_orders) == 3

    # --- 3. Component COMP-C (Lead time = 2) ---
    comp_c_result = calculate_net_requirements(
        item_id="COMP-C",
        periods=periods,
        gross_requirements=[25.0, 20.0, 30.0, 0.0],
        on_hand=10.0,
        allocated=0.0,
        safety_stock=5.0,
        lead_time=2,
    )
    assert comp_c_result.initial_pab == 10.0
    assert comp_c_result.net_requirements == [20.0, 20.0, 30.0, 0.0]
    assert comp_c_result.planned_order_receipts == [20.0, 20.0, 30.0, 0.0]
    assert comp_c_result.projected_available_balance == [5.0, 5.0, 5.0, 5.0]

    # Two past due orders: P1 with LT=2 -> release -1; P2 with LT=2 -> release 0
    orders_c = comp_c_result.planned_orders
    assert orders_c[0].release_period == -1
    assert orders_c[0].past_due is True
    assert orders_c[1].release_period == 0
    assert orders_c[1].past_due is True
    assert len(comp_c_result.exceptions) == 2


def test_zero_demand_stability():
    periods = [1, 2, 3, 4]
    result = calculate_net_requirements(
        item_id="STABLE-1",
        periods=periods,
        gross_requirements=[0.0, 0.0, 0.0, 0.0],
        on_hand=50.0,
        allocated=10.0,
        safety_stock=20.0,
        lead_time=1,
    )
    assert result.initial_pab == 40.0
    assert result.net_requirements == [0.0, 0.0, 0.0, 0.0]
    assert result.projected_available_balance == [40.0, 40.0, 40.0, 40.0]
    assert len(result.planned_orders) == 0
    assert len(result.exceptions) == 0


def test_custom_lot_sizing_callback():
    def batch_50_sizer(net_req, rule, params):
        # Round up to multiple of 50
        import math
        return math.ceil(net_req / 50.0) * 50.0

    periods = [1, 2]
    result = calculate_net_requirements(
        item_id="BATCH-ITEM",
        periods=periods,
        gross_requirements=[30.0, 60.0],
        on_hand=0.0,
        safety_stock=0.0,
        lead_time=1,
        lot_sizing_rule="MULTIPLE_50",
        lot_sizing_func=batch_50_sizer,
    )
    # P1: Net Req 30 -> batch to 50 -> ending PAB = 20
    # P2: GR 60 -> Available before order = 20 - 60 = -40 -> Net Req 40 -> batch to 50 -> ending PAB = 10
    assert result.net_requirements == [30.0, 40.0]
    assert result.planned_order_receipts == [50.0, 50.0]
    assert result.projected_available_balance == [20.0, 10.0]


def test_netting_result_exports_and_validations():
    periods = [1, 2]
    res = calculate_net_requirements(
        item_id="EXP-1",
        periods=periods,
        gross_requirements=[10.0, 20.0],
        on_hand=5.0,
    )
    df = res.to_dataframe()
    assert len(df) == 2
    assert "pab" in df.columns

    d = res.to_dict()
    assert d["item_id"] == "EXP-1"

    # Mismatched length checks
    with pytest.raises(ValueError, match="gross_requirements length"):
        calculate_net_requirements("BAD", periods, [10.0], 0.0)

    with pytest.raises(ValueError, match="scheduled_receipts length"):
        calculate_net_requirements("BAD", periods, [10.0, 20.0], 0.0, scheduled_receipts=[1.0])

    # Empty mps aggregation
    assert aggregate_gross_requirements([], [1, 2]) == {}
    assert aggregate_gross_requirements(pd.DataFrame(), [1, 2]) == {}
