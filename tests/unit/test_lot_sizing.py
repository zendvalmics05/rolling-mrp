"""Unit tests for Lot Sizing Algorithms and Benchmarks (src/core/lot_sizing.py)."""

import json
from pathlib import Path
import pytest
from src.core.lot_sizing import (
    compute_lot_sizing,
    economic_order_quantity,
    lot_for_lot,
    periodic_order_quantity,
    silver_meal,
    wagner_whitin,
    apply_batch_constraints,
)

VALIDATION_DIR = Path(__file__).resolve().parent.parent.parent / "validation"


def test_tc003_lot_sizing_benchmarks_all_8_items():
    """Verify that Wagner-Whitin and L4L reproduce exact TC-003 hand-solved benchmark costs."""
    tc_file = VALIDATION_DIR / "TC-003" / "expected_output.json"
    with open(tc_file, "r", encoding="utf-8") as f:
        bench = json.load(f)

    for item_id, data in bench["items"].items():
        demands = data["demands"]
        setup_cost = data["setup_cost"]
        holding_cost = data["holding_cost_per_period"]

        # 1. Lot-for-Lot
        l4l_res = lot_for_lot(demands, setup_cost, holding_cost)
        assert l4l_res.total_cost == pytest.approx(data["l4l"]["total_cost"], rel=1e-3)
        assert l4l_res.orders == data["l4l"]["orders"]

        # 2. Wagner-Whitin (Optimal)
        ww_res = wagner_whitin(demands, setup_cost, holding_cost)
        assert ww_res.total_cost == pytest.approx(data["wagner_whitin"]["total_cost"], rel=1e-3)
        assert ww_res.orders == data["wagner_whitin"]["orders"]

        # Principle 3: W-W cost must always be <= L4L cost
        assert ww_res.total_cost <= l4l_res.total_cost


def test_berry_1972_silver_meal_and_eoq():
    # Classical Berry 1972 demand vector: [50, 60, 90, 70, 30, 100, 60, 40], S=100, h=2
    demands = [50, 60, 90, 70, 30, 100, 60, 40]
    S = 100.0
    h = 2.0

    sm_res = silver_meal(demands, S, h)
    # Silver-Meal should match Wagner-Whitin optimal on Berry 1972
    assert sm_res.total_cost == pytest.approx(740.0, rel=1e-3)
    assert sm_res.orders == [50, 60, 90, 100, 0, 100, 100, 0]

    eoq_res = economic_order_quantity(demands, S, h)
    assert eoq_res.total_cost > 0
    assert len(eoq_res.orders) == 8


def test_batch_constraints_min_and_multiple():
    # Raw order 35 -> min_lot 50 -> 50
    assert apply_batch_constraints(35.0, min_lot=50.0) == 50.0
    # Raw order 65 -> multiple_lot 25 -> 75
    assert apply_batch_constraints(65.0, multiple_lot=25.0) == 75.0
    # Both: raw order 15 -> min 40, mult 25 -> max(15, 40) = 40 -> ceil(40/25)*25 = 50
    assert apply_batch_constraints(15.0, min_lot=40.0, multiple_lot=25.0) == 50.0
    # Zero quantity stays 0
    assert apply_batch_constraints(0.0, min_lot=100.0) == 0.0


def test_unified_compute_lot_sizing():
    demands = [20, 0, 40, 10]
    res_l4l = compute_lot_sizing("L4L", demands, setup_cost=50.0, holding_cost=2.0)
    assert res_l4l.orders == [20, 0, 40, 10]

    res_poq = compute_lot_sizing("POQ", demands, setup_cost=50.0, holding_cost=2.0)
    assert len(res_poq.orders) == 4

    # With multiple_lot constraint applied to L4L
    res_batch = compute_lot_sizing("L4L", demands, setup_cost=50.0, holding_cost=2.0, multiple_lot=25.0)
    assert res_batch.orders[0] == 25.0  # 20 rounded up to 25

