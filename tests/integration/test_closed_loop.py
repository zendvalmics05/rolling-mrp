"""Integration tests for Phase 6.3 — Closed-Loop Planning Controller.

Tests run a 10-period closed-loop simulation with a minimal 3-item factory:
  FG-01 (finished good, LLC 0)
    └── COMP-A (component, LLC 1)

Assertions:
  - Inventory balances hold (no negative stock)
  - No future-information leakage (forecast start > current period)
  - Nervousness tracker has one record per cycle
  - DES export files are created for every cycle
  - Simulation summary aggregation is consistent
"""

import json
import os
import tempfile
from typing import Dict, List

import pytest

from src.core.explosion import BOMExplosionEngine
from src.integration.controller import (
    ClosedLoopPlanningController,
    InventoryState,
    SimulationSummary,
)
from src.integration.des_link import ExecutionRecord, OrderStatus
from src.integration.forecast_api import ForecastApiClient, PeriodForecastItem


# ---------------------------------------------------------------------------
# Shared factory configuration (3-item, 1-BOM-level minimal factory)
# ---------------------------------------------------------------------------

ITEM_IDS = ["FG-01", "FG-02"]

BOM_DATA = [
    # parent_item_id, component_item_id, quantity_per (required BOMExplosionEngine columns)
    {"parent_item_id": "FG-01", "component_item_id": "COMP-A", "quantity_per": 2.0},
    {"parent_item_id": "FG-02", "component_item_id": "COMP-B", "quantity_per": 1.0},
]

ITEMS_DATA = [
    {"item_id": "FG-01",  "lead_time": 1, "safety_stock": 5.0,  "lot_sizing_rule": "LFL",
     "quantity_on_hand": 20.0, "quantity_allocated": 0.0},
    {"item_id": "FG-02",  "lead_time": 2, "safety_stock": 10.0, "lot_sizing_rule": "LFL",
     "quantity_on_hand": 30.0, "quantity_allocated": 0.0},
    {"item_id": "COMP-A", "lead_time": 1, "safety_stock": 0.0,  "lot_sizing_rule": "LFL",
     "quantity_on_hand": 50.0, "quantity_allocated": 0.0},
    {"item_id": "COMP-B", "lead_time": 1, "safety_stock": 0.0,  "lot_sizing_rule": "LFL",
     "quantity_on_hand": 40.0, "quantity_allocated": 0.0},
]


def _make_mock_forecast_provider(item_ids, qty_per_item=10.0):
    """Return a mock provider callable that generates flat forecasts."""
    def provider(items, start_period, horizon_length):
        result = []
        for iid in items:
            for h in range(horizon_length):
                result.append(PeriodForecastItem(
                    item_id=iid,
                    period=start_period + h,
                    quantity=qty_per_item,
                    confidence_interval_lower=qty_per_item * 0.8,
                    confidence_interval_upper=qty_per_item * 1.2,
                ))
        return result
    return provider


def _make_controller(tmp_dir: str, item_ids=None) -> ClosedLoopPlanningController:
    """Construct a minimal ClosedLoopPlanningController for testing."""
    if item_ids is None:
        item_ids = ITEM_IDS

    explosion_engine = BOMExplosionEngine(bom_data=BOM_DATA, items_data=ITEMS_DATA)

    forecast_client = ForecastApiClient(
        cache_dir=os.path.join(tmp_dir, "forecast_cache"),
        mock_provider=_make_mock_forecast_provider(item_ids, qty_per_item=10.0),
    )

    inventory_states = {
        "FG-01": InventoryState(on_hand=20.0, allocated=0.0, safety_stock=5.0, lead_time=1),
        "FG-02": InventoryState(on_hand=30.0, allocated=0.0, safety_stock=10.0, lead_time=2),
    }

    ctrl = ClosedLoopPlanningController(
        item_ids=item_ids,
        inventory_states=inventory_states,
        explosion_engine=explosion_engine,
        rccp_engine=None,  # No capacity pipeline in unit integration test
        forecast_client=forecast_client,
        planning_horizon=12,
        frozen_periods=2,
        slushy_periods=3,
        des_export_dir=os.path.join(tmp_dir, "des_exports"),
        target_fill_rate=0.90,
    )
    return ctrl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClosedLoopSingleCycle:
    """Verify one planning cycle produces all required outputs."""

    def test_cycle_returns_audit_record(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        audit = ctrl.run_planning_cycle(t=1)
        assert audit.cycle == 1
        assert audit.planning_period == 1
        assert audit.timestamp is not None
        assert audit.forecast_status in {"SUCCESS", "FALLBACK_CACHED", "FALLBACK_GENERATED"}

    def test_cycle_creates_des_export_file(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        audit = ctrl.run_planning_cycle(t=1)
        assert audit.des_export_path is not None
        assert os.path.exists(audit.des_export_path)

    def test_des_export_valid_json(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        audit = ctrl.run_planning_cycle(t=1)
        with open(audit.des_export_path, "r") as fh:
            data = json.load(fh)
        assert "orders" in data
        assert data["format"] == "ROLLINGMRP_DES_EXPORT_v1"

    def test_nervousness_recorded_after_cycle(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        ctrl.run_planning_cycle(t=1)
        assert len(ctrl.nervousness_history) == 1

    def test_audit_trail_grows_with_cycles(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        for t in range(1, 4):
            ctrl.run_planning_cycle(t=t)
        assert len(ctrl.audit_trail) == 3

    def test_no_negative_nervousness_index(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        audit = ctrl.run_planning_cycle(t=1)
        assert audit.nervousness_index >= 0.0

    def test_fill_rate_none_when_no_actuals(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        audit = ctrl.run_planning_cycle(t=1, execution_results=None)
        assert audit.fill_rate is None

    def test_fill_rate_computed_from_actuals(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        # Provide partial execution feedback
        actuals = [
            ExecutionRecord("PO-X", "FG-01", planned_quantity=10.0, actual_quantity=9.0,
                            planned_period=2, actual_period=2, status=OrderStatus.PARTIAL),
        ]
        audit = ctrl.run_planning_cycle(t=1, execution_results=actuals)
        assert audit.fill_rate == pytest.approx(0.9, abs=1e-3)


class TestClosedLoopForecastLeakage:
    """Verify no future information is used in planning."""

    def test_forecast_start_period_strictly_greater_than_t(self, tmp_path):
        """The forecast must start at t+1 or later — never at or before t."""
        captured_start: List[int] = []

        def spy_provider(items, start_period, horizon_length):
            captured_start.append(start_period)
            return _make_mock_forecast_provider(items)(items, start_period, horizon_length)

        explosion_engine = BOMExplosionEngine(bom_data=BOM_DATA, items_data=ITEMS_DATA)
        forecast_client = ForecastApiClient(
            cache_dir=str(tmp_path / "cache"),
            mock_provider=spy_provider,
        )
        inventory_states = {
            "FG-01": InventoryState(on_hand=20.0),
            "FG-02": InventoryState(on_hand=30.0),
        }
        ctrl = ClosedLoopPlanningController(
            item_ids=ITEM_IDS,
            inventory_states=inventory_states,
            explosion_engine=explosion_engine,
            forecast_client=forecast_client,
            des_export_dir=str(tmp_path / "des"),
        )
        for t in [1, 3, 7]:
            ctrl.run_planning_cycle(t=t)

        for t, start in zip([1, 3, 7], captured_start):
            assert start > t, f"Forecast leaked: start_period={start} ≤ t={t}"


class TestClosedLoopSimulation:
    """Full 10-period rolling simulation tests."""

    @pytest.fixture
    def simulation_result(self, tmp_path) -> SimulationSummary:
        ctrl = _make_controller(str(tmp_path))
        return ctrl.run_simulation(num_periods=10, start_period=1)

    def test_simulation_runs_all_periods(self, simulation_result):
        assert simulation_result.num_periods == 10

    def test_simulation_nervousness_trend_length(self, simulation_result):
        assert len(simulation_result.nervousness_trend) == 10

    def test_simulation_nervousness_all_non_negative(self, simulation_result):
        for ni in simulation_result.nervousness_trend:
            assert ni >= 0.0

    def test_simulation_total_released_orders_non_negative(self, simulation_result):
        assert simulation_result.total_released_orders >= 0

    def test_simulation_avg_nervousness_matches_trend(self, simulation_result):
        import numpy as np
        expected = round(float(np.mean(simulation_result.nervousness_trend)), 4)
        assert simulation_result.avg_nervousness_index == pytest.approx(expected, abs=1e-3)

    def test_simulation_cycle_audit_records_complete(self, simulation_result):
        assert len(simulation_result.cycle_audit_records) == 10
        for rec in simulation_result.cycle_audit_records:
            assert "cycle" in rec
            assert "planning_period" in rec
            assert "nervousness_index" in rec
            assert "des_export_path" in rec

    def test_simulation_des_exports_all_created(self, simulation_result, tmp_path):
        for rec in simulation_result.cycle_audit_records:
            path = rec["des_export_path"]
            assert os.path.exists(path), f"DES export missing: {path}"

    def test_simulation_nervousness_tracker_has_10_records(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        ctrl.run_simulation(num_periods=10, start_period=1)
        assert len(ctrl.nervousness_history) == 10

    def test_simulation_with_execution_feedback(self, tmp_path):
        """Provide DES actuals for periods 2 and 4; fill_rate is computed."""
        def make_actuals(period):
            return [
                ExecutionRecord(
                    f"PO-FG01-{period}", "FG-01",
                    planned_quantity=10.0, actual_quantity=10.0,
                    planned_period=period, actual_period=period,
                    status=OrderStatus.COMPLETED,
                )
            ]

        ctrl = _make_controller(str(tmp_path))
        summary = ctrl.run_simulation(
            num_periods=5,
            execution_results_by_period={2: make_actuals(2), 4: make_actuals(4)},
            start_period=1,
        )
        fill_rates = [
            r["fill_rate"] for r in summary.cycle_audit_records if r["fill_rate"] is not None
        ]
        assert len(fill_rates) == 2
        assert all(fr == pytest.approx(1.0) for fr in fill_rates)


class TestInventoryState:
    """Unit tests for InventoryState helper."""

    def test_available_balance(self):
        inv = InventoryState(on_hand=100.0, allocated=20.0)
        assert inv.available_balance() == pytest.approx(80.0)

    def test_apply_issue_floors_at_zero(self):
        inv = InventoryState(on_hand=10.0)
        inv.apply_issue(50.0)  # more than on hand
        assert inv.on_hand == pytest.approx(0.0)

    def test_apply_receipt_accumulates(self):
        inv = InventoryState(on_hand=0.0)
        inv.apply_receipt(5, 30.0)
        inv.apply_receipt(5, 20.0)
        assert inv.scheduled_receipts[5] == pytest.approx(50.0)

    def test_apply_adjustment_negative_floors_at_zero(self):
        inv = InventoryState(on_hand=10.0)
        inv.apply_adjustment(-50.0)
        assert inv.on_hand == pytest.approx(0.0)

    def test_apply_adjustment_positive(self):
        inv = InventoryState(on_hand=10.0)
        inv.apply_adjustment(15.0)
        assert inv.on_hand == pytest.approx(25.0)

    def test_to_mrp_dict_has_required_keys(self):
        inv = InventoryState(on_hand=100.0, allocated=5.0, safety_stock=10.0, lead_time=2)
        d = inv.to_mrp_dict()
        assert "on_hand" in d
        assert "allocated" in d
        assert "safety_stock" in d
        assert "lead_time" in d
        assert "lot_sizing_rule" in d


class TestInventoryFeedbackLoop:
    """Verify ingest_execution_feedback updates on-hand correctly."""

    def test_completed_order_credits_inventory(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        initial_oh = ctrl.inventory_states["FG-01"].on_hand

        recs = [ExecutionRecord(
            "PO-X", "FG-01", planned_quantity=20.0, actual_quantity=20.0,
            planned_period=3, actual_period=3, status=OrderStatus.COMPLETED,
        )]
        ctrl.ingest_execution_feedback(recs, update_inventory=True)
        assert ctrl.inventory_states["FG-01"].on_hand == pytest.approx(initial_oh + 20.0)

    def test_no_inventory_update_when_flag_false(self, tmp_path):
        ctrl = _make_controller(str(tmp_path))
        initial_oh = ctrl.inventory_states["FG-01"].on_hand

        recs = [ExecutionRecord(
            "PO-X", "FG-01", planned_quantity=20.0, actual_quantity=20.0,
            planned_period=3, actual_period=3, status=OrderStatus.COMPLETED,
        )]
        ctrl.ingest_execution_feedback(recs, update_inventory=False)
        assert ctrl.inventory_states["FG-01"].on_hand == pytest.approx(initial_oh)
