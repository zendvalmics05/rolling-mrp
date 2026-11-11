"""Unit tests for Phase 6.4 — Reconciliation Engine (src/validation/reconcile.py)."""

import pytest

from src.integration.controller import InventoryState
from src.integration.des_link import (
    ExecutionRecord,
    OrderStatus,
    PlannedOrderExport,
)
from src.validation.reconcile import (
    AdherenceMetrics,
    BiasAnalysis,
    FeedbackAdjustment,
    ReconciliationEngine,
    ReconciliationReport,
    VarianceRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Reconciliation engine with known unit costs."""
    return ReconciliationEngine(
        unit_costs={"FG-01": 10.0, "FG-02": 20.0, "FG-03": 5.0},
        scrap_penalty_factor=1.5,
        bias_threshold_periods=0.5,
        bias_threshold_pct=5.0,
        safety_stock_z=1.65,
    )


@pytest.fixture
def planned_orders():
    return [
        PlannedOrderExport("PO-1", "FG-01", 1, 2, 100.0, lot_sizing_rule="LFL"),
        PlannedOrderExport("PO-2", "FG-01", 3, 4, 80.0, lot_sizing_rule="LFL"),
        PlannedOrderExport("PO-3", "FG-02", 2, 3, 50.0, lot_sizing_rule="EOQ"),
        PlannedOrderExport("PO-4", "FG-03", 5, 6, 200.0, lot_sizing_rule="LFL"),
    ]


@pytest.fixture
def full_on_time_actuals():
    """All orders on time, full quantity."""
    return [
        ExecutionRecord("PO-1", "FG-01", 100.0, 100.0, 2, 2),
        ExecutionRecord("PO-2", "FG-01", 80.0, 80.0, 4, 4),
        ExecutionRecord("PO-3", "FG-02", 50.0, 50.0, 3, 3),
        ExecutionRecord("PO-4", "FG-03", 200.0, 200.0, 6, 6),
    ]


@pytest.fixture
def mixed_actuals():
    """Mix of on-time, late, partial, and one missing (cancelled)."""
    return [
        ExecutionRecord("PO-1", "FG-01", 100.0, 100.0, 2, 2),          # on-time, full
        ExecutionRecord("PO-2", "FG-01", 80.0, 60.0, 4, 5),            # late +1, partial
        ExecutionRecord("PO-3", "FG-02", 50.0, 50.0, 3, 2),            # early -1, full
        # PO-4 is missing → cancelled
    ]


@pytest.fixture
def late_biased_actuals():
    """All orders consistently 2 periods late — systematic lead-time bias."""
    return [
        ExecutionRecord("PO-1", "FG-01", 100.0, 100.0, 2, 4, delay_periods=2),
        ExecutionRecord("PO-2", "FG-01", 80.0, 80.0, 4, 6, delay_periods=2),
        ExecutionRecord("PO-3", "FG-02", 50.0, 50.0, 3, 5, delay_periods=2),
        ExecutionRecord("PO-4", "FG-03", 200.0, 200.0, 6, 8, delay_periods=2),
    ]


@pytest.fixture
def under_quantity_actuals():
    """All orders deliver 20% below planned — systematic quantity bias."""
    return [
        ExecutionRecord("PO-1", "FG-01", 100.0, 80.0, 2, 2),
        ExecutionRecord("PO-2", "FG-01", 80.0, 64.0, 4, 4),
        ExecutionRecord("PO-3", "FG-02", 50.0, 40.0, 3, 3),
        ExecutionRecord("PO-4", "FG-03", 200.0, 160.0, 6, 6),
    ]


@pytest.fixture
def inventory_states():
    return {
        "FG-01": InventoryState(on_hand=50.0, safety_stock=10.0, lead_time=1,
                                demand_history=[100, 110, 95, 105, 90, 100, 115, 80]),
        "FG-02": InventoryState(on_hand=30.0, safety_stock=15.0, lead_time=2,
                                demand_history=[50, 55, 48, 52, 60, 45, 50, 53]),
        "FG-03": InventoryState(on_hand=100.0, safety_stock=20.0, lead_time=1,
                                demand_history=[200, 195, 210, 205, 190, 200, 215, 180]),
    }


# ---------------------------------------------------------------------------
# Test: Adherence Metrics
# ---------------------------------------------------------------------------


class TestAdherence:

    def test_all_on_time_full(self, engine, planned_orders, full_on_time_actuals):
        adh = engine.compute_adherence(planned_orders, full_on_time_actuals)
        assert adh.total_orders == 4
        assert adh.on_time_count == 4
        assert adh.on_time_pct == pytest.approx(100.0)
        assert adh.late_count == 0
        assert adh.cancelled_count == 0
        assert adh.quantity_fill_rate == pytest.approx(1.0)
        assert adh.perfect_order_pct == pytest.approx(100.0)

    def test_mixed_scenario(self, engine, planned_orders, mixed_actuals):
        adh = engine.compute_adherence(planned_orders, mixed_actuals)
        assert adh.total_orders == 4
        assert adh.on_time_count == 1    # PO-1
        assert adh.late_count == 1       # PO-2
        assert adh.early_count == 1      # PO-3
        assert adh.cancelled_count == 1  # PO-4 missing
        assert adh.partial_count == 1    # PO-2 (60 < 80)

    def test_fill_rate_with_partial(self, engine, planned_orders, mixed_actuals):
        adh = engine.compute_adherence(planned_orders, mixed_actuals)
        # min(actual, planned): 100 + 60 + 50 + 0 = 210
        # planned: 100 + 80 + 50 + 200 = 430
        expected = 210.0 / 430.0
        assert adh.quantity_fill_rate == pytest.approx(expected, abs=1e-3)

    def test_perfect_order_excludes_late(self, engine, planned_orders, mixed_actuals):
        adh = engine.compute_adherence(planned_orders, mixed_actuals)
        # Only PO-1 is on-time and full-quantity
        # 1 perfect out of 3 evaluated (1 cancelled)
        assert adh.perfect_order_pct == pytest.approx(1.0 / 3.0 * 100.0, abs=0.1)

    def test_scrapped_order_counted(self, engine):
        planned = [PlannedOrderExport("PO-X", "FG-01", 1, 2, 50.0)]
        actuals = [ExecutionRecord("PO-X", "FG-01", 50.0, 0.0, 2, 2,
                                   scrap_quantity=50.0, status=OrderStatus.SCRAPPED)]
        adh = engine.compute_adherence(planned, actuals)
        assert adh.scrapped_count == 1

    def test_empty_inputs(self, engine):
        adh = engine.compute_adherence([], [])
        assert adh.total_orders == 0
        assert adh.on_time_pct == pytest.approx(0.0)
        assert adh.quantity_fill_rate == pytest.approx(1.0)

    def test_all_cancelled(self, engine, planned_orders):
        adh = engine.compute_adherence(planned_orders, [])
        assert adh.cancelled_count == 4
        assert adh.on_time_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: Variance Computation
# ---------------------------------------------------------------------------


class TestVariance:

    def test_full_on_time_zero_variance(self, engine, planned_orders, full_on_time_actuals):
        records = engine.compute_variance(planned_orders, full_on_time_actuals)
        assert len(records) == 4
        for v in records:
            assert v.quantity_variance == pytest.approx(0.0, abs=1e-3)
            assert v.timing_variance == 0

    def test_partial_late_variance(self, engine, planned_orders, mixed_actuals):
        records = engine.compute_variance(planned_orders, mixed_actuals)
        by_id = {v.order_id: v for v in records}

        # PO-2: actual 60 vs planned 80, late by 1
        v2 = by_id["PO-2"]
        assert v2.quantity_variance == pytest.approx(-20.0)
        assert v2.quantity_variance_pct == pytest.approx(-25.0)
        assert v2.timing_variance == 1

        # PO-3: on-time but early (-1)
        v3 = by_id["PO-3"]
        assert v3.timing_variance == -1

        # PO-4: cancelled
        v4 = by_id["PO-4"]
        assert v4.quantity_variance == pytest.approx(-200.0)
        assert v4.quantity_variance_pct == pytest.approx(-100.0)

    def test_cost_variance_with_unit_costs(self, engine, planned_orders, mixed_actuals):
        records = engine.compute_variance(planned_orders, mixed_actuals)
        by_id = {v.order_id: v for v in records}

        # PO-1 FG-01: 100 units × $10 = $1000 planned & actual
        v1 = by_id["PO-1"]
        assert v1.planned_cost == pytest.approx(1000.0)
        assert v1.actual_cost == pytest.approx(1000.0)
        assert v1.cost_variance == pytest.approx(0.0)

        # PO-2 FG-01: planned $800, actual 60×$10 = $600
        v2 = by_id["PO-2"]
        assert v2.planned_cost == pytest.approx(800.0)
        assert v2.actual_cost == pytest.approx(600.0)
        assert v2.cost_variance == pytest.approx(-200.0)

    def test_scrap_penalty_in_cost(self, engine):
        planned = [PlannedOrderExport("PO-X", "FG-01", 1, 2, 100.0)]
        actuals = [ExecutionRecord("PO-X", "FG-01", 100.0, 80.0, 2, 2, scrap_quantity=20.0)]
        records = engine.compute_variance(planned, actuals)
        v = records[0]
        # actual_cost = 80 × $10 + 20 × $10 × 1.5 = 800 + 300 = 1100
        assert v.actual_cost == pytest.approx(1100.0)
        # cost_variance = 1100 - 1000 = +100
        assert v.cost_variance == pytest.approx(100.0)

    def test_cancelled_order_cost(self, engine, planned_orders):
        records = engine.compute_variance(planned_orders, [])
        for v in records:
            assert v.actual_cost == pytest.approx(0.0)
            assert v.cost_variance < 0  # negative = under-spend


# ---------------------------------------------------------------------------
# Test: Bias Analysis
# ---------------------------------------------------------------------------


class TestBiasAnalysis:

    def test_no_bias_on_time_full(self, engine, planned_orders, full_on_time_actuals):
        records = engine.compute_variance(planned_orders, full_on_time_actuals)
        bias = engine.analyze_bias(records)
        assert bias.lead_time_bias == pytest.approx(0.0)
        assert bias.quantity_bias == pytest.approx(0.0)
        assert bias.is_lead_time_biased is False
        assert bias.is_quantity_biased is False
        assert bias.bias_direction_timing == "neutral"
        assert bias.bias_direction_quantity == "neutral"

    def test_late_bias_detected(self, engine, planned_orders, late_biased_actuals):
        records = engine.compute_variance(planned_orders, late_biased_actuals)
        bias = engine.analyze_bias(records)
        assert bias.lead_time_bias == pytest.approx(2.0)
        assert bias.is_lead_time_biased is True
        assert bias.bias_direction_timing == "late"
        assert bias.sample_size == 4

    def test_under_quantity_bias_detected(self, engine, planned_orders, under_quantity_actuals):
        records = engine.compute_variance(planned_orders, under_quantity_actuals)
        bias = engine.analyze_bias(records)
        assert bias.quantity_bias == pytest.approx(-20.0)
        assert bias.is_quantity_biased is True
        assert bias.bias_direction_quantity == "under"

    def test_per_item_biases(self, engine, planned_orders, late_biased_actuals):
        records = engine.compute_variance(planned_orders, late_biased_actuals)
        bias = engine.analyze_bias(records)
        assert "FG-01" in bias.item_biases
        assert "FG-02" in bias.item_biases
        # FG-01 has 2 orders both late by 2
        assert bias.item_biases["FG-01"]["lead_time_bias"] == pytest.approx(2.0)

    def test_empty_records_neutral(self, engine):
        bias = engine.analyze_bias([])
        assert bias.sample_size == 0
        assert bias.is_lead_time_biased is False
        assert bias.is_quantity_biased is False

    def test_mixed_timing_neutral(self, engine, planned_orders, mixed_actuals):
        records = engine.compute_variance(planned_orders, mixed_actuals)
        bias = engine.analyze_bias(records)
        # PO-1: 0, PO-2: +1, PO-3: -1 → mean ~0 → neutral
        assert bias.bias_direction_timing == "neutral"


# ---------------------------------------------------------------------------
# Test: Feedback Loop
# ---------------------------------------------------------------------------


class TestFeedbackLoop:

    def test_lead_time_increased_on_late_bias(
        self, engine, planned_orders, late_biased_actuals, inventory_states
    ):
        records = engine.compute_variance(planned_orders, late_biased_actuals)
        bias = engine.analyze_bias(records)
        feedback = engine.generate_feedback(bias, inventory_states)

        # All items are consistently 2 periods late → expect lead-time increase
        fb_map = {f.item_id: f for f in feedback}
        assert "FG-01" in fb_map
        assert fb_map["FG-01"].lead_time_delta > 0
        assert fb_map["FG-01"].recommended_lead_time > fb_map["FG-01"].current_lead_time

    def test_safety_stock_updated(
        self, engine, planned_orders, late_biased_actuals, inventory_states
    ):
        records = engine.compute_variance(planned_orders, late_biased_actuals)
        bias = engine.analyze_bias(records)
        feedback = engine.generate_feedback(bias, inventory_states)

        # With increased lead time, safety stock should change
        fb_map = {f.item_id: f for f in feedback}
        for iid in fb_map:
            adj = fb_map[iid]
            if adj.safety_stock_delta != 0:
                assert adj.recommended_safety_stock >= 0

    def test_no_feedback_without_inventory_states(self, engine, planned_orders, late_biased_actuals):
        records = engine.compute_variance(planned_orders, late_biased_actuals)
        bias = engine.analyze_bias(records)
        feedback = engine.generate_feedback(bias, inventory_states=None)
        assert feedback == []

    def test_no_feedback_when_no_bias(
        self, engine, planned_orders, full_on_time_actuals, inventory_states
    ):
        records = engine.compute_variance(planned_orders, full_on_time_actuals)
        bias = engine.analyze_bias(records)
        feedback = engine.generate_feedback(bias, inventory_states)
        # No systematic bias → no adjustments
        assert len(feedback) == 0

    def test_feedback_with_external_demand_history(
        self, engine, planned_orders, late_biased_actuals, inventory_states
    ):
        records = engine.compute_variance(planned_orders, late_biased_actuals)
        bias = engine.analyze_bias(records)
        demand_history = {
            "FG-01": [100, 105, 110, 95, 100, 108, 92, 103],
            "FG-02": [50, 52, 48, 55, 51, 49, 53, 50],
        }
        feedback = engine.generate_feedback(bias, inventory_states, demand_history)
        assert len(feedback) > 0


# ---------------------------------------------------------------------------
# Test: Apply Feedback
# ---------------------------------------------------------------------------


class TestApplyFeedback:

    def test_apply_updates_inventory(self, engine, inventory_states):
        adjustments = [
            FeedbackAdjustment(
                item_id="FG-01",
                current_lead_time=1,
                recommended_lead_time=3,
                lead_time_delta=2,
                current_safety_stock=10.0,
                recommended_safety_stock=25.0,
                safety_stock_delta=15.0,
                rationale="Test adjustment.",
            )
        ]
        applied = engine.apply_feedback(adjustments, inventory_states)
        assert "FG-01" in applied
        assert inventory_states["FG-01"].lead_time == 3
        assert inventory_states["FG-01"].safety_stock == pytest.approx(25.0)

    def test_apply_skips_unknown_items(self, engine, inventory_states):
        adjustments = [
            FeedbackAdjustment("UNKNOWN", 1, 2, 1, 0.0, 5.0, 5.0, "")
        ]
        applied = engine.apply_feedback(adjustments, inventory_states)
        assert "UNKNOWN" not in applied

    def test_apply_lead_time_only(self, engine, inventory_states):
        adjustments = [
            FeedbackAdjustment("FG-02", 2, 4, 2, 15.0, 15.0, 0.0, "LT only.")
        ]
        applied = engine.apply_feedback(adjustments, inventory_states)
        assert applied["FG-02"]["lead_time"]["new"] == 4
        assert "safety_stock" not in applied["FG-02"]  # no SS change


# ---------------------------------------------------------------------------
# Test: Full Reconciliation Report
# ---------------------------------------------------------------------------


class TestReconciliationReport:

    def test_full_reconciliation_structure(
        self, engine, planned_orders, mixed_actuals, inventory_states
    ):
        report = engine.reconcile(planned_orders, mixed_actuals, inventory_states)
        assert isinstance(report, ReconciliationReport)
        assert isinstance(report.adherence, AdherenceMetrics)
        assert isinstance(report.bias_analysis, BiasAnalysis)
        assert len(report.variance_records) == 4
        assert "total_planned_cost" in report.cost_summary
        assert "total_actual_cost" in report.cost_summary
        assert "total_cost_variance" in report.cost_summary

    def test_reconciliation_to_dict(
        self, engine, planned_orders, full_on_time_actuals
    ):
        report = engine.reconcile(planned_orders, full_on_time_actuals)
        d = report.to_dict()
        assert "timestamp" in d
        assert "adherence" in d
        assert "bias_analysis" in d
        assert "cost_summary" in d
        assert isinstance(d["variance_records"], list)

    def test_cost_summary_consistency(
        self, engine, planned_orders, mixed_actuals
    ):
        report = engine.reconcile(planned_orders, mixed_actuals)
        cs = report.cost_summary
        assert cs["total_cost_variance"] == pytest.approx(
            cs["total_actual_cost"] - cs["total_planned_cost"], abs=0.01
        )

    def test_reconciliation_with_demand_history(
        self, engine, planned_orders, late_biased_actuals, inventory_states
    ):
        demand_history = {
            "FG-01": [100, 105, 95, 110, 90, 100, 108, 92],
        }
        report = engine.reconcile(
            planned_orders, late_biased_actuals,
            inventory_states, demand_history
        )
        assert report.bias_analysis.is_lead_time_biased is True
        assert len(report.feedback_adjustments) > 0

