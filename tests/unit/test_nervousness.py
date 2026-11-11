"""Unit tests for Phase 3.1 & 3.2: Nervousness Quantification & Trend Analysis."""

import pytest
import numpy as np

from src.rolling.nervousness import (
    NervousnessAnalyzer,
    NervousnessMetrics,
    TrendDirection,
    TrendAnalysisReport,
)
from src.rolling.frozen_window import PlanChangeRecord, ChangeType


@pytest.fixture
def sample_plans():
    """Create paired previous and new plans with deliberate known changes."""
    prev_plan = [
        {"item_id": "FG-001", "due_period": 1, "quantity": 100.0},
        {"item_id": "FG-001", "due_period": 2, "quantity": 100.0},
        {"item_id": "FG-001", "due_period": 3, "quantity": 100.0},
        {"item_id": "FG-002", "due_period": 1, "quantity": 50.0},
        {"item_id": "FG-002", "due_period": 4, "quantity": 50.0},
    ]

    # New plan changes:
    # 1. FG-001 period 1: unchanged (100)
    # 2. FG-001 period 2: quantity changed from 100 to 120 (delta: +20)
    # 3. FG-001 period 3: shifted to period 5 (shift = 2 > 1) -> timing churn
    # 4. FG-002 period 1: unchanged (50)
    # 5. FG-002 period 4: cancelled (missing in new plan)
    # 6. New order FG-002 period 6: quantity 60
    new_plan = [
        {"item_id": "FG-001", "due_period": 1, "quantity": 100.0},
        {"item_id": "FG-001", "due_period": 2, "quantity": 120.0},
        {"item_id": "FG-001", "due_period": 5, "quantity": 100.0},
        {"item_id": "FG-002", "due_period": 1, "quantity": 50.0},
        {"item_id": "FG-002", "due_period": 6, "quantity": 60.0},
    ]

    return prev_plan, new_plan


def test_identical_plans_zero_nervousness():
    """Identical plans should produce 0 churn, 0 timing churn, 0 cancellation, 0 nervousness."""
    analyzer = NervousnessAnalyzer()
    plan = [
        {"item_id": "FG-001", "due_period": 1, "quantity": 100.0},
        {"item_id": "FG-002", "due_period": 2, "quantity": 50.0},
    ]

    metrics = analyzer.calculate_metrics(previous_plan=plan, new_plan=plan, cycle=1)

    assert metrics.order_churn == 0.0
    assert metrics.timing_churn == 0.0
    assert metrics.cancellation_rate == 0.0
    assert metrics.nervousness_index == 0.0
    assert 0.0 <= metrics.nervousness_index <= 1.0


def test_metrics_calculation_and_bounds(sample_plans):
    """Verify that all four metrics calculate correctly and are strictly bounded [0, 1]."""
    prev_plan, new_plan = sample_plans
    analyzer = NervousnessAnalyzer()

    metrics = analyzer.calculate_metrics(
        previous_plan=prev_plan,
        new_plan=new_plan,
        cycle=2,
    )

    # Check bounds
    assert 0.0 <= metrics.order_churn <= 1.0
    assert 0.0 <= metrics.timing_churn <= 1.0
    assert 0.0 <= metrics.cancellation_rate <= 1.0
    assert 0.0 <= metrics.nervousness_index <= 1.0

    # Verify order churn > 0 due to quantity variations
    assert metrics.order_churn > 0.0
    # Verify cancellation detected
    assert metrics.cancelled_orders >= 1
    assert metrics.cancellation_rate > 0.0
    # Verify timing shift > 1 detected
    assert metrics.shifted_orders >= 1
    assert metrics.timing_churn > 0.0

    # Composite formula check: 0.5 * churn + 0.3 * timing + 0.2 * cancellation
    expected_index = (
        0.5 * metrics.order_churn
        + 0.3 * metrics.timing_churn
        + 0.2 * metrics.cancellation_rate
    )
    assert pytest.approx(metrics.nervousness_index, abs=1e-3) == expected_index


def test_calculate_from_change_records():
    """Verify calculation directly from PlanChangeRecord instances."""
    analyzer = NervousnessAnalyzer()

    prev_plan = [
        {"item_id": "FG-001", "due_period": 1, "quantity": 100.0},
        {"item_id": "FG-001", "due_period": 2, "quantity": 100.0},
        {"item_id": "FG-001", "due_period": 3, "quantity": 100.0},
    ]
    new_plan = [
        {"item_id": "FG-001", "due_period": 1, "quantity": 100.0},
        {"item_id": "FG-001", "due_period": 2, "quantity": 110.0},
        {"item_id": "FG-001", "due_period": 5, "quantity": 100.0},
    ]

    records = [
        PlanChangeRecord(
            item_id="FG-001",
            change_type=ChangeType.QUANTITY_CHANGE.value,
            zone="LIQUID",
            status="ALLOWED",
            old_period=2,
            new_period=2,
            old_quantity=100.0,
            new_quantity=110.0,
            quantity_delta=10.0,
        ),
        PlanChangeRecord(
            item_id="FG-001",
            change_type=ChangeType.TIMING_CHANGE.value,
            zone="LIQUID",
            status="ALLOWED",
            old_period=3,
            new_period=5,
            old_quantity=100.0,
            new_quantity=100.0,
            quantity_delta=0.0,
        ),
    ]

    metrics = analyzer.calculate_from_changes(records, prev_plan, new_plan, cycle=3)
    assert 0.0 <= metrics.nervousness_index <= 1.0
    assert metrics.shifted_orders == 1  # 5 - 3 = 2 > 1
    assert metrics.quantity_changes == 1


def test_trend_analysis_improving_declining_stable():
    """Verify trend direction classification: improving, stable, declining."""
    analyzer = NervousnessAnalyzer(alert_threshold=0.15)

    # 1. Declining trend (nervousness rising over cycles)
    analyzer.history = [
        NervousnessMetrics(cycle=1, order_churn=0.05, timing_churn=0.0, cancellation_rate=0.0, nervousness_index=0.03),
        NervousnessMetrics(cycle=2, order_churn=0.10, timing_churn=0.05, cancellation_rate=0.0, nervousness_index=0.07),
        NervousnessMetrics(cycle=3, order_churn=0.18, timing_churn=0.10, cancellation_rate=0.05, nervousness_index=0.13),
        NervousnessMetrics(cycle=4, order_churn=0.25, timing_churn=0.15, cancellation_rate=0.10, nervousness_index=0.19),
    ]

    report = analyzer.analyze_trends(lookback_runs=4)
    assert report.trend_direction == TrendDirection.DECLINING.value
    assert report.is_alert is True  # 0.19 > 0.15
    assert report.slope > 0

    # 2. Improving trend (nervousness falling over cycles)
    analyzer.history = [
        NervousnessMetrics(cycle=1, order_churn=0.30, timing_churn=0.20, cancellation_rate=0.15, nervousness_index=0.24),
        NervousnessMetrics(cycle=2, order_churn=0.20, timing_churn=0.15, cancellation_rate=0.10, nervousness_index=0.16),
        NervousnessMetrics(cycle=3, order_churn=0.12, timing_churn=0.08, cancellation_rate=0.05, nervousness_index=0.09),
        NervousnessMetrics(cycle=4, order_churn=0.05, timing_churn=0.02, cancellation_rate=0.02, nervousness_index=0.04),
    ]

    report = analyzer.analyze_trends(lookback_runs=4)
    assert report.trend_direction == TrendDirection.IMPROVING.value
    assert report.is_alert is False  # 0.04 < 0.15
    assert report.slope < 0

    # 3. Stable trend (minimal variation)
    analyzer.history = [
        NervousnessMetrics(cycle=1, order_churn=0.05, timing_churn=0.05, cancellation_rate=0.02, nervousness_index=0.05),
        NervousnessMetrics(cycle=2, order_churn=0.05, timing_churn=0.05, cancellation_rate=0.02, nervousness_index=0.051),
        NervousnessMetrics(cycle=3, order_churn=0.05, timing_churn=0.05, cancellation_rate=0.02, nervousness_index=0.049),
        NervousnessMetrics(cycle=4, order_churn=0.05, timing_churn=0.05, cancellation_rate=0.02, nervousness_index=0.05),
    ]

    report = analyzer.analyze_trends(lookback_runs=4)
    assert report.trend_direction == TrendDirection.STABLE.value


def test_visualization_data_and_persistence(tmp_path):
    """Verify visualization data formatting and CSV save/load roundtrip."""
    analyzer = NervousnessAnalyzer(alert_threshold=0.15)
    for c in range(1, 6):
        analyzer.history.append(
            NervousnessMetrics(
                cycle=c,
                order_churn=0.05 * c,
                timing_churn=0.02 * c,
                cancellation_rate=0.01 * c,
                nervousness_index=round(0.05 * c, 4),
            )
        )

    vis_data = analyzer.prepare_visualization_data()
    assert len(vis_data["cycles"]) == 5
    assert len(vis_data["nervousness_index"]) == 5
    assert vis_data["summary"]["runs_exceeding_threshold"] == 2  # cycles 4 (0.20) and 5 (0.25)

    # Save to CSV and verify reload
    csv_file = str(tmp_path / "nervousness_history.csv")
    analyzer.save_history(csv_file)

    new_analyzer = NervousnessAnalyzer()
    new_analyzer.load_history(csv_file)
    assert len(new_analyzer.history) == 5
    assert new_analyzer.history[-1].cycle == 5
    assert new_analyzer.history[-1].nervousness_index == 0.25

