"""Unit tests for Phase 3.3 & 3.4: Nervousness Mitigation Policies and Experiment E-01."""

import os
import pytest
import numpy as np
import pandas as pd

from src.rolling.nervousness import MitigationPolicyManager
from src.rolling.frozen_window import PlanChangeRecord, ChangeType


def test_change_threshold_filtering():
    """Verify that minor perturbations below percentage and dollar thresholds are suppressed."""
    manager = MitigationPolicyManager(percent_threshold=0.05, value_threshold=50.0)

    # 1. Minor change: 3 units out of 100 = 3% (< 5%), unit cost = $10 -> $30 (< $50)
    # 2. Major change: 10 units out of 100 = 10% (> 5%)
    # 3. Large dollar change: 4 units out of 100 = 4% (< 5%), but unit cost = $20 -> $80 (> $50)
    records = [
        PlanChangeRecord(
            item_id="RM-001",
            change_type=ChangeType.QUANTITY_CHANGE.value,
            zone="LIQUID",
            status="ALLOWED",
            old_period=2,
            new_period=2,
            old_quantity=100.0,
            new_quantity=103.0,
            quantity_delta=3.0,
        ),
        PlanChangeRecord(
            item_id="RM-001",
            change_type=ChangeType.QUANTITY_CHANGE.value,
            zone="LIQUID",
            status="ALLOWED",
            old_period=3,
            new_period=3,
            old_quantity=100.0,
            new_quantity=110.0,
            quantity_delta=10.0,
        ),
        PlanChangeRecord(
            item_id="RM-002",
            change_type=ChangeType.QUANTITY_CHANGE.value,
            zone="LIQUID",
            status="ALLOWED",
            old_period=4,
            new_period=4,
            old_quantity=100.0,
            new_quantity=104.0,
            quantity_delta=4.0,
        ),
    ]

    unit_costs = {"RM-001": 10.0, "RM-002": 20.0}
    filtered = manager.filter_changes_by_threshold(records, unit_costs)

    assert len(filtered) == 3
    # Record 0 should be suppressed
    assert filtered[0].status == "FILTERED_BELOW_THRESHOLD"
    assert filtered[0].quantity_delta == 0.0
    assert filtered[0].change_type == ChangeType.UNCHANGED.value

    # Record 1 should remain (10% exceeds 5%)
    assert filtered[1].status == "ALLOWED"
    assert filtered[1].quantity_delta == 10.0

    # Record 2 should remain ($80 exceeds $50)
    assert filtered[2].status == "ALLOWED"
    assert filtered[2].quantity_delta == 4.0


def test_demand_smoothing():
    """Verify exponential smoothing damps high-frequency variability."""
    manager = MitigationPolicyManager(smoothing_alpha=0.3)
    raw_demand = [100.0, 150.0, 80.0, 140.0, 90.0]

    smoothed = manager.smooth_demand_series(raw_demand)

    assert len(smoothed) == len(raw_demand)
    assert smoothed[0] == 100.0
    # Expected s_1 = 0.3 * 150 + 0.7 * 100 = 45 + 70 = 115.0
    assert pytest.approx(smoothed[1], abs=0.05) == 115.0
    # Smoothed series variance must be strictly less than raw series variance
    assert np.std(smoothed) < np.std(raw_demand)


def test_dynamic_safety_stock_calculation():
    """Verify dynamic safety stock responds to demand variance and lead time."""
    manager = MitigationPolicyManager(safety_stock_z=1.65)

    low_var_demand = [100.0, 102.0, 99.0, 101.0, 100.0]
    high_var_demand = [100.0, 150.0, 60.0, 140.0, 80.0]

    ss_low = manager.calculate_dynamic_safety_stock(low_var_demand, lead_time_periods=1)
    ss_high = manager.calculate_dynamic_safety_stock(high_var_demand, lead_time_periods=1)

    assert ss_high > ss_low
    # Increased lead time should increase safety stock
    ss_high_lt4 = manager.calculate_dynamic_safety_stock(high_var_demand, lead_time_periods=4)
    assert ss_high_lt4 > ss_high


def test_frozen_window_dynamic_extension():
    """Verify extension recommendation when nervousness breaches threshold."""
    manager = MitigationPolicyManager()

    # Nominal nervousness -> no extension
    ext_norm = manager.evaluate_frozen_window_extension(current_frozen_periods=2, nervousness_index=0.08)
    assert ext_norm == 2

    # High nervousness -> extension recommended
    ext_high = manager.evaluate_frozen_window_extension(current_frozen_periods=2, nervousness_index=0.25, threshold=0.15)
    assert ext_high > 2

