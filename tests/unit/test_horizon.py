"""Unit tests for Rolling-Horizon Window and Forecast Evolution (src/rolling/horizon.py)."""

import pytest
import numpy as np
from src.rolling.horizon import (
    DemandEvolutionModel,
    RollingHorizonManager,
    PeriodDemand,
)


def test_forecast_evolution_accuracy_decay():
    model = DemandEvolutionModel(base_demand=100.0, trend=0.0, seasonal_amplitude=0.0, seed=123)

    # At current period = 1, forecast for period 1 must match ground truth exactly
    actual_p1 = model.get_ground_truth(1)
    fc_p1 = model.generate_forecast(current_period=1, target_period=1)
    assert fc_p1 == actual_p1

    # Over many replications, forecast variance at horizon 8 should be larger than horizon 1
    errors_h1 = []
    errors_h8 = []
    gt_2 = model.get_ground_truth(2)
    gt_9 = model.get_ground_truth(9)

    for s in range(50):
        m = DemandEvolutionModel(base_demand=100.0, trend=0.0, seasonal_amplitude=0.0, seed=s)
        errors_h1.append(abs(m.generate_forecast(current_period=1, target_period=2) - gt_2))
        errors_h8.append(abs(m.generate_forecast(current_period=1, target_period=9) - gt_9))

    # Mean absolute error at horizon 8 should be strictly larger than horizon 1
    assert np.mean(errors_h8) > np.mean(errors_h1)


def test_rolling_window_shift_and_demand_composition():
    manager = RollingHorizonManager(
        item_id="FG-TEST",
        window_length=12,
        firm_window_length=2,
        initial_on_hand=50.0,
        seed=42,
    )

    # Initial window: 1..12
    assert manager.get_window_periods() == list(range(1, 13))

    # Check demand composition
    demands = manager.get_window_demand()
    assert len(demands) == 12
    # Period 1: actual
    assert demands[0].demand_type == "actual"
    # Period 2: firm_planned (within firm_window_length=2)
    assert demands[1].demand_type == "firm_planned"
    # Periods 3..12: forecast
    for d in demands[2:]:
        assert d.demand_type == "forecast"

    # Advance period
    step_rec = manager.advance_period()
    assert step_rec.period == 1
    assert manager.current_period == 2

    # Shifted window: 2..13
    assert manager.get_window_periods() == list(range(2, 14))

    # New demand composition: Period 2 is now actual!
    new_demands = manager.get_window_demand()
    assert new_demands[0].period == 2
    assert new_demands[0].demand_type == "actual"
    assert new_demands[1].period == 3
    assert new_demands[1].demand_type == "firm_planned"


def test_state_carry_forward_and_scheduled_receipts():
    manager = RollingHorizonManager(
        item_id="FG-TEST",
        window_length=12,
        initial_on_hand=30.0,
        initial_allocated=5.0,
        safety_stock=10.0,
        seed=42,
    )
    # Available PAB = 30 - 5 = 25
    assert manager.current_pab == 25.0

    # Schedule incoming order of 40 units due in Period 2
    manager.record_scheduled_receipt(period=2, quantity=40.0)

    # Step Period 1
    step1 = manager.advance_period()
    assert step1.period == 1
    # Check that stock was debited
    assert manager.on_hand >= manager.allocated

    # Step Period 2 (scheduled receipt of 40 units arrives)
    pre_step_on_hand = manager.on_hand
    step2 = manager.advance_period()
    assert step2.period == 2
    assert step2.order_receipts_realized == 40.0

    # Verify simulation summary output
    summary = manager.get_simulation_summary()
    assert len(summary) == 2
    assert "demand_realized" in summary.columns
    assert "units_served" in summary.columns

