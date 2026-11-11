"""Unit tests for Frozen Window, Time Fences, and Plan Comparison (src/rolling/frozen_window.py)."""

import pytest
from src.rolling.frozen_window import (
    ChangeType,
    PlanChangeRecord,
    PlanComparator,
    PlanningZone,
    TimeFencePolicy,
)


def test_time_fence_zone_classification():
    policy = TimeFencePolicy(frozen_periods=2, slushy_periods=3)
    curr_period = 1

    # Period 1 and 2: FROZEN
    assert policy.get_zone(curr_period, 1) == PlanningZone.FROZEN
    assert policy.get_zone(curr_period, 2) == PlanningZone.FROZEN

    # Period 3, 4, 5: SLUSHY
    assert policy.get_zone(curr_period, 3) == PlanningZone.SLUSHY
    assert policy.get_zone(curr_period, 4) == PlanningZone.SLUSHY
    assert policy.get_zone(curr_period, 5) == PlanningZone.SLUSHY

    # Period 6+: LIQUID
    assert policy.get_zone(curr_period, 6) == PlanningZone.LIQUID
    assert policy.get_zone(curr_period, 10) == PlanningZone.LIQUID

    # Past due / negative relative offset: FROZEN
    assert policy.get_zone(curr_period, 0) == PlanningZone.FROZEN


def test_plan_comparator_change_detection():
    comparator = PlanComparator(TimeFencePolicy(frozen_periods=2, slushy_periods=2))
    curr_period = 1

    previous_plan = [
        {"item_id": "FG-1", "due_period": 2, "quantity": 50.0},
        {"item_id": "FG-1", "due_period": 4, "quantity": 30.0},
        {"item_id": "SA-1", "due_period": 3, "quantity": 25.0},
    ]

    new_plan = [
        # Quantity change in period 2 (FROZEN zone)
        {"item_id": "FG-1", "due_period": 2, "quantity": 70.0},
        # SA-1 in period 3 cancelled (missing in new plan)
        # Timing shift for FG-1 from period 4 to period 5
        {"item_id": "FG-1", "due_period": 5, "quantity": 30.0},
        # New order introduced in period 6 (LIQUID zone)
        {"item_id": "SA-2", "due_period": 6, "quantity": 40.0},
    ]

    changes = comparator.compare_plans(previous_plan, new_plan, current_period=curr_period)
    change_types = {c.change_type for c in changes}

    assert ChangeType.QUANTITY_CHANGE.value in change_types
    assert ChangeType.TIMING_CHANGE.value in change_types
    assert ChangeType.CANCELLED.value in change_types
    assert ChangeType.NEW_ORDER.value in change_types

    # Verify frozen zone flags action as BLOCKED
    frozen_change = [c for c in changes if c.new_period == 2][0]
    assert frozen_change.zone == PlanningZone.FROZEN.value
    assert frozen_change.status == "BLOCKED"


def test_frozen_window_strict_enforcement():
    comparator = PlanComparator(TimeFencePolicy(frozen_periods=2, slushy_periods=2))
    curr_period = 1

    previous_plan = [
        {"item_id": "FG-1", "due_period": 1, "quantity": 20.0},
        {"item_id": "FG-1", "due_period": 2, "quantity": 30.0},
        {"item_id": "FG-1", "due_period": 3, "quantity": 40.0},
        {"item_id": "FG-1", "due_period": 5, "quantity": 50.0},
    ]

    new_plan = [
        # Unauthorized attempt to modify Period 2 (Frozen) from 30 -> 80
        {"item_id": "FG-1", "due_period": 1, "quantity": 20.0},
        {"item_id": "FG-1", "due_period": 2, "quantity": 80.0},
        # Unauthorized attempt to change Period 3 (Slushy) from 40 -> 60
        {"item_id": "FG-1", "due_period": 3, "quantity": 60.0},
        # Legitimate replanning in Period 5 (Liquid) from 50 -> 90
        {"item_id": "FG-1", "due_period": 5, "quantity": 90.0},
    ]

    # Without slushy approval
    enforced, audit = comparator.enforce_time_fences(
        new_plan, previous_plan, current_period=curr_period, allow_slushy_changes=False
    )

    enforced_dict = {o["due_period"]: o["quantity"] for o in enforced}
    # Period 2 (Frozen) must be strictly reverted to 30.0
    assert enforced_dict[2] == 30.0
    # Period 3 (Slushy) unapproved must be reverted to 40.0
    assert enforced_dict[3] == 40.0
    # Period 5 (Liquid) must accept new 90.0
    assert enforced_dict[5] == 90.0

    # With slushy approval
    enforced_slushy, _ = comparator.enforce_time_fences(
        new_plan, previous_plan, current_period=curr_period, allow_slushy_changes=True
    )
    enforced_slushy_dict = {o["due_period"]: o["quantity"] for o in enforced_slushy}
    # Period 2 remains reverted to 30.0 (frozen is never breached)
    assert enforced_slushy_dict[2] == 30.0
    # Period 3 accepts the approved 60.0
    assert enforced_slushy_dict[3] == 60.0
    # Period 5 accepts 90.0
    assert enforced_slushy_dict[5] == 90.0
