"""Frozen Window, Time Fences, and Plan Comparison Engine for ROLLINGMRP.

Implements:
- Phase 2.3: Time Fence Zones and Enforcement:
  - Frozen (Weeks 0–1): Hard constraint; no changes permitted to released/firm orders
  - Slushy (Weeks 2–4): Controlled flexibility; modifications require explicit planner approval
  - Liquid (Weeks 5+): Unconstrained replanning horizon to absorb forecast evolution
- Phase 2.4: Plan Comparison and Change Detection:
  - Plan diffing between planning cycles: new plan vs. previous plan
  - Detection of quantity changes, schedule shifts (timing churn), cancellations, and new orders
  - Change classification and time fence policy enforcement
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


class PlanningZone(str, Enum):
    """Planning horizon time zones."""
    FROZEN = "FROZEN"
    SLUSHY = "SLUSHY"
    LIQUID = "LIQUID"


class ChangeType(str, Enum):
    """Classification of plan modifications across planning cycles."""
    UNCHANGED = "UNCHANGED"
    NEW_ORDER = "NEW_ORDER"
    CANCELLED = "CANCELLED"
    QUANTITY_CHANGE = "QUANTITY_CHANGE"
    TIMING_CHANGE = "TIMING_CHANGE"


@dataclass
class PlanChangeRecord:
    """Audit record capturing a detected modification between two planning runs."""
    item_id: str
    change_type: str
    zone: str
    status: str  # 'BLOCKED', 'APPROVED_CHANGE', 'ALLOWED', 'REQUIRES_APPROVAL'
    old_period: Optional[int] = None
    new_period: Optional[int] = None
    old_quantity: float = 0.0
    new_quantity: float = 0.0
    quantity_delta: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TimeFencePolicy:
    """Defines and computes boundary fences relative to current simulation period."""

    def __init__(
        self,
        frozen_periods: int = 2,  # Current period and next period (e.g. Weeks 0-1)
        slushy_periods: int = 3,  # Following 3 periods (e.g. Weeks 2-4)
    ):
        self.frozen_periods = frozen_periods
        self.slushy_periods = slushy_periods

    def get_zone(self, current_period: int, target_period: int) -> PlanningZone:
        """Classify target_period into FROZEN, SLUSHY, or LIQUID relative to current_period."""
        horizon_offset = target_period - current_period
        if horizon_offset < 0:
            # Past due or immediate release
            return PlanningZone.FROZEN
        elif horizon_offset < self.frozen_periods:
            return PlanningZone.FROZEN
        elif horizon_offset < (self.frozen_periods + self.slushy_periods):
            return PlanningZone.SLUSHY
        else:
            return PlanningZone.LIQUID


class PlanComparator:
    """Compares consecutive MRP plans, identifies changes, and enforces time fence policies."""

    def __init__(self, policy: Optional[TimeFencePolicy] = None):
        self.policy = policy or TimeFencePolicy()

    def compare_plans(
        self,
        previous_plan: List[Dict[str, Any]],
        new_plan: List[Dict[str, Any]],
        current_period: int,
        allow_slushy_changes: bool = True,
    ) -> List[PlanChangeRecord]:
        """Compare previous plan against new plan and classify all modifications."""
        changes: List[PlanChangeRecord] = []

        # Index orders: (item_id, due_period) -> order_dict
        prev_map: Dict[Tuple[str, int], Dict[str, Any]] = {
            (o["item_id"], o["due_period"]): o for o in previous_plan
        }
        new_map: Dict[Tuple[str, int], Dict[str, Any]] = {
            (o["item_id"], o["due_period"]): o for o in new_plan
        }

        # 1. Check existing orders in new plan against previous
        for (item_id, period), new_order in new_map.items():
            zone = self.policy.get_zone(current_period, period)
            new_qty = float(new_order["quantity"])

            if (item_id, period) in prev_map:
                old_order = prev_map[(item_id, period)]
                old_qty = float(old_order["quantity"])

                if abs(new_qty - old_qty) > 1e-4:
                    delta = new_qty - old_qty
                    status = self._resolve_status(zone, allow_slushy_changes)
                    changes.append(PlanChangeRecord(
                        item_id=item_id,
                        change_type=ChangeType.QUANTITY_CHANGE.value,
                        zone=zone.value,
                        status=status,
                        old_period=period,
                        new_period=period,
                        old_quantity=old_qty,
                        new_quantity=new_qty,
                        quantity_delta=delta,
                        description=f"Quantity modified in period {period} from {old_qty} to {new_qty} (delta: {delta:+0.1f})",
                    ))
            else:
                # Check if this represents a timing shift of an order from a nearby period
                # Look for matching item in previous plan that is missing in new plan
                timing_candidate = None
                for (p_item, p_period), p_ord in prev_map.items():
                    if p_item == item_id and (p_item, p_period) not in new_map:
                        timing_candidate = (p_period, p_ord)
                        break

                if timing_candidate and abs(float(timing_candidate[1]["quantity"]) - new_qty) < 1e-4:
                    old_p, old_ord = timing_candidate
                    status = self._resolve_status(zone, allow_slushy_changes)
                    changes.append(PlanChangeRecord(
                        item_id=item_id,
                        change_type=ChangeType.TIMING_CHANGE.value,
                        zone=zone.value,
                        status=status,
                        old_period=old_p,
                        new_period=period,
                        old_quantity=float(old_ord["quantity"]),
                        new_quantity=new_qty,
                        quantity_delta=0.0,
                        description=f"Order schedule shifted from period {old_p} to {period}",
                    ))
                else:
                    # Genuinely new order
                    status = self._resolve_status(zone, allow_slushy_changes)
                    changes.append(PlanChangeRecord(
                        item_id=item_id,
                        change_type=ChangeType.NEW_ORDER.value,
                        zone=zone.value,
                        status=status,
                        old_period=None,
                        new_period=period,
                        old_quantity=0.0,
                        new_quantity=new_qty,
                        quantity_delta=new_qty,
                        description=f"New order introduced in period {period} for {new_qty} units",
                    ))

        # 2. Check orders present in previous plan that were cancelled in new plan
        for (item_id, period), old_order in prev_map.items():
            if (item_id, period) not in new_map:
                # If not already matched as a timing shift
                already_matched = any(
                    c.item_id == item_id and c.old_period == period and c.change_type == ChangeType.TIMING_CHANGE.value
                    for c in changes
                )
                if not already_matched:
                    zone = self.policy.get_zone(current_period, period)
                    old_qty = float(old_order["quantity"])
                    status = self._resolve_status(zone, allow_slushy_changes)
                    changes.append(PlanChangeRecord(
                        item_id=item_id,
                        change_type=ChangeType.CANCELLED.value,
                        zone=zone.value,
                        status=status,
                        old_period=period,
                        new_period=None,
                        old_quantity=old_qty,
                        new_quantity=0.0,
                        quantity_delta=-old_qty,
                        description=f"Order in period {period} for {old_qty} units cancelled",
                    ))

        return changes

    def _resolve_status(self, zone: PlanningZone, allow_slushy_changes: bool) -> str:
        """Resolve policy action based on planning zone."""
        if zone == PlanningZone.FROZEN:
            return "BLOCKED"
        elif zone == PlanningZone.SLUSHY:
            return "APPROVED_CHANGE" if allow_slushy_changes else "REQUIRES_APPROVAL"
        else:
            return "ALLOWED"

    def enforce_time_fences(
        self,
        new_plan: List[Dict[str, Any]],
        previous_plan: List[Dict[str, Any]],
        current_period: int,
        allow_slushy_changes: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[PlanChangeRecord]]:
        """Enforce frozen time fence policies by reverting unauthorized changes.

        - FROZEN changes: Reverted back to previous plan order.
        - SLUSHY changes: Reverted unless allow_slushy_changes is True.
        - LIQUID changes: Retained without restriction.

        Returns:
            Tuple of (enforced_plan_orders, change_audit_records).
        """
        changes = self.compare_plans(previous_plan, new_plan, current_period, allow_slushy_changes)

        # Build enforced plan starting from allowed/liquid new orders
        enforced_orders: List[Dict[str, Any]] = []

        # Index previous orders by (item_id, due_period)
        prev_map = {(o["item_id"], o["due_period"]): dict(o) for o in previous_plan}
        new_map = {(o["item_id"], o["due_period"]): dict(o) for o in new_plan}

        # Keep all orders in LIQUID zone from new plan
        # For FROZEN zone: copy orders from previous plan
        # For SLUSHY zone: copy from new plan if approved, else from previous plan
        all_keys = set(prev_map.keys()).union(set(new_map.keys()))

        for (item_id, period) in all_keys:
            zone = self.policy.get_zone(current_period, period)

            if zone == PlanningZone.FROZEN:
                # Strict enforcement: order must match previous plan
                if (item_id, period) in prev_map:
                    order = dict(prev_map[(item_id, period)])
                    order["status"] = "firm"
                    enforced_orders.append(order)
            elif zone == PlanningZone.SLUSHY:
                if allow_slushy_changes:
                    if (item_id, period) in new_map:
                        enforced_orders.append(new_map[(item_id, period)])
                else:
                    # Not approved -> retain previous
                    if (item_id, period) in prev_map:
                        enforced_orders.append(prev_map[(item_id, period)])
            else:  # LIQUID
                if (item_id, period) in new_map:
                    enforced_orders.append(new_map[(item_id, period)])

        # Sort cleanly by item and due_period
        enforced_orders.sort(key=lambda o: (o["item_id"], o["due_period"]))
        return enforced_orders, changes

