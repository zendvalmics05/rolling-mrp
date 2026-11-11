"""Lot Sizing Optimization and Heuristics Module for ROLLINGMRP.

Implements Phase 1.6:
- Lot-for-Lot (L4L): Exact demand matching, zero holding inventory
- Economic Order Quantity (EOQ): Classical continuous approximation adapted to discrete periods
- Wagner-Whitin (W-W): Provably optimal dynamic programming forward algorithm
- Silver-Meal (SM): Near-optimal greedy heuristic minimizing average cost per period
- Periodic Order Quantity (POQ): Time-based batching derived from EOQ cycle time
- Batch Constraints: Minimum lot size and multiple lot (batch rounding) enforcement
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LotSizingResult:
    """Encapsulates order decisions and cost breakdown from a lot sizing run."""
    rule: str
    orders: List[float]
    ending_inventory: List[float]
    setup_cost: float
    holding_cost: float
    total_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "orders": self.orders,
            "ending_inventory": self.ending_inventory,
            "setup_cost": round(self.setup_cost, 2),
            "holding_cost": round(self.holding_cost, 2),
            "total_cost": round(self.total_cost, 2),
        }


def apply_batch_constraints(
    quantity: float,
    min_lot: Optional[float] = None,
    multiple_lot: Optional[float] = None,
) -> float:
    """Apply minimum lot size and multiple lot constraints to an order quantity."""
    if quantity <= 0.0:
        return 0.0

    qty = quantity
    if min_lot and min_lot > 0:
        qty = max(qty, float(min_lot))

    if multiple_lot and multiple_lot > 0:
        qty = math.ceil(qty / float(multiple_lot)) * float(multiple_lot)

    return qty


def lot_for_lot(
    demands: List[float],
    setup_cost: float = 0.0,
    holding_cost: float = 0.0,
) -> LotSizingResult:
    """Lot-for-Lot (L4L): Order exact net requirement in each period."""
    orders = [float(d) for d in demands]
    n = len(demands)
    ending_inv = [0.0] * n
    num_setups = sum(1 for d in demands if d > 0)
    total_setup = num_setups * setup_cost
    total_hold = 0.0

    return LotSizingResult(
        rule="L4L",
        orders=orders,
        ending_inventory=ending_inv,
        setup_cost=total_setup,
        holding_cost=total_hold,
        total_cost=total_setup + total_hold,
    )


def economic_order_quantity(
    demands: List[float],
    setup_cost: float,
    holding_cost: float,
    min_lot: Optional[float] = None,
    multiple_lot: Optional[float] = None,
) -> LotSizingResult:
    """Economic Order Quantity (EOQ) heuristic:
    Calculates EOQ batch size from average demand, ordering whenever net available drops below zero.
    """
    n = len(demands)
    total_demand = sum(demands)
    if total_demand <= 0 or holding_cost <= 0:
        return lot_for_lot(demands, setup_cost, holding_cost)

    avg_demand = total_demand / n
    eoq_calc = math.sqrt((2.0 * avg_demand * setup_cost) / holding_cost)
    batch_size = max(1.0, round(eoq_calc))
    batch_size = apply_batch_constraints(batch_size, min_lot, multiple_lot)

    orders = [0.0] * n
    ending_inv = [0.0] * n
    curr_inv = 0.0
    total_setup = 0.0
    total_hold = 0.0

    for t in range(n):
        net_needed = demands[t] - curr_inv
        if net_needed > 0:
            num_batches = math.ceil(net_needed / batch_size)
            order_qty = num_batches * batch_size
            orders[t] = order_qty
            total_setup += setup_cost
            curr_inv += order_qty

        curr_inv -= demands[t]
        ending_inv[t] = curr_inv
        total_hold += curr_inv * holding_cost

    return LotSizingResult(
        rule="EOQ",
        orders=orders,
        ending_inventory=ending_inv,
        setup_cost=total_setup,
        holding_cost=total_hold,
        total_cost=total_setup + total_hold,
    )


def wagner_whitin(
    demands: List[float],
    setup_cost: float,
    holding_cost: float,
) -> LotSizingResult:
    """Wagner-Whitin Dynamic Programming Algorithm:
    Computes provably optimal time-phased lot sizes minimizing total setup and holding costs.

    Cost recurrence:
    f(t) = min_{1 <= j <= t} [ f(j-1) + S + sum_{k=j}^t (k - j) * h * D[k] ]
    """
    n = len(demands)
    if n == 0:
        return LotSizingResult("WAGNER_WHITIN", [], [], 0.0, 0.0, 0.0)

    # Precompute holding cost matrix C[j][t]: cost to satisfy periods j..t with an order in period j
    C = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for t in range(j, n):
            h_cost = sum((k - j) * holding_cost * demands[k] for k in range(j, t + 1))
            C[j][t] = setup_cost + h_cost

    # Dynamic programming forward pass
    f = [float("inf")] * (n + 1)
    f[0] = 0.0
    last_order_period = [0] * (n + 1)

    for t in range(1, n + 1):
        for j in range(1, t + 1):
            cost = f[j - 1] + C[j - 1][t - 1]
            if cost < f[t]:
                f[t] = cost
                last_order_period[t] = j

    # Backtrack optimal order placements
    orders = [0.0] * n
    curr = n
    while curr > 0:
        j = last_order_period[curr]
        order_qty = sum(demands[k] for k in range(j - 1, curr))
        orders[j - 1] = float(order_qty)
        curr = j - 1

    # Simulate resulting inventory trajectory
    ending_inv = [0.0] * n
    curr_inv = 0.0
    total_hold = 0.0
    for t in range(n):
        curr_inv += orders[t] - demands[t]
        ending_inv[t] = curr_inv
        total_hold += curr_inv * holding_cost

    num_setups = sum(1 for q in orders if q > 0)
    total_setup = num_setups * setup_cost

    return LotSizingResult(
        rule="WAGNER_WHITIN",
        orders=orders,
        ending_inventory=ending_inv,
        setup_cost=total_setup,
        holding_cost=total_hold,
        total_cost=f[n],
    )


def silver_meal(
    demands: List[float],
    setup_cost: float,
    holding_cost: float,
) -> LotSizingResult:
    """Silver-Meal Heuristic:
    Evaluates average cost per period for orders covering 1, 2, ..., k periods.
    Stops and places an order when the average cost per period increases.
    """
    n = len(demands)
    orders = [0.0] * n
    t = 0

    while t < n:
        if demands[t] <= 0:
            t += 1
            continue

        best_span = 1
        cum_h = 0.0
        prev_ac = setup_cost  # (S + 0) / 1

        for span in range(2, n - t + 1):
            next_d = demands[t + span - 1]
            cum_h += (span - 1) * holding_cost * next_d
            ac = (setup_cost + cum_h) / span
            if ac < prev_ac:
                prev_ac = ac
                best_span = span
            else:
                break

        order_qty = sum(demands[t : t + best_span])
        orders[t] = float(order_qty)
        t += best_span

    # Compute inventory and costs
    ending_inv = [0.0] * n
    curr_inv = 0.0
    total_hold = 0.0
    for idx in range(n):
        curr_inv += orders[idx] - demands[idx]
        ending_inv[idx] = curr_inv
        total_hold += curr_inv * holding_cost

    num_setups = sum(1 for q in orders if q > 0)
    total_setup = num_setups * setup_cost

    return LotSizingResult(
        rule="SILVER_MEAL",
        orders=orders,
        ending_inventory=ending_inv,
        setup_cost=total_setup,
        holding_cost=total_hold,
        total_cost=total_setup + total_hold,
    )


def periodic_order_quantity(
    demands: List[float],
    setup_cost: float,
    holding_cost: float,
) -> LotSizingResult:
    """Periodic Order Quantity (POQ):
    Converts EOQ into an integer number of demand periods: P = round(EOQ / avg_demand).
    Orders every P periods to cover requirements over the cycle.
    """
    n = len(demands)
    total_demand = sum(demands)
    if total_demand <= 0 or holding_cost <= 0:
        return lot_for_lot(demands, setup_cost, holding_cost)

    avg_demand = total_demand / n
    eoq = math.sqrt((2.0 * avg_demand * setup_cost) / holding_cost)
    poq_period = max(1, round(eoq / avg_demand))

    orders = [0.0] * n
    t = 0
    while t < n:
        if demands[t] <= 0:
            t += 1
            continue
        span = min(poq_period, n - t)
        order_qty = sum(demands[t : t + span])
        orders[t] = float(order_qty)
        t += span

    ending_inv = [0.0] * n
    curr_inv = 0.0
    total_hold = 0.0
    for idx in range(n):
        curr_inv += orders[idx] - demands[idx]
        ending_inv[idx] = curr_inv
        total_hold += curr_inv * holding_cost

    num_setups = sum(1 for q in orders if q > 0)
    total_setup = num_setups * setup_cost

    return LotSizingResult(
        rule="POQ",
        orders=orders,
        ending_inventory=ending_inv,
        setup_cost=total_setup,
        holding_cost=total_hold,
        total_cost=total_setup + total_hold,
    )


def compute_lot_sizing(
    rule: str,
    demands: List[float],
    setup_cost: float,
    holding_cost: float,
    min_lot: Optional[float] = None,
    multiple_lot: Optional[float] = None,
) -> LotSizingResult:
    """Unified entrypoint for all lot sizing strategies."""
    rule_upper = rule.upper()
    if rule_upper in ("L4L", "LOT_FOR_LOT"):
        res = lot_for_lot(demands, setup_cost, holding_cost)
    elif rule_upper == "EOQ":
        res = economic_order_quantity(demands, setup_cost, holding_cost, min_lot, multiple_lot)
    elif rule_upper in ("WAGNER_WHITIN", "W-W", "WW"):
        res = wagner_whitin(demands, setup_cost, holding_cost)
    elif rule_upper in ("SILVER_MEAL", "SM"):
        res = silver_meal(demands, setup_cost, holding_cost)
    elif rule_upper == "POQ":
        res = periodic_order_quantity(demands, setup_cost, holding_cost)
    else:
        # Default to L4L
        res = lot_for_lot(demands, setup_cost, holding_cost)

    # Post-apply min/multiple lot constraints if rule didn't handle internally
    if (min_lot or multiple_lot) and rule_upper not in ("EOQ",):
        adjusted_orders = [apply_batch_constraints(q, min_lot, multiple_lot) for q in res.orders]
        # Re-simulate inventory with adjusted orders
        curr_inv = 0.0
        total_hold = 0.0
        ending_inv = [0.0] * len(demands)
        for t in range(len(demands)):
            curr_inv += adjusted_orders[t] - demands[t]
            ending_inv[t] = curr_inv
            total_hold += curr_inv * holding_cost
        num_setups = sum(1 for q in adjusted_orders if q > 0)
        res = LotSizingResult(
            rule=res.rule,
            orders=adjusted_orders,
            ending_inventory=ending_inv,
            setup_cost=num_setups * setup_cost,
            holding_cost=total_hold,
            total_cost=num_setups * setup_cost + total_hold,
        )

    return res

