"""Gross-to-Net Requirements Engine for ROLLINGMRP.

Implements:
- Phase 1.2: Gross requirements aggregation from MPS across periods and demand types
- Phase 1.3: Time-phased Net Requirements Calculation:
  - Initial PAB(0) = on_hand - allocated
  - Time-phased PAB update: PAB(t) = PAB(t-1) + SR(t) - GR(t)
  - Safety stock deficit detection and net requirement triggers
  - Planned Order Receipts (PORec) and Lead-Time Offsetting to Planned Order Releases (PORel)
  - Exception detection (past-due releases, safety stock breaches)
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd


@dataclass
class PlannedOrder:
    """Represents a planned order generated during netting."""
    item_id: str
    due_period: int
    release_period: int
    quantity: float
    lot_sizing_rule: str
    past_due: bool = False
    status: str = "planned"
    pegging_chain: Optional[Dict[str, Any]] = None


@dataclass
class NettingResult:
    """Encapsulates the complete time-phased MRP netting grid for an item."""
    item_id: str
    periods: List[int]
    initial_on_hand: float
    allocated: float
    initial_pab: float
    safety_stock: float
    lead_time: int
    lot_sizing_rule: str
    gross_requirements: List[float]
    scheduled_receipts: List[float]
    projected_available_balance: List[float]
    net_requirements: List[float]
    planned_order_receipts: List[float]
    planned_orders: List[PlannedOrder] = field(default_factory=list)
    exceptions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Export netting grid as a tabular pandas DataFrame."""
        return pd.DataFrame({
            "period": self.periods,
            "gross_req": self.gross_requirements,
            "scheduled_receipts": self.scheduled_receipts,
            "pab": self.projected_available_balance,
            "net_req": self.net_requirements,
            "po_receipts": self.planned_order_receipts,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        res = asdict(self)
        res["planned_orders"] = [asdict(po) for po in self.planned_orders]
        return res


def aggregate_gross_requirements(
    mps_data: Union[List[Dict[str, Any]], pd.DataFrame],
    periods: List[int],
    item_id: Optional[str] = None,
    demand_types: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """Aggregate MPS gross requirements across specified periods and demand types.

    Args:
        mps_data: List of MPS dicts or DataFrame containing item_id, period_number, quantity, demand_type.
        periods: List of planning periods (e.g. [1, 2, ..., 12]).
        item_id: Optional specific item filter.
        demand_types: Optional list of demand types to include (e.g. ['actual', 'forecast']).

    Returns:
        Dict mapping item_id -> list of float gross requirements aligned with `periods`.
    """
    if isinstance(mps_data, list):
        if not mps_data:
            return {}
        df = pd.DataFrame(mps_data)
    else:
        df = mps_data.copy()

    if df.empty:
        return {}

    if item_id:
        df = df[df["item_id"] == item_id]

    if demand_types:
        df = df[df["demand_type"].isin(demand_types)]

    result: Dict[str, List[float]] = {}
    unique_items = df["item_id"].unique()

    for itm in unique_items:
        itm_df = df[df["item_id"] == itm]
        # Sum quantities by period
        qty_by_period = itm_df.groupby("period_number")["quantity"].sum().to_dict()
        reqs = [float(qty_by_period.get(p, 0.0)) for p in periods]
        result[itm] = reqs

    return result


def calculate_net_requirements(
    item_id: str,
    periods: List[int],
    gross_requirements: List[float],
    on_hand: float,
    allocated: float = 0.0,
    safety_stock: float = 0.0,
    lead_time: int = 0,
    scheduled_receipts: Optional[List[float]] = None,
    lot_sizing_rule: str = "L4L",
    lot_sizing_func: Optional[Callable[[float, str, Dict[str, Any]], float]] = None,
    lot_sizing_params: Optional[Dict[str, Any]] = None,
) -> NettingResult:
    """Calculate standard time-phased MRP gross-to-net requirements.

    Args:
        item_id: Target item identifier.
        periods: Ordered list of time periods (e.g. [1, 2, 3, 4]).
        gross_requirements: Gross demand per period.
        on_hand: Starting physical inventory.
        allocated: Quantity allocated to existing commitments.
        safety_stock: Minimum required buffer stock.
        lead_time: Lead time in periods for order offsetting.
        scheduled_receipts: Known incoming orders per period.
        lot_sizing_rule: Identifier ('L4L', 'EOQ', etc.).
        lot_sizing_func: Optional custom callable to batch net requirements.
        lot_sizing_params: Extra parameters for lot sizing.

    Returns:
        NettingResult containing full time-phased trajectory, orders, and exceptions.
    """
    n = len(periods)
    if len(gross_requirements) != n:
        raise ValueError(f"gross_requirements length ({len(gross_requirements)}) must match periods ({n})")

    if scheduled_receipts is None:
        sr_list = [0.0] * n
    else:
        if len(scheduled_receipts) != n:
            raise ValueError(f"scheduled_receipts length ({len(scheduled_receipts)}) must match periods ({n})")
        sr_list = [float(x) for x in scheduled_receipts]

    gr_list = [float(x) for x in gross_requirements]
    pab_0 = float(on_hand - allocated)
    current_pab = pab_0

    pab_history: List[float] = []
    net_requirements: List[float] = []
    po_receipts: List[float] = []
    planned_orders: List[PlannedOrder] = []
    exceptions: List[Dict[str, Any]] = []

    for idx, period in enumerate(periods):
        gr = gr_list[idx]
        sr = sr_list[idx]

        # 1. Unconstrained PAB before new planned order receipt
        available_before_order = current_pab + sr - gr

        # 2. Check if net requirement is triggered
        if available_before_order < safety_stock:
            net_req = safety_stock - available_before_order
            # Apply lot sizing
            if lot_sizing_func:
                po_rec = lot_sizing_func(net_req, lot_sizing_rule, lot_sizing_params or {})
            else:
                # Default L4L
                po_rec = net_req

            po_rec = max(0.0, po_rec)
            end_pab = available_before_order + po_rec

            # 3. Lead time offsetting
            rel_period = period - lead_time
            is_past_due = rel_period <= 0

            po = PlannedOrder(
                item_id=item_id,
                due_period=period,
                release_period=rel_period,
                quantity=po_rec,
                lot_sizing_rule=lot_sizing_rule,
                past_due=is_past_due,
            )
            planned_orders.append(po)

            if is_past_due:
                exceptions.append({
                    "item_id": item_id,
                    "period_due": period,
                    "release_period": rel_period,
                    "type": "expedite",
                    "quantity": po_rec,
                    "description": f"Order for {item_id} due in period {period} requires release in period {rel_period} (past due). Expedite needed.",
                })
        else:
            net_req = 0.0
            po_rec = 0.0
            end_pab = available_before_order

        current_pab = end_pab
        pab_history.append(end_pab)
        net_requirements.append(net_req)
        po_receipts.append(po_rec)

    return NettingResult(
        item_id=item_id,
        periods=periods,
        initial_on_hand=on_hand,
        allocated=allocated,
        initial_pab=pab_0,
        safety_stock=safety_stock,
        lead_time=lead_time,
        lot_sizing_rule=lot_sizing_rule,
        gross_requirements=gr_list,
        scheduled_receipts=sr_list,
        projected_available_balance=pab_history,
        net_requirements=net_requirements,
        planned_order_receipts=po_receipts,
        planned_orders=planned_orders,
        exceptions=exceptions,
    )

