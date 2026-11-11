"""Rough-Cut Capacity Planning (RCCP) Engine for ROLLINGMRP.

Implements Phase 4.1:
- Routing retrieval: Operations by item, sequence, work center.
- Load calculation: required_hours = sum(setup_time + qty * run_time) for positive production.
- Work center aggregation: Time-phased load by work center by period/week.
- Calendar-aware capacity: effective_available_hours = nominal_capacity * efficiency * (calendar_days / nominal_days).
- Utilization calculation: utilization_pct = (load_hours / available_hours) * 100%.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


@dataclass
class OperationRouting:
    """Manufacturing operation for an item at a specific work center."""
    item_id: str
    work_center_id: str
    operation_sequence: int
    setup_time_hours: float
    run_time_hours_per_unit: float
    routing_id: Optional[str] = None

    def calculate_operation_load(self, quantity: float) -> float:
        """Calculate required machine/labor hours for a production batch."""
        if quantity <= 0:
            return 0.0
        return self.setup_time_hours + (quantity * self.run_time_hours_per_unit)


@dataclass
class WorkCenterProfile:
    """Capacity definition and operating characteristics for a work center."""
    work_center_id: str
    capacity_hours_per_week: float
    efficiency_factor: float = 1.0
    calendar_id: Optional[str] = "CAL-STD-5D"
    description: Optional[str] = ""

    def get_effective_capacity(
        self,
        working_days: Optional[float] = None,
        nominal_days: float = 5.0,
    ) -> float:
        """Calculate effective available hours adjusted for efficiency and working calendar.

        effective_hours = capacity_hours * efficiency * (working_days / nominal_days)
        """
        base_capacity = self.capacity_hours_per_week * self.efficiency_factor
        if working_days is not None and nominal_days > 0:
            calendar_factor = working_days / nominal_days
            return base_capacity * calendar_factor
        return base_capacity


@dataclass
class PeriodWorkCenterLoad:
    """Detailed load and utilization metrics for a work center in a specific planning period."""
    work_center_id: str
    period: int
    load_hours: float
    available_hours: float
    utilization_pct: float
    overload_hours: float  # max(0.0, load_hours - available_hours)
    slack_hours: float     # max(0.0, available_hours - load_hours)
    contributing_orders: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RCCPEngine:
    """Rough-Cut Capacity Planning calculation engine."""

    def __init__(
        self,
        routings_data: Union[pd.DataFrame, List[Dict[str, Any]], List[OperationRouting]],
        work_centers_data: Union[pd.DataFrame, List[Dict[str, Any]], List[WorkCenterProfile]],
    ):
        """Initialize engine with routings and work center master specifications."""
        self.routings: Dict[str, List[OperationRouting]] = {}
        self.work_centers: Dict[str, WorkCenterProfile] = {}
        self._load_routings(routings_data)
        self._load_work_centers(work_centers_data)

    def _load_routings(self, data: Any) -> None:
        """Load and index routings by item_id sorted by operation sequence."""
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError("routings_data must be a pandas DataFrame or list of dicts/OperationRouting")

        for r in records:
            if isinstance(r, OperationRouting):
                item_id = r.item_id
                routing_obj = r
            else:
                item_id = str(r["item_id"])
                routing_obj = OperationRouting(
                    item_id=item_id,
                    work_center_id=str(r["work_center_id"]),
                    operation_sequence=int(r.get("operation_sequence", 10)),
                    setup_time_hours=float(r.get("setup_time_hours", 0.0)),
                    run_time_hours_per_unit=float(r.get("run_time_hours_per_unit", 0.0)),
                    routing_id=str(r.get("routing_id", "")),
                )
            if item_id not in self.routings:
                self.routings[item_id] = []
            self.routings[item_id].append(routing_obj)

        for iid in self.routings:
            self.routings[iid].sort(key=lambda op: op.operation_sequence)

    def _load_work_centers(self, data: Any) -> None:
        """Load work centers map."""
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError("work_centers_data must be a pandas DataFrame or list")

        for wc in records:
            if isinstance(wc, WorkCenterProfile):
                self.work_centers[wc.work_center_id] = wc
            else:
                wc_id = str(wc["work_center_id"])
                # Handle different column namings gracefully (capacity_hours vs capacity_hours_per_week)
                cap = float(wc.get("capacity_hours_per_week", wc.get("capacity_hours", 40.0)))
                eff = float(wc.get("efficiency_factor", 1.0))
                cal = str(wc.get("calendar_id", "CAL-STD-5D"))
                desc = str(wc.get("description", ""))
                self.work_centers[wc_id] = WorkCenterProfile(
                    work_center_id=wc_id,
                    capacity_hours_per_week=cap,
                    efficiency_factor=eff,
                    calendar_id=cal,
                    description=desc,
                )

    def get_item_routings(self, item_id: str) -> List[OperationRouting]:
        """Retrieve sequential routing operations for a given manufactured item."""
        return self.routings.get(item_id, [])

    def calculate_plan_load(
        self,
        planned_orders: List[Dict[str, Any]],
        periods: Optional[List[int]] = None,
        period_working_days: Optional[Dict[int, float]] = None,
        timing_key: str = "due_period",
    ) -> Dict[str, Dict[int, PeriodWorkCenterLoad]]:
        """Calculate time-phased capacity load across all work centers and planning periods.

        Args:
            planned_orders: List of planned orders or production quantities.
                            Must contain 'item_id', 'quantity', and timing_key (e.g. 'due_period' or 'period').
            periods: Sequence of periods to compute. If None, inferred from planned_orders.
            period_working_days: Optional mapping of period -> working days for calendar adjustment.
            timing_key: Dictionary key for the planning period ('due_period', 'period', or 'release_period').

        Returns:
            Nested dict: {work_center_id: {period: PeriodWorkCenterLoad}}
        """
        # Determine periods
        if periods is not None:
            active_periods = sorted(list(periods))
        else:
            observed_periods = {
                int(o.get(timing_key, o.get("period", 1)))
                for o in planned_orders
                if float(o.get("quantity", 0.0)) > 0
            }
            active_periods = sorted(list(observed_periods)) if observed_periods else [1]

        # Initialize load table structure
        load_matrix: Dict[str, Dict[int, PeriodWorkCenterLoad]] = {}
        for wc_id, wc_profile in self.work_centers.items():
            load_matrix[wc_id] = {}
            for p in active_periods:
                w_days = period_working_days.get(p) if period_working_days else None
                available = wc_profile.get_effective_capacity(working_days=w_days)
                load_matrix[wc_id][p] = PeriodWorkCenterLoad(
                    work_center_id=wc_id,
                    period=p,
                    load_hours=0.0,
                    available_hours=available,
                    utilization_pct=0.0,
                    overload_hours=0.0,
                    slack_hours=available,
                    contributing_orders=[],
                )

        # Aggregate loads by item routing operations
        for order in planned_orders:
            qty = float(order.get("quantity", 0.0))
            if qty <= 0:
                continue

            item_id = str(order.get("item_id"))
            p = int(order.get(timing_key, order.get("period", 1)))

            # If period is outside our monitored horizon, skip or allocate if within active_periods
            if p not in active_periods:
                continue

            # Retrieve routing operations for this item
            ops = self.get_item_routings(item_id)
            for op in ops:
                wc_id = op.work_center_id
                if wc_id not in load_matrix:
                    # Dynamically register unlisted work center with default 40h
                    self.work_centers[wc_id] = WorkCenterProfile(work_center_id=wc_id, capacity_hours_per_week=40.0)
                    load_matrix[wc_id] = {}
                    for per in active_periods:
                        load_matrix[wc_id][per] = PeriodWorkCenterLoad(
                            work_center_id=wc_id,
                            period=per,
                            load_hours=0.0,
                            available_hours=40.0,
                            utilization_pct=0.0,
                            overload_hours=0.0,
                            slack_hours=40.0,
                            contributing_orders=[],
                        )

                op_load = op.calculate_operation_load(qty)
                cell = load_matrix[wc_id][p]
                cell.load_hours += op_load
                cell.contributing_orders.append({
                    "item_id": item_id,
                    "order_id": order.get("order_id", ""),
                    "quantity": qty,
                    "setup_hours": op.setup_time_hours,
                    "run_hours": qty * op.run_time_hours_per_unit,
                    "total_hours": op_load,
                })

        # Finalize metrics: round numbers, compute utilization, overload, and slack
        for wc_id in load_matrix:
            for p in active_periods:
                cell = load_matrix[wc_id][p]
                cell.load_hours = round(cell.load_hours, 2)
                avail = cell.available_hours
                if avail > 0:
                    cell.utilization_pct = round((cell.load_hours / avail) * 100.0, 1)
                else:
                    cell.utilization_pct = 999.0 if cell.load_hours > 0 else 0.0

                cell.overload_hours = round(max(0.0, cell.load_hours - avail), 2)
                cell.slack_hours = round(max(0.0, avail - cell.load_hours), 2)

        return load_matrix

    def to_dataframe(
        self,
        load_matrix: Dict[str, Dict[int, PeriodWorkCenterLoad]],
    ) -> pd.DataFrame:
        """Convert load matrix into a clean summary DataFrame."""
        rows = []
        for wc_id, periods in load_matrix.items():
            for p, cell in periods.items():
                rows.append({
                    "work_center_id": wc_id,
                    "period": p,
                    "load_hours": cell.load_hours,
                    "available_hours": cell.available_hours,
                    "utilization_pct": cell.utilization_pct,
                    "overload_hours": cell.overload_hours,
                    "slack_hours": cell.slack_hours,
                    "order_count": len(cell.contributing_orders),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(by=["work_center_id", "period"], inplace=True)
        return df

