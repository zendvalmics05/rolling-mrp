"""Bottleneck Detection and Root Cause Tracing Engine for ROLLINGMRP.

Implements Phase 4.2:
- Overload identification: utilization > threshold (default 100%).
- Severity ranking: based on peak overload magnitude, total overload hours, and duration.
- Root cause tracing: identifying which items and orders drive the work center overload.
- Bottleneck report: detailed summary ranking top constrained work centers with actionable analytics.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from src.capacity.rccp import PeriodWorkCenterLoad, RCCPEngine


@dataclass
class OverloadIncident:
    """An overload occurrence at a specific work center and period."""
    work_center_id: str
    period: int
    load_hours: float
    available_hours: float
    overload_hours: float
    utilization_pct: float
    severity_score: float
    top_drivers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BottleneckSummary:
    """Consolidated bottleneck profile for a constrained work center across the planning horizon."""
    work_center_id: str
    rank: int
    max_utilization_pct: float
    total_overload_hours: float
    overloaded_periods_count: int
    overloaded_periods: List[int]
    total_horizon_load: float
    total_horizon_capacity: float
    average_utilization_pct: float
    primary_cost_driver_items: List[str]
    incidents: List[OverloadIncident] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BottleneckDetector:
    """Identifies, ranks, and diagnoses manufacturing capacity bottlenecks."""

    def __init__(self, utilization_threshold_pct: float = 100.0):
        """
        Args:
            utilization_threshold_pct: Utilization percentage that qualifies as overloaded (default 100.0%).
        """
        self.utilization_threshold_pct = utilization_threshold_pct

    def detect_bottlenecks(
        self,
        load_matrix: Dict[str, Dict[int, PeriodWorkCenterLoad]],
    ) -> List[BottleneckSummary]:
        """Analyze load matrix to identify and rank all overloaded work centers.

        Args:
            load_matrix: Nested dictionary {work_center_id: {period: PeriodWorkCenterLoad}}.

        Returns:
            List of BottleneckSummary objects sorted by severity (most critical first).
        """
        summaries: List[BottleneckSummary] = []

        for wc_id, period_dict in load_matrix.items():
            incidents: List[OverloadIncident] = []
            total_load = 0.0
            total_capacity = 0.0
            max_util = 0.0
            total_overload = 0.0
            item_hours_map: Dict[str, float] = {}

            for p in sorted(period_dict.keys()):
                cell = period_dict[p]
                total_load += cell.load_hours
                total_capacity += cell.available_hours
                if cell.utilization_pct > max_util:
                    max_util = cell.utilization_pct

                # Accumulate item hours across horizon for this work center
                for ord_info in cell.contributing_orders:
                    itm = ord_info["item_id"]
                    item_hours_map[itm] = item_hours_map.get(itm, 0.0) + ord_info["total_hours"]

                # Check if overloaded
                if cell.utilization_pct > self.utilization_threshold_pct or cell.overload_hours > 1e-4:
                    total_overload += cell.overload_hours

                    # Trace drivers for this period
                    drivers = sorted(
                        cell.contributing_orders,
                        key=lambda x: x["total_hours"],
                        reverse=True,
                    )
                    top_drivers = [
                        {
                            "item_id": d["item_id"],
                            "quantity": d["quantity"],
                            "hours": round(d["total_hours"], 2),
                            "percent_of_load": round((d["total_hours"] / cell.load_hours) * 100.0, 1)
                            if cell.load_hours > 0 else 0.0,
                        }
                        for d in drivers
                    ]

                    # Severity score: combination of peak overload magnitude and excess utilization
                    severity = round(cell.overload_hours * (cell.utilization_pct / 100.0), 2)

                    incidents.append(OverloadIncident(
                        work_center_id=wc_id,
                        period=p,
                        load_hours=cell.load_hours,
                        available_hours=cell.available_hours,
                        overload_hours=cell.overload_hours,
                        utilization_pct=cell.utilization_pct,
                        severity_score=severity,
                        top_drivers=top_drivers,
                    ))

            # If work center has overloads, construct summary
            if incidents:
                # Top contributing items across whole horizon
                top_items = sorted(item_hours_map.keys(), key=lambda k: item_hours_map[k], reverse=True)
                avg_util = round((total_load / total_capacity) * 100.0, 1) if total_capacity > 0 else 0.0

                summaries.append(BottleneckSummary(
                    work_center_id=wc_id,
                    rank=0,  # Will be assigned after sorting
                    max_utilization_pct=round(max_util, 1),
                    total_overload_hours=round(total_overload, 2),
                    overloaded_periods_count=len(incidents),
                    overloaded_periods=[inc.period for inc in incidents],
                    total_horizon_load=round(total_load, 2),
                    total_horizon_capacity=round(total_capacity, 2),
                    average_utilization_pct=avg_util,
                    primary_cost_driver_items=top_items[:3],
                    incidents=incidents,
                ))

        # Rank bottlenecks: sort descending by total overload hours, then max utilization
        summaries.sort(
            key=lambda b: (b.total_overload_hours, b.max_utilization_pct),
            reverse=True,
        )

        for i, s in enumerate(summaries, start=1):
            s.rank = i

        return summaries

    def generate_bottleneck_report(
        self,
        bottlenecks: List[BottleneckSummary],
    ) -> pd.DataFrame:
        """Convert bottleneck summaries into a structured diagnostic DataFrame."""
        if not bottlenecks:
            return pd.DataFrame(columns=[
                "rank", "work_center_id", "max_utilization_pct", "total_overload_hours",
                "overloaded_periods_count", "overloaded_periods", "primary_drivers",
            ])

        rows = []
        for b in bottlenecks:
            rows.append({
                "rank": b.rank,
                "work_center_id": b.work_center_id,
                "max_utilization_pct": b.max_utilization_pct,
                "total_overload_hours": b.total_overload_hours,
                "overloaded_periods_count": b.overloaded_periods_count,
                "overloaded_periods": str(b.overloaded_periods),
                "primary_drivers": ", ".join(b.primary_cost_driver_items),
            })
        return pd.DataFrame(rows)

