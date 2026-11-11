"""Closed-Loop MRP with Capacity-Aware Validation and Leveling (Phase 4.4).

Implements:
- Post-MRP capacity validation: checks generated planned orders against work center capacities.
- Capacity exception generation: creates standardized exception messages for work center overloads.
- Auto-leveling workflow: optionally runs leveling heuristics and re-verifies feasibility.
- Iterative loop: MRP -> RCCP -> Bottleneck Detection -> Leveling Heuristics -> Re-verify.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.core.explosion import BOMExplosionEngine
from src.core.netting import NettingResult
from src.capacity.rccp import RCCPEngine, PeriodWorkCenterLoad
from src.capacity.bottleneck import BottleneckDetector, BottleneckSummary
from src.capacity.leveling import CapacityLevelingEngine, LevelingResult, LevelingAction


@dataclass
class CapacityExceptionMessage:
    """Actionable warning produced when capacity is violated."""
    exception_id: str
    work_center_id: str
    period: int
    overload_hours: float
    utilization_pct: float
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    suggested_action: str
    impacted_items: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapacityAwareMRPResult:
    """Consolidated output from capacity-aware MRP closed loop."""
    is_capacity_feasible: bool
    initial_planned_orders: List[Dict[str, Any]]
    final_planned_orders: List[Dict[str, Any]]
    initial_bottlenecks: List[BottleneckSummary]
    final_bottlenecks: List[BottleneckSummary]
    capacity_exceptions: List[CapacityExceptionMessage]
    leveling_actions: List[LevelingAction]
    initial_load_matrix: Dict[str, Dict[int, PeriodWorkCenterLoad]]
    final_load_matrix: Dict[str, Dict[int, PeriodWorkCenterLoad]]
    total_prebuild_units: float = 0.0
    recommended_overtime: Dict[str, Dict[int, float]] = field(default_factory=dict)


class CapacityAwareMRPPipeline:
    """Coordinates iterative closed-loop planning between MRP and finite capacity constraints."""

    def __init__(
        self,
        explosion_engine: BOMExplosionEngine,
        rccp_engine: RCCPEngine,
        leveling_engine: Optional[CapacityLevelingEngine] = None,
        utilization_threshold_pct: float = 100.0,
    ):
        self.explosion_engine = explosion_engine
        self.rccp_engine = rccp_engine
        self.leveling_engine = leveling_engine or CapacityLevelingEngine(rccp_engine)
        self.detector = BottleneckDetector(utilization_threshold_pct=utilization_threshold_pct)

    def run_capacity_aware_mrp(
        self,
        mps_gross_requirements: Dict[str, List[float]],
        periods: List[int],
        inventory_status: Optional[Dict[str, Dict[str, Any]]] = None,
        scheduled_receipts: Optional[Dict[str, Dict[int, float]]] = None,
        auto_level: bool = True,
        allow_overtime: bool = True,
        timing_key: str = "due_period",
    ) -> CapacityAwareMRPResult:
        """Execute closed-loop MRP: run MRP explosion, check RCCP, level overloads, and re-verify.

        Args:
            mps_gross_requirements: Dict mapping finished good item_id -> list of gross demands.
            periods: Sequence of period indices for planning horizon.
            inventory_status: Inventory starting balances and safety stock.
            scheduled_receipts: Pre-existing scheduled receipts.
            auto_level: Whether to automatically apply leveling heuristics to resolve overloads.
            allow_overtime: Whether to allow overtime hours if pre-build capacity is insufficient.
            timing_key: Period timing reference ('due_period' or 'release_period').

        Returns:
            CapacityAwareMRPResult instance.
        """
        # Step 1: Run unconstrained multi-level MRP
        mrp_results, pegging_report = self.explosion_engine.run_multi_level_mrp(
            mps_gross_requirements=mps_gross_requirements,
            periods=periods,
            inventory_status=inventory_status,
            scheduled_receipts=scheduled_receipts,
        )

        # Step 2: Extract planned orders
        initial_orders: List[Dict[str, Any]] = []
        for itm, res in mrp_results.items():
            for po in res.planned_orders:
                initial_orders.append({
                    "item_id": po.item_id,
                    "order_id": getattr(po, "order_id", f"PO-{po.item_id}-{po.due_period}"),
                    "due_period": po.due_period,
                    "release_period": po.release_period,
                    "period": po.due_period if timing_key == "due_period" else po.release_period,
                    "quantity": po.quantity,
                    "status": po.status,
                })

        # Step 3: Compute initial RCCP loads & detect bottlenecks
        initial_loads = self.rccp_engine.calculate_plan_load(
            initial_orders,
            periods=periods,
            timing_key=timing_key,
        )
        initial_bottlenecks = self.detector.detect_bottlenecks(initial_loads)

        # Step 4: Generate Capacity Exception Messages
        exceptions: List[CapacityExceptionMessage] = []
        exc_count = 0
        for b in initial_bottlenecks:
            for inc in b.incidents:
                exc_count += 1
                severity = "CRITICAL" if inc.utilization_pct > 120.0 else "HIGH"
                drivers = [d["item_id"] for d in inc.top_drivers]
                msg = (
                    f"Overload of {inc.overload_hours:.1f} hrs on {inc.work_center_id} "
                    f"in period {inc.period} ({inc.utilization_pct:.1f}%). "
                    f"Consider backward pre-building or authorizing overtime."
                )
                exceptions.append(CapacityExceptionMessage(
                    exception_id=f"CAP-EXC-{exc_count:04d}",
                    work_center_id=inc.work_center_id,
                    period=inc.period,
                    overload_hours=inc.overload_hours,
                    utilization_pct=inc.utilization_pct,
                    severity=severity,
                    suggested_action=msg,
                    impacted_items=drivers,
                ))

        # Step 5: Leveling (if auto_level enabled and overloads detected)
        if auto_level and initial_bottlenecks:
            level_res = self.leveling_engine.level_capacity_plan(
                planned_orders=initial_orders,
                periods=periods,
                timing_key=timing_key,
                allow_overtime=allow_overtime,
            )
            final_orders = level_res.leveled_orders
            leveling_actions = level_res.actions
            final_loads = self.rccp_engine.calculate_plan_load(
                final_orders,
                periods=periods,
                timing_key=timing_key,
            )
            final_bottlenecks = self.detector.detect_bottlenecks(final_loads)
            is_feasible = level_res.is_feasible
            total_prebuild = level_res.total_prebuilt_inventory_units
            ot_recs = level_res.recommended_overtime_hours
        else:
            final_orders = initial_orders
            leveling_actions = []
            final_loads = initial_loads
            final_bottlenecks = initial_bottlenecks
            is_feasible = len(initial_bottlenecks) == 0
            total_prebuild = 0.0
            ot_recs = {}

        return CapacityAwareMRPResult(
            is_capacity_feasible=is_feasible,
            initial_planned_orders=initial_orders,
            final_planned_orders=final_orders,
            initial_bottlenecks=initial_bottlenecks,
            final_bottlenecks=final_bottlenecks,
            capacity_exceptions=exceptions,
            leveling_actions=leveling_actions,
            initial_load_matrix=initial_loads,
            final_load_matrix=final_loads,
            total_prebuild_units=total_prebuild,
            recommended_overtime=ot_recs,
        )
