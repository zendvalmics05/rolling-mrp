"""Plan-vs-Actual Reconciliation Engine for ROLLINGMRP.

Implements Phase 6.4:
- Plan vs. actual comparison: quantity, timing, cost.
- Adherence metrics: % orders on time, % quantity fulfilled, service level.
- Variance analysis: identify systematic biases (lead-time bias, quantity bias,
  item-level bias) via statistical tests and trend detection.
- Feedback loop: update lead times and safety stock based on observed actuals.

Mathematical foundations:
  - On-time delivery rate  OTD = N_on_time / N_total
  - Quantity fill rate     QFR = Σ(min(actual_qty, planned_qty)) / Σ(planned_qty)
  - Lead-time bias         LTB = mean(actual_period − planned_period)
  - Quantity bias           QB = mean((actual_qty − planned_qty) / planned_qty)
  - Safety stock update    SS = z · σ_d · √(L + δL)
    where δL is the observed mean lead-time deviation and σ_d is demand std-dev.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.integration.controller import InventoryState
from src.integration.des_link import (
    DeviationAnalyzer,
    DeviationReport,
    ExecutionRecord,
    OrderStatus,
    PlannedOrderExport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class AdherenceMetrics:
    """Aggregate plan-adherence KPIs.

    Attributes:
        total_orders: Total planned orders evaluated.
        on_time_count: Orders completed in the planned period.
        on_time_pct: On-time delivery percentage.
        late_count: Orders completed after the planned period.
        late_pct: Late delivery percentage.
        early_count: Orders completed before the planned period.
        quantity_fill_rate: Σ min(actual, planned) / Σ planned.
        perfect_order_pct: % of orders that are both on-time AND full-quantity.
        partial_count: Orders with actual < planned quantity.
        scrapped_count: Orders fully scrapped.
        cancelled_count: Orders with no execution record.
    """
    total_orders: int
    on_time_count: int
    on_time_pct: float
    late_count: int
    late_pct: float
    early_count: int
    quantity_fill_rate: float
    perfect_order_pct: float
    partial_count: int
    scrapped_count: int
    cancelled_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VarianceRecord:
    """Per-order variance detail.

    Attributes:
        order_id: Order reference.
        item_id: Item reference.
        planned_quantity: Planned quantity.
        actual_quantity: Actual quantity produced/received.
        quantity_variance: actual − planned.
        quantity_variance_pct: Signed % variance.
        planned_period: Due period.
        actual_period: Actual completion period.
        timing_variance: actual_period − planned_period (signed, in periods).
        planned_cost: Planned cost (quantity × unit_cost).
        actual_cost: Actual cost (actual_qty × unit_cost + scrap penalty).
        cost_variance: actual_cost − planned_cost.
    """
    order_id: str
    item_id: str
    planned_quantity: float
    actual_quantity: float
    quantity_variance: float
    quantity_variance_pct: float
    planned_period: int
    actual_period: int
    timing_variance: int
    planned_cost: float = 0.0
    actual_cost: float = 0.0
    cost_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BiasAnalysis:
    """Statistical bias analysis across all orders.

    Attributes:
        lead_time_bias: Mean signed timing deviation (positive = consistently late).
        lead_time_std: Std-dev of timing deviations.
        quantity_bias: Mean signed quantity deviation as % of planned.
        quantity_bias_std: Std-dev of quantity deviation %.
        is_lead_time_biased: True if |lead_time_bias| > bias_threshold_periods.
        is_quantity_biased: True if |quantity_bias| > bias_threshold_pct.
        bias_direction_timing: 'late', 'early', or 'neutral'.
        bias_direction_quantity: 'over', 'under', or 'neutral'.
        item_biases: Per-item bias summary {item_id → {lead_time_bias, quantity_bias}}.
        sample_size: Number of data points used.
    """
    lead_time_bias: float
    lead_time_std: float
    quantity_bias: float
    quantity_bias_std: float
    is_lead_time_biased: bool
    is_quantity_biased: bool
    bias_direction_timing: str
    bias_direction_quantity: str
    item_biases: Dict[str, Dict[str, float]]
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackAdjustment:
    """Recommended parameter adjustments from the feedback loop.

    Attributes:
        item_id: Item for which the adjustment applies.
        current_lead_time: Current lead-time parameter (periods).
        recommended_lead_time: Adjusted lead time incorporating observed bias.
        lead_time_delta: Change in lead time (signed).
        current_safety_stock: Current safety stock level.
        recommended_safety_stock: New safety stock from SS = z · σ_d · √(L_adj).
        safety_stock_delta: Change in safety stock (signed).
        rationale: Human-readable explanation.
    """
    item_id: str
    current_lead_time: int
    recommended_lead_time: int
    lead_time_delta: int
    current_safety_stock: float
    recommended_safety_stock: float
    safety_stock_delta: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationReport:
    """Full reconciliation output combining all analysis facets.

    Attributes:
        timestamp: ISO-8601 wall-clock time of the reconciliation.
        adherence: Aggregate adherence metrics.
        variance_records: Per-order variance details.
        bias_analysis: Statistical bias analysis.
        feedback_adjustments: Recommended parameter changes.
        cost_summary: {total_planned_cost, total_actual_cost, total_cost_variance}.
    """
    timestamp: str
    adherence: AdherenceMetrics
    variance_records: List[VarianceRecord]
    bias_analysis: BiasAnalysis
    feedback_adjustments: List[FeedbackAdjustment]
    cost_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "adherence": self.adherence.to_dict(),
            "variance_records": [v.to_dict() for v in self.variance_records],
            "bias_analysis": self.bias_analysis.to_dict(),
            "feedback_adjustments": [f.to_dict() for f in self.feedback_adjustments],
            "cost_summary": self.cost_summary,
        }


# ---------------------------------------------------------------------------
# Reconciliation Engine
# ---------------------------------------------------------------------------


class ReconciliationEngine:
    """Compares planned orders against execution actuals and feeds back parameter updates.

    Parameters
    ----------
    unit_costs:
        Dict[item_id → cost_per_unit] for cost-variance calculations.
        If not supplied, cost analysis defaults to 0.
    scrap_penalty_factor:
        Multiplier applied to scrap quantity for cost penalty (default 1.5×).
    bias_threshold_periods:
        Minimum absolute mean timing deviation to flag systematic bias (default 0.5).
    bias_threshold_pct:
        Minimum absolute mean quantity deviation % to flag systematic bias (default 5.0).
    safety_stock_z:
        Z-score for safety-stock calculation (default 1.65 ≈ 95% CSL).
    """

    def __init__(
        self,
        unit_costs: Optional[Dict[str, float]] = None,
        scrap_penalty_factor: float = 1.5,
        bias_threshold_periods: float = 0.5,
        bias_threshold_pct: float = 5.0,
        safety_stock_z: float = 1.65,
    ) -> None:
        self.unit_costs = unit_costs or {}
        self.scrap_penalty_factor = scrap_penalty_factor
        self.bias_threshold_periods = bias_threshold_periods
        self.bias_threshold_pct = bias_threshold_pct
        self.safety_stock_z = safety_stock_z

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(
        self,
        planned: List[PlannedOrderExport],
        actuals: List[ExecutionRecord],
        inventory_states: Optional[Dict[str, InventoryState]] = None,
        demand_history: Optional[Dict[str, List[float]]] = None,
    ) -> ReconciliationReport:
        """Run full reconciliation: adherence → variance → bias → feedback.

        Parameters
        ----------
        planned:
            List of planned orders from the MRP system.
        actuals:
            List of execution records from DES / shop floor.
        inventory_states:
            Current inventory state per item (used to read current lead_time
            and safety_stock for the feedback loop).
        demand_history:
            Dict[item_id → List[float]] historical demand observations per
            item, used to compute demand std-dev for safety-stock updates.

        Returns
        -------
        :class:`ReconciliationReport` with all analysis facets.
        """
        timestamp = datetime.datetime.now().isoformat()

        # 1. Adherence
        adherence = self.compute_adherence(planned, actuals)

        # 2. Variance (per-order)
        variance_records = self.compute_variance(planned, actuals)

        # 3. Cost summary
        cost_summary = self._aggregate_cost(variance_records)

        # 4. Bias analysis
        bias_analysis = self.analyze_bias(variance_records)

        # 5. Feedback loop
        feedback = self.generate_feedback(
            bias_analysis=bias_analysis,
            inventory_states=inventory_states,
            demand_history=demand_history,
        )

        return ReconciliationReport(
            timestamp=timestamp,
            adherence=adherence,
            variance_records=variance_records,
            bias_analysis=bias_analysis,
            feedback_adjustments=feedback,
            cost_summary=cost_summary,
        )

    def compute_adherence(
        self,
        planned: List[PlannedOrderExport],
        actuals: List[ExecutionRecord],
    ) -> AdherenceMetrics:
        """Compute aggregate adherence metrics.

        OTD = on_time_count / total evaluated
        QFR = Σ min(actual, planned) / Σ planned
        Perfect order = on-time AND actual ≥ planned
        """
        planned_map = {o.order_id: o for o in planned}
        actual_map = {r.order_id: r for r in actuals}

        total = len(planned)
        on_time = late = early = partial = scrapped = cancelled = perfect = 0
        sum_min_actual_planned = 0.0
        sum_planned_qty = 0.0

        for po in planned:
            sum_planned_qty += po.quantity
            rec = actual_map.get(po.order_id)

            if rec is None:
                cancelled += 1
                continue

            sum_min_actual_planned += min(rec.actual_quantity, po.quantity)

            if rec.status == OrderStatus.SCRAPPED:
                scrapped += 1
                continue

            delay = rec.delay_periods
            is_on_time = (delay == 0)
            is_full = (rec.actual_quantity >= po.quantity - 1e-6)

            if is_on_time:
                on_time += 1
            elif delay > 0:
                late += 1
            else:
                early += 1

            if rec.actual_quantity < po.quantity - 1e-6:
                partial += 1

            if is_on_time and is_full:
                perfect += 1

        evaluated = total - cancelled
        on_time_pct = (on_time / evaluated * 100.0) if evaluated > 0 else 0.0
        late_pct = (late / evaluated * 100.0) if evaluated > 0 else 0.0
        qfr = (sum_min_actual_planned / sum_planned_qty) if sum_planned_qty > 0 else 1.0
        perfect_pct = (perfect / evaluated * 100.0) if evaluated > 0 else 0.0

        return AdherenceMetrics(
            total_orders=total,
            on_time_count=on_time,
            on_time_pct=round(on_time_pct, 2),
            late_count=late,
            late_pct=round(late_pct, 2),
            early_count=early,
            quantity_fill_rate=round(qfr, 4),
            perfect_order_pct=round(perfect_pct, 2),
            partial_count=partial,
            scrapped_count=scrapped,
            cancelled_count=cancelled,
        )

    def compute_variance(
        self,
        planned: List[PlannedOrderExport],
        actuals: List[ExecutionRecord],
    ) -> List[VarianceRecord]:
        """Compute per-order variance details including cost.

        For each planned order with a matching execution record:
          quantity_variance = actual − planned
          timing_variance   = actual_period − planned_period
          planned_cost      = planned_qty × unit_cost
          actual_cost       = actual_qty × unit_cost + scrap × unit_cost × scrap_factor
          cost_variance     = actual_cost − planned_cost
        """
        actual_map = {r.order_id: r for r in actuals}
        records: List[VarianceRecord] = []

        for po in planned:
            rec = actual_map.get(po.order_id)

            if rec is None:
                # Cancelled / no execution — full negative variance
                unit_cost = self.unit_costs.get(po.item_id, 0.0)
                planned_cost = po.quantity * unit_cost
                records.append(VarianceRecord(
                    order_id=po.order_id,
                    item_id=po.item_id,
                    planned_quantity=po.quantity,
                    actual_quantity=0.0,
                    quantity_variance=-po.quantity,
                    quantity_variance_pct=-100.0,
                    planned_period=po.due_period,
                    actual_period=po.due_period,  # N/A
                    timing_variance=0,
                    planned_cost=round(planned_cost, 2),
                    actual_cost=0.0,
                    cost_variance=round(-planned_cost, 2),
                ))
                continue

            qty_var = rec.actual_quantity - po.quantity
            qty_var_pct = (qty_var / po.quantity * 100.0) if po.quantity > 0 else 0.0
            timing_var = rec.delay_periods

            unit_cost = self.unit_costs.get(po.item_id, 0.0)
            planned_cost = po.quantity * unit_cost
            scrap_cost = rec.scrap_quantity * unit_cost * self.scrap_penalty_factor
            actual_cost = rec.actual_quantity * unit_cost + scrap_cost
            cost_var = actual_cost - planned_cost

            records.append(VarianceRecord(
                order_id=po.order_id,
                item_id=po.item_id,
                planned_quantity=po.quantity,
                actual_quantity=rec.actual_quantity,
                quantity_variance=round(qty_var, 4),
                quantity_variance_pct=round(qty_var_pct, 2),
                planned_period=po.due_period,
                actual_period=rec.actual_period,
                timing_variance=timing_var,
                planned_cost=round(planned_cost, 2),
                actual_cost=round(actual_cost, 2),
                cost_variance=round(cost_var, 2),
            ))

        return records

    def analyze_bias(
        self,
        variance_records: List[VarianceRecord],
    ) -> BiasAnalysis:
        """Identify systematic biases across all variance records.

        Lead-time bias: mean signed timing variance.
        Quantity bias: mean signed quantity variance %.

        A bias is flagged when the absolute mean exceeds the configured threshold.
        """
        # Filter to records that have actual execution (not cancelled)
        executed = [v for v in variance_records if v.actual_quantity > 0 or v.timing_variance != 0]

        if not executed:
            return BiasAnalysis(
                lead_time_bias=0.0,
                lead_time_std=0.0,
                quantity_bias=0.0,
                quantity_bias_std=0.0,
                is_lead_time_biased=False,
                is_quantity_biased=False,
                bias_direction_timing="neutral",
                bias_direction_quantity="neutral",
                item_biases={},
                sample_size=0,
            )

        timing_devs = [float(v.timing_variance) for v in executed]
        qty_dev_pcts = [float(v.quantity_variance_pct) for v in executed]

        lt_bias = float(np.mean(timing_devs))
        lt_std = float(np.std(timing_devs, ddof=1)) if len(timing_devs) > 1 else 0.0
        q_bias = float(np.mean(qty_dev_pcts))
        q_std = float(np.std(qty_dev_pcts, ddof=1)) if len(qty_dev_pcts) > 1 else 0.0

        is_lt_biased = abs(lt_bias) > self.bias_threshold_periods
        is_q_biased = abs(q_bias) > self.bias_threshold_pct

        if lt_bias > self.bias_threshold_periods:
            lt_direction = "late"
        elif lt_bias < -self.bias_threshold_periods:
            lt_direction = "early"
        else:
            lt_direction = "neutral"

        if q_bias > self.bias_threshold_pct:
            q_direction = "over"
        elif q_bias < -self.bias_threshold_pct:
            q_direction = "under"
        else:
            q_direction = "neutral"

        # Per-item bias
        item_timing: Dict[str, List[float]] = {}
        item_qty: Dict[str, List[float]] = {}
        for v in executed:
            item_timing.setdefault(v.item_id, []).append(float(v.timing_variance))
            item_qty.setdefault(v.item_id, []).append(float(v.quantity_variance_pct))

        item_biases: Dict[str, Dict[str, float]] = {}
        for iid in item_timing:
            item_biases[iid] = {
                "lead_time_bias": round(float(np.mean(item_timing[iid])), 3),
                "quantity_bias": round(float(np.mean(item_qty[iid])), 3),
                "sample_size": len(item_timing[iid]),
            }

        return BiasAnalysis(
            lead_time_bias=round(lt_bias, 3),
            lead_time_std=round(lt_std, 3),
            quantity_bias=round(q_bias, 3),
            quantity_bias_std=round(q_std, 3),
            is_lead_time_biased=is_lt_biased,
            is_quantity_biased=is_q_biased,
            bias_direction_timing=lt_direction,
            bias_direction_quantity=q_direction,
            item_biases=item_biases,
            sample_size=len(executed),
        )

    def generate_feedback(
        self,
        bias_analysis: BiasAnalysis,
        inventory_states: Optional[Dict[str, InventoryState]] = None,
        demand_history: Optional[Dict[str, List[float]]] = None,
    ) -> List[FeedbackAdjustment]:
        """Generate recommended lead-time and safety-stock adjustments.

        For each item with observed bias:
          - Lead-time adjustment: L_new = max(1, L_current + ceil(lead_time_bias))
            Applied only when the item shows a significant lead-time bias.
          - Safety-stock adjustment: SS = z · σ_d · √(L_adj)
            where σ_d is computed from demand_history (or estimated from
            quantity bias when no history is available), and L_adj includes the
            lead-time bias correction.

        Parameters
        ----------
        bias_analysis:
            Output of :meth:`analyze_bias`.
        inventory_states:
            Current inventory parameters per item.
        demand_history:
            Historical demand per item (needed for σ_d computation).

        Returns
        -------
        List of :class:`FeedbackAdjustment` — one per item with a non-trivial
        adjustment. Empty if no systematic bias is detected.
        """
        if not inventory_states:
            return []

        adjustments: List[FeedbackAdjustment] = []
        demand_hist = demand_history or {}

        for item_id, inv in inventory_states.items():
            item_bias = bias_analysis.item_biases.get(item_id)
            if item_bias is None:
                continue

            item_lt_bias = item_bias.get("lead_time_bias", 0.0)
            item_q_bias = item_bias.get("quantity_bias", 0.0)

            # --- Lead-time feedback ---
            current_lt = inv.lead_time
            lt_delta = 0
            rationale_parts: List[str] = []

            if abs(item_lt_bias) > self.bias_threshold_periods:
                # Round towards the bias direction
                lt_delta = int(np.ceil(abs(item_lt_bias))) * (1 if item_lt_bias > 0 else -1)
                recommended_lt = max(1, current_lt + lt_delta)
                lt_delta = recommended_lt - current_lt
                rationale_parts.append(
                    f"Lead-time bias of {item_lt_bias:+.2f} periods detected; "
                    f"adjusting L from {current_lt} to {recommended_lt}."
                )
            else:
                recommended_lt = current_lt

            # Only calculate/adjust safety stock if there is a detected bias (lead time or quantity)
            has_item_bias = (
                abs(item_lt_bias) > self.bias_threshold_periods
                or abs(item_q_bias) > self.bias_threshold_pct
            )

            current_ss = inv.safety_stock
            recommended_ss = current_ss
            ss_delta = 0.0

            if has_item_bias:
                hist = demand_hist.get(item_id, inv.demand_history)

                if len(hist) >= 2:
                    sigma_d = float(np.std(hist, ddof=1))
                else:
                    # Estimate sigma from quantity bias: assume CV ≈ |q_bias|/100
                    avg_demand = float(np.mean(hist)) if hist else max(1.0, current_ss)
                    sigma_d = avg_demand * max(0.05, abs(item_q_bias) / 100.0)

                l_adj = max(1, recommended_lt)
                calc_ss = round(self.safety_stock_z * sigma_d * np.sqrt(l_adj), 2)
                diff_ss = round(calc_ss - current_ss, 2)

                if abs(diff_ss) > 0.5:
                    recommended_ss = calc_ss
                    ss_delta = diff_ss
                    rationale_parts.append(
                        f"Safety stock: SS = {self.safety_stock_z} × {sigma_d:.2f} × √{l_adj} "
                        f"= {recommended_ss:.2f} (Δ{ss_delta:+.2f})."
                    )

            # Only emit adjustment if something changed
            if lt_delta != 0 or ss_delta != 0.0:
                adjustments.append(FeedbackAdjustment(
                    item_id=item_id,
                    current_lead_time=current_lt,
                    recommended_lead_time=recommended_lt,
                    lead_time_delta=lt_delta,
                    current_safety_stock=current_ss,
                    recommended_safety_stock=recommended_ss,
                    safety_stock_delta=ss_delta,
                    rationale=" ".join(rationale_parts) if rationale_parts else "No significant bias.",
                ))

        return adjustments

    def apply_feedback(
        self,
        adjustments: List[FeedbackAdjustment],
        inventory_states: Dict[str, InventoryState],
    ) -> Dict[str, Dict[str, Any]]:
        """Apply recommended feedback adjustments to inventory state parameters.

        Parameters
        ----------
        adjustments:
            Output of :meth:`generate_feedback`.
        inventory_states:
            Mutable inventory states to update in place.

        Returns
        -------
        Dict of applied changes: {item_id → {field → {old, new}}}.
        """
        applied: Dict[str, Dict[str, Any]] = {}

        for adj in adjustments:
            inv = inventory_states.get(adj.item_id)
            if inv is None:
                continue

            changes: Dict[str, Any] = {}

            if adj.lead_time_delta != 0:
                old_lt = inv.lead_time
                inv.lead_time = adj.recommended_lead_time
                changes["lead_time"] = {"old": old_lt, "new": inv.lead_time}

            if adj.safety_stock_delta != 0.0:
                old_ss = inv.safety_stock
                inv.safety_stock = adj.recommended_safety_stock
                changes["safety_stock"] = {"old": old_ss, "new": inv.safety_stock}

            if changes:
                applied[adj.item_id] = changes
                logger.info(
                    f"Feedback applied to {adj.item_id}: {changes} — {adj.rationale}"
                )

        return applied

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_cost(
        variance_records: List[VarianceRecord],
    ) -> Dict[str, float]:
        """Sum planned, actual, and variance costs."""
        total_planned = sum(v.planned_cost for v in variance_records)
        total_actual = sum(v.actual_cost for v in variance_records)
        return {
            "total_planned_cost": round(total_planned, 2),
            "total_actual_cost": round(total_actual, 2),
            "total_cost_variance": round(total_actual - total_planned, 2),
        }

