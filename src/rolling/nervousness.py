"""Nervousness Quantification and Plan Stability Engine for ROLLINGMRP.

Implements:
- Phase 3.1: Nervousness Metrics:
  - Order Churn: Σ|Q_new - Q_old| / ΣQ_old across overlapping periods
  - Timing Churn: Count of due-date shifts > 1 period (normalized to [0, 1])
  - Cancellation Rate: cancelled orders / total orders in previous plan
  - Nervousness Index: Weighted composite (0.5 * churn + 0.3 * timing + 0.2 * cancellation)
  - Time-series metric history storage (DataFrame / CSV / Database)
- Phase 3.2: Trend Analysis and Alerting:
  - Trend direction detection (improving, stable, declining)
  - Threshold alerting (flags when nervousness_index > threshold, default 0.15)
  - Visualization data preparation for dashboard charts
- Phase 3.3: Mitigation Policies (foundation):
  - Change threshold filtering
  - Demand smoothing (exponential smoothing)
  - Dynamic safety stock buffer
  - Frozen window extension recommendation
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.rolling.frozen_window import PlanChangeRecord, ChangeType


@dataclass
class NervousnessMetrics:
    """Calculated nervousness metrics for a planning cycle or plan comparison."""
    cycle: int
    order_churn: float
    timing_churn: float
    cancellation_rate: float
    nervousness_index: float
    total_previous_orders: int = 0
    total_new_orders: int = 0
    cancelled_orders: int = 0
    shifted_orders: int = 0
    quantity_changes: int = 0
    unaltered_orders: int = 0
    timestamp: Optional[str] = None
    item_id: Optional[str] = None  # None if aggregate across all items

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class TrendAnalysisReport:
    """Report describing nervousness trajectory over recent planning cycles."""
    current_cycle: int
    current_index: float
    trend_direction: str  # improving, stable, declining
    slope: float
    is_alert: bool
    threshold: float
    average_index: float
    min_index: float
    max_index: float
    recommendation: str
    recent_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NervousnessAnalyzer:
    """Quantifies schedule instability (nervousness) across rolling planning cycles."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        alert_threshold: float = 0.15,
    ):
        """
        Args:
            weights: Dictionary of weights for composite index.
                     Defaults to {'order_churn': 0.5, 'timing_churn': 0.3, 'cancellation_rate': 0.2}.
            alert_threshold: Nervousness index threshold triggering an alert (default 0.15).
        """
        self.weights = weights or {
            "order_churn": 0.5,
            "timing_churn": 0.3,
            "cancellation_rate": 0.2,
        }
        # Normalize weights to sum to 1.0
        w_sum = sum(self.weights.values())
        if w_sum > 0:
            self.weights = {k: v / w_sum for k, v in self.weights.items()}
        self.alert_threshold = alert_threshold
        self.history: List[NervousnessMetrics] = []

    def calculate_metrics(
        self,
        previous_plan: List[Dict[str, Any]],
        new_plan: List[Dict[str, Any]],
        cycle: int = 1,
        overlapping_periods: Optional[List[int]] = None,
        item_id: Optional[str] = None,
    ) -> NervousnessMetrics:
        """Calculate order churn, timing churn, cancellation rate, and composite index.

        Args:
            previous_plan: List of planned orders from previous cycle.
                           Must have 'item_id', 'due_period', 'quantity'.
            new_plan: List of planned orders from current cycle.
            cycle: Planning cycle index or simulation period.
            overlapping_periods: Specific periods to evaluate. If None, uses intersection
                                of periods present in both plans or common horizon.
            item_id: Optional item filter to compute item-specific metrics.

        Returns:
            NervousnessMetrics instance.
        """
        # Filter by item if requested
        prev_orders = [
            dict(o) for o in previous_plan
            if item_id is None or o.get("item_id") == item_id
        ]
        curr_orders = [
            dict(o) for o in new_plan
            if item_id is None or o.get("item_id") == item_id
        ]

        if not prev_orders and not curr_orders:
            metrics = NervousnessMetrics(
                cycle=cycle,
                order_churn=0.0,
                timing_churn=0.0,
                cancellation_rate=0.0,
                nervousness_index=0.0,
                item_id=item_id,
            )
            self.history.append(metrics)
            return metrics

        # Index orders by (item_id, due_period) -> quantity
        prev_dict: Dict[Tuple[str, int], float] = {}
        for o in prev_orders:
            key = (o["item_id"], int(o["due_period"]))
            prev_dict[key] = prev_dict.get(key, 0.0) + float(o["quantity"])

        curr_dict: Dict[Tuple[str, int], float] = {}
        for o in curr_orders:
            key = (o["item_id"], int(o["due_period"]))
            curr_dict[key] = curr_dict.get(key, 0.0) + float(o["quantity"])

        # Determine evaluation periods
        prev_periods = {k[1] for k in prev_dict.keys()}
        curr_periods = {k[1] for k in curr_dict.keys()}
        all_periods = prev_periods.union(curr_periods)

        if overlapping_periods is not None:
            eval_periods = set(overlapping_periods)
        else:
            # Overlapping periods between previous and new plan
            eval_periods = prev_periods.intersection(curr_periods)
            if not eval_periods:
                # If no direct period intersection, fall back to all observed periods
                eval_periods = all_periods

        # 1. Order Churn: Σ|Q_new − Q_old| / ΣQ_old across overlapping periods
        # For order churn, we evaluate items in the overlapping periods
        relevant_keys = {
            k for k in set(prev_dict.keys()).union(set(curr_dict.keys()))
            if k[1] in eval_periods
        }

        sum_abs_delta = 0.0
        sum_q_old = 0.0
        qty_change_count = 0

        for k in relevant_keys:
            q_old = prev_dict.get(k, 0.0)
            q_new = curr_dict.get(k, 0.0)
            delta = abs(q_new - q_old)
            sum_abs_delta += delta
            sum_q_old += q_old
            if q_old > 0 and q_new > 0 and abs(q_new - q_old) > 1e-4:
                qty_change_count += 1

        if sum_q_old > 0:
            raw_churn = sum_abs_delta / sum_q_old
            # Order churn can exceed 1.0 if new demand doubles, bound for index: min(1.0, raw_churn)
            order_churn = min(1.0, raw_churn)
        elif sum_abs_delta > 0:
            order_churn = 1.0
        else:
            order_churn = 0.0

        # 2. Timing Churn and Cancellations:
        # Match orders across cycles by item and approximate quantity
        # An order shifted by > 1 period counts towards timing churn
        prev_unmatched = list(prev_orders)
        curr_unmatched = list(curr_orders)

        shifts_gt_1 = 0
        unaltered_count = 0

        # Exact matches (same item, same period, same qty)
        i = 0
        while i < len(prev_unmatched):
            p_ord = prev_unmatched[i]
            matched_idx = None
            for j, c_ord in enumerate(curr_unmatched):
                if (
                    p_ord["item_id"] == c_ord["item_id"]
                    and int(p_ord["due_period"]) == int(c_ord["due_period"])
                    and abs(float(p_ord["quantity"]) - float(c_ord["quantity"])) < 1e-4
                ):
                    matched_idx = j
                    break
            if matched_idx is not None:
                unaltered_count += 1
                prev_unmatched.pop(i)
                curr_unmatched.pop(matched_idx)
            else:
                i += 1

        # Only consider it a timing shift if the shift is reasonably local (e.g. within 3 periods)
        # or if the target period is within the planning window.
        i = 0
        while i < len(prev_unmatched):
            p_ord = prev_unmatched[i]
            matched_idx = None
            for j, c_ord in enumerate(curr_unmatched):
                if p_ord["item_id"] == c_ord["item_id"]:
                    shift_amt = abs(int(c_ord["due_period"]) - int(p_ord["due_period"]))
                    # Match if quantities are identical/very close and shift is reasonable (<= 3 periods)
                    if abs(float(p_ord["quantity"]) - float(c_ord["quantity"])) < 1e-4 and shift_amt <= 3:
                        matched_idx = j
                        if shift_amt > 1:
                            shifts_gt_1 += 1
                        break
            if matched_idx is not None:
                prev_unmatched.pop(i)
                curr_unmatched.pop(matched_idx)
            else:
                i += 1

        # Remaining previous orders are treated as cancelled
        cancelled_orders = len(prev_unmatched)
        total_prev_orders = max(1, len(prev_orders))

        # Cancellation rate: cancelled orders / total orders in previous plan
        cancellation_rate = min(1.0, cancelled_orders / total_prev_orders)
        timing_churn = min(1.0, shifts_gt_1 / total_prev_orders)

        # Composite Nervousness Index bounded [0, 1]
        nervousness_index = (
            self.weights["order_churn"] * order_churn
            + self.weights["timing_churn"] * timing_churn
            + self.weights["cancellation_rate"] * cancellation_rate
        )
        nervousness_index = max(0.0, min(1.0, nervousness_index))

        metrics = NervousnessMetrics(
            cycle=cycle,
            order_churn=round(order_churn, 4),
            timing_churn=round(timing_churn, 4),
            cancellation_rate=round(cancellation_rate, 4),
            nervousness_index=round(nervousness_index, 4),
            total_previous_orders=len(prev_orders),
            total_new_orders=len(curr_orders),
            cancelled_orders=cancelled_orders,
            shifted_orders=shifts_gt_1,
            quantity_changes=qty_change_count,
            unaltered_orders=unaltered_count,
            item_id=item_id,
        )

        self.history.append(metrics)
        return metrics

    def calculate_from_changes(
        self,
        change_records: List[PlanChangeRecord],
        previous_plan: List[Dict[str, Any]],
        new_plan: List[Dict[str, Any]],
        cycle: int = 1,
    ) -> NervousnessMetrics:
        """Calculate nervousness metrics directly from PlanChangeRecord audit trail."""
        total_prev = len(previous_plan)
        sum_q_old = sum(float(o.get("quantity", 0.0)) for o in previous_plan)

        cancelled_count = 0
        shifted_count = 0
        qty_change_count = 0
        sum_abs_delta = 0.0

        for r in change_records:
            if r.change_type == ChangeType.CANCELLED.value:
                cancelled_count += 1
                sum_abs_delta += abs(r.quantity_delta)
            elif r.change_type == ChangeType.TIMING_CHANGE.value:
                if r.old_period is not None and r.new_period is not None:
                    if abs(r.new_period - r.old_period) > 1:
                        shifted_count += 1
            elif r.change_type == ChangeType.QUANTITY_CHANGE.value:
                qty_change_count += 1
                sum_abs_delta += abs(r.quantity_delta)
            elif r.change_type == ChangeType.NEW_ORDER.value:
                sum_abs_delta += abs(r.new_quantity)

        order_churn = min(1.0, sum_abs_delta / sum_q_old) if sum_q_old > 0 else (1.0 if sum_abs_delta > 0 else 0.0)
        cancellation_rate = min(1.0, cancelled_count / total_prev) if total_prev > 0 else 0.0
        timing_churn = min(1.0, shifted_count / total_prev) if total_prev > 0 else 0.0

        nervousness_index = (
            self.weights["order_churn"] * order_churn
            + self.weights["timing_churn"] * timing_churn
            + self.weights["cancellation_rate"] * cancellation_rate
        )
        nervousness_index = max(0.0, min(1.0, nervousness_index))

        metrics = NervousnessMetrics(
            cycle=cycle,
            order_churn=round(order_churn, 4),
            timing_churn=round(timing_churn, 4),
            cancellation_rate=round(cancellation_rate, 4),
            nervousness_index=round(nervousness_index, 4),
            total_previous_orders=total_prev,
            total_new_orders=len(new_plan),
            cancelled_orders=cancelled_count,
            shifted_orders=shifted_count,
            quantity_changes=qty_change_count,
        )
        self.history.append(metrics)
        return metrics

    def analyze_trends(
        self,
        lookback_runs: int = 5,
        threshold: Optional[float] = None,
    ) -> TrendAnalysisReport:
        """Analyze nervousness trend across recent planning runs.

        Args:
            lookback_runs: Number of recent runs to evaluate.
            threshold: Alert threshold (overrides default if provided).

        Returns:
            TrendAnalysisReport detailing trend direction, slope, and alert status.
        """
        thresh = threshold if threshold is not None else self.alert_threshold

        if not self.history:
            return TrendAnalysisReport(
                current_cycle=0,
                current_index=0.0,
                trend_direction=TrendDirection.STABLE.value,
                slope=0.0,
                is_alert=False,
                threshold=thresh,
                average_index=0.0,
                min_index=0.0,
                max_index=0.0,
                recommendation="No history available to analyze trends.",
            )

        recent = self.history[-lookback_runs:]
        indices = [m.nervousness_index for m in recent]
        cycles = [m.cycle for m in recent]
        current_idx = indices[-1]
        current_cyc = cycles[-1]

        # Calculate slope via linear regression if >= 2 points
        if len(indices) >= 2:
            x = np.array(range(len(indices)), dtype=float)
            y = np.array(indices, dtype=float)
            # Covariance / Variance
            slope, _ = np.polyfit(x, y, 1)
        else:
            slope = 0.0

        # Classify trend direction
        # Negative slope means nervousness is dropping -> IMPROVING
        # Positive slope means nervousness is rising -> DECLINING
        # Near zero -> STABLE
        if slope < -0.015:
            trend_direction = TrendDirection.IMPROVING.value
        elif slope > 0.015:
            trend_direction = TrendDirection.DECLINING.value
        else:
            trend_direction = TrendDirection.STABLE.value

        is_alert = current_idx > thresh

        # Formulate actionable recommendation
        if is_alert and trend_direction == TrendDirection.DECLINING.value:
            recommendation = (
                f"CRITICAL: Nervousness index ({current_idx:.3f}) exceeds threshold ({thresh:.3f}) "
                "and is worsening. Extend frozen window or enforce change thresholds immediately."
            )
        elif is_alert:
            recommendation = (
                f"ALERT: Nervousness index ({current_idx:.3f}) exceeds threshold ({thresh:.3f}). "
                "Review slushy approvals and consider input demand smoothing."
            )
        elif trend_direction == TrendDirection.DECLINING.value:
            recommendation = (
                f"CAUTION: Plan stability is declining (slope: {slope:+.3f}/run). "
                "Monitor upcoming replanning cycles for forecast fluctuations."
            )
        elif trend_direction == TrendDirection.IMPROVING.value:
            recommendation = (
                f"OPTIMAL: Plan stability is improving (slope: {slope:+.3f}/run). "
                "Time fences and lot-sizing rules are effectively damping perturbations."
            )
        else:
            recommendation = (
                f"NOMINAL: Plan stability is stable around index {np.mean(indices):.3f}. "
                "Operating within normal control limits."
            )

        return TrendAnalysisReport(
            current_cycle=current_cyc,
            current_index=round(current_idx, 4),
            trend_direction=trend_direction,
            slope=round(float(slope), 4),
            is_alert=is_alert,
            threshold=thresh,
            average_index=round(float(np.mean(indices)), 4),
            min_index=round(float(np.min(indices)), 4),
            max_index=round(float(np.max(indices)), 4),
            recommendation=recommendation,
            recent_history=[m.to_dict() for m in recent],
        )

    def prepare_visualization_data(self) -> Dict[str, Any]:
        """Prepare structured data ready for Streamlit, Plotly, or Matplotlib charts."""
        if not self.history:
            return {"cycles": [], "nervousness_index": [], "order_churn": [], "timing_churn": [], "cancellation_rate": []}

        df = pd.DataFrame([m.to_dict() for m in self.history])
        return {
            "cycles": df["cycle"].tolist(),
            "nervousness_index": df["nervousness_index"].tolist(),
            "order_churn": df["order_churn"].tolist(),
            "timing_churn": df["timing_churn"].tolist(),
            "cancellation_rate": df["cancellation_rate"].tolist(),
            "alert_threshold": [self.alert_threshold] * len(df),
            "summary": {
                "mean_index": float(df["nervousness_index"].mean()),
                "max_index": float(df["nervousness_index"].max()),
                "runs_exceeding_threshold": int((df["nervousness_index"] > self.alert_threshold).sum()),
            }
        }

    def save_history(self, filepath: str) -> None:
        """Save metric history time-series to CSV or JSON."""
        if not self.history:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = [m.to_dict() for m in self.history]
        if filepath.endswith(".csv"):
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    def load_history(self, filepath: str) -> None:
        """Load metric history time-series from CSV or JSON."""
        if not os.path.exists(filepath):
            return
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
            self.history = [NervousnessMetrics(**row) for row in df.to_dict(orient="records")]
        else:
            with open(filepath, "r") as f:
                data = json.load(f)
                self.history = [NervousnessMetrics(**d) for d in data]


# ==============================================================================
# Phase 3.3: Nervousness Mitigation Policies
# ==============================================================================

class MitigationPolicyManager:
    """Implements and evaluates plan stability mitigation strategies.

    Strategies supported:
    1. Change Thresholding: Ignore / filter minor plan modifications (e.g. < X% or < $Y).
    2. Demand Smoothing: Exponential smoothing on input demand to damp high-frequency noise.
    3. Dynamic Safety Stock Buffering: Dynamically scales safety stock buffers based on
       demand volatility / forecast errors.
    4. Frozen Window Dynamic Extension: Extends the frozen fence when nervousness index breaches alert limits.
    """

    def __init__(
        self,
        percent_threshold: float = 0.05,  # 5% change threshold
        value_threshold: float = 50.0,    # $50 value threshold
        smoothing_alpha: float = 0.3,     # Exponential smoothing parameter
        safety_stock_z: float = 1.65,     # 95% service level
    ):
        self.percent_threshold = percent_threshold
        self.value_threshold = value_threshold
        self.smoothing_alpha = smoothing_alpha
        self.safety_stock_z = safety_stock_z

    def filter_changes_by_threshold(
        self,
        changes: List[PlanChangeRecord],
        item_unit_costs: Optional[Dict[str, float]] = None,
    ) -> List[PlanChangeRecord]:
        """Filter out negligible changes whose delta is below percent and value thresholds.

        Minor changes that are filtered are marked as 'FILTERED_BELOW_THRESHOLD'
        or converted to 'UNCHANGED' so that the shop-floor plan is not disrupted.

        Args:
            changes: List of PlanChangeRecords generated by PlanComparator.
            item_unit_costs: Mapping of item_id -> unit cost to evaluate dollar impact.

        Returns:
            Filtered list of PlanChangeRecord where minor modifications are suppressed.
        """
        unit_costs = item_unit_costs or {}
        filtered: List[PlanChangeRecord] = []

        for c in changes:
            if c.change_type == ChangeType.QUANTITY_CHANGE.value and c.old_quantity > 0:
                pct_change = abs(c.quantity_delta) / c.old_quantity
                cost = unit_costs.get(c.item_id, 1.0)
                dollar_impact = abs(c.quantity_delta) * cost

                # If below both percentage and value thresholds, damp this change
                if pct_change < self.percent_threshold and dollar_impact < self.value_threshold:
                    # Suppress minor perturbation
                    c_copy = PlanChangeRecord(
                        item_id=c.item_id,
                        change_type=ChangeType.UNCHANGED.value,
                        zone=c.zone,
                        status="FILTERED_BELOW_THRESHOLD",
                        old_period=c.old_period,
                        new_period=c.new_period,
                        old_quantity=c.old_quantity,
                        new_quantity=c.old_quantity,  # revert to old quantity
                        quantity_delta=0.0,
                        description=f"Change of {pct_change:.1%} (${dollar_impact:.2f}) suppressed below threshold",
                    )
                    filtered.append(c_copy)
                    continue

            filtered.append(c)

        return filtered

    def smooth_demand_series(
        self,
        demand_series: List[float],
        alpha: Optional[float] = None,
    ) -> List[float]:
        """Apply single exponential smoothing to an input demand time series:
        S_t = alpha * Y_t + (1 - alpha) * S_{t-1}

        Args:
            demand_series: Raw time-phased demand quantities.
            alpha: Smoothing factor in (0, 1]. Defaults to self.smoothing_alpha.

        Returns:
            Smoothed demand time series of identical length.
        """
        if not demand_series:
            return []

        a = alpha if alpha is not None else self.smoothing_alpha
        a = max(0.01, min(1.0, a))

        smoothed = [float(demand_series[0])]
        for val in demand_series[1:]:
            s_t = a * float(val) + (1.0 - a) * smoothed[-1]
            smoothed.append(round(s_t, 2))

        return smoothed

    def calculate_dynamic_safety_stock(
        self,
        demand_history: List[float],
        lead_time_periods: int = 1,
        z_score: Optional[float] = None,
        min_safety_stock: float = 0.0,
    ) -> float:
        """Calculate dynamic safety stock buffer based on demand variability:
        SS = Z * sigma_D * sqrt(L)

        Args:
            demand_history: Historical or forecasted demand series.
            lead_time_periods: Replenishment lead time in periods.
            z_score: Service level factor (default 1.65 ~ 95% cycle service level).
            min_safety_stock: Floor for safety stock.

        Returns:
            Computed dynamic safety stock quantity.
        """
        if len(demand_history) < 2:
            return float(min_safety_stock)

        z = z_score if z_score is not None else self.safety_stock_z
        sigma_d = float(np.std(demand_history, ddof=1))
        lt = max(1, lead_time_periods)

        dynamic_ss = z * sigma_d * np.sqrt(lt)
        return float(max(min_safety_stock, round(dynamic_ss, 2)))

    def evaluate_frozen_window_extension(
        self,
        current_frozen_periods: int,
        nervousness_index: float,
        threshold: float = 0.15,
        max_extension: int = 2,
    ) -> int:
        """Determine whether to dynamically extend the frozen window fence
        when plan nervousness exceeds critical operating limits.

        Args:
            current_frozen_periods: Base frozen window length (e.g. 1 or 2).
            nervousness_index: Recent plan nervousness index.
            threshold: Alert threshold triggering extension (default 0.15).
            max_extension: Maximum additional periods allowed.

        Returns:
            Recommended effective frozen window length.
        """
        if nervousness_index > threshold:
            # Scale extension proportionally to excess nervousness
            excess_ratio = (nervousness_index - threshold) / threshold
            extension = min(max_extension, max(1, int(np.ceil(excess_ratio))))
            return current_frozen_periods + extension
        return current_frozen_periods

