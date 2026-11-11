"""Supplier Reliability Scoring and Performance Evaluation for ROLLINGMRP.

Implements Phase 5.1:
- Metric calculators:
  - On-Time Delivery (OTD): % of purchase order deliveries on or before quoted lead time
  - Fill Rate: % of ordered units successfully delivered
  - Quality Reject Rate: % defective / rejected units out of total received
  - Lead Time Variability: stddev of actual delivery lead time vs. quoted lead time
- Composite score: Weighted multi-criteria index (default: 30% OTD, 25% Fill, 25% Quality, 20% LT Variability)
- Normalization: Scaled [0, 100] with configurable performance tiers (Preferred, Approved, Conditional, High Risk)
- Trajectory / trend analysis over historical purchase receipt records
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SupplierTier(str, Enum):
    PREFERRED = "Preferred"        # Score >= 90
    APPROVED = "Approved"          # 75 <= Score < 90
    CONDITIONAL = "Conditional"    # 60 <= Score < 75
    HIGH_RISK = "High Risk"        # Score < 60


@dataclass
class SupplierMetrics:
    """Individual performance metrics and composite score for a supplier."""
    supplier_id: str
    name: str
    on_time_delivery_pct: float      # 0 - 100%
    fill_rate_pct: float             # 0 - 100%
    quality_acceptance_pct: float    # 100% - reject_rate% (0 - 100%)
    lead_time_stability_pct: float   # 0 - 100% (penalizes high stddev)
    composite_score: float           # 0 - 100
    tier: str                        # Preferred, Approved, Conditional, High Risk
    sample_order_count: int = 0
    actual_lead_time_mean_days: float = 0.0
    actual_lead_time_stddev_days: float = 0.0
    quality_reject_rate_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupplierReliabilityScorer:
    """Calculates supplier reliability, scorecard metrics, and historical trajectories."""

    DEFAULT_WEIGHTS = {
        "on_time_delivery": 0.30,
        "fill_rate": 0.25,
        "quality": 0.25,
        "lead_time_variability": 0.20,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_acceptable_stddev_days: float = 5.0,
    ):
        """
        Args:
            weights: Dictionary of weights summing to 1.0.
            max_acceptable_stddev_days: Maximum lead time stddev that corresponds to 0% stability.
        """
        raw_weights = weights or self.DEFAULT_WEIGHTS
        w_sum = sum(raw_weights.values())
        self.weights = {k: v / w_sum for k, v in raw_weights.items()}
        self.max_acceptable_stddev_days = max(1.0, max_acceptable_stddev_days)

    def calculate_score_from_summary(
        self,
        supplier_id: str,
        name: str,
        on_time_delivery_rate: float,
        fill_rate: float,
        quality_reject_rate: float,
        actual_lead_time_stddev: float,
        quoted_lead_time_days: float = 7.0,
        actual_lead_time_mean: Optional[float] = None,
        sample_order_count: int = 50,
    ) -> SupplierMetrics:
        """Calculate supplier composite score from aggregate KPIs.

        Rates can be passed as fractions [0, 1] or percentages [0, 100].
        """
        # Normalize inputs to 0-100 scale
        otd_pct = on_time_delivery_rate * 100.0 if on_time_delivery_rate <= 1.0 else on_time_delivery_rate
        fill_pct = fill_rate * 100.0 if fill_rate <= 1.0 else fill_rate
        reject_pct = quality_reject_rate * 100.0 if quality_reject_rate <= 1.0 else quality_reject_rate
        quality_pct = max(0.0, min(100.0, 100.0 - reject_pct))

        # LT Stability: 100% when stddev = 0, dropping to 0% as stddev reaches max_acceptable_stddev_days
        stability_ratio = max(0.0, 1.0 - (actual_lead_time_stddev / self.max_acceptable_stddev_days))
        stability_pct = round(stability_ratio * 100.0, 2)

        composite = (
            self.weights["on_time_delivery"] * otd_pct
            + self.weights["fill_rate"] * fill_pct
            + self.weights["quality"] * quality_pct
            + self.weights["lead_time_variability"] * stability_pct
        )
        composite = round(max(0.0, min(100.0, composite)), 2)

        # Classify Tier
        if composite >= 90.0:
            tier = SupplierTier.PREFERRED.value
        elif composite >= 75.0:
            tier = SupplierTier.APPROVED.value
        elif composite >= 60.0:
            tier = SupplierTier.CONDITIONAL.value
        else:
            tier = SupplierTier.HIGH_RISK.value

        return SupplierMetrics(
            supplier_id=supplier_id,
            name=name,
            on_time_delivery_pct=round(otd_pct, 2),
            fill_rate_pct=round(fill_pct, 2),
            quality_acceptance_pct=round(quality_pct, 2),
            lead_time_stability_pct=stability_pct,
            composite_score=composite,
            tier=tier,
            sample_order_count=sample_order_count,
            actual_lead_time_mean_days=actual_lead_time_mean or quoted_lead_time_days,
            actual_lead_time_stddev_days=round(actual_lead_time_stddev, 2),
            quality_reject_rate_pct=round(reject_pct, 2),
        )

    def calculate_score_from_history(
        self,
        supplier_id: str,
        name: str,
        purchase_orders_df: pd.DataFrame,
    ) -> SupplierMetrics:
        """Calculate score from a historical DataFrame of completed purchase orders.

        Expected columns:
        - 'quoted_lead_time_days'
        - 'actual_lead_time_days'
        - 'ordered_quantity'
        - 'received_quantity'
        - 'defective_quantity'
        """
        if purchase_orders_df.empty:
            return self.calculate_score_from_summary(supplier_id, name, 1.0, 1.0, 0.0, 0.0, sample_order_count=0)

        df = purchase_orders_df[purchase_orders_df["supplier_id"] == supplier_id]
        if df.empty:
            df = purchase_orders_df

        # 1. On-Time Delivery: actual_lead_time <= quoted_lead_time
        on_time = (df["actual_lead_time_days"] <= df["quoted_lead_time_days"]).mean() * 100.0

        # 2. Fill Rate: total received / total ordered
        tot_ordered = df["ordered_quantity"].sum()
        tot_received = df["received_quantity"].sum()
        fill_pct = (tot_received / tot_ordered * 100.0) if tot_ordered > 0 else 100.0

        # 3. Quality Reject Rate: defective / received
        tot_defective = df["defective_quantity"].sum() if "defective_quantity" in df.columns else 0.0
        reject_pct = (tot_defective / tot_received * 100.0) if tot_received > 0 else 0.0

        # 4. Lead Time Variability
        stddev_lt = float(df["actual_lead_time_days"].std(ddof=1)) if len(df) > 1 else 0.0
        mean_lt = float(df["actual_lead_time_days"].mean())

        return self.calculate_score_from_summary(
            supplier_id=supplier_id,
            name=name,
            on_time_delivery_rate=on_time,
            fill_rate=fill_pct,
            quality_reject_rate=reject_pct,
            actual_lead_time_stddev=stddev_lt,
            actual_lead_time_mean=mean_lt,
            sample_order_count=len(df),
        )

    def evaluate_supplier_portfolio(
        self,
        suppliers_df: pd.DataFrame,
    ) -> List[SupplierMetrics]:
        """Compute scorecards for all suppliers in the supplier master table."""
        scorecards = []
        for _, row in suppliers_df.iterrows():
            sid = str(row["supplier_id"])
            sname = str(row.get("name", sid))
            quoted_lt = float(row.get("quoted_lead_time_days", 7.0))
            mean_lt = float(row.get("actual_lead_time_mean", quoted_lt))
            std_lt = float(row.get("actual_lead_time_stddev", 1.0))
            fill = float(row.get("fill_rate", 0.95))
            reject = float(row.get("quality_reject_rate", 0.02))

            # Approximate OTD: prob(actual <= quoted) using normal distribution approximation
            if std_lt > 1e-4:
                z = (quoted_lt - mean_lt) / std_lt
                # Logistic approximation of standard normal CDF
                otd_est = 1.0 / (1.0 + np.exp(-1.7 * z))
            else:
                otd_est = 1.0 if mean_lt <= quoted_lt else 0.0

            metric = self.calculate_score_from_summary(
                supplier_id=sid,
                name=sname,
                on_time_delivery_rate=otd_est,
                fill_rate=fill,
                quality_reject_rate=reject,
                actual_lead_time_stddev=std_lt,
                quoted_lead_time_days=quoted_lt,
                actual_lead_time_mean=mean_lt,
            )
            scorecards.append(metric)

        scorecards.sort(key=lambda s: s.composite_score, reverse=True)
        return scorecards


# ==============================================================================
# Phase 5.4: Dual Sourcing & Order Allocation Optimization
# ==============================================================================

@dataclass
class DualSourcingSplit:
    """Recommended split allocation for a purchased item across two qualified suppliers."""
    item_id: str
    total_order_quantity: float
    primary_supplier_id: str
    primary_supplier_name: str
    primary_share_pct: float
    primary_quantity: float
    backup_supplier_id: str
    backup_supplier_name: str
    backup_share_pct: float
    backup_quantity: float
    single_source_cost: float
    dual_source_cost: float
    cost_premium: float
    cost_premium_pct: float
    portfolio_risk_reduction_pct: float
    is_diversified: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DualSourcingManager:
    """Manages dual-sourcing qualification, risk-optimal order splitting, and cost-risk tradeoffs."""

    def __init__(
        self,
        min_split_pct: float = 20.0,           # Minimum volume share to keep backup supplier active
        max_primary_pct: float = 80.0,         # Maximum volume share for primary supplier
        risk_penalty_weight: float = 2.0,      # Weight of supplier risk penalty vs unit cost
    ):
        self.min_split_pct = min_split_pct
        self.max_primary_pct = max_primary_pct
        self.risk_penalty_weight = risk_penalty_weight

    def calculate_order_split(
        self,
        item_id: str,
        total_quantity: float,
        primary_metrics: SupplierMetrics,
        backup_metrics: SupplierMetrics,
        primary_unit_cost: float = 10.0,
        backup_unit_cost: float = 11.0,
    ) -> DualSourcingSplit:
        """Calculate order allocation between primary and backup suppliers based on risk scores.

        Optimization Logic:
        Higher composite score receives higher allocation share, but bounded by
        [min_split_pct, max_primary_pct] to ensure dual-sourcing resilience.
        """
        score_p = max(1.0, primary_metrics.composite_score)
        score_b = max(1.0, backup_metrics.composite_score)

        # Baseline split proportional to composite scores squared (convex preference for reliability)
        weight_p = score_p ** 1.5
        weight_b = score_b ** 1.5
        raw_share_p = weight_p / (weight_p + weight_b) * 100.0

        # Enforce bounds
        share_p = max(self.min_split_pct, min(self.max_primary_pct, raw_share_p))
        share_b = 100.0 - share_p

        # Quantities
        qty_p = round((share_p / 100.0) * total_quantity, 2)
        qty_b = round(total_quantity - qty_p, 2)

        # Cost comparison
        single_cost = round(total_quantity * primary_unit_cost, 2)
        dual_cost = round(qty_p * primary_unit_cost + qty_b * backup_unit_cost, 2)
        premium = round(max(0.0, dual_cost - single_cost), 2)
        premium_pct = round((premium / single_cost * 100.0) if single_cost > 0 else 0.0, 2)

        # Portfolio risk reduction:
        # Variance of a dual-sourced supply pool is substantially lower than single sourcing
        # Risk index: Var(delivery) = w_p^2 * std_p^2 + w_b^2 * std_b^2 (assuming independent delays)
        std_p = max(0.2, primary_metrics.actual_lead_time_stddev_days)
        std_b = max(0.2, backup_metrics.actual_lead_time_stddev_days)

        var_single = std_p ** 2
        var_dual = ((share_p / 100.0) ** 2) * (std_p ** 2) + ((share_b / 100.0) ** 2) * (std_b ** 2)

        risk_reduction = max(0.0, (1.0 - np.sqrt(var_dual) / std_p) * 100.0)

        return DualSourcingSplit(
            item_id=item_id,
            total_order_quantity=total_quantity,
            primary_supplier_id=primary_metrics.supplier_id,
            primary_supplier_name=primary_metrics.name,
            primary_share_pct=round(share_p, 1),
            primary_quantity=qty_p,
            backup_supplier_id=backup_metrics.supplier_id,
            backup_supplier_name=backup_metrics.name,
            backup_share_pct=round(share_b, 1),
            backup_quantity=qty_b,
            single_source_cost=single_cost,
            dual_source_cost=dual_cost,
            cost_premium=premium,
            cost_premium_pct=premium_pct,
            portfolio_risk_reduction_pct=round(risk_reduction, 1),
            is_diversified=(qty_b > 0),
        )


