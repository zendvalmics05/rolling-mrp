"""Dynamic Risk-Adjusted Safety Stock and Lead Time Buffering for ROLLINGMRP.

Implements Phase 5.2 & 5.3:
- Risk-based safety stock calculation:
  SS = Z * sigma_demand * sqrt(L + R) * risk_multiplier
- Risk multiplier:
  Scales from 1.0 (for reliable suppliers with score ~ 100) up to 2.0 (for high-risk suppliers)
- Periodic recalculation responding to dynamic supplier performance changes
- Lead time buffer calculation:
  effective_lead_time = quoted_lead_time + k * lead_time_stddev
  where k is a risk tolerance factor (typically 1.0 - 2.0)
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.risk.supplier import SupplierMetrics, SupplierReliabilityScorer


@dataclass
class SafetyStockRecommendation:
    """Dynamic safety stock and lead time parameters for an item and supplier."""
    item_id: str
    supplier_id: str
    supplier_score: float
    service_level_target_pct: float
    z_score: float
    base_safety_stock: float
    risk_multiplier: float
    dynamic_safety_stock: float
    quoted_lead_time_days: int
    lead_time_buffer_days: int
    effective_lead_time_days: int
    stock_increase_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DynamicSafetyStockEngine:
    """Calculates risk-adjusted safety stock buffers and lead times."""

    def __init__(
        self,
        default_service_level: float = 0.95,  # 95% cycle service level
        max_risk_multiplier: float = 2.0,     # Max 2.0x for worst supplier
        min_risk_multiplier: float = 1.0,     # 1.0x for perfect supplier
        buffer_k_factor: float = 1.5,         # k factor for lead time stddev buffer
    ):
        self.default_service_level = default_service_level
        self.max_risk_multiplier = max_risk_multiplier
        self.min_risk_multiplier = min_risk_multiplier
        self.buffer_k_factor = buffer_k_factor

        # Standard normal inverse CDF lookup
        self.z_table = {
            0.90: 1.282,
            0.95: 1.645,
            0.975: 1.960,
            0.99: 2.326,
            0.999: 3.090,
        }

    def get_z_score(self, service_level: float) -> float:
        """Retrieve standard normal quantile for requested service level."""
        # Find closest match or approximate
        for sl, z in sorted(self.z_table.items()):
            if abs(service_level - sl) < 0.01:
                return z
        # Approximation for arbitrary service level
        p = max(0.5, min(0.999, service_level))
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z_approx = t - ((c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t))
        return round(float(z_approx), 3)

    def calculate_supplier_risk_multiplier(self, supplier_score: float) -> float:
        """Map supplier composite score (0-100) to safety stock risk multiplier [1.0, max_multiplier].

        Score 100 -> multiplier = 1.0
        Score 0   -> multiplier = max_risk_multiplier (default 2.0)
        Linear scaling: 1.0 + (1.0 - score/100) * (max_multiplier - 1.0)
        """
        score = max(0.0, min(100.0, supplier_score))
        risk_fraction = (100.0 - score) / 100.0
        mult = self.min_risk_multiplier + risk_fraction * (self.max_risk_multiplier - self.min_risk_multiplier)
        return round(float(mult), 3)

    def calculate_dynamic_safety_stock(
        self,
        item_id: str,
        demand_series: List[float],
        lead_time_periods: float,
        review_period_periods: float = 1.0,
        supplier_score: float = 100.0,
        service_level: Optional[float] = None,
        quoted_lead_time_days: int = 7,
        actual_lead_time_stddev_days: float = 0.0,
    ) -> SafetyStockRecommendation:
        """Calculate risk-adjusted safety stock and buffered replenishment lead time.

        Formula:
        SS_base = Z * sigma_D * sqrt(L + R)
        SS_dynamic = SS_base * risk_multiplier
        """
        sl = service_level if service_level is not None else self.default_service_level
        z = self.get_z_score(sl)

        # Demand variability
        if len(demand_series) >= 2:
            sigma_d = float(np.std(demand_series, ddof=1))
        elif len(demand_series) == 1:
            sigma_d = float(demand_series[0]) * 0.2  # 20% default coefficient of variation
        else:
            sigma_d = 10.0

        exposure_periods = max(0.1, lead_time_periods + review_period_periods)
        base_ss = z * sigma_d * np.sqrt(exposure_periods)
        base_ss = max(1.0, round(base_ss, 2))

        # Risk multiplier
        risk_mult = self.calculate_supplier_risk_multiplier(supplier_score)
        dynamic_ss = round(base_ss * risk_mult, 2)

        stock_inc_pct = round(((dynamic_ss - base_ss) / base_ss) * 100.0, 1)

        # Lead time buffer
        # Buffer addition: k * stddev_days (rounded to integer days)
        lt_buffer_days = int(np.ceil(self.buffer_k_factor * actual_lead_time_stddev_days)) if actual_lead_time_stddev_days > 0 else 0
        effective_lt_days = quoted_lead_time_days + lt_buffer_days

        return SafetyStockRecommendation(
            item_id=item_id,
            supplier_id="",
            supplier_score=round(supplier_score, 1),
            service_level_target_pct=round(sl * 100.0, 1),
            z_score=z,
            base_safety_stock=base_ss,
            risk_multiplier=risk_mult,
            dynamic_safety_stock=dynamic_ss,
            quoted_lead_time_days=quoted_lead_time_days,
            lead_time_buffer_days=lt_buffer_days,
            effective_lead_time_days=effective_lt_days,
            stock_increase_pct=stock_inc_pct,
        )

    def evaluate_item_portfolio(
        self,
        items_df: pd.DataFrame,
        supplier_metrics_map: Dict[str, SupplierMetrics],
        demand_history_map: Optional[Dict[str, List[float]]] = None,
        default_demand: float = 50.0,
    ) -> List[SafetyStockRecommendation]:
        """Generate risk-adjusted recommendations across factory item master."""
        recommendations = []
        demands = demand_history_map or {}

        for _, row in items_df.iterrows():
            iid = str(row["item_id"])
            sid = str(row.get("supplier_id", ""))
            s_metric = supplier_metrics_map.get(sid)
            s_score = s_metric.composite_score if s_metric else 85.0
            s_stddev = s_metric.actual_lead_time_stddev_days if s_metric else 1.0

            quoted_lt = int(row.get("lead_time_days", 7))
            lt_periods = float(row.get("lead_time_periods", max(1, quoted_lt // 7)))

            item_dem = demands.get(iid, [default_demand] * 12)

            rec = self.calculate_dynamic_safety_stock(
                item_id=iid,
                demand_series=item_dem,
                lead_time_periods=lt_periods,
                supplier_score=s_score,
                quoted_lead_time_days=quoted_lt,
                actual_lead_time_stddev_days=s_stddev,
            )
            rec.supplier_id = sid
            recommendations.append(rec)

        return recommendations

