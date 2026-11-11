"""Rolling-Horizon Simulator and Forecast Evolution for ROLLINGMRP.

Implements:
- Phase 2.1: Dynamic Planning Window Management:
  - 12-week configurable rolling window
  - Time-phased demand composition: Actual (past/firm) -> Forecast (decaying uncertainty)
  - Window shift logic: advance current period, drop elapsed period, append new horizon forecast
  - State preservation: carry forward physical on-hand, allocated, and on-order quantities
- Phase 2.2: Forecast Evolution & Stochastic Demand Generation:
  - Forecast accuracy decay with horizon (MAPE increases as horizon h increases)
  - Stochastic noise injection (no future information leakage; actual demand stays hidden until realized)
  - Actual demand revelation as time advances
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd


@dataclass
class PeriodDemand:
    """Represents demand in a specific period within the planning window."""
    period: int
    quantity: float
    demand_type: str  # 'actual', 'firm_planned', or 'forecast'
    source: str       # 'customer_order' or 'forecast'
    realized_actual: Optional[float] = None  # None until period arrives


@dataclass
class SimulationStepRecord:
    """Audit log of a single step in the rolling simulation."""
    period: int
    demand_planned: float
    demand_realized: float
    units_served: float
    stockout_units: float
    starting_pab: float
    ending_pab: float
    order_receipts_realized: float
    planned_orders_active: List[Dict[str, Any]] = field(default_factory=list)


class DemandEvolutionModel:
    """Generates ground-truth realized demand and decaying-accuracy forecasts.

    Guarantees Principle 1 & 5:
    Actual demand for future periods remains hidden until the simulation clock reaches that period.
    Forecast uncertainty increases with horizon h: sigma_h = base_mape + decay_rate * h.
    """

    def __init__(
        self,
        base_demand: float = 50.0,
        trend: float = 1.0,
        seasonal_amplitude: float = 10.0,
        seasonal_period: int = 12,
        base_mape: float = 0.08,
        decay_rate_per_period: float = 0.03,
        seed: int = 42,
    ):
        self.base_demand = base_demand
        self.trend = trend
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.base_mape = base_mape
        self.decay_rate_per_period = decay_rate_per_period
        self.rng = np.random.default_rng(seed)

        # Memoized ground truth demands generated lazily per period
        self._ground_truth: Dict[int, float] = {}

    def get_ground_truth(self, period: int) -> float:
        """Get or lazily generate the true realized demand for a period."""
        if period not in self._ground_truth:
            t = period
            trend_val = self.trend * t
            seasonal_val = self.seasonal_amplitude * math.sin(2.0 * math.pi * t / self.seasonal_period)
            noise_val = self.rng.normal(0, 3.0)
            actual_val = max(5.0, round(self.base_demand + trend_val + seasonal_val + noise_val, 1))
            self._ground_truth[period] = actual_val
        return self._ground_truth[period]

    def generate_forecast(self, current_period: int, target_period: int) -> float:
        """Generate forecast for target_period viewed from current_period.

        Forecast horizon h = target_period - current_period >= 0.
        When h == 0 (current period): actual demand is known.
        When h >= 1: forecast error standard deviation grows with h.
        """
        ground_truth = self.get_ground_truth(target_period)
        if target_period <= current_period:
            return ground_truth

        horizon = target_period - current_period
        # Standard error scales with horizon: e.g. 8% + 3% * h
        effective_sigma_pct = self.base_mape + (self.decay_rate_per_period * horizon)
        error_pct = self.rng.normal(0, effective_sigma_pct)

        forecast_val = max(1.0, round(ground_truth * (1.0 + error_pct), 1))
        return forecast_val


class RollingHorizonManager:
    """Manages the time-phased rolling planning window and state carry-forward."""

    def __init__(
        self,
        item_id: str,
        window_length: int = 12,
        firm_window_length: int = 2,
        initial_on_hand: float = 0.0,
        initial_allocated: float = 0.0,
        safety_stock: float = 0.0,
        lead_time_periods: int = 1,
        demand_model: Optional[DemandEvolutionModel] = None,
        seed: int = 42,
    ):
        self.item_id = item_id
        self.window_length = window_length
        self.firm_window_length = firm_window_length
        self.current_period = 1

        # Inventory State
        self.on_hand = float(initial_on_hand)
        self.allocated = float(initial_allocated)
        self.safety_stock = float(safety_stock)
        self.lead_time_periods = int(lead_time_periods)

        # Scheduled receipts arriving in future periods: period -> quantity
        self.scheduled_receipts: Dict[int, float] = defaultdict(float)

        self.demand_model = demand_model or DemandEvolutionModel(seed=seed)
        self.simulation_log: List[SimulationStepRecord] = []

    @property
    def current_pab(self) -> float:
        """Physical unallocated on-hand stock."""
        return self.on_hand - self.allocated

    def get_window_periods(self) -> List[int]:
        """Return list of period numbers currently inside the rolling window."""
        return list(range(self.current_period, self.current_period + self.window_length))

    def get_window_demand(self) -> List[PeriodDemand]:
        """Construct the demand vector for the active planning window.

        Demand composition:
        - Periods within [current, current + firm_window_length - 1]:
          Treated as 'actual' or 'firm_planned' customer commitments.
        - Periods beyond firm fence:
          Treated as 'forecast' with horizon uncertainty.
        """
        demands: List[PeriodDemand] = []
        for p in self.get_window_periods():
            horizon = p - self.current_period

            if horizon == 0:
                # Immediate current period: actual demand is revealed
                qty = self.demand_model.get_ground_truth(p)
                demands.append(PeriodDemand(
                    period=p,
                    quantity=qty,
                    demand_type="actual",
                    source="customer_order",
                    realized_actual=qty,
                ))
            elif horizon < self.firm_window_length:
                # Firm window: known customer orders + baseline forecast
                qty = self.demand_model.generate_forecast(self.current_period, p)
                demands.append(PeriodDemand(
                    period=p,
                    quantity=qty,
                    demand_type="firm_planned",
                    source="customer_order",
                ))
            else:
                # Liquid planning horizon: forecast
                qty = self.demand_model.generate_forecast(self.current_period, p)
                demands.append(PeriodDemand(
                    period=p,
                    quantity=qty,
                    demand_type="forecast",
                    source="forecast",
                ))

        return demands

    def record_scheduled_receipt(self, period: int, quantity: float) -> None:
        """Schedule an incoming receipt (e.g. from a released order)."""
        self.scheduled_receipts[period] += float(quantity)

    def advance_period(self, active_planned_orders: Optional[List[Dict[str, Any]]] = None) -> SimulationStepRecord:
        """Execute one simulation step:
        1. Receive scheduled orders due in current_period into on_hand stock.
        2. Realize actual customer demand for current_period.
        3. Fulfill demand from available on-hand; log any stockout / backlog.
        4. Advance current_period by 1, shifting the rolling window.
        """
        p = self.current_period
        start_pab = self.current_pab

        # 1. Receive incoming deliveries
        receipts_realized = float(self.scheduled_receipts.pop(p, 0.0))
        self.on_hand += receipts_realized

        # 2. Reveal true demand
        demand_realized = self.demand_model.get_ground_truth(p)

        # 3. Fulfill demand from inventory
        available_stock = self.on_hand - self.allocated
        if available_stock >= demand_realized:
            units_served = demand_realized
            stockout_units = 0.0
            self.on_hand -= demand_realized
        else:
            units_served = max(0.0, available_stock)
            stockout_units = demand_realized - units_served
            self.on_hand = self.allocated  # Stock exhausted down to allocated commitments

        end_pab = self.current_pab

        record = SimulationStepRecord(
            period=p,
            demand_planned=demand_realized,
            demand_realized=demand_realized,
            units_served=units_served,
            stockout_units=stockout_units,
            starting_pab=start_pab,
            ending_pab=end_pab,
            order_receipts_realized=receipts_realized,
            planned_orders_active=active_planned_orders or [],
        )
        self.simulation_log.append(record)

        # 4. Advance clock by 1 period
        self.current_period += 1
        return record

    def get_simulation_summary(self) -> pd.DataFrame:
        """Export simulation execution history as a DataFrame."""
        records = [
            {
                "period": r.period,
                "demand_realized": r.demand_realized,
                "units_served": r.units_served,
                "stockout_units": r.stockout_units,
                "starting_pab": r.starting_pab,
                "receipts_realized": r.order_receipts_realized,
                "ending_pab": r.ending_pab,
            }
            for r in self.simulation_log
        ]
        return pd.DataFrame(records)
