"""Lead-Time Offsetting and Calendar Module for ROLLINGMRP.

Implements Phase 1.7:
- Basic period and date offsetting: release = due - lead_time
- Working day business calendar (skipping Saturdays, Sundays, and public holidays)
- Supplier-specific lead time lookup from item/supplier data
- Risk-adjusted lead time calculation under stochastic uncertainty:
  LT_adj = ceil(mean_LT + k * std_LT)
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union


class OffsettingCalendar:
    """Business calendar supporting working days calculation and holiday exclusion."""

    def __init__(self, holidays: Optional[Set[Union[str, date]]] = None):
        self.holidays: Set[date] = set()
        if holidays:
            for h in holidays:
                if isinstance(h, str):
                    self.holidays.add(datetime.strptime(h, "%Y-%m-%d").date())
                elif isinstance(h, date):
                    self.holidays.add(h)

    def is_business_day(self, d: date) -> bool:
        """Check if date falls on a weekday (Monday-Friday) and is not a holiday."""
        # 0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday
        if d.weekday() >= 5:
            return False
        if d in self.holidays:
            return False
        return True

    def add_business_days(self, start_date: Union[str, date], days: int) -> date:
        """Add N business days to start_date, skipping weekends and holidays."""
        curr = datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
        added = 0
        step = 1 if days >= 0 else -1
        target = abs(days)

        while added < target:
            curr += timedelta(days=step)
            if self.is_business_day(curr):
                added += 1

        return curr

    def subtract_business_days(self, due_date: Union[str, date], days: int) -> date:
        """Subtract N business days from due_date."""
        return self.add_business_days(due_date, -days)

    def count_business_days(self, start_date: Union[str, date], end_date: Union[str, date]) -> int:
        """Count working business days between start_date and end_date (inclusive of start, exclusive of end)."""
        d_start = datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
        d_end = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date

        if d_start > d_end:
            d_start, d_end = d_end, d_start

        count = 0
        curr = d_start
        while curr < d_end:
            if self.is_business_day(curr):
                count += 1
            curr += timedelta(days=1)
        return count


class LeadTimeOffsetter:
    """Manages lead time calculations, calendar offsetting, and risk adjustments."""

    def __init__(self, calendar: Optional[OffsettingCalendar] = None):
        self.calendar = calendar or OffsettingCalendar()

    def offset_period(self, due_period: int, lead_time_periods: int) -> int:
        """Calculate release period by discrete period offsetting."""
        return due_period - lead_time_periods

    def offset_date(self, due_date: Union[str, date], lead_time_days: int) -> date:
        """Calculate release date using the business calendar."""
        return self.calendar.subtract_business_days(due_date, lead_time_days)

    def calculate_risk_adjusted_lead_time(
        self,
        mean_lead_time: float,
        std_lead_time: float,
        service_factor_k: float = 1.645,  # 95% service level normal buffer
    ) -> int:
        """Calculate risk-adjusted lead time under supplier lead time variability (links to Phase 5).

        Formula: ceil(mean + k * stddev)
        """
        import math
        buffered_lt = mean_lead_time + (service_factor_k * std_lead_time)
        return max(1, math.ceil(buffered_lt))

    def get_effective_lead_time(
        self,
        item_master: Dict[str, Any],
        supplier_data: Optional[Dict[str, Any]] = None,
        use_risk_adjustment: bool = False,
        service_factor_k: float = 1.645,
    ) -> int:
        """Determine effective lead time days from item and supplier profiles."""
        base_lt = int(item_master.get("lead_time_days", 0))
        if not use_risk_adjustment or not supplier_data:
            return base_lt

        mean_lt = float(supplier_data.get("actual_lead_time_mean", base_lt))
        std_lt = float(supplier_data.get("actual_lead_time_stddev", 0.0))
        return self.calculate_risk_adjusted_lead_time(mean_lt, std_lt, service_factor_k)

