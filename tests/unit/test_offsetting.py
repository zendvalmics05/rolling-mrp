"""Unit tests for Lead-Time Offsetting and Calendar Operations (src/core/offsetting.py)."""

from datetime import date
import pytest
from src.core.offsetting import OffsettingCalendar, LeadTimeOffsetter


def test_discrete_period_offsetting():
    offsetter = LeadTimeOffsetter()
    assert offsetter.offset_period(due_period=5, lead_time_periods=2) == 3
    assert offsetter.offset_period(due_period=1, lead_time_periods=1) == 0  # Past due


def test_business_calendar_skips_weekends():
    cal = OffsettingCalendar()
    # 2026-11-06 is Friday. Subtract 1 business day -> Thursday 2026-11-05
    assert cal.subtract_business_days("2026-11-06", 1) == date(2026, 11, 5)
    # Subtract 2 business days from Monday 2026-11-09 -> skips Sun & Sat -> Thursday 2026-11-05
    assert cal.subtract_business_days("2026-11-09", 2) == date(2026, 11, 5)
    # Add 1 business day from Friday 2026-11-06 -> Monday 2026-11-09
    assert cal.add_business_days("2026-11-06", 1) == date(2026, 11, 9)


def test_business_calendar_with_holiday():
    # Holiday on Wednesday 2026-11-11
    holidays = {"2026-11-11"}
    cal = OffsettingCalendar(holidays=holidays)

    # From Friday 2026-11-13, subtract 3 business days:
    # Day 1: Thursday Nov 12
    # Day 2: skips Wed Nov 11 (Holiday) -> Tuesday Nov 10
    # Day 3: Monday Nov 9
    assert cal.subtract_business_days("2026-11-13", 3) == date(2026, 11, 9)


def test_risk_adjusted_lead_time():
    offsetter = LeadTimeOffsetter()
    # mean = 10 days, std = 2.5 days, k = 1.645
    # buffered = 10 + 1.645 * 2.5 = 10 + 4.1125 = 14.1125 -> ceil = 15 days
    adj_lt = offsetter.calculate_risk_adjusted_lead_time(mean_lead_time=10.0, std_lead_time=2.5)
    assert adj_lt == 15


def test_effective_lead_time_with_supplier():
    offsetter = LeadTimeOffsetter()
    item = {"item_id": "RM-01", "lead_time_days": 7}
    supplier = {
        "supplier_id": "SUP-04",
        "quoted_lead_time_days": 10,
        "actual_lead_time_mean": 13.5,
        "actual_lead_time_stddev": 4.2,
    }

    # Standard (no risk adjustment): uses base lead time
    assert offsetter.get_effective_lead_time(item, supplier, use_risk_adjustment=False) == 7

    # Risk-adjusted: 13.5 + 1.645 * 4.2 = 13.5 + 6.909 = 20.409 -> ceil = 21 days
    assert offsetter.get_effective_lead_time(item, supplier, use_risk_adjustment=True) == 21


def test_business_calendar_count_and_edge_cases():
    cal = OffsettingCalendar(holidays={"2026-11-11"})
    # Mon 2026-11-09 to Fri 2026-11-13 (Nov 11 is holiday) -> 3 business days (Nov 9, 10, 12)
    assert cal.count_business_days("2026-11-09", "2026-11-13") == 3
    # Reverse order count
    assert cal.count_business_days("2026-11-13", "2026-11-09") == 3
    # Negative days in add_business_days
    assert cal.add_business_days("2026-11-13", -3) == date(2026, 11, 9)
    # Check date objects passed directly
    d1 = date(2026, 11, 9)
    d2 = date(2026, 11, 13)
    assert cal.count_business_days(d1, d2) == 3
