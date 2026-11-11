"""Unit tests for Pegging and Traceability (src/core/pegging.py)."""

import pytest
import pandas as pd
from src.core.pegging import PeggingTracker, PeggingLink


def test_pegging_record_and_bidirectional_trace():
    tracker = PeggingTracker()

    # Record Level 0 (FG to MPS)
    tracker.record_link(
        child_item_id="FG-1",
        child_period=1,
        child_quantity=20.0,
        mps_item_id="FG-1",
        mps_period=2,
    )

    # Record Level 1 (SA to FG)
    tracker.record_link(
        child_item_id="SA-1",
        child_period=1,
        child_quantity=20.0,
        parent_item_id="FG-1",
        parent_period=1,
        mps_item_id="FG-1",
        mps_period=2,
    )

    # Record Level 2 (RM to SA)
    tracker.record_link(
        child_item_id="RM-1",
        child_period=0,
        child_quantity=40.0,
        parent_item_id="SA-1",
        parent_period=1,
        mps_item_id="FG-1",
        mps_period=2,
    )

    # 1. Trace up from RM-1
    up_trace = tracker.trace_up("RM-1")
    assert len(up_trace) == 1
    assert up_trace[0]["parent_item_id"] == "SA-1"
    assert up_trace[0]["mps_item_id"] == "FG-1"
    assert up_trace[0]["mps_period"] == 2

    # 2. Trace down from MPS FG-1 in period 2
    down_trace = tracker.trace_down("FG-1", mps_period=2)
    assert len(down_trace) == 3
    child_items = {d["child_item_id"] for d in down_trace}
    assert child_items == {"FG-1", "SA-1", "RM-1"}

    # 3. Pegging report
    df_report = tracker.generate_pegging_report()
    assert isinstance(df_report, pd.DataFrame)
    assert len(df_report) == 3
    assert set(df_report.columns) >= {"child_item_id", "parent_item_id", "mps_item_id"}


def test_empty_pegging_report():
    tracker = PeggingTracker()
    df = tracker.generate_pegging_report()
    assert df.empty
    assert "child_item_id" in df.columns

