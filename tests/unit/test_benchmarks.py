"""Unit tests for hand-solved benchmarks (TC-001 through TC-004)."""

import json
from pathlib import Path
import pytest

VALIDATION_DIR = Path(__file__).resolve().parent.parent.parent / "validation"


def test_tc001_netting_benchmark_integrity():
    tc_dir = VALIDATION_DIR / "TC-001"
    assert (tc_dir / "README.md").exists()
    assert (tc_dir / "expected_output.json").exists()

    with open(tc_dir / "expected_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_case_id"] == "TC-001"
    assert "FG-A" in data["items"]
    assert "COMP-B" in data["items"]
    assert "COMP-C" in data["items"]
    assert len(data["exceptions"]) >= 3


def test_tc002_explosion_pegging_benchmark_integrity():
    tc_dir = VALIDATION_DIR / "TC-002"
    assert (tc_dir / "README.md").exists()
    assert (tc_dir / "expected_output.json").exists()

    with open(tc_dir / "expected_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_case_id"] == "TC-002"
    items = data["items"]
    assert len(items) == 5
    # RM-1 must have commonality (pegged to SA-1 and SA-2)
    rm1 = items["RM-1"]
    p2_order = [o for o in rm1["planned_order_releases"] if o["period"] == 0][0]
    pegged_parents = {p["parent_item"] for p in p2_order["pegged_to"]}
    assert pegged_parents == {"SA-1", "SA-2"}


def test_tc003_lot_sizing_benchmark_integrity():
    tc_dir = VALIDATION_DIR / "TC-003"
    assert (tc_dir / "README.md").exists()
    assert (tc_dir / "expected_output.json").exists()

    with open(tc_dir / "expected_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_case_id"] == "TC-003"
    assert len(data["items"]) == 8
    # Principle 3 check: Wagner-Whitin cost must always be <= L4L cost
    for item_id, item_data in data["items"].items():
        assert item_data["wagner_whitin"]["total_cost"] <= item_data["l4l"]["total_cost"]


def test_tc004_capacity_benchmark_integrity():
    tc_dir = VALIDATION_DIR / "TC-004"
    assert (tc_dir / "README.md").exists()
    assert (tc_dir / "expected_output.json").exists()

    with open(tc_dir / "expected_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_case_id"] == "TC-004"
    # Unconstrained plan has Period 3 overload
    unconstrained = data["unconstrained_plan"]
    assert len(unconstrained["overloaded_periods"]) == 2
    # Leveled plan eliminates all overloads
    leveled = data["leveled_plan"]
    assert len(leveled["overloaded_periods"]) == 0
    assert all(util <= 100.0 for util in leveled["utilization_pct"]["WC-B"])

