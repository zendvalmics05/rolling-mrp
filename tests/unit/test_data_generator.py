"""Unit tests for synthetic data generator."""

import pytest
import numpy as np
from src.integration.data_generator import generate_synthetic_data


def test_generator_reproducibility():
    data_1 = generate_synthetic_data(seed=42)
    data_2 = generate_synthetic_data(seed=42)

    for key in data_1:
        assert data_1[key].equals(data_2[key]), f"Mismatch in {key} between identical seeds"


def test_item_counts_and_types():
    data = generate_synthetic_data(seed=42)
    items = data["items"]

    assert len(items) == 25
    assert (items["type"] == "finished_good").sum() == 5
    assert (items["type"] == "subassembly").sum() == 8
    assert (items["type"] == "raw_material").sum() == 12


def test_bom_depth_and_commonality():
    data = generate_synthetic_data(seed=42)
    bom = data["bom"]
    items = data["items"].set_index("item_id")

    fg_ids = set(data["items"][data["items"]["type"] == "finished_good"]["item_id"])
    sa_ids = set(data["items"][data["items"]["type"] == "subassembly"]["item_id"])
    rm_ids = set(data["items"][data["items"]["type"] == "raw_material"]["item_id"])

    # Level 1 edges must be FG -> SA
    level_1_edges = bom[bom["parent_item_id"].isin(fg_ids)]
    assert set(level_1_edges["component_item_id"]).issubset(sa_ids)

    # Level 2 edges must be SA -> RM
    level_2_edges = bom[bom["parent_item_id"].isin(sa_ids)]
    assert set(level_2_edges["component_item_id"]).issubset(rm_ids)

    # Component commonality: verify some raw materials appear in >1 subassembly
    rm_counts = level_2_edges["component_item_id"].value_counts()
    common_rms = rm_counts[rm_counts > 1]
    assert len(common_rms) >= 4, f"Expected at least 4 common components, found {len(common_rms)}"


def test_mps_horizon_and_integrity():
    data = generate_synthetic_data(seed=42)
    mps = data["mps"]

    # 5 finished goods * 12 periods = 60 entries
    assert len(mps) == 60
    assert set(mps["period_number"].unique()) == set(range(1, 13))
    assert (mps["quantity"] > 0).all()


def test_suppliers_and_work_centers():
    data = generate_synthetic_data(seed=42)
    suppliers = data["suppliers"]
    work_centers = data["work_centers"]

    assert len(suppliers) == 4
    assert len(work_centers) == 3
    # Check reliability profiles vary
    assert suppliers["reliability_score"].nunique() == 4

