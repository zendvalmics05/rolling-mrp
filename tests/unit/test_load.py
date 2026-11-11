"""Unit tests for data ingestion, cleaning, and BOM validation."""

import pytest
import pandas as pd
from src.integration.load import (
    clean_and_validate_dataset,
    detect_bom_cycles,
    validate_bom_structure,
    initialize_inventory_status,
    BOMValidationError,
    DataQualityError,
    run_pipeline,
)
from src.integration.data_generator import generate_synthetic_data


def test_clean_and_validate_valid_dataset():
    data = generate_synthetic_data(seed=42)
    cleaned, report = clean_and_validate_dataset(data)

    assert report["summary"]["critical_issues_count"] == 0
    assert len(cleaned["items"]) == 25
    assert len(cleaned["bom"]) == 45


def test_bom_cycle_detection_direct():
    # Direct 2-node cycle: A -> B -> A
    df_bom = pd.DataFrame([
        {"parent_item_id": "A", "component_item_id": "B", "quantity_per": 1.0, "scrap_factor": 0.0},
        {"parent_item_id": "B", "component_item_id": "A", "quantity_per": 2.0, "scrap_factor": 0.0},
    ])
    cycle = detect_bom_cycles(df_bom)
    assert cycle is not None
    assert set(cycle) == {"A", "B"}


def test_bom_cycle_detection_indirect():
    # 3-node cycle: A -> B -> C -> A
    df_bom = pd.DataFrame([
        {"parent_item_id": "A", "component_item_id": "B", "quantity_per": 1.0, "scrap_factor": 0.0},
        {"parent_item_id": "B", "component_item_id": "C", "quantity_per": 1.0, "scrap_factor": 0.0},
        {"parent_item_id": "C", "component_item_id": "A", "quantity_per": 1.0, "scrap_factor": 0.0},
    ])
    cycle = detect_bom_cycles(df_bom)
    assert cycle is not None
    assert set(cycle) == {"A", "B", "C"}

    df_items = pd.DataFrame([{"item_id": x} for x in ["A", "B", "C"]])
    with pytest.raises(BOMValidationError, match="Cycle detected in BOM structure"):
        validate_bom_structure(df_bom, df_items)


def test_bom_self_reference_rejection():
    df_bom = pd.DataFrame([
        {"parent_item_id": "A", "component_item_id": "A", "quantity_per": 1.0, "scrap_factor": 0.0},
    ])
    df_items = pd.DataFrame([{"item_id": "A"}])
    with pytest.raises(BOMValidationError, match="Self-referential BOM edges detected"):
        validate_bom_structure(df_bom, df_items)


def test_bom_orphan_component_rejection():
    df_bom = pd.DataFrame([
        {"parent_item_id": "FG-01", "component_item_id": "GHOST-PART", "quantity_per": 1.0, "scrap_factor": 0.0},
    ])
    df_items = pd.DataFrame([{"item_id": "FG-01"}])
    with pytest.raises(BOMValidationError, match="BOM components do not exist in item master"):
        validate_bom_structure(df_bom, df_items)


def test_bom_invalid_quantity_or_scrap():
    df_items = pd.DataFrame([{"item_id": "A"}, {"item_id": "B"}])

    # Non-positive quantity_per
    bad_qty = pd.DataFrame([
        {"parent_item_id": "A", "component_item_id": "B", "quantity_per": 0.0, "scrap_factor": 0.0},
    ])
    with pytest.raises(BOMValidationError, match="non-positive quantity_per"):
        validate_bom_structure(bad_qty, df_items)

    # Scrap factor >= 1.0
    bad_scrap = pd.DataFrame([
        {"parent_item_id": "A", "component_item_id": "B", "quantity_per": 1.0, "scrap_factor": 1.05},
    ])
    with pytest.raises(BOMValidationError, match="invalid scrap_factor"):
        validate_bom_structure(bad_scrap, df_items)


def test_inventory_initialization_calculations():
    df_items = pd.DataFrame([
        {"item_id": "ITEM-1", "type": "finished_good", "safety_stock": 20.0, "standard_cost": 50.0, "lot_sizing_rule": "L4L", "lead_time_days": 7},
    ])
    df_inv = pd.DataFrame([
        {"item_id": "ITEM-1", "location_id": "WH-MAIN", "quantity_on_hand": 100.0, "quantity_allocated": 30.0, "quantity_on_order": 0.0},
    ])
    res = initialize_inventory_status(df_inv, df_items)

    assert res.iloc[0]["type"] == "finished_good"
    assert res.iloc[0]["net_available_start"] == 70.0
    assert res.iloc[0]["inventory_value"] == 5000.0
    assert res.iloc[0]["safety_stock"] == 20.0


def test_data_quality_error_on_corrupt_data():
    data = generate_synthetic_data(seed=42)
    # Corrupt with negative cost
    data["items"].loc[0, "standard_cost"] = -10.0

    with pytest.raises(DataQualityError, match="Negative standard_cost detected"):
        clean_and_validate_dataset(data)

