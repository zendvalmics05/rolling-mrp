"""Unit tests for Data Access Layer (src/integration/database.py)."""

import pytest
from pathlib import Path
from src.integration.db import init_db
from src.integration.database import DatabaseManager, get_db


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_dal.db"
    init_db(db_path=db_file)
    DatabaseManager.reset_instance()
    db = get_db(db_path=db_file)
    yield db
    DatabaseManager.reset_instance()


def test_singleton_behavior(tmp_path):
    db_file = tmp_path / "singleton_test.db"
    db1 = DatabaseManager.get_instance(db_file)
    db2 = DatabaseManager.get_instance(db_file)
    assert db1 is db2


def test_transaction_rollback(test_db):
    # Insert valid item
    test_db.create_item({
        "item_id": "TEST-ITEM-1",
        "description": "Test Item 1",
        "type": "finished_good",
        "unit_of_measure": "EA",
        "standard_cost": 100.0,
        "lot_sizing_rule": "L4L",
        "lead_time_days": 7,
    })

    # Attempt transaction that fails halfway
    with pytest.raises(Exception):
        with test_db.transaction() as cur:
            cur.execute(
                "INSERT INTO items (item_id, description, type, unit_of_measure, standard_cost, lot_sizing_rule, lead_time_days) "
                "VALUES ('TEST-ITEM-2', 'Desc', 'finished_good', 'EA', 50.0, 'L4L', 7)"
            )
            # Intentionally cause a foreign key / check violation or invalid query
            cur.execute("INSERT INTO items (item_id, description, type) VALUES ('INVALID', 'Desc', 'invalid_type')")

    # TEST-ITEM-2 must have been rolled back
    assert test_db.get_item("TEST-ITEM-2") is None


def test_item_crud_lifecycle(test_db):
    # 1. Create
    item_id = test_db.create_item({
        "item_id": "FG-TEST",
        "description": "Finished Good Unit Test",
        "type": "finished_good",
        "unit_of_measure": "EA",
        "standard_cost": 250.0,
        "lot_sizing_rule": "EOQ",
        "safety_stock": 20.0,
        "lead_time_days": 5,
    })
    assert item_id == "FG-TEST"

    # 2. Read
    item = test_db.get_item("FG-TEST")
    assert item is not None
    assert item["description"] == "Finished Good Unit Test"
    assert item["standard_cost"] == 250.0
    assert item["safety_stock"] == 20.0

    # 3. Update
    updated = test_db.update_item("FG-TEST", {"standard_cost": 275.50, "safety_stock": 25.0})
    assert updated is True
    re_read = test_db.get_item("FG-TEST")
    assert re_read["standard_cost"] == 275.50
    assert re_read["safety_stock"] == 25.0

    # 4. Get all
    all_fg = test_db.get_all_items(item_type="finished_good")
    assert len(all_fg) == 1
    assert all_fg[0]["item_id"] == "FG-TEST"

    # 5. Delete
    deleted = test_db.delete_item("FG-TEST")
    assert deleted is True
    assert test_db.get_item("FG-TEST") is None


def test_bom_queries(test_db):
    # Setup parent and 2 components
    test_db.create_item({"item_id": "P-1", "description": "Parent", "type": "finished_good", "unit_of_measure": "EA", "standard_cost": 100.0, "lot_sizing_rule": "L4L", "lead_time_days": 7})
    test_db.create_item({"item_id": "C-1", "description": "Comp 1", "type": "component", "unit_of_measure": "EA", "standard_cost": 20.0, "lot_sizing_rule": "L4L", "lead_time_days": 3})
    test_db.create_item({"item_id": "C-2", "description": "Comp 2", "type": "component", "unit_of_measure": "EA", "standard_cost": 30.0, "lot_sizing_rule": "POQ", "lead_time_days": 4})

    test_db.create_bom_entry({"bom_id": "B-1", "parent_item_id": "P-1", "component_item_id": "C-1", "quantity_per": 2.0, "scrap_factor": 0.01})
    test_db.create_bom_entry({"bom_id": "B-2", "parent_item_id": "P-1", "component_item_id": "C-2", "quantity_per": 4.0, "scrap_factor": 0.0})

    children = test_db.get_bom_children("P-1")
    assert len(children) == 2
    comp_ids = {c["component_item_id"] for c in children}
    assert comp_ids == {"C-1", "C-2"}
    # Verify metadata join
    c1_meta = [c for c in children if c["component_item_id"] == "C-1"][0]
    assert c1_meta["lead_time_days"] == 3
    assert c1_meta["quantity_per"] == 2.0

    parents = test_db.get_bom_parents("C-1")
    assert len(parents) == 1
    assert parents[0]["parent_item_id"] == "P-1"


def test_inventory_operations(test_db):
    test_db.create_item({"item_id": "INV-ITEM", "description": "Item", "type": "raw_material", "unit_of_measure": "KG", "standard_cost": 15.0, "lot_sizing_rule": "EOQ", "lead_time_days": 5})

    test_db.upsert_inventory({
        "item_id": "INV-ITEM",
        "location_id": "WH-MAIN",
        "quantity_on_hand": 100.0,
        "quantity_allocated": 25.0,
        "quantity_on_order": 50.0,
    })

    inv = test_db.get_inventory("INV-ITEM", "WH-MAIN")
    assert inv["quantity_on_hand"] == 100.0
    assert inv["quantity_allocated"] == 25.0
    assert inv["quantity_on_order"] == 50.0

    # Update quantities
    updated = test_db.update_inventory_quantities("INV-ITEM", on_hand=120.0, allocated=30.0)
    assert updated is True
    re_inv = test_db.get_inventory("INV-ITEM")
    assert re_inv["quantity_on_hand"] == 120.0
    assert re_inv["quantity_allocated"] == 30.0
    assert re_inv["quantity_on_order"] == 50.0  # unchanged


def test_mps_queries_and_batch_operations(test_db):
    test_db.create_item({"item_id": "FG-MPS", "description": "Item", "type": "finished_good", "unit_of_measure": "EA", "standard_cost": 300.0, "lot_sizing_rule": "L4L", "lead_time_days": 7})

    records = [
        {"mps_id": f"MPS-{p}", "item_id": "FG-MPS", "period_number": p, "quantity": 10.0 * p, "demand_type": "forecast", "source": "forecast"}
        for p in range(1, 13)
    ]
    test_db.batch_insert_mps(records)

    # Query all
    all_mps = test_db.get_mps("FG-MPS")
    assert len(all_mps) == 12

    # Query range
    sub_mps = test_db.get_mps("FG-MPS", period_range=(3, 6))
    assert len(sub_mps) == 4
    assert [r["period_number"] for r in sub_mps] == [3, 4, 5, 6]
    assert sub_mps[0]["quantity"] == 30.0

