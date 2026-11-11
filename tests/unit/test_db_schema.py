"""Unit tests for database schema creation and validation."""

import sqlite3
import pytest
from src.integration.db import init_db, get_connection


def test_schema_creates_all_expected_tables(tmp_path):
    db_file = tmp_path / "test_mrp.db"
    init_db(db_path=db_file)

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%';")
    tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {
        "suppliers",
        "items",
        "bom",
        "inventory",
        "work_centers",
        "routings",
        "mps",
        "planned_orders",
        "exception_messages",
    }

    assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"
    conn.close()


def test_foreign_keys_and_constraints(tmp_path):
    db_file = tmp_path / "test_mrp.db"
    init_db(db_path=db_file)

    conn = get_connection(db_file)
    with conn:
        # Insert supplier
        conn.execute(
            """INSERT INTO suppliers (supplier_id, name, reliability_score, quoted_lead_time_days,
               actual_lead_time_mean, actual_lead_time_stddev, fill_rate, quality_reject_rate)
               VALUES ('SUP-01', 'Test Supplier', 0.95, 7, 7.2, 1.1, 0.98, 0.02)"""
        )

        # Insert item
        conn.execute(
            """INSERT INTO items (item_id, description, type, unit_of_measure, standard_cost,
               shelf_life_days, lot_sizing_rule, safety_stock, lead_time_days, supplier_id)
               VALUES ('FG-100', 'Finished Good 100', 'finished_good', 'EA', 120.50, NULL, 'L4L', 10.0, 7, 'SUP-01')"""
        )

    # Test invalid FK rejection when foreign keys are enabled
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                """INSERT INTO items (item_id, description, type, unit_of_measure, standard_cost,
                   shelf_life_days, lot_sizing_rule, safety_stock, lead_time_days, supplier_id)
                   VALUES ('FG-200', 'Invalid Supplier Item', 'finished_good', 'EA', 50.0, NULL, 'L4L', 0.0, 3, 'NON_EXISTENT_SUPPLIER')"""
            )

    conn.close()

