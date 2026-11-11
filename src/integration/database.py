"""Data Access Layer (DAL) for ROLLINGMRP.

Provides:
- Thread-safe / process-aware Connection Manager singleton with context manager
- Atomic transaction management with automatic rollback on error
- CRUD operations for ITEM, BOM, INVENTORY, and MPS tables
- High-level query methods: get_bom_children, get_inventory, get_mps
- Batch operations for high-throughput planning updates
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "rollingmrp.db"


class DatabaseManager:
    """Connection Manager and DAL provider implementing a singleton pattern."""

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "DatabaseManager":
        """Get or initialize singleton instance."""
        with cls._lock:
            if cls._instance is None or (db_path and Path(db_path) != cls._instance.db_path):
                cls._instance = cls(db_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (primarily for testing with custom tmp databases)."""
        with cls._lock:
            cls._instance = None

    def _create_connection(self) -> sqlite3.Connection:
        """Create new sqlite connection with foreign keys and dict-like row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an open SQLite connection."""
        conn = self._create_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager providing an atomic transaction (auto-commit / auto-rollback)."""
        conn = self._create_connection()
        cursor = conn.cursor()
        try:
            conn.execute("BEGIN TRANSACTION;")
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    # =========================================================================
    # ITEM CRUD Operations
    # =========================================================================
    def create_item(self, item_data: Dict[str, Any]) -> str:
        """Insert a new item record. Returns item_id."""
        query = """
            INSERT INTO items (
                item_id, description, type, unit_of_measure, standard_cost,
                shelf_life_days, lot_sizing_rule, safety_stock, lead_time_days, supplier_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            item_data["item_id"],
            item_data["description"],
            item_data["type"],
            item_data["unit_of_measure"],
            float(item_data["standard_cost"]),
            item_data.get("shelf_life_days"),
            item_data["lot_sizing_rule"],
            float(item_data.get("safety_stock", 0.0)),
            int(item_data["lead_time_days"]),
            item_data.get("supplier_id"),
        )
        with self.transaction() as cur:
            cur.execute(query, params)
        return item_data["item_id"]

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single item by item_id."""
        query = "SELECT * FROM items WHERE item_id = ?"
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (item_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_all_items(self, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve items, optionally filtered by type."""
        if item_type:
            query = "SELECT * FROM items WHERE type = ? ORDER BY item_id"
            params = (item_type,)
        else:
            query = "SELECT * FROM items ORDER BY item_id"
            params = ()

        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def update_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update item fields dynamically. Returns True if row updated."""
        if not updates:
            return False
        fields = [f"{k} = ?" for k in updates.keys()]
        values = list(updates.values()) + [item_id]
        query = f"UPDATE items SET {', '.join(fields)} WHERE item_id = ?"

        with self.transaction() as cur:
            cur.execute(query, values)
            return cur.rowcount > 0

    def delete_item(self, item_id: str) -> bool:
        """Delete item by item_id. Returns True if deleted."""
        with self.transaction() as cur:
            cur.execute("DELETE FROM items WHERE item_id = ?", (item_id,))
            return cur.rowcount > 0

    # =========================================================================
    # BOM CRUD & Query Operations
    # =========================================================================
    def create_bom_entry(self, bom_data: Dict[str, Any]) -> str:
        """Insert a BOM relationship edge. Returns bom_id."""
        query = """
            INSERT INTO bom (
                bom_id, parent_item_id, component_item_id, quantity_per,
                scrap_factor, effective_date, obsolete_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            bom_data["bom_id"],
            bom_data["parent_item_id"],
            bom_data["component_item_id"],
            float(bom_data["quantity_per"]),
            float(bom_data.get("scrap_factor", 0.0)),
            bom_data.get("effective_date"),
            bom_data.get("obsolete_date"),
        )
        with self.transaction() as cur:
            cur.execute(query, params)
        return bom_data["bom_id"]

    def get_bom_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """Query direct child components for given parent_id with item metadata joined."""
        query = """
            SELECT b.*, i.description, i.type, i.lead_time_days, i.lot_sizing_rule, i.safety_stock, i.standard_cost
            FROM bom b
            JOIN items i ON b.component_item_id = i.item_id
            WHERE b.parent_item_id = ?
            ORDER BY b.component_item_id
        """
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (parent_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_bom_parents(self, component_id: str) -> List[Dict[str, Any]]:
        """Query all direct parents that consume given component_id (supports pegging)."""
        query = """
            SELECT b.*, i.description, i.type
            FROM bom b
            JOIN items i ON b.parent_item_id = i.item_id
            WHERE b.component_item_id = ?
            ORDER BY b.parent_item_id
        """
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (component_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_all_bom(self) -> List[Dict[str, Any]]:
        """Retrieve all BOM relationships."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM bom ORDER BY parent_item_id, component_item_id")
            return [dict(row) for row in cur.fetchall()]

    # =========================================================================
    # INVENTORY Operations
    # =========================================================================
    def upsert_inventory(self, inv_data: Dict[str, Any]) -> int:
        """Insert or update inventory for an item at a location."""
        query = """
            INSERT INTO inventory (item_id, location_id, quantity_on_hand, quantity_allocated, quantity_on_order, expiration_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id, location_id) DO UPDATE SET
                quantity_on_hand = excluded.quantity_on_hand,
                quantity_allocated = excluded.quantity_allocated,
                quantity_on_order = excluded.quantity_on_order,
                expiration_date = excluded.expiration_date,
                updated_at = CURRENT_TIMESTAMP
        """
        params = (
            inv_data["item_id"],
            inv_data.get("location_id", "WH-MAIN"),
            float(inv_data.get("quantity_on_hand", 0.0)),
            float(inv_data.get("quantity_allocated", 0.0)),
            float(inv_data.get("quantity_on_order", 0.0)),
            inv_data.get("expiration_date"),
        )
        with self.transaction() as cur:
            cur.execute(query, params)
            return cur.lastrowid

    def get_inventory(self, item_id: str, location_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve inventory status for an item."""
        if location_id:
            query = "SELECT * FROM inventory WHERE item_id = ? AND location_id = ?"
            params = (item_id, location_id)
        else:
            query = "SELECT * FROM inventory WHERE item_id = ? ORDER BY location_id LIMIT 1"
            params = (item_id,)

        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def update_inventory_quantities(
        self,
        item_id: str,
        location_id: str = "WH-MAIN",
        on_hand: Optional[float] = None,
        allocated: Optional[float] = None,
        on_order: Optional[float] = None,
    ) -> bool:
        """Update specific inventory quantities for an item."""
        updates: List[str] = []
        params: List[Any] = []
        if on_hand is not None:
            updates.append("quantity_on_hand = ?")
            params.append(float(on_hand))
        if allocated is not None:
            updates.append("quantity_allocated = ?")
            params.append(float(allocated))
        if on_order is not None:
            updates.append("quantity_on_order = ?")
            params.append(float(on_order))

        if not updates:
            return False

        params.extend([item_id, location_id])
        query = f"UPDATE inventory SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE item_id = ? AND location_id = ?"
        with self.transaction() as cur:
            cur.execute(query, params)
            return cur.rowcount > 0

    # =========================================================================
    # MPS Operations
    # =========================================================================
    def create_mps_entry(self, mps_data: Dict[str, Any]) -> str:
        """Insert or replace a master production schedule demand record."""
        query = """
            INSERT INTO mps (mps_id, item_id, period_number, quantity, demand_type, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id, period_number, demand_type) DO UPDATE SET
                quantity = excluded.quantity,
                source = excluded.source
        """
        params = (
            mps_data["mps_id"],
            mps_data["item_id"],
            int(mps_data["period_number"]),
            float(mps_data["quantity"]),
            mps_data["demand_type"],
            mps_data["source"],
        )
        with self.transaction() as cur:
            cur.execute(query, params)
        return mps_data["mps_id"]

    def get_mps(
        self,
        item_id: Optional[str] = None,
        period_range: Optional[Tuple[int, int]] = None,
        demand_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve MPS demand records with optional filtering by item, period range, and demand type."""
        clauses = []
        params: List[Any] = []

        if item_id:
            clauses.append("item_id = ?")
            params.append(item_id)
        if period_range:
            start_p, end_p = period_range
            clauses.append("period_number BETWEEN ? AND ?")
            params.extend([start_p, end_p])
        if demand_type:
            clauses.append("demand_type = ?")
            params.append(demand_type)

        where_stmt = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM mps {where_stmt} ORDER BY item_id, period_number"

        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def batch_insert_mps(self, mps_records: List[Dict[str, Any]]) -> int:
        """Batch insert MPS records inside a single transaction."""
        query = """
            INSERT INTO mps (mps_id, item_id, period_number, quantity, demand_type, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id, period_number, demand_type) DO UPDATE SET
                quantity = excluded.quantity,
                source = excluded.source
        """
        param_list = [
            (
                r["mps_id"],
                r["item_id"],
                int(r["period_number"]),
                float(r["quantity"]),
                r["demand_type"],
                r["source"],
            )
            for r in mps_records
        ]
        with self.transaction() as cur:
            cur.executemany(query, param_list)
            return cur.rowcount


def get_db(db_path: Optional[Path] = None) -> DatabaseManager:
    """Convenience factory returning DatabaseManager instance."""
    return DatabaseManager.get_instance(db_path)

