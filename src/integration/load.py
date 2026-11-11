"""Data Ingestion, Validation, and BOM Integrity Module for ROLLINGMRP.

Handles:
- Ingestion of raw factory datasets
- Data hygiene & cleaning (missing values, duplicates, range checks)
- BOM structural validation (cycle detection, orphan components, self-references)
- Inventory status initialization (PAB_0, allocated netting, safety stock merge)
- Data quality reporting
- Processed data export to `data/processed/`
"""

import sys
from pathlib import Path

# Add project root to sys.path if running as standalone script
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np


class BOMValidationError(Exception):
    """Exception raised when Bill of Materials fails structural integrity checks."""
    pass


class DataQualityError(Exception):
    """Exception raised when critical data quality violations are detected."""
    pass


def detect_bom_cycles(df_bom: pd.DataFrame) -> Optional[List[str]]:
    """Detect cycles in BOM graph using Depth-First Search (DFS) with node coloring.

    Returns:
        List of item_ids representing the detected cycle (e.g. ['A', 'B', 'A']),
        or None if the BOM is a valid Directed Acyclic Graph (DAG).
    """
    # Build adjacency list: parent -> list of components
    adj: Dict[str, List[str]] = {}
    for _, row in df_bom.iterrows():
        p, c = str(row["parent_item_id"]), str(row["component_item_id"])
        if p not in adj:
            adj[p] = []
        adj[p].append(c)

    # 0 = unvisited, 1 = visiting (in current DFS call stack), 2 = completely visited
    visited_state: Dict[str, int] = {}
    cycle_path: List[str] = []

    def dfs(node: str, path: List[str]) -> bool:
        visited_state[node] = 1
        path.append(node)

        for child in adj.get(node, []):
            if visited_state.get(child, 0) == 1:
                # Cycle found
                cycle_start_idx = path.index(child)
                cycle_path.extend(path[cycle_start_idx:] + [child])
                return True
            elif visited_state.get(child, 0) == 0:
                if dfs(child, path):
                    return True

        path.pop()
        visited_state[node] = 2
        return False

    all_nodes = set(df_bom["parent_item_id"].astype(str)).union(set(df_bom["component_item_id"].astype(str)))
    for node in all_nodes:
        if visited_state.get(node, 0) == 0:
            if dfs(node, []):
                return cycle_path

    return None


def validate_bom_structure(df_bom: pd.DataFrame, df_items: pd.DataFrame) -> List[str]:
    """Validate BOM structural constraints:
    - No self-referential edges
    - No orphan components (all parent & component IDs must exist in items)
    - Quantity sanity (quantity_per > 0, 0 <= scrap_factor < 1.0)
    - Directed Acyclic Graph (DAG) validation (no circular dependencies)

    Raises:
        BOMValidationError: If any critical BOM violation or cycle is detected.
    Returns:
        List of warning/notice strings if non-critical issues exist.
    """
    warnings: List[str] = []
    item_ids = set(df_items["item_id"].astype(str))

    # 1. Self-reference check
    self_refs = df_bom[df_bom["parent_item_id"] == df_bom["component_item_id"]]
    if not self_refs.empty:
        bad_items = self_refs["parent_item_id"].tolist()
        raise BOMValidationError(f"Self-referential BOM edges detected for items: {bad_items}")

    # 2. Orphan check
    missing_parents = set(df_bom["parent_item_id"].astype(str)) - item_ids
    if missing_parents:
        raise BOMValidationError(f"BOM parents do not exist in item master: {missing_parents}")

    missing_components = set(df_bom["component_item_id"].astype(str)) - item_ids
    if missing_components:
        raise BOMValidationError(f"BOM components do not exist in item master: {missing_components}")

    # 3. Quantity sanity
    invalid_qty = df_bom[df_bom["quantity_per"] <= 0]
    if not invalid_qty.empty:
        raise BOMValidationError(f"BOM edges with non-positive quantity_per: {invalid_qty[['parent_item_id', 'component_item_id']].to_dict('records')}")

    invalid_scrap = df_bom[(df_bom["scrap_factor"] < 0) | (df_bom["scrap_factor"] >= 1.0)]
    if not invalid_scrap.empty:
        raise BOMValidationError(f"BOM edges with invalid scrap_factor (<0 or >=1.0): {invalid_scrap[['parent_item_id', 'component_item_id']].to_dict('records')}")

    # 4. Cycle detection
    cycle = detect_bom_cycles(df_bom)
    if cycle:
        cycle_str = " -> ".join(cycle)
        raise BOMValidationError(f"Cycle detected in BOM structure: {cycle_str}")

    return warnings


def clean_and_validate_dataset(raw_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Clean raw dataset and perform comprehensive data quality validation.

    Returns:
        Tuple of (cleaned_data_dict, quality_report)
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    report: Dict[str, Any] = {
        "critical_issues": [],
        "warnings": [],
        "summary": {},
    }

    # --- 1. Suppliers ---
    df_sup = raw_data["suppliers"].copy().drop_duplicates(subset=["supplier_id"])
    if (df_sup["reliability_score"] < 0).any() or (df_sup["reliability_score"] > 1).any():
        report["critical_issues"].append("Supplier reliability_score out of [0, 1] range")
    if (df_sup["quoted_lead_time_days"] < 0).any():
        report["critical_issues"].append("Negative supplier quoted lead time detected")
    cleaned["suppliers"] = df_sup

    # --- 2. Items ---
    df_items = raw_data["items"].copy().drop_duplicates(subset=["item_id"])
    valid_types = {"raw_material", "component", "subassembly", "finished_good"}
    invalid_types = set(df_items["type"]) - valid_types
    if invalid_types:
        report["critical_issues"].append(f"Invalid item types found: {invalid_types}")
    if (df_items["standard_cost"] < 0).any():
        report["critical_issues"].append("Negative standard_cost detected in items")
    if (df_items["lead_time_days"] < 0).any():
        report["critical_issues"].append("Negative lead_time_days detected in items")
    if (df_items["safety_stock"] < 0).any():
        report["critical_issues"].append("Negative safety_stock detected in items")
    cleaned["items"] = df_items

    # --- 3. BOM ---
    df_bom = raw_data["bom"].copy().drop_duplicates(subset=["parent_item_id", "component_item_id"])
    try:
        bom_warnings = validate_bom_structure(df_bom, df_items)
        report["warnings"].extend(bom_warnings)
    except BOMValidationError as e:
        report["critical_issues"].append(f"BOM Validation Error: {str(e)}")
    cleaned["bom"] = df_bom

    # --- 4. Inventory ---
    df_inv = raw_data["inventory"].copy().drop_duplicates(subset=["item_id", "location_id"])
    if (df_inv["quantity_on_hand"] < 0).any():
        report["critical_issues"].append("Negative quantity_on_hand detected in inventory")
    if (df_inv["quantity_allocated"] < 0).any():
        report["critical_issues"].append("Negative quantity_allocated detected in inventory")
    if (df_inv["quantity_on_order"] < 0).any():
        report["critical_issues"].append("Negative quantity_on_order detected in inventory")
    # Check over-allocation warning
    over_allocated = df_inv[df_inv["quantity_allocated"] > df_inv["quantity_on_hand"]]
    if not over_allocated.empty:
        report["warnings"].append(f"{len(over_allocated)} items have allocated quantity > on_hand quantity")
    cleaned["inventory"] = df_inv

    # --- 5. Work Centers ---
    df_wc = raw_data["work_centers"].copy().drop_duplicates(subset=["work_center_id"])
    if (df_wc["capacity_hours_per_week"] <= 0).any():
        report["critical_issues"].append("Non-positive work center weekly capacity detected")
    if (df_wc["efficiency_factor"] <= 0).any():
        report["critical_issues"].append("Non-positive work center efficiency detected")
    cleaned["work_centers"] = df_wc

    # --- 6. Routings ---
    df_routings = raw_data["routings"].copy().drop_duplicates(subset=["item_id", "operation_sequence"])
    if (df_routings["setup_time_hours"] < 0).any() or (df_routings["run_time_hours_per_unit"] < 0).any():
        report["critical_issues"].append("Negative routing setup or run times detected")
    cleaned["routings"] = df_routings

    # --- 7. MPS ---
    df_mps = raw_data["mps"].copy().drop_duplicates(subset=["item_id", "period_number", "demand_type"])
    if (df_mps["quantity"] < 0).any():
        report["critical_issues"].append("Negative MPS quantities detected")
    if (df_mps["period_number"] < 1).any():
        report["critical_issues"].append("Invalid MPS period numbers (< 1) detected")
    cleaned["mps"] = df_mps

    # Summary
    report["summary"] = {
        "suppliers_count": len(cleaned["suppliers"]),
        "items_count": len(cleaned["items"]),
        "bom_edges_count": len(cleaned["bom"]),
        "inventory_records": len(cleaned["inventory"]),
        "work_centers_count": len(cleaned["work_centers"]),
        "routings_count": len(cleaned["routings"]),
        "mps_records": len(cleaned["mps"]),
        "critical_issues_count": len(report["critical_issues"]),
        "warnings_count": len(report["warnings"]),
    }

    if report["critical_issues"]:
        raise DataQualityError(f"Data quality violations encountered: {report['critical_issues']}")

    return cleaned, report


def initialize_inventory_status(df_inventory: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:
    """Initialize starting inventory status table.

    Computes:
    - initial_available_unallocated = quantity_on_hand - quantity_allocated
    - merges type, description, safety_stock, standard_cost from item master
    - initial_inventory_value = quantity_on_hand * standard_cost
    """
    candidate_cols = ["item_id", "type", "description", "safety_stock", "standard_cost", "lot_sizing_rule", "lead_time_days"]
    merge_cols = [c for c in candidate_cols if c in df_items.columns]
    merged = df_inventory.merge(
        df_items[merge_cols],
        on="item_id",
        how="left",
    )
    merged["net_available_start"] = merged["quantity_on_hand"] - merged["quantity_allocated"]
    merged["inventory_value"] = merged["quantity_on_hand"] * merged["standard_cost"]
    return merged


def save_processed_data(processed_data: Dict[str, pd.DataFrame], output_dir: Optional[Path] = None) -> Path:
    """Save cleaned datasets to data/processed/ directory."""
    out_dir = output_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in processed_data.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return out_dir


def run_pipeline(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """End-to-end ingestion and validation pipeline from raw CSVs."""
    r_dir = raw_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "raw")
    tables = ["suppliers", "items", "bom", "inventory", "work_centers", "routings", "mps"]

    raw_data: Dict[str, pd.DataFrame] = {}
    for table in tables:
        csv_file = r_dir / f"{table}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Raw data file missing: {csv_file}")
        raw_data[table] = pd.read_csv(csv_file)

    cleaned_data, report = clean_and_validate_dataset(raw_data)
    # Compute initialized inventory status
    cleaned_data["inventory_initialized"] = initialize_inventory_status(cleaned_data["inventory"], cleaned_data["items"])

    save_processed_data(cleaned_data, output_dir=processed_dir)
    return cleaned_data, report


if __name__ == "__main__":
    print("Running Data Ingestion and Validation Pipeline...")
    cleaned, quality_report = run_pipeline()
    print("Pipeline completed successfully!")
    print("Summary:", quality_report["summary"])
    print("Critical issues:", quality_report["summary"]["critical_issues_count"])
    print("Warnings:", quality_report["summary"]["warnings_count"])
