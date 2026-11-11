"""Synthetic Factory Data Generator for ROLLINGMRP.

Generates realistic, reproducible test data for:
- 4 Suppliers with distinct reliability & lead time profiles
- 25 Items across 3 BOM levels (5 Finished Goods, 8 Subassemblies, 12 Raw Materials)
- 3-level BOM tree with component commonality
- 3 Work Centers with varying capacities and efficiencies
- Routings with operation sequences, setup times, and run times per unit
- Initial Inventory state (on-hand, allocated, on-order)
- 12-week MPS demand with trend, seasonality, and stochastic noise
"""

import sys
from pathlib import Path

# Add project root to sys.path if running as standalone script
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd
from src.integration.db import get_connection, init_db


def generate_synthetic_data(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Generate reproducible synthetic dataset for the ROLLINGMRP factory."""
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # 1. SUPPLIERS (4 suppliers with distinct risk & capability profiles)
    # -------------------------------------------------------------------------
    suppliers_data = [
        {
            "supplier_id": "SUP-01",
            "name": "Apex Precision Metalworks",
            "reliability_score": 0.98,
            "quoted_lead_time_days": 7,
            "actual_lead_time_mean": 7.1,
            "actual_lead_time_stddev": 0.6,
            "fill_rate": 0.99,
            "quality_reject_rate": 0.01,
        },
        {
            "supplier_id": "SUP-02",
            "name": "Global ElectroComponent Co",
            "reliability_score": 0.88,
            "quoted_lead_time_days": 14,
            "actual_lead_time_mean": 15.2,
            "actual_lead_time_stddev": 2.8,
            "fill_rate": 0.91,
            "quality_reject_rate": 0.04,
        },
        {
            "supplier_id": "SUP-03",
            "name": "FastTrack Polymers & Fasteners",
            "reliability_score": 0.94,
            "quoted_lead_time_days": 5,
            "actual_lead_time_mean": 5.2,
            "actual_lead_time_stddev": 0.9,
            "fill_rate": 0.96,
            "quality_reject_rate": 0.02,
        },
        {
            "supplier_id": "SUP-04",
            "name": "Budget Wholesale Raw Materials",
            "reliability_score": 0.74,
            "quoted_lead_time_days": 10,
            "actual_lead_time_mean": 13.5,
            "actual_lead_time_stddev": 4.2,
            "fill_rate": 0.80,
            "quality_reject_rate": 0.08,
        },
    ]
    df_suppliers = pd.DataFrame(suppliers_data)

    # -------------------------------------------------------------------------
    # 2. ITEMS (25 items: 5 Finished Goods, 8 Subassemblies, 12 Raw Materials)
    # -------------------------------------------------------------------------
    items_data = [
        # --- Finished Goods (5 items: Level 0) ---
        {
            "item_id": "FG-01",
            "description": "Industrial Robot Arm (Alpha)",
            "type": "finished_good",
            "unit_of_measure": "EA",
            "standard_cost": 550.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "L4L",
            "safety_stock": 25.0,
            "lead_time_days": 7,  # 1 week production
            "supplier_id": None,
        },
        {
            "item_id": "FG-02",
            "description": "Precision Actuator Module (Beta)",
            "type": "finished_good",
            "unit_of_measure": "EA",
            "standard_cost": 420.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "L4L",
            "safety_stock": 30.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "FG-03",
            "description": "Conveyor Drive Unit (Gamma)",
            "type": "finished_good",
            "unit_of_measure": "EA",
            "standard_cost": 610.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "L4L",
            "safety_stock": 20.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "FG-04",
            "description": "Automated Feeder Unit (Delta)",
            "type": "finished_good",
            "unit_of_measure": "EA",
            "standard_cost": 380.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "L4L",
            "safety_stock": 15.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "FG-05",
            "description": "Smart Sensor Hub (Epsilon)",
            "type": "finished_good",
            "unit_of_measure": "EA",
            "standard_cost": 290.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 35.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        # --- Subassemblies (8 items: Level 1) ---
        {
            "item_id": "SA-01",
            "description": "Servo Motor Core Assembly",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 140.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "L4L",
            "safety_stock": 20.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "SA-02",
            "description": "Planetary Gearbox Reducer",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 115.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "POQ",
            "safety_stock": 15.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "SA-03",
            "description": "Structural Robotic Joint Frame",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 85.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "L4L",
            "safety_stock": 10.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "SA-04",
            "description": "Microcontroller Circuit Board Assembly",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 95.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "WAGNER_WHITIN",
            "safety_stock": 25.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "SA-05",
            "description": "Linear Motion Carriage",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 75.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 15.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "SA-06",
            "description": "Heavy-Duty Base Mounting Plate",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 65.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "POQ",
            "safety_stock": 10.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "SA-07",
            "description": "Power Inverter & Filter Unit",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 105.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "L4L",
            "safety_stock": 20.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        {
            "item_id": "SA-08",
            "description": "Multi-Channel Optical Sensor Array",
            "type": "subassembly",
            "unit_of_measure": "EA",
            "standard_cost": 80.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "SILVER_MEAL",
            "safety_stock": 15.0,
            "lead_time_days": 7,
            "supplier_id": None,
        },
        # --- Raw Materials (12 items: Level 2) ---
        {
            "item_id": "RM-01",
            "description": "CNC Machined Alloy Rod 6061",
            "type": "raw_material",
            "unit_of_measure": "KG",
            "standard_cost": 18.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 100.0,
            "lead_time_days": 7,
            "supplier_id": "SUP-01",
        },
        {
            "item_id": "RM-02",
            "description": "High-Grade Stainless Steel Plate",
            "type": "raw_material",
            "unit_of_measure": "KG",
            "standard_cost": 22.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 80.0,
            "lead_time_days": 7,
            "supplier_id": "SUP-01",
        },
        {
            "item_id": "RM-03",
            "description": "Hardened Steel Ball Bearings Pack",
            "type": "raw_material",
            "unit_of_measure": "EA",
            "standard_cost": 8.5,
            "shelf_life_days": None,
            "lot_sizing_rule": "POQ",
            "safety_stock": 150.0,
            "lead_time_days": 7,
            "supplier_id": "SUP-01",
        },
        {
            "item_id": "RM-04",
            "description": "SMD Microcontroller IC Chip",
            "type": "raw_material",
            "unit_of_measure": "EA",
            "standard_cost": 28.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 120.0,
            "lead_time_days": 14,
            "supplier_id": "SUP-02",
        },
        {
            "item_id": "RM-05",
            "description": "Power MOSFET Module",
            "type": "raw_material",
            "unit_of_measure": "EA",
            "standard_cost": 14.0,
            "shelf_life_days": None,
            "lot_sizing_rule": "POQ",
            "safety_stock": 90.0,
            "lead_time_days": 14,
            "supplier_id": "SUP-02",
        },
        {
            "item_id": "RM-06",
            "description": "Multi-Layer Ceramic Capacitors Reel",
            "type": "raw_material",
            "unit_of_measure": "EA",
            "standard_cost": 4.5,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 250.0,
            "lead_time_days": 14,
            "supplier_id": "SUP-02",
        },
        {
            "item_id": "RM-07",
            "description": "Engineering Thermoplastic Housing Resin",
            "type": "raw_material",
            "unit_of_measure": "KG",
            "standard_cost": 9.0,
            "shelf_life_days": 365,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 140.0,
            "lead_time_days": 5,
            "supplier_id": "SUP-03",
        },
        {
            "item_id": "RM-08",
            "description": "Precision Fastener Kit (M4/M6 Bolts)",
            "type": "raw_material",
            "unit_of_measure": "EA",
            "standard_cost": 2.5,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 300.0,
            "lead_time_days": 5,
            "supplier_id": "SUP-03",
        },
        {
            "item_id": "RM-09",
            "description": "EPDM Rubber Sealing Gaskets",
            "type": "raw_material",
            "unit_of_measure": "EA",
            "standard_cost": 3.8,
            "shelf_life_days": 730,
            "lot_sizing_rule": "POQ",
            "safety_stock": 100.0,
            "lead_time_days": 5,
            "supplier_id": "SUP-03",
        },
        {
            "item_id": "RM-10",
            "description": "Industrial Copper Magnet Wire",
            "type": "raw_material",
            "unit_of_measure": "KG",
            "standard_cost": 16.5,
            "shelf_life_days": None,
            "lot_sizing_rule": "EOQ",
            "safety_stock": 70.0,
            "lead_time_days": 10,
            "supplier_id": "SUP-04",
        },
        {
            "item_id": "RM-11",
            "description": "Neodymium Rotor Magnets",
            "type": "raw_material",
            "unit_of_measure": "EA",
            "standard_cost": 19.5,
            "shelf_life_days": None,
            "lot_sizing_rule": "POQ",
            "safety_stock": 60.0,
            "lead_time_days": 10,
            "supplier_id": "SUP-04",
        },
        {
            "item_id": "RM-12",
            "description": "Silicone Thermal Conductive Paste",
            "type": "raw_material",
            "unit_of_measure": "KG",
            "standard_cost": 11.0,
            "shelf_life_days": 180,
            "lot_sizing_rule": "L4L",
            "safety_stock": 40.0,
            "lead_time_days": 10,
            "supplier_id": "SUP-04",
        },
    ]
    df_items = pd.DataFrame(items_data)

    # -------------------------------------------------------------------------
    # 3. BILL OF MATERIALS (BOM) — 3 Levels with component commonality
    # -------------------------------------------------------------------------
    # Level 0 -> Level 1 (Finished Goods -> Subassemblies)
    # Level 1 -> Level 2 (Subassemblies -> Raw Materials)
    bom_data = [
        # --- FG-01: Robot Arm ---
        {"bom_id": "BOM-001", "parent_item_id": "FG-01", "component_item_id": "SA-01", "quantity_per": 2.0, "scrap_factor": 0.01},
        {"bom_id": "BOM-002", "parent_item_id": "FG-01", "component_item_id": "SA-02", "quantity_per": 2.0, "scrap_factor": 0.01},
        {"bom_id": "BOM-003", "parent_item_id": "FG-01", "component_item_id": "SA-03", "quantity_per": 1.0, "scrap_factor": 0.02},
        {"bom_id": "BOM-004", "parent_item_id": "FG-01", "component_item_id": "SA-04", "quantity_per": 1.0, "scrap_factor": 0.01},

        # --- FG-02: Actuator Module ---
        {"bom_id": "BOM-005", "parent_item_id": "FG-02", "component_item_id": "SA-01", "quantity_per": 1.0, "scrap_factor": 0.01}, # common SA-01
        {"bom_id": "BOM-006", "parent_item_id": "FG-02", "component_item_id": "SA-05", "quantity_per": 1.0, "scrap_factor": 0.01},
        {"bom_id": "BOM-007", "parent_item_id": "FG-02", "component_item_id": "SA-04", "quantity_per": 1.0, "scrap_factor": 0.01}, # common SA-04

        # --- FG-03: Conveyor Drive ---
        {"bom_id": "BOM-008", "parent_item_id": "FG-03", "component_item_id": "SA-01", "quantity_per": 2.0, "scrap_factor": 0.02}, # common SA-01
        {"bom_id": "BOM-009", "parent_item_id": "FG-03", "component_item_id": "SA-02", "quantity_per": 1.0, "scrap_factor": 0.01}, # common SA-02
        {"bom_id": "BOM-010", "parent_item_id": "FG-03", "component_item_id": "SA-06", "quantity_per": 1.0, "scrap_factor": 0.01},
        {"bom_id": "BOM-011", "parent_item_id": "FG-03", "component_item_id": "SA-07", "quantity_per": 1.0, "scrap_factor": 0.01},

        # --- FG-04: Feeder Unit ---
        {"bom_id": "BOM-012", "parent_item_id": "FG-04", "component_item_id": "SA-03", "quantity_per": 1.0, "scrap_factor": 0.01}, # common SA-03
        {"bom_id": "BOM-013", "parent_item_id": "FG-04", "component_item_id": "SA-05", "quantity_per": 1.0, "scrap_factor": 0.01}, # common SA-05
        {"bom_id": "BOM-014", "parent_item_id": "FG-04", "component_item_id": "SA-06", "quantity_per": 1.0, "scrap_factor": 0.02}, # common SA-06

        # --- FG-05: Sensor Hub ---
        {"bom_id": "BOM-015", "parent_item_id": "FG-05", "component_item_id": "SA-04", "quantity_per": 1.0, "scrap_factor": 0.01}, # common SA-04
        {"bom_id": "BOM-016", "parent_item_id": "FG-05", "component_item_id": "SA-07", "quantity_per": 1.0, "scrap_factor": 0.01}, # common SA-07
        {"bom_id": "BOM-017", "parent_item_id": "FG-05", "component_item_id": "SA-08", "quantity_per": 2.0, "scrap_factor": 0.01},

        # --- SA-01 Components (Servo Motor Core) ---
        {"bom_id": "BOM-018", "parent_item_id": "SA-01", "component_item_id": "RM-01", "quantity_per": 1.5, "scrap_factor": 0.02},
        {"bom_id": "BOM-019", "parent_item_id": "SA-01", "component_item_id": "RM-03", "quantity_per": 2.0, "scrap_factor": 0.01},
        {"bom_id": "BOM-020", "parent_item_id": "SA-01", "component_item_id": "RM-10", "quantity_per": 1.2, "scrap_factor": 0.03},
        {"bom_id": "BOM-021", "parent_item_id": "SA-01", "component_item_id": "RM-11", "quantity_per": 4.0, "scrap_factor": 0.01},

        # --- SA-02 Components (Planetary Gearbox) ---
        {"bom_id": "BOM-022", "parent_item_id": "SA-02", "component_item_id": "RM-01", "quantity_per": 2.0, "scrap_factor": 0.02}, # common RM-01
        {"bom_id": "BOM-023", "parent_item_id": "SA-02", "component_item_id": "RM-03", "quantity_per": 4.0, "scrap_factor": 0.01}, # common RM-03
        {"bom_id": "BOM-024", "parent_item_id": "SA-02", "component_item_id": "RM-08", "quantity_per": 8.0, "scrap_factor": 0.01},

        # --- SA-03 Components (Robotic Joint Frame) ---
        {"bom_id": "BOM-025", "parent_item_id": "SA-03", "component_item_id": "RM-01", "quantity_per": 1.0, "scrap_factor": 0.01}, # common RM-01
        {"bom_id": "BOM-026", "parent_item_id": "SA-03", "component_item_id": "RM-02", "quantity_per": 2.5, "scrap_factor": 0.02},
        {"bom_id": "BOM-027", "parent_item_id": "SA-03", "component_item_id": "RM-08", "quantity_per": 6.0, "scrap_factor": 0.01}, # common RM-08

        # --- SA-04 Components (Microcontroller Circuit) ---
        {"bom_id": "BOM-028", "parent_item_id": "SA-04", "component_item_id": "RM-04", "quantity_per": 1.0, "scrap_factor": 0.01},
        {"bom_id": "BOM-029", "parent_item_id": "SA-04", "component_item_id": "RM-05", "quantity_per": 2.0, "scrap_factor": 0.01},
        {"bom_id": "BOM-030", "parent_item_id": "SA-04", "component_item_id": "RM-06", "quantity_per": 10.0, "scrap_factor": 0.02},
        {"bom_id": "BOM-031", "parent_item_id": "SA-04", "component_item_id": "RM-12", "quantity_per": 0.1, "scrap_factor": 0.01},

        # --- SA-05 Components (Linear Motion Carriage) ---
        {"bom_id": "BOM-032", "parent_item_id": "SA-05", "component_item_id": "RM-02", "quantity_per": 1.8, "scrap_factor": 0.02}, # common RM-02
        {"bom_id": "BOM-033", "parent_item_id": "SA-05", "component_item_id": "RM-03", "quantity_per": 2.0, "scrap_factor": 0.01}, # common RM-03
        {"bom_id": "BOM-034", "parent_item_id": "SA-05", "component_item_id": "RM-08", "quantity_per": 4.0, "scrap_factor": 0.01}, # common RM-08

        # --- SA-06 Components (Heavy Base Plate) ---
        {"bom_id": "BOM-035", "parent_item_id": "SA-06", "component_item_id": "RM-02", "quantity_per": 4.0, "scrap_factor": 0.01}, # common RM-02
        {"bom_id": "BOM-036", "parent_item_id": "SA-06", "component_item_id": "RM-08", "quantity_per": 6.0, "scrap_factor": 0.01}, # common RM-08
        {"bom_id": "BOM-037", "parent_item_id": "SA-06", "component_item_id": "RM-09", "quantity_per": 2.0, "scrap_factor": 0.02},

        # --- SA-07 Components (Power Inverter Unit) ---
        {"bom_id": "BOM-038", "parent_item_id": "SA-07", "component_item_id": "RM-05", "quantity_per": 4.0, "scrap_factor": 0.01}, # common RM-05
        {"bom_id": "BOM-039", "parent_item_id": "SA-07", "component_item_id": "RM-06", "quantity_per": 8.0, "scrap_factor": 0.02}, # common RM-06
        {"bom_id": "BOM-040", "parent_item_id": "SA-07", "component_item_id": "RM-07", "quantity_per": 0.8, "scrap_factor": 0.02},
        {"bom_id": "BOM-041", "parent_item_id": "SA-07", "component_item_id": "RM-12", "quantity_per": 0.2, "scrap_factor": 0.01}, # common RM-12

        # --- SA-08 Components (Sensor Array) ---
        {"bom_id": "BOM-042", "parent_item_id": "SA-08", "component_item_id": "RM-04", "quantity_per": 1.0, "scrap_factor": 0.01}, # common RM-04
        {"bom_id": "BOM-043", "parent_item_id": "SA-08", "component_item_id": "RM-07", "quantity_per": 0.5, "scrap_factor": 0.02}, # common RM-07
        {"bom_id": "BOM-044", "parent_item_id": "SA-08", "component_item_id": "RM-08", "quantity_per": 4.0, "scrap_factor": 0.01}, # common RM-08
        {"bom_id": "BOM-045", "parent_item_id": "SA-08", "component_item_id": "RM-09", "quantity_per": 1.0, "scrap_factor": 0.01}, # common RM-09
    ]
    df_bom = pd.DataFrame(bom_data)
    df_bom["effective_date"] = "2026-01-01"
    df_bom["obsolete_date"] = None

    # -------------------------------------------------------------------------
    # 4. WORK CENTERS (3 centers: Machining, Subassembly, Final Assembly)
    # -------------------------------------------------------------------------
    work_centers_data = [
        {
            "work_center_id": "WC-100",
            "description": "CNC Machining & Forming Center",
            "capacity_hours_per_week": 160.0,
            "efficiency_factor": 0.95,
            "calendar_id": "CAL-STD-5D",
        },
        {
            "work_center_id": "WC-200",
            "description": "Subassembly & Electronics SMT Line",
            "capacity_hours_per_week": 120.0,
            "efficiency_factor": 0.90,
            "calendar_id": "CAL-STD-5D",
        },
        {
            "work_center_id": "WC-300",
            "description": "Final Assembly & Systems Testing Cell",
            "capacity_hours_per_week": 80.0,  # critical bottleneck candidate
            "efficiency_factor": 0.85,
            "calendar_id": "CAL-STD-5D",
        },
    ]
    df_work_centers = pd.DataFrame(work_centers_data)

    # -------------------------------------------------------------------------
    # 5. ROUTINGS
    # -------------------------------------------------------------------------
    routings_data = []
    # Finished goods routings (Final Assembly & QA in WC-300)
    for idx, fg in enumerate(["FG-01", "FG-02", "FG-03", "FG-04", "FG-05"], start=1):
        routings_data.append({
            "routing_id": f"RTG-{fg}-10",
            "item_id": fg,
            "operation_sequence": 10,
            "work_center_id": "WC-300",
            "setup_time_hours": 1.5 + (idx * 0.2),
            "run_time_hours_per_unit": 0.6 + (idx * 0.1),
        })

    # Subassembly routings (Operations in WC-100 and/or WC-200)
    sa_wc_map = {
        "SA-01": [("WC-100", 1.0, 0.25), ("WC-200", 0.5, 0.20)],
        "SA-02": [("WC-100", 1.2, 0.30)],
        "SA-03": [("WC-100", 1.5, 0.35)],
        "SA-04": [("WC-200", 1.0, 0.22)],
        "SA-05": [("WC-100", 0.8, 0.18)],
        "SA-06": [("WC-100", 1.0, 0.20)],
        "SA-07": [("WC-200", 1.2, 0.25)],
        "SA-08": [("WC-200", 0.8, 0.15)],
    }
    for sa, ops in sa_wc_map.items():
        for seq_idx, (wc_id, setup_h, run_h) in enumerate(ops, start=1):
            routings_data.append({
                "routing_id": f"RTG-{sa}-{seq_idx*10}",
                "item_id": sa,
                "operation_sequence": seq_idx * 10,
                "work_center_id": wc_id,
                "setup_time_hours": setup_h,
                "run_time_hours_per_unit": run_h,
            })
    df_routings = pd.DataFrame(routings_data)

    # -------------------------------------------------------------------------
    # 6. INITIAL INVENTORY
    # -------------------------------------------------------------------------
    inventory_data = []
    location_id = "WH-MAIN"
    for item in items_data:
        item_id = item["item_id"]
        item_type = item["type"]
        ss = item["safety_stock"]

        # Realistic buffer above safety stock
        if item_type == "finished_good":
            on_hand = float(rng.integers(int(ss * 0.8), int(ss * 1.5)))
            allocated = float(rng.integers(0, int(ss * 0.4)))
            on_order = 0.0
        elif item_type == "subassembly":
            on_hand = float(rng.integers(int(ss * 1.0), int(ss * 2.0)))
            allocated = float(rng.integers(0, int(ss * 0.5)))
            on_order = float(rng.choice([0.0, 20.0, 30.0], p=[0.6, 0.2, 0.2]))
        else:  # raw_material
            on_hand = float(rng.integers(int(ss * 1.2), int(ss * 2.5)))
            allocated = float(rng.integers(0, int(ss * 0.3)))
            on_order = float(rng.choice([0.0, 50.0, 100.0], p=[0.5, 0.3, 0.2]))

        inventory_data.append({
            "item_id": item_id,
            "location_id": location_id,
            "quantity_on_hand": on_hand,
            "quantity_allocated": allocated,
            "quantity_on_order": on_order,
            "expiration_date": None if item["shelf_life_days"] is None else "2027-12-31",
        })
    df_inventory = pd.DataFrame(inventory_data)

    # -------------------------------------------------------------------------
    # 7. 12-WEEK MPS DEMAND (Finished Goods: Trend + Seasonality + Noise)
    # -------------------------------------------------------------------------
    mps_data = []
    fg_items = [it["item_id"] for it in items_data if it["type"] == "finished_good"]

    # Parameter settings per finished good
    fg_params = {
        "FG-01": {"base": 45.0, "trend": 1.2, "amp": 8.0, "noise_sd": 3.0},
        "FG-02": {"base": 35.0, "trend": 0.8, "amp": 6.0, "noise_sd": 2.5},
        "FG-03": {"base": 50.0, "trend": 1.5, "amp": 10.0, "noise_sd": 4.0},
        "FG-04": {"base": 30.0, "trend": 0.5, "amp": 5.0, "noise_sd": 2.0},
        "FG-05": {"base": 60.0, "trend": 2.0, "amp": 12.0, "noise_sd": 5.0},
    }

    mps_counter = 1
    for period in range(1, 13):
        # Weeks 1-2: firm customer orders, Weeks 3-12: forecasts
        if period <= 2:
            d_type = "actual"
            src = "customer_order"
        elif period <= 4:
            d_type = "firm_planned"
            src = "customer_order"
        else:
            d_type = "forecast"
            src = "forecast"

        for fg in fg_items:
            params = fg_params[fg]
            trend_val = params["trend"] * period
            season_val = params["amp"] * math.sin(2.0 * math.pi * period / 12.0)
            noise_val = rng.normal(0, params["noise_sd"])
            raw_qty = params["base"] + trend_val + season_val + noise_val
            final_qty = max(5.0, round(raw_qty, 0))

            mps_data.append({
                "mps_id": f"MPS-{mps_counter:04d}",
                "item_id": fg,
                "period_number": period,
                "quantity": float(final_qty),
                "demand_type": d_type,
                "source": src,
            })
            mps_counter += 1

    df_mps = pd.DataFrame(mps_data)

    return {
        "suppliers": df_suppliers,
        "items": df_items,
        "bom": df_bom,
        "inventory": df_inventory,
        "work_centers": df_work_centers,
        "routings": df_routings,
        "mps": df_mps,
    }


def save_synthetic_data_to_csv(data: Dict[str, pd.DataFrame], output_dir: Optional[Path] = None) -> Path:
    """Save all generated DataFrames as CSV files in the raw data directory."""
    out_path = output_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "raw")
    out_path.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(out_path / f"{name}.csv", index=False)
    return out_path


def load_synthetic_data_into_db(data: Dict[str, pd.DataFrame], db_path: Optional[Path] = None) -> None:
    """Initialize SQLite database and populate with synthetic data."""
    conn = get_connection(db_path)
    # Ensure fresh schema
    init_db(db_path=db_path)

    # Insert into tables respecting foreign key order
    with conn:
        data["suppliers"].to_sql("suppliers", conn, if_exists="append", index=False)
        data["items"].to_sql("items", conn, if_exists="append", index=False)
        data["bom"].to_sql("bom", conn, if_exists="append", index=False)
        data["inventory"].to_sql("inventory", conn, if_exists="append", index=False)
        data["work_centers"].to_sql("work_centers", conn, if_exists="append", index=False)
        data["routings"].to_sql("routings", conn, if_exists="append", index=False)
        data["mps"].to_sql("mps", conn, if_exists="append", index=False)

    conn.close()


if __name__ == "__main__":
    print("Generating synthetic factory dataset with seed=42...")
    dataset = generate_synthetic_data(seed=42)
    csv_dir = save_synthetic_data_to_csv(dataset)
    print(f"Saved CSV files to: {csv_dir}")
    load_synthetic_data_into_db(dataset)
    print("Loaded synthetic dataset successfully into data/rollingmrp.db!")
    print(f"Items: {len(dataset['items'])} (FG: {(dataset['items']['type']=='finished_good').sum()}, "
          f"SA: {(dataset['items']['type']=='subassembly').sum()}, "
          f"RM: {(dataset['items']['type']=='raw_material').sum()})")
    print(f"BOM edges: {len(dataset['bom'])}")
    print(f"MPS records: {len(dataset['mps'])} (periods 1-12)")
