"""What-If Scenario Simulation Engine for ROLLINGMRP.

Implements Phase 7.2:
- Parameter Sliders / Adjusters:
    - Demand shift (±%): uniform or SKU-specific demand scaling
    - Lead time delta (±days or ±periods): shifts component & FG replenishment lead times
    - Safety stock shift (±%): expands or contracts buffer stock requirements
    - Work center capacity shift (±hours or ±%): models overtime, machine breakdown, or shifts
- Scenario Runner:
    - Executes isolated re-run of Capacity-Aware MRP with modified parameter set
    - Measures resulting planned orders, capacity utilization, bottleneck overloads, and inventory cost
- Comparison View:
    - Side-by-side comparison of baseline vs. scenario KPIs:
        - Total Planned Orders & Units
        - Peak & Mean Work Center Utilization (%)
        - Total Bottleneck Overload Hours
        - Total Inventory Investment ($)
        - Capacity Feasibility Status
        - Delta & percentage changes
- Persistence:
    - Save named scenarios to disk (JSON)
    - Load, list, and delete saved scenarios
- Export:
    - Export scenario run results and side-by-side comparisons to CSV or Excel (openpyxl if available)
"""

from __future__ import annotations

import copy
import datetime
from dataclasses import asdict, dataclass, field
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.capacity.integration import CapacityAwareMRPPipeline, CapacityAwareMRPResult
from src.capacity.rccp import RCCPEngine
from src.core.explosion import BOMExplosionEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScenarioParameters:
    """Parameters defining a what-if planning simulation.

    Attributes:
        scenario_id: Unique identifier.
        name: Short descriptive name (e.g., 'Demand Spike +20%').
        description: Extended commentary on assumptions.
        demand_pct_change: Percentage change in gross requirements (e.g. +20.0 or -15.0).
        lead_time_days_delta: Uniform shift in lead time days across all items (e.g. +7 or -3).
        lead_time_period_delta: Discrete period shift in lead time (e.g. +1 period).
        safety_stock_pct_change: Percentage change in safety stocks across all items.
        capacity_hours_delta: Absolute shift in available capacity hours per work center per week (e.g. +10.0 or -20.0).
        capacity_pct_change: Percentage shift in work center capacity (e.g. +15.0 for overtime).
        item_specific_demand_pct: Dict of SKU-specific demand overrides {item_id: pct_change}.
        wc_specific_capacity_delta: Dict of WC-specific capacity hour shifts {wc_id: hours_delta}.
        created_at: ISO timestamp of creation.
    """
    scenario_id: str
    name: str
    description: str = ""
    demand_pct_change: float = 0.0
    lead_time_days_delta: int = 0
    lead_time_period_delta: int = 0
    safety_stock_pct_change: float = 0.0
    capacity_hours_delta: float = 0.0
    capacity_pct_change: float = 0.0
    item_specific_demand_pct: Dict[str, float] = field(default_factory=dict)
    wc_specific_capacity_delta: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioParameters":
        return cls(**data)


@dataclass
class ScenarioKPIs:
    """Consolidated key performance indicators from a scenario run."""
    scenario_id: str
    scenario_name: str
    total_planned_orders: int
    total_planned_units: float
    total_inventory_cost: float
    is_capacity_feasible: bool
    total_overload_hours: float
    max_wc_utilization_pct: float
    mean_wc_utilization_pct: float
    bottleneck_count: int
    capacity_exceptions_count: int
    computation_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioRunResult:
    """Full execution output from a scenario run."""
    parameters: ScenarioParameters
    kpis: ScenarioKPIs
    planned_orders: List[Dict[str, Any]]
    capacity_exceptions: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]
    utilization_matrix: Dict[str, Dict[int, float]]  # {wc_id: {period: util_pct}}
    executed_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters.to_dict(),
            "kpis": self.kpis.to_dict(),
            "planned_orders": self.planned_orders,
            "capacity_exceptions": self.capacity_exceptions,
            "bottlenecks": self.bottlenecks,
            "utilization_matrix": self.utilization_matrix,
            "executed_at": self.executed_at,
        }


# ---------------------------------------------------------------------------
# Scenario Engine
# ---------------------------------------------------------------------------

class ScenarioEngine:
    """Executes what-if parameter perturbations and comparative MRP runs."""

    def __init__(
        self,
        base_items_df: pd.DataFrame,
        base_bom_df: pd.DataFrame,
        base_routings_df: pd.DataFrame,
        base_wc_df: pd.DataFrame,
        base_mps_requirements: Dict[str, List[float]],
        periods: List[int],
        unit_costs: Optional[Dict[str, float]] = None,
        storage_dir: str = "data/scenarios",
    ):
        self.base_items_df = base_items_df.copy()
        self.base_bom_df = base_bom_df.copy()
        self.base_routings_df = base_routings_df.copy()
        self.base_wc_df = base_wc_df.copy()
        self.base_mps = copy.deepcopy(base_mps_requirements)
        self.periods = list(periods)
        self.unit_costs = unit_costs or dict(
            zip(base_items_df["item_id"], base_items_df.get("standard_cost", [10.0] * len(base_items_df)))
        )
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._saved_scenarios: Dict[str, ScenarioParameters] = {}
        self._cached_results: Dict[str, ScenarioRunResult] = {}
        self._base_result: Optional[ScenarioRunResult] = None

        self._load_all_saved_scenarios()

    # ------------------------------------------------------------------
    # Parameter Application
    # ------------------------------------------------------------------

    def apply_parameters(
        self, params: ScenarioParameters
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[float]]]:
        """Apply scenario parameter perturbations to items, work centers, and MPS."""
        # 1. Perturb Items (Lead time & Safety stock)
        items_mod = self.base_items_df.copy()
        
        # Lead time delta
        if params.lead_time_days_delta != 0 and "lead_time_days" in items_mod.columns:
            items_mod["lead_time_days"] = (
                items_mod["lead_time_days"] + params.lead_time_days_delta
            ).clip(lower=1)
        if params.lead_time_period_delta != 0 and "lead_time" in items_mod.columns:
            items_mod["lead_time"] = (
                items_mod["lead_time"] + params.lead_time_period_delta
            ).clip(lower=1)

        # Safety stock multiplier
        if params.safety_stock_pct_change != 0 and "safety_stock" in items_mod.columns:
            factor = max(0.0, 1.0 + params.safety_stock_pct_change / 100.0)
            items_mod["safety_stock"] = (items_mod["safety_stock"] * factor).round(1)

        # 2. Perturb Work Centers (Capacity shifts)
        wc_mod = self.base_wc_df.copy()
        if "capacity_hours_per_week" in wc_mod.columns:
            # Uniform % shift
            if params.capacity_pct_change != 0:
                wc_mod["capacity_hours_per_week"] = (
                    wc_mod["capacity_hours_per_week"] * (1.0 + params.capacity_pct_change / 100.0)
                ).clip(lower=1.0)
            # Uniform absolute hours shift
            if params.capacity_hours_delta != 0:
                wc_mod["capacity_hours_per_week"] = (
                    wc_mod["capacity_hours_per_week"] + params.capacity_hours_delta
                ).clip(lower=1.0)
            # Work-center specific shifts
            for wc_id, delta in params.wc_specific_capacity_delta.items():
                mask = wc_mod["work_center_id"] == wc_id
                wc_mod.loc[mask, "capacity_hours_per_week"] = (
                    wc_mod.loc[mask, "capacity_hours_per_week"] + delta
                ).clip(lower=1.0)

        # 3. Perturb MPS Gross Requirements (Demand shifts)
        mps_mod = copy.deepcopy(self.base_mps)
        uniform_mult = 1.0 + params.demand_pct_change / 100.0

        for item_id, reqs in mps_mod.items():
            item_mult = uniform_mult
            if item_id in params.item_specific_demand_pct:
                item_mult = 1.0 + params.item_specific_demand_pct[item_id] / 100.0
            item_mult = max(0.0, item_mult)
            mps_mod[item_id] = [round(r * item_mult, 1) for r in reqs]

        return items_mod, wc_mod, mps_mod

    # ------------------------------------------------------------------
    # Scenario Execution
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        params: ScenarioParameters,
        auto_level: bool = True,
        use_cache: bool = True,
    ) -> ScenarioRunResult:
        """Run an isolated Capacity-Aware MRP planning cycle with scenario parameters."""
        if use_cache and params.scenario_id in self._cached_results:
            return self._cached_results[params.scenario_id]

        start_time = datetime.datetime.now()

        # Generate perturbed inputs
        items_mod, wc_mod, mps_mod = self.apply_parameters(params)

        # Setup isolated engines
        explosion_engine = BOMExplosionEngine(
            bom_data=self.base_bom_df,
            items_data=items_mod,
        )
        rccp_engine = RCCPEngine(
            routings_data=self.base_routings_df,
            work_centers_data=wc_mod,
        )
        pipeline = CapacityAwareMRPPipeline(
            explosion_engine=explosion_engine,
            rccp_engine=rccp_engine,
        )

        # Prepare inventory status from perturbed items
        inventory_status = {}
        for _, row in items_mod.iterrows():
            iid = str(row["item_id"])
            inventory_status[iid] = {
                "on_hand": float(row.get("quantity_on_hand", 0.0)),
                "allocated": float(row.get("quantity_allocated", 0.0)),
                "safety_stock": float(row.get("safety_stock", 0.0)),
                "lead_time": int(row.get("lead_time", row.get("lead_time_days", 1))),
                "lot_sizing_rule": str(row.get("lot_sizing_rule", "LFL")),
            }

        # Run pipeline
        mrp_result: CapacityAwareMRPResult = pipeline.run_capacity_aware_mrp(
            mps_gross_requirements=mps_mod,
            periods=self.periods,
            inventory_status=inventory_status,
            auto_level=auto_level,
            allow_overtime=True,
            timing_key="due_period",
        )

        duration = (datetime.datetime.now() - start_time).total_seconds()

        # Calculate KPIs
        orders = mrp_result.final_planned_orders
        total_orders = len(orders)
        total_units = sum(float(o.get("quantity", 0.0)) for o in orders)
        
        # Total inventory value of planned orders
        total_inv_cost = sum(
            float(o.get("quantity", 0.0)) * self.unit_costs.get(o.get("item_id", ""), 10.0)
            for o in orders
        )

        # RCCP utilization and overloads
        util_matrix: Dict[str, Dict[int, float]] = {}
        all_utils: List[float] = []
        total_overload = 0.0

        for wc_id, p_dict in mrp_result.final_load_matrix.items():
            util_matrix[wc_id] = {}
            for p, load in p_dict.items():
                util_pct = load.utilization_pct
                util_matrix[wc_id][p] = round(util_pct, 2)
                all_utils.append(util_pct)
                if load.overload_hours > 0:
                    total_overload += load.overload_hours

        max_util = max(all_utils) if all_utils else 0.0
        mean_util = float(np.mean(all_utils)) if all_utils else 0.0

        kpis = ScenarioKPIs(
            scenario_id=params.scenario_id,
            scenario_name=params.name,
            total_planned_orders=total_orders,
            total_planned_units=round(total_units, 1),
            total_inventory_cost=round(total_inv_cost, 2),
            is_capacity_feasible=mrp_result.is_capacity_feasible,
            total_overload_hours=round(total_overload, 1),
            max_wc_utilization_pct=round(max_util, 1),
            mean_wc_utilization_pct=round(mean_util, 1),
            bottleneck_count=len(mrp_result.final_bottlenecks),
            capacity_exceptions_count=len(mrp_result.capacity_exceptions),
            computation_time_seconds=round(duration, 3),
        )

        res = ScenarioRunResult(
            parameters=params,
            kpis=kpis,
            planned_orders=orders,
            capacity_exceptions=[e.to_dict() for e in mrp_result.capacity_exceptions],
            bottlenecks=[b.to_dict() for b in mrp_result.final_bottlenecks],
            utilization_matrix=util_matrix,
        )

        self._cached_results[params.scenario_id] = res
        return res

    def get_base_scenario_result(self, force_refresh: bool = False) -> ScenarioRunResult:
        """Run or return the baseline (unperturbed) scenario."""
        if self._base_result is None or force_refresh:
            base_params = ScenarioParameters(
                scenario_id="BASELINE",
                name="Baseline Plan",
                description="Unperturbed master production schedule and factory baseline.",
            )
            self._base_result = self.run_scenario(base_params, use_cache=not force_refresh)
        return self._base_result

    # ------------------------------------------------------------------
    # Comparative Analytics
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        scenario_res: ScenarioRunResult,
        base_res: Optional[ScenarioRunResult] = None,
    ) -> pd.DataFrame:
        """Create side-by-side KPI comparison view between baseline and scenario."""
        base = base_res or self.get_base_scenario_result()
        b_kpi = base.kpis
        s_kpi = scenario_res.kpis

        def _calc_delta(s_val: float, b_val: float) -> Tuple[float, str]:
            delta = s_val - b_val
            pct = (delta / b_val * 100.0) if b_val != 0 else 0.0
            return delta, f"{pct:+.1f}%"

        rows = []
        
        # Total Planned Orders
        d_ord, p_ord = _calc_delta(s_kpi.total_planned_orders, b_kpi.total_planned_orders)
        rows.append({
            "KPI Metric": "Total Planned Orders",
            "Baseline": f"{b_kpi.total_planned_orders:,}",
            f"Scenario: {s_kpi.scenario_name}": f"{s_kpi.total_planned_orders:,}",
            "Delta": f"{int(d_ord):+,}",
            "Delta %": p_ord,
        })

        # Total Units
        d_units, p_units = _calc_delta(s_kpi.total_planned_units, b_kpi.total_planned_units)
        rows.append({
            "KPI Metric": "Total Planned Units",
            "Baseline": f"{b_kpi.total_planned_units:,.1f}",
            f"Scenario: {s_kpi.scenario_name}": f"{s_kpi.total_planned_units:,.1f}",
            "Delta": f"{d_units:+,.1f}",
            "Delta %": p_units,
        })

        # Inventory Value
        d_cost, p_cost = _calc_delta(s_kpi.total_inventory_cost, b_kpi.total_inventory_cost)
        rows.append({
            "KPI Metric": "Total Inventory Cost ($)",
            "Baseline": f"${b_kpi.total_inventory_cost:,.2f}",
            f"Scenario: {s_kpi.scenario_name}": f"${s_kpi.total_inventory_cost:,.2f}",
            "Delta": f"${d_cost:+,.2f}",
            "Delta %": p_cost,
        })

        # Overload Hours
        d_ovld, p_ovld = _calc_delta(s_kpi.total_overload_hours, b_kpi.total_overload_hours)
        rows.append({
            "KPI Metric": "Bottleneck Overload (hrs)",
            "Baseline": f"{b_kpi.total_overload_hours:,.1f} hrs",
            f"Scenario: {s_kpi.scenario_name}": f"{s_kpi.total_overload_hours:,.1f} hrs",
            "Delta": f"{d_ovld:+,.1f} hrs",
            "Delta %": p_ovld,
        })

        # Peak Utilization
        d_maxu, p_maxu = _calc_delta(s_kpi.max_wc_utilization_pct, b_kpi.max_wc_utilization_pct)
        rows.append({
            "KPI Metric": "Peak Work Center Utilization (%)",
            "Baseline": f"{b_kpi.max_wc_utilization_pct:.1f}%",
            f"Scenario: {s_kpi.scenario_name}": f"{s_kpi.max_wc_utilization_pct:.1f}%",
            "Delta": f"{d_maxu:+.1f}%",
            "Delta %": "-",
        })

        # Mean Utilization
        d_meanu, _ = _calc_delta(s_kpi.mean_wc_utilization_pct, b_kpi.mean_wc_utilization_pct)
        rows.append({
            "KPI Metric": "Mean Work Center Utilization (%)",
            "Baseline": f"{b_kpi.mean_wc_utilization_pct:.1f}%",
            f"Scenario: {s_kpi.scenario_name}": f"{s_kpi.mean_wc_utilization_pct:.1f}%",
            "Delta": f"{d_meanu:+.1f}%",
            "Delta %": "-",
        })

        # Capacity Feasible
        rows.append({
            "KPI Metric": "Capacity Feasible",
            "Baseline": "Yes" if b_kpi.is_capacity_feasible else "No",
            f"Scenario: {s_kpi.scenario_name}": "Yes" if s_kpi.is_capacity_feasible else "No",
            "Delta": "Unchanged" if b_kpi.is_capacity_feasible == s_kpi.is_capacity_feasible else "Changed",
            "Delta %": "-",
        })

        # Exceptions Count
        d_exc, p_exc = _calc_delta(s_kpi.capacity_exceptions_count, b_kpi.capacity_exceptions_count)
        rows.append({
            "KPI Metric": "Capacity Exceptions Count",
            "Baseline": f"{b_kpi.capacity_exceptions_count}",
            f"Scenario: {s_kpi.scenario_name}": f"{s_kpi.capacity_exceptions_count}",
            "Delta": f"{int(d_exc):+}",
            "Delta %": p_exc,
        })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence (Save, Load, Delete, List)
    # ------------------------------------------------------------------

    def save_scenario(self, params: ScenarioParameters) -> Path:
        """Persist scenario parameters to disk as JSON."""
        self._saved_scenarios[params.scenario_id] = params
        target_path = self.storage_dir / f"{params.scenario_id}.json"
        with open(target_path, "w", encoding="utf-8") as fh:
            json.dump(params.to_dict(), fh, indent=2)
        logger.info(f"Saved scenario {params.scenario_id} to {target_path}")
        return target_path

    def load_scenario(self, scenario_id: str) -> Optional[ScenarioParameters]:
        """Load scenario parameters from disk or in-memory registry."""
        if scenario_id in self._saved_scenarios:
            return self._saved_scenarios[scenario_id]

        target_path = self.storage_dir / f"{scenario_id}.json"
        if not target_path.exists():
            return None

        try:
            with open(target_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            params = ScenarioParameters.from_dict(data)
            self._saved_scenarios[scenario_id] = params
            return params
        except Exception as exc:
            logger.error(f"Failed to load scenario {scenario_id}: {exc}")
            return None

    def delete_scenario(self, scenario_id: str) -> bool:
        """Delete scenario from registry and disk."""
        self._saved_scenarios.pop(scenario_id, None)
        self._cached_results.pop(scenario_id, None)
        target_path = self.storage_dir / f"{scenario_id}.json"
        if target_path.exists():
            target_path.unlink()
            return True
        return False

    def list_saved_scenarios(self) -> List[ScenarioParameters]:
        """List all currently persisted scenarios."""
        return list(self._saved_scenarios.values())

    def _load_all_saved_scenarios(self) -> None:
        """Scan storage directory and load existing scenario files."""
        for p in self.storage_dir.glob("*.json"):
            sid = p.stem
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._saved_scenarios[sid] = ScenarioParameters.from_dict(data)
            except Exception as exc:
                logger.warning(f"Could not parse scenario file {p}: {exc}")

    # ------------------------------------------------------------------
    # Data Export (CSV & Excel)
    # ------------------------------------------------------------------

    def export_scenario_results(
        self,
        result: ScenarioRunResult,
        output_format: str = "csv",
        output_path: Optional[str] = None,
    ) -> Union[str, bytes]:
        """Export scenario planned orders and summary KPIs to CSV or Excel.

        Args:
            result: The ScenarioRunResult to export.
            output_format: 'csv' or 'excel'.
            output_path: Optional file destination.

        Returns:
            CSV string (if format='csv') or bytes (if format='excel').
        """
        fmt = output_format.lower()
        orders_df = pd.DataFrame(result.planned_orders)
        kpi_df = pd.DataFrame([result.kpis.to_dict()])

        if fmt == "csv":
            buf = io.StringIO()
            buf.write("# --- SCENARIO KPIS ---\n")
            kpi_df.to_csv(buf, index=False)
            buf.write("\n# --- PLANNED ORDERS ---\n")
            orders_df.to_csv(buf, index=False)
            content = buf.getvalue()

            if output_path:
                with open(output_path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(content)
            return content

        elif fmt in ("excel", "xlsx"):
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    kpi_df.to_excel(writer, sheet_name="KPI Summary", index=False)
                    orders_df.to_excel(writer, sheet_name="Planned Orders", index=False)
                    # Add utilization matrix sheet
                    util_df = pd.DataFrame(result.utilization_matrix).T
                    util_df.to_excel(writer, sheet_name="WC Utilization")
                bytes_content = buf.getvalue()
            except ImportError:
                # Fallback to CSV format if openpyxl is not installed
                logger.warning("openpyxl not installed. Exporting CSV format instead.")
                csv_str = self.export_scenario_results(result, output_format="csv")
                bytes_content = csv_str.encode("utf-8")

            if output_path:
                with open(output_path, "wb") as fh:
                    fh.write(bytes_content)
            return bytes_content

        else:
            raise ValueError(f"Unsupported export format '{output_format}'. Use 'csv' or 'excel'.")

    def export_comparison_view(
        self,
        comparison_df: pd.DataFrame,
        output_format: str = "csv",
        output_path: Optional[str] = None,
    ) -> Union[str, bytes]:
        """Export side-by-side comparison table to CSV or Excel."""
        fmt = output_format.lower()
        if fmt == "csv":
            csv_str = comparison_df.to_csv(index=False)
            if output_path:
                with open(output_path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(csv_str)
            return csv_str
        elif fmt in ("excel", "xlsx"):
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    comparison_df.to_excel(writer, sheet_name="Scenario Comparison", index=False)
                bytes_content = buf.getvalue()
            except ImportError:
                csv_str = comparison_df.to_csv(index=False)
                bytes_content = csv_str.encode("utf-8")
            if output_path:
                with open(output_path, "wb") as fh:
                    fh.write(bytes_content)
            return bytes_content
        else:
            raise ValueError(f"Unsupported format '{output_format}'.")

