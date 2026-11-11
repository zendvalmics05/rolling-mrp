"""ROLLINGMRP Interactive Planner Workbench.

Implements Phase 7.4:
- Streamlit application providing interactive planner controls:
  - Sidebar: Data controls, horizon selector, run triggers, quick filters.
  - Tabs:
      1. Overview: Executive KPIs, PAB summary, high-level stability and workload metrics.
      2. MRP Results: Multi-level explosion netting grid, order schedule Gantt chart, CSV downloads.
      3. Exceptions Inbox: Interactive action workbench, priority badges, batch approve/reject/defer, audit trail.
      4. What-If Scenarios: Interactive parameter sliders (demand, lead time, safety stock, capacity),
         re-run solver, side-by-side comparative analytics, persistence.
      5. Capacity (RCCP): Work center load heatmap, bottleneck analysis, leveling recommendations.
      6. Supplier Risk: Scorecards, reliability ranking, risk matrix, dynamic safety stock suggestions.
- Session State: Preserves user filters, approvals, and scenario results across reruns.
"""

from __future__ import annotations

import copy
import datetime
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.exception_inbox import (
    ActionStatus,
    ExceptionInboxManager,
    ExceptionMessage,
    ExceptionPriority,
    ExceptionType,
)
from app.scenario_engine import (
    ScenarioEngine,
    ScenarioParameters,
    ScenarioRunResult,
)
from src.capacity.integration import CapacityAwareMRPPipeline, CapacityAwareMRPResult
from src.capacity.rccp import RCCPEngine
from src.core.explosion import BOMExplosionEngine
from src.integration.data_generator import generate_synthetic_data
from src.risk.safety_stock import DynamicSafetyStockEngine
from src.risk.supplier import SupplierReliabilityScorer, SupplierTier
from src.visualization.visualizers import (
    plot_capacity_heatmap,
    plot_gantt_chart,
    plot_inventory_waterfall,
    plot_nervousness_trend,
    plot_pareto_exceptions,
    plot_supplier_scorecard,
)

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ROLLINGMRP — Planner Workbench",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data Caching & Session Initialization
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_factory_data(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Load and prepare factory dataset."""
    data = generate_synthetic_data(seed=seed)
    items_df = data["items"]
    inv_df = data["inventory"]
    merged_items = items_df.merge(inv_df, on="item_id", how="left")
    merged_items["quantity_on_hand"] = merged_items["quantity_on_hand"].fillna(0.0)
    merged_items["quantity_allocated"] = merged_items["quantity_allocated"].fillna(0.0)
    merged_items["lead_time"] = (merged_items["lead_time_days"] / 7.0).round().astype(int).clip(lower=1)
    data["items_merged"] = merged_items
    return data


def init_session_state():
    """Initialize persistent session variables."""
    if "factory_data" not in st.session_state:
        st.session_state.factory_data = load_factory_data(seed=42)

    data = st.session_state.factory_data
    merged_items = data["items_merged"]
    fg_items = merged_items[merged_items["type"] == "finished_good"]["item_id"].tolist()

    if "planning_periods" not in st.session_state:
        st.session_state.planning_periods = list(range(1, 13))

    if "base_mps" not in st.session_state:
        # Default 30 units per week with a small peak in period 6
        mps = {}
        for fg in fg_items:
            reqs = [25.0] * 12
            reqs[5] = 40.0  # week 6 spike
            mps[fg] = reqs
        st.session_state.base_mps = mps

    if "inbox_manager" not in st.session_state:
        st.session_state.inbox_manager = ExceptionInboxManager(
            audit_log_path="data/audit_log.jsonl"
        )

    if "scenario_engine" not in st.session_state:
        st.session_state.scenario_engine = ScenarioEngine(
            base_items_df=merged_items,
            base_bom_df=data["bom"],
            base_routings_df=data["routings"],
            base_wc_df=data["work_centers"],
            base_mps_requirements=st.session_state.base_mps,
            periods=st.session_state.planning_periods,
            storage_dir="data/scenarios",
        )

    if "mrp_result" not in st.session_state:
        # Auto-run initial baseline MRP run
        re_run_base_mrp()


def re_run_base_mrp():
    """Execute baseline Capacity-Aware MRP."""
    data = st.session_state.factory_data
    merged_items = data["items_merged"]
    periods = st.session_state.planning_periods
    base_mps = st.session_state.base_mps

    explosion_engine = BOMExplosionEngine(bom_data=data["bom"], items_data=merged_items)
    rccp_engine = RCCPEngine(routings_data=data["routings"], work_centers_data=data["work_centers"])
    pipeline = CapacityAwareMRPPipeline(explosion_engine=explosion_engine, rccp_engine=rccp_engine)

    inventory_status = {
        str(row["item_id"]): {
            "on_hand": float(row["quantity_on_hand"]),
            "allocated": float(row["quantity_allocated"]),
            "safety_stock": float(row["safety_stock"]),
            "lead_time": int(row["lead_time"]),
            "lot_sizing_rule": str(row["lot_sizing_rule"]),
        }
        for _, row in merged_items.iterrows()
    }

    mrp_res: CapacityAwareMRPResult = pipeline.run_capacity_aware_mrp(
        mps_gross_requirements=base_mps,
        periods=periods,
        inventory_status=inventory_status,
        auto_level=True,
        allow_overtime=True,
        timing_key="due_period",
    )

    st.session_state.mrp_result = mrp_res

    # Ingest exceptions to inbox
    inbox: ExceptionInboxManager = st.session_state.inbox_manager
    costs = dict(zip(merged_items["item_id"], merged_items.get("standard_cost", 10.0)))
    
    # Ingest capacity exceptions
    if mrp_res.capacity_exceptions:
        inbox.ingest_capacity_exceptions(mrp_res.capacity_exceptions, hourly_rate=85.0)

    # Ingest past due / expedite netting exceptions
    for po in mrp_res.initial_planned_orders:
        if isinstance(po, dict) and po.get("release_period", 1) <= 0:
            inbox.create_and_add_exception(
                exception_type=ExceptionType.EXPEDITE,
                item_id=str(po.get("item_id")),
                period=int(po.get("due_period", 1)),
                quantity=float(po.get("quantity", 0.0)),
                unit_cost=float(costs.get(str(po.get("item_id")), 10.0)),
                description=f"Past due order for {po.get('item_id')} requires release in period {po.get('release_period')}.",
                recommended_action="Expedite supplier order or line start.",
                order_id=str(po.get("order_id")),
            )


# Initialize state
init_session_state()

# ---------------------------------------------------------------------------
# Sidebar: Navigation & Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ ROLLINGMRP")
    st.caption("Production-Grade Rolling-Horizon MRP System")
    st.divider()

    st.subheader("Planning Horizon")
    horizon_weeks = st.slider("Planning Horizon (Weeks)", min_value=6, max_value=24, value=12, step=1)
    if horizon_weeks != len(st.session_state.planning_periods):
        st.session_state.planning_periods = list(range(1, horizon_weeks + 1))
        # Adjust base MPS length
        for fg in st.session_state.base_mps:
            curr = st.session_state.base_mps[fg]
            if len(curr) < horizon_weeks:
                curr.extend([25.0] * (horizon_weeks - len(curr)))
            else:
                st.session_state.base_mps[fg] = curr[:horizon_weeks]

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Re-Run MRP", use_container_width=True, type="primary"):
            with st.spinner("Solving multi-level MRP & leveling..."):
                re_run_base_mrp()
            st.success("Plan regenerated!")
    with col_btn2:
        if st.button("🎲 New Data Seed", use_container_width=True):
            st.cache_data.clear()
            st.session_state.factory_data = load_factory_data(seed=int(datetime.datetime.now().timestamp()) % 1000)
            re_run_base_mrp()
            st.rerun()

    st.divider()
    st.subheader("System Status")
    res: CapacityAwareMRPResult = st.session_state.mrp_result
    if res.is_capacity_feasible:
        st.success("✅ Capacity Feasible")
    else:
        st.error(f"⚠️ {len(res.capacity_exceptions)} Capacity Overloads")

    inbox_metrics = st.session_state.inbox_manager.get_summary_metrics()
    st.metric("Pending Exceptions", inbox_metrics["pending_count"])
    st.metric("Pending Exposure ($)", f"${inbox_metrics['total_pending_dollar_impact']:,.0f}")


# ---------------------------------------------------------------------------
# Main Panel: Multi-Tab Layout
# ---------------------------------------------------------------------------

tab_overview, tab_mrp, tab_inbox, tab_scenarios, tab_capacity, tab_suppliers = st.tabs([
    "📊 Overview",
    "📦 MRP Results",
    "🚨 Exceptions Inbox",
    "🔬 What-If Scenarios",
    "⚡ Capacity (RCCP)",
    "🤝 Supplier Risk",
])

data = st.session_state.factory_data
merged_items = data["items_merged"]
res: CapacityAwareMRPResult = st.session_state.mrp_result
periods = st.session_state.planning_periods

# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab_overview:
    st.subheader("Executive Planning Summary")

    total_orders = len(res.final_planned_orders)
    total_units = sum(float(o.get("quantity", 0.0)) for o in res.final_planned_orders)
    cost_map = dict(zip(merged_items["item_id"], merged_items.get("standard_cost", 10.0)))
    total_val = sum(float(o.get("quantity", 0.0)) * cost_map.get(o.get("item_id", ""), 10.0) for o in res.final_planned_orders)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Planned Orders", f"{total_orders:,}")
    col2.metric("Total Output Units", f"{total_units:,.0f}")
    col3.metric("Planned Spend ($)", f"${total_val:,.0f}")
    col4.metric("Bottlenecks Detected", len(res.final_bottlenecks))
    col5.metric("Leveling Actions", len(res.leveling_actions))

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Production & Purchase Order Schedule (Gantt)")
        fig_gantt = plot_gantt_chart(res.final_planned_orders, max_orders=30)
        st.plotly_chart(fig_gantt, use_container_width=True)

    with col_right:
        st.subheader("Rough-Cut Work Center Load (%)")
        util_matrix = {wc: {p: l.utilization_pct for p, l in pmap.items()} for wc, pmap in res.final_load_matrix.items()}
        fig_heat = plot_capacity_heatmap(util_matrix)
        st.plotly_chart(fig_heat, use_container_width=True)


# ===========================================================================
# TAB 2: MRP RESULTS
# ===========================================================================
with tab_mrp:
    st.subheader("Material Requirements Planning (MRP) Orders & Netting")

    col_flt1, col_flt2, col_flt3 = st.columns(3)
    with col_flt1:
        sel_sku = st.selectbox("Select SKU for Detailed PAB Waterfall", merged_items["item_id"].tolist())
    with col_flt2:
        filter_status = st.multiselect("Filter Lifecycle Status", ["planned", "released", "firm"], default=["planned", "released"])
    with col_flt3:
        sort_col = st.selectbox("Sort Orders Table By", ["due_period", "release_period", "quantity", "item_id"])

    # Detailed Waterfall for Selected SKU
    sku_row = merged_items[merged_items["item_id"] == sel_sku].iloc[0]
    on_hand = float(sku_row["quantity_on_hand"])
    ss = float(sku_row["safety_stock"])
    
    # Calculate synthetic netting for waterfall view
    sku_orders = [o for o in res.final_planned_orders if o.get("item_id") == sel_sku]
    sku_gross = st.session_state.base_mps.get(sel_sku, [0.0] * len(periods))
    
    # Simple simulated PAB trajectory for plotting
    pab_traj = []
    curr_pab = on_hand
    por_list = []
    for p in periods:
        p_orders = [o for o in sku_orders if o.get("due_period") == p]
        qty_in = sum(float(o.get("quantity", 0.0)) for o in p_orders)
        por_list.append(qty_in)
        demand = sku_gross[p - 1] if p - 1 < len(sku_gross) else 0.0
        curr_pab = curr_pab + qty_in - demand
        pab_traj.append(curr_pab)

    fig_waterfall = plot_inventory_waterfall(
        item_id=sel_sku,
        periods=periods,
        gross_requirements=sku_gross[:len(periods)],
        scheduled_receipts=[0.0] * len(periods),
        planned_receipts=por_list,
        pab_trajectory=pab_traj,
        safety_stock=ss,
        initial_on_hand=on_hand,
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Orders Table & CSV Download
    orders_df = pd.DataFrame(res.final_planned_orders)
    if not orders_df.empty and "status" in orders_df.columns:
        orders_df = orders_df[orders_df["status"].isin(filter_status)]
        if sort_col in orders_df.columns:
            orders_df = orders_df.sort_values(by=sort_col)

    st.subheader("Planned & Dispatched Orders Ledger")
    st.dataframe(orders_df, use_container_width=True)

    csv_data = orders_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Orders (CSV)", data=csv_data, file_name="mrp_planned_orders.csv", mime="text/csv")


# ===========================================================================
# TAB 3: EXCEPTIONS INBOX
# ===========================================================================
with tab_inbox:
    st.subheader("Planning Exceptions Action Workbench")
    inbox: ExceptionInboxManager = st.session_state.inbox_manager

    # Summary Cards
    m_info = inbox.get_summary_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Exceptions", m_info["total_exceptions"])
    c2.metric("Pending Action", m_info["pending_count"])
    c3.metric("Critical Exposure ($)", f"${m_info['total_pending_dollar_impact']:,.0f}")
    c4.metric("Approved / Resolved", m_info["approved_count"])

    st.markdown("---")

    # Filters
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        f_status = st.selectbox("Action Status", ["ALL", "PENDING", "APPROVED", "REJECTED", "DEFERRED"])
    with f_col2:
        f_priority = st.selectbox("Priority Tier", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with f_col3:
        f_type = st.selectbox("Exception Type", ["ALL", "CAPACITY_OVERLOAD", "EXPEDITE", "DEFER", "SAFETY_STOCK", "CANCEL"])
    with f_col4:
        min_impact = st.number_input("Min Dollar Exposure ($)", min_value=0.0, value=0.0, step=500.0)

    # Filter messages
    status_filter = None if f_status == "ALL" else ActionStatus(f_status)
    priority_filter = None if f_priority == "ALL" else ExceptionPriority(f_priority)
    type_filter = None if f_type == "ALL" else ExceptionType(f_type)

    filtered_msgs = inbox.list_exceptions(
        status=status_filter,
        priority=priority_filter,
        exception_type=type_filter,
        min_dollar_impact=min_impact,
        sort_by="dollar_impact",
    )

    col_tbl, col_actions = st.columns([3, 1])

    with col_tbl:
        if filtered_msgs:
            df_inbox = pd.DataFrame([m.to_dict() for m in filtered_msgs])
            show_cols = ["exception_id", "priority", "exception_type", "item_id", "period", "quantity", "dollar_impact", "status", "description"]
            st.dataframe(
                df_inbox[[c for c in show_cols if c in df_inbox.columns]],
                use_container_width=True,
                height=380,
            )
        else:
            st.info("No exceptions matching filter criteria.")

    with col_actions:
        st.markdown("#### Take Action on Exception")
        target_ids = [m.exception_id for m in filtered_msgs] if filtered_msgs else []
        selected_exc_id = st.selectbox("Select Exception ID", target_ids if target_ids else ["None"])
        user_name = st.text_input("Planner Username", value="planner_1")
        action_comment = st.text_input("Action Comment", value="Approved per capacity slack")

        act_c1, act_c2, act_c3 = st.columns(3)
        with act_c1:
            if st.button("✅ Approve", use_container_width=True) and selected_exc_id != "None":
                inbox.approve(selected_exc_id, user=user_name, comment=action_comment)
                st.success(f"{selected_exc_id} Approved")
                st.rerun()
        with act_c2:
            if st.button("❌ Reject", use_container_width=True) and selected_exc_id != "None":
                inbox.reject(selected_exc_id, user=user_name, comment=action_comment)
                st.warning(f"{selected_exc_id} Rejected")
                st.rerun()
        with act_c3:
            if st.button("⏳ Defer", use_container_width=True) and selected_exc_id != "None":
                inbox.defer(selected_exc_id, user=user_name, comment=action_comment)
                st.info(f"{selected_exc_id} Deferred")
                st.rerun()

        st.markdown("---")
        st.markdown("##### Bulk Operations")
        if st.button("⚡ Bulk Approve All Filtered", use_container_width=True):
            inbox.batch_action(target_ids, ActionStatus.APPROVED, user=user_name, comment="Bulk approval via UI")
            st.success(f"Approved {len(target_ids)} exceptions!")
            st.rerun()

    # Audit Trail Sub-section
    with st.expander("📜 View Audit Trail Log"):
        audit_df = inbox.get_audit_trail_dataframe()
        st.dataframe(audit_df, use_container_width=True)


# ===========================================================================
# TAB 4: WHAT-IF SCENARIOS
# ===========================================================================
with tab_scenarios:
    st.subheader("What-If Scenario Simulation & Parameter Sensitivity")
    scen_engine: ScenarioEngine = st.session_state.scenario_engine

    col_sliders, col_results = st.columns([1, 2])

    with col_sliders:
        st.markdown("#### Scenario Controls")
        scen_name = st.text_input("Scenario Name", value="Demand Surge +25%")
        scen_id = st.text_input("Scenario ID", value=f"SCEN_{int(datetime.datetime.now().timestamp()) % 10000}")
        
        st.markdown("##### Parameter Perturbations")
        d_shift = st.slider("Demand Shift (%)", min_value=-50.0, max_value=100.0, value=25.0, step=5.0)
        lt_shift = st.slider("Lead Time Shift (Periods)", min_value=-2, max_value=3, value=0, step=1)
        ss_shift = st.slider("Safety Stock Shift (%)", min_value=-50.0, max_value=100.0, value=0.0, step=10.0)
        cap_shift = st.slider("Capacity Shift (Hours/Week)", min_value=-30.0, max_value=40.0, value=0.0, step=5.0)

        col_run, col_save = st.columns(2)
        with col_run:
            run_btn = st.button("🚀 Run Scenario", type="primary", use_container_width=True)
        with col_save:
            save_btn = st.button("💾 Save Scenario", use_container_width=True)

    with col_results:
        st.markdown("#### Simulation Output & Comparison View")
        
        current_params = ScenarioParameters(
            scenario_id=scen_id,
            name=scen_name,
            demand_pct_change=d_shift,
            lead_time_period_delta=lt_shift,
            safety_stock_pct_change=ss_shift,
            capacity_hours_delta=cap_shift,
        )

        if save_btn:
            scen_engine.save_scenario(current_params)
            st.success(f"Scenario '{scen_name}' saved to disk!")

        # Execute scenario
        with st.spinner("Executing scenario MRP & capacity run..."):
            scen_res: ScenarioRunResult = scen_engine.run_scenario(current_params, use_cache=not run_btn)
            comp_df = scen_engine.compare_scenarios(scen_res)

        st.dataframe(comp_df, use_container_width=True)

        # Export Buttons
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_str = scen_engine.export_comparison_view(comp_df, output_format="csv")
            st.download_button("📥 Export Comparison (CSV)", data=csv_str.encode("utf-8"), file_name=f"{scen_id}_comparison.csv", mime="text/csv")
        with col_exp2:
            orders_csv = scen_engine.export_scenario_results(scen_res, output_format="csv")
            st.download_button("📥 Export Scenario Orders (CSV)", data=orders_csv.encode("utf-8"), file_name=f"{scen_id}_orders.csv", mime="text/csv")


# ===========================================================================
# TAB 5: CAPACITY (RCCP)
# ===========================================================================
with tab_capacity:
    st.subheader("Rough-Cut Capacity Planning (RCCP) & Bottlenecks")

    util_matrix = {wc: {p: l.utilization_pct for p, l in pmap.items()} for wc, pmap in res.final_load_matrix.items()}
    fig_heat = plot_capacity_heatmap(util_matrix, title="Work Center Load Heatmap Across Horizon (%)")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Detected Work Center Bottlenecks & Overloads")
    if res.final_bottlenecks:
        b_records = []
        for b in res.final_bottlenecks:
            for inc in b.incidents:
                b_records.append({
                    "Work Center": inc.work_center_id,
                    "Period": inc.period,
                    "Utilization (%)": f"{inc.utilization_pct:.1f}%",
                    "Overload (Hours)": f"{inc.overload_hours:.1f}",
                    "Available Capacity (Hours)": f"{inc.available_hours:.1f}",
                    "Top Driver SKUs": ", ".join(d["item_id"] for d in inc.top_drivers),
                })
        st.dataframe(pd.DataFrame(b_records), use_container_width=True)
    else:
        st.success("🎉 All work centers are capacity-feasible! No bottleneck overloads detected.")


# ===========================================================================
# TAB 6: SUPPLIER RISK
# ===========================================================================
with tab_suppliers:
    st.subheader("Supplier Reliability & Risk Portfolio")
    suppliers_df = data["suppliers"]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### Supplier Performance Scorecard Matrix")
        fig_sup_scatter = plot_supplier_scorecard(suppliers_df, scorecard_type="scatter")
        st.plotly_chart(fig_sup_scatter, use_container_width=True)

    with col_s2:
        st.markdown("#### Reliability Ranking & Tiers")
        fig_sup_rank = plot_supplier_scorecard(suppliers_df, scorecard_type="ranking")
        st.plotly_chart(fig_sup_rank, use_container_width=True)

    st.subheader("Supplier Portfolio Database")
    st.dataframe(suppliers_df, use_container_width=True)

