"""Interactive Visualizations for ROLLINGMRP Planner Workbench.

Implements Phase 7.3:
1. plot_gantt_chart: Planned orders over time, horizontally positioned from release_period
   to due_period, color-coded by lifecycle status (PLANNED, RELEASED, IN_PROGRESS, etc.).
2. plot_inventory_waterfall: Projected Available Balance (PAB) trajectory showing starting stock,
   gross requirements (depletion), scheduled/planned receipts (replenishment), and safety stock floor.
3. plot_capacity_heatmap: Work center utilization percentage across planning weeks/periods
   with color scale (green < 85%, yellow 85-100%, red > 100% overload).
4. plot_nervousness_trend: Plan stability & churn metrics across rolling cycles with alert threshold line.
5. plot_pareto_exceptions: Pareto analysis of exceptions (expedite, defer, cancel, overload)
   with bar volumes and cumulative percentage line (80/20 rule).
6. plot_supplier_scorecard: Supplier risk vs. reliability radar/scatter or ranking bar chart
   colored by risk tier (PREFERRED, STANDARD, CONDITIONAL, HIGH_RISK).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# 1. Gantt Chart: Planned & Released Orders Over Time
# ---------------------------------------------------------------------------

def plot_gantt_chart(
    orders: Union[List[Dict[str, Any]], pd.DataFrame],
    title: str = "Planned & Released Production / Purchase Orders Schedule",
    max_orders: int = 50,
) -> go.Figure:
    """Render horizontal Gantt chart of planned and released orders.

    Each bar begins at release_period and finishes at due_period.
    Color-coded by status (PLANNED, RELEASED, IN_PROGRESS, COMPLETED, FIRM).

    Args:
        orders: List of order dictionaries or DataFrame with keys:
                item_id, release_period, due_period, quantity, status (optional order_id).
        title: Plot title.
        max_orders: Maximum number of orders to display for readability.
    """
    df = orders if isinstance(orders, pd.DataFrame) else pd.DataFrame(orders)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Orders Available", template="plotly_white")
        return fig

    # Normalize fields
    df = df.copy()
    if "release_period" not in df.columns:
        df["release_period"] = df.get("period", 1) - 1
    if "due_period" not in df.columns:
        df["due_period"] = df.get("period", 1)
    if "status" not in df.columns:
        df["status"] = "PLANNED"
    if "order_id" not in df.columns:
        df["order_id"] = [f"PO-{r.item_id}-{r.due_period}" for _, r in df.iterrows()]

    # Limit orders for readability
    if len(df) > max_orders:
        df = df.iloc[:max_orders]

    df["status_upper"] = df["status"].astype(str).str.upper()
    df["duration"] = (df["due_period"] - df["release_period"]).clip(lower=0.5)

    color_discrete_map = {
        "PLANNED": "#3366cc",
        "RELEASED": "#109618",
        "FIRM": "#ff9900",
        "IN_PROGRESS": "#990099",
        "COMPLETED": "#0099c6",
        "CANCELLED": "#dc3912",
    }

    fig = px.bar(
        df,
        base="release_period",
        x="duration",
        y="order_id",
        color="status_upper",
        color_discrete_map=color_discrete_map,
        orientation="h",
        hover_data=["item_id", "quantity", "release_period", "due_period"],
        labels={
            "duration": "Planning Periods (Duration)",
            "order_id": "Order Identifier",
            "status_upper": "Order Status",
        },
        title=title,
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Planning Horizon (Periods)",
        yaxis_title="Order Reference",
        legend_title="Lifecycle Status",
        barmode="stack",
        height=max(400, min(800, len(df) * 22)),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Inventory Waterfall: PAB Trajectory
# ---------------------------------------------------------------------------

def plot_inventory_waterfall(
    item_id: str,
    periods: List[int],
    gross_requirements: List[float],
    scheduled_receipts: List[float],
    planned_receipts: List[float],
    pab_trajectory: List[float],
    safety_stock: float = 0.0,
    initial_on_hand: float = 0.0,
) -> go.Figure:
    """Plot time-phased Projected Available Balance (PAB) waterfall with demand and receipts.

    Args:
        item_id: Item identifier.
        periods: List of horizon period indices (e.g. [1, 2, 3...]).
        gross_requirements: Gross requirements per period.
        scheduled_receipts: Pre-existing scheduled receipts.
        planned_receipts: MRP planned order receipts.
        pab_trajectory: Time-phased projected available balance after netting.
        safety_stock: Safety stock buffer floor.
        initial_on_hand: Starting inventory balance at t=0.
    """
    fig = go.Figure()

    # Total incoming replenishment
    total_receipts = [
        sr + pr for sr, pr in zip(scheduled_receipts, planned_receipts)
    ]

    # Bar: Inflow (Receipts)
    fig.add_trace(go.Bar(
        x=periods,
        y=total_receipts,
        name="Replenishment Receipts (SR + PORec)",
        marker_color="#2ca02c",
        opacity=0.75,
    ))

    # Bar: Outflow (Gross Requirements)
    neg_demand = [-d for d in gross_requirements]
    fig.add_trace(go.Bar(
        x=periods,
        y=neg_demand,
        name="Gross Requirements (Demand)",
        marker_color="#d62728",
        opacity=0.75,
    ))

    # Line: PAB trajectory
    fig.add_trace(go.Scatter(
        x=periods,
        y=pab_trajectory,
        mode="lines+markers",
        name="Projected Available Balance (PAB)",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8),
    ))

    # Safety stock line
    fig.add_trace(go.Scatter(
        x=periods,
        y=[safety_stock] * len(periods),
        mode="lines",
        name=f"Safety Stock ({safety_stock:,.0f})",
        line=dict(color="#e377c2", width=2, dash="dash"),
    ))

    # Zero stock line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    fig.update_layout(
        title=f"Inventory Waterfall & PAB Trajectory — SKU: {item_id} (Initial On-Hand: {initial_on_hand:,.0f})",
        xaxis_title="Planning Period",
        yaxis_title="Inventory Units",
        template="plotly_white",
        barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Capacity Heatmap: Work Center Utilization Matrix
# ---------------------------------------------------------------------------

def plot_capacity_heatmap(
    utilization_matrix: Dict[str, Dict[int, float]],
    title: str = "Rough-Cut Capacity Planning (RCCP) — Work Center Utilization Heatmap (%)",
    threshold_pct: float = 100.0,
) -> go.Figure:
    """Render 2D heatmap of work center capacity utilization (%) by period.

    Colors:
      - Green: < 80% (Slack)
      - Yellow: 80% - 100% (Balanced)
      - Red: > 100% (Overload / Bottleneck)

    Args:
        utilization_matrix: Dict of {work_center_id: {period: util_pct}}.
        title: Chart title.
        threshold_pct: Overload threshold (default 100%).
    """
    if not utilization_matrix:
        fig = go.Figure()
        fig.update_layout(title="No Capacity Data Available", template="plotly_white")
        return fig

    df_util = pd.DataFrame(utilization_matrix).T  # rows: WC, cols: periods
    df_util = df_util.sort_index()
    # Sort period columns numerically
    df_util = df_util.reindex(columns=sorted(df_util.columns, key=lambda x: int(x)))

    work_centers = df_util.index.tolist()
    periods = [f"Period {c}" for c in df_util.columns]
    z_values = df_util.values

    # Custom color scale: Green (0-80) -> Yellow (100) -> Red (>120)
    colorscale = [
        [0.0, "#2ca02c"],    # Green: low util
        [0.6, "#ffbb78"],    # Warm: medium util
        [0.8, "#ff7f0e"],    # Orange: close to 100%
        [0.833, "#d62728"],  # Red: 100%+ overload
        [1.0, "#7f0000"],    # Dark red: severe overload
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=periods,
        y=work_centers,
        colorscale=colorscale,
        zmin=0,
        zmax=150,
        text=[[f"{val:.1f}%" for val in row] for row in z_values],
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Utilization %"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Planning Period",
        yaxis_title="Work Center ID",
        template="plotly_white",
        height=max(360, len(work_centers) * 60 + 150),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Nervousness Trend: Plan Churn & Stability
# ---------------------------------------------------------------------------

def plot_nervousness_trend(
    cycles: List[int],
    nervousness_indices: List[float],
    order_churn: Optional[List[float]] = None,
    timing_churn: Optional[List[float]] = None,
    alert_threshold: float = 0.15,
) -> go.Figure:
    """Plot multi-period plan nervousness trend with component metrics and alert threshold.

    Args:
        cycles: List of planning cycle numbers (1, 2, 3...).
        nervousness_indices: Composite nervousness indices.
        order_churn: Quantity churn metrics (optional).
        timing_churn: Timing churn metrics (optional).
        alert_threshold: Threshold line triggering planner alerts (default 0.15).
    """
    fig = go.Figure()

    # Area + line for composite nervousness
    fig.add_trace(go.Scatter(
        x=cycles,
        y=nervousness_indices,
        mode="lines+markers",
        name="Composite Nervousness Index",
        line=dict(color="#9467bd", width=3),
        marker=dict(size=8, symbol="diamond"),
        fill="tozeroy",
        fillcolor="rgba(148, 103, 189, 0.15)",
    ))

    # Optional component lines
    if order_churn:
        fig.add_trace(go.Scatter(
            x=cycles,
            y=order_churn,
            mode="lines",
            name="Order Quantity Churn",
            line=dict(color="#1f77b4", width=1.5, dash="dot"),
        ))

    if timing_churn:
        fig.add_trace(go.Scatter(
            x=cycles,
            y=timing_churn,
            mode="lines",
            name="Timing Churn",
            line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        ))

    # Threshold alert line
    fig.add_hline(
        y=alert_threshold,
        line_dash="dash",
        line_color="#d62728",
        line_width=2,
        annotation_text=f"Alert Threshold ({alert_threshold})",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Rolling Horizon Plan Stability & Nervousness Trend",
        xaxis_title="Planning Cycle / Day (Period t)",
        yaxis_title="Nervousness Index",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Pareto Analysis: Exception Messages
# ---------------------------------------------------------------------------

def plot_pareto_exceptions(
    exceptions: Union[List[Dict[str, Any]], pd.DataFrame],
    group_by: str = "item_id",
    metric: str = "dollar_impact",
    top_n: int = 15,
) -> go.Figure:
    """Render Pareto 80/20 chart of planning exceptions.

    Bars show individual driver impact (dollar impact or count);
    line shows cumulative percentage (0 to 100%).

    Args:
        exceptions: List of exception dictionaries or DataFrame.
        group_by: Grouping dimension ('item_id', 'exception_type', or 'work_center_id').
        metric: Metric to aggregate ('dollar_impact' or 'count').
        top_n: Limit top N categories on x-axis.
    """
    df = exceptions if isinstance(exceptions, pd.DataFrame) else pd.DataFrame(exceptions)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Exceptions Data Available", template="plotly_white")
        return fig

    df = df.copy()
    if group_by not in df.columns:
        group_by = "exception_type"

    # Aggregation
    if metric == "dollar_impact" and "dollar_impact" in df.columns:
        agg = df.groupby(group_by)["dollar_impact"].sum().reset_index()
        y_col = "dollar_impact"
        y_title = "Total Financial Impact ($)"
    else:
        agg = df.groupby(group_by).size().reset_index(name="count")
        y_col = "count"
        y_title = "Exception Count"

    # Sort descending
    agg = agg.sort_values(by=y_col, ascending=False).head(top_n)
    total_val = agg[y_col].sum()
    agg["cumulative_pct"] = (agg[y_col].cumsum() / total_val * 100.0) if total_val > 0 else 0.0

    # Create 2-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar trace
    fig.add_trace(
        go.Bar(
            x=agg[group_by].astype(str),
            y=agg[y_col],
            name=y_title,
            marker_color="#1f77b4",
        ),
        secondary_y=False,
    )

    # Cumulative line trace
    fig.add_trace(
        go.Scatter(
            x=agg[group_by].astype(str),
            y=agg["cumulative_pct"],
            name="Cumulative %",
            mode="lines+markers",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )

    # 80% reference line
    fig.add_hline(
        y=80.0,
        line_dash="dot",
        line_color="#ff7f0e",
        secondary_y=True,
        annotation_text="80% Threshold",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Pareto Analysis of Planning Exceptions by {group_by.replace('_', ' ').title()}",
        xaxis_title=group_by.replace("_", " ").title(),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    fig.update_yaxes(title_text=y_title, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 6. Supplier Scorecard: Reliability & Risk Ranking
# ---------------------------------------------------------------------------

def plot_supplier_scorecard(
    suppliers_data: Union[List[Dict[str, Any]], pd.DataFrame],
    scorecard_type: str = "scatter",
) -> go.Figure:
    """Render interactive supplier risk & reliability visualization.

    Types:
      - 'scatter': Scatter plot of On-Time Delivery % vs Quality Reject Rate %,
        sized by Lead Time, colored by Risk Tier.
      - 'ranking': Horizontal bar chart ranking suppliers by Composite Score.

    Args:
        suppliers_data: List of supplier dicts or DataFrame containing fields:
                        supplier_id, name, composite_score or reliability_score,
                        fill_rate, quality_reject_rate, tier.
        scorecard_type: 'scatter' or 'ranking'.
    """
    df = suppliers_data if isinstance(suppliers_data, pd.DataFrame) else pd.DataFrame(suppliers_data)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Supplier Data Available", template="plotly_white")
        return fig

    df = df.copy()
    score_col = "composite_score" if "composite_score" in df.columns else "reliability_score"
    if score_col not in df.columns:
        df[score_col] = 80.0

    if "tier" not in df.columns:
        # Auto-assign tiers if missing
        df["tier"] = df[score_col].apply(
            lambda s: "PREFERRED" if s >= 85 else ("STANDARD" if s >= 70 else "HIGH_RISK")
        )

    tier_color_map = {
        "PREFERRED": "#2ca02c",    # Green
        "STANDARD": "#1f77b4",     # Blue
        "CONDITIONAL": "#ff7f0e",  # Orange
        "HIGH_RISK": "#d62728",    # Red
    }

    if scorecard_type == "ranking":
        df_sorted = df.sort_values(by=score_col, ascending=True)
        fig = px.bar(
            df_sorted,
            x=score_col,
            y="name" if "name" in df_sorted.columns else "supplier_id",
            color="tier",
            color_discrete_map=tier_color_map,
            orientation="h",
            text=score_col,
            labels={score_col: "Reliability Score", "tier": "Supplier Tier"},
            title="Supplier Performance & Risk Tier Ranking",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="inside")
        fig.update_layout(template="plotly_white", height=max(400, len(df) * 35 + 100))
        return fig

    # Default: Scatter plot (OTD vs Quality / Reliability)
    x_col = "fill_rate" if "fill_rate" in df.columns else score_col
    y_col = "quality_reject_rate" if "quality_reject_rate" in df.columns else score_col
    size_col = "quoted_lead_time_days" if "quoted_lead_time_days" in df.columns else None

    # Normalise fill_rate to percentage if in decimal [0, 1]
    if x_col == "fill_rate" and df[x_col].max() <= 1.0:
        df[x_col] = df[x_col] * 100.0

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color="tier",
        color_discrete_map=tier_color_map,
        hover_name="name" if "name" in df.columns else "supplier_id",
        hover_data=["supplier_id", score_col],
        labels={
            x_col: "Order Fill Rate (%)" if x_col == "fill_rate" else "Score",
            y_col: "Quality Reject Rate (%)" if y_col == "quality_reject_rate" else "Score",
            "tier": "Risk Tier",
        },
        title="Supplier Risk Matrix: Fill Rate vs. Quality Rejection",
    )

    fig.update_layout(
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

