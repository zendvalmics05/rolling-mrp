"""Unit tests for Phase 7.3 — Interactive Visualizations (src/visualization/visualizers.py)."""

import pytest
import pandas as pd
import plotly.graph_objects as go

from src.visualization.visualizers import (
    plot_gantt_chart,
    plot_inventory_waterfall,
    plot_capacity_heatmap,
    plot_nervousness_trend,
    plot_pareto_exceptions,
    plot_supplier_scorecard,
)


class TestGanttChart:

    def test_gantt_chart_renders_orders(self):
        orders = [
            {"order_id": "PO-1", "item_id": "FG-01", "release_period": 1, "due_period": 3, "quantity": 100.0, "status": "PLANNED"},
            {"order_id": "PO-2", "item_id": "FG-02", "release_period": 2, "due_period": 4, "quantity": 50.0, "status": "RELEASED"},
        ]
        fig = plot_gantt_chart(orders)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert "Planned & Released" in fig.layout.title.text

    def test_gantt_chart_empty_dataframe(self):
        fig = plot_gantt_chart([])
        assert isinstance(fig, go.Figure)
        assert "No Orders Available" in fig.layout.title.text


class TestInventoryWaterfall:

    def test_inventory_waterfall_renders_all_traces(self):
        periods = [1, 2, 3, 4]
        gr = [50.0, 40.0, 60.0, 30.0]
        sr = [10.0, 0.0, 0.0, 0.0]
        por = [40.0, 40.0, 60.0, 30.0]
        pab = [20.0, 20.0, 20.0, 20.0]

        fig = plot_inventory_waterfall(
            item_id="FG-01",
            periods=periods,
            gross_requirements=gr,
            scheduled_receipts=sr,
            planned_receipts=por,
            pab_trajectory=pab,
            safety_stock=20.0,
            initial_on_hand=20.0,
        )
        assert isinstance(fig, go.Figure)
        # Receipts bar, Demand bar, PAB line, Safety Stock line
        assert len(fig.data) == 4
        assert "FG-01" in fig.layout.title.text


class TestCapacityHeatmap:

    def test_capacity_heatmap_renders_matrix(self):
        matrix = {
            "WC-100": {1: 75.0, 2: 95.0, 3: 115.0},
            "WC-200": {1: 60.0, 2: 80.0, 3: 100.0},
        }
        fig = plot_capacity_heatmap(matrix)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)
        assert list(fig.data[0].y) == ["WC-100", "WC-200"]

    def test_capacity_heatmap_empty(self):
        fig = plot_capacity_heatmap({})
        assert isinstance(fig, go.Figure)
        assert "No Capacity Data" in fig.layout.title.text


class TestNervousnessTrend:

    def test_nervousness_trend_renders_traces(self):
        cycles = [1, 2, 3, 4, 5]
        ni = [0.45, 0.25, 0.18, 0.12, 0.11]
        churn = [0.40, 0.20, 0.15, 0.10, 0.09]
        timing = [0.05, 0.05, 0.03, 0.02, 0.02]

        fig = plot_nervousness_trend(
            cycles=cycles,
            nervousness_indices=ni,
            order_churn=churn,
            timing_churn=timing,
            alert_threshold=0.15,
        )
        assert isinstance(fig, go.Figure)
        # Composite line, churn line, timing line
        assert len(fig.data) == 3
        assert "Plan Stability & Nervousness Trend" in fig.layout.title.text


class TestParetoExceptions:

    def test_pareto_exceptions_by_item(self):
        exceptions = [
            {"item_id": "FG-01", "exception_type": "EXPEDITE", "dollar_impact": 15000.0},
            {"item_id": "FG-02", "exception_type": "DEFER", "dollar_impact": 5000.0},
            {"item_id": "COMP-A", "exception_type": "SAFETY_STOCK", "dollar_impact": 2000.0},
        ]
        fig = plot_pareto_exceptions(exceptions, group_by="item_id")
        assert isinstance(fig, go.Figure)
        # Bar trace + Cumulative % line trace
        assert len(fig.data) == 2
        assert "Pareto Analysis" in fig.layout.title.text

    def test_pareto_exceptions_empty(self):
        fig = plot_pareto_exceptions([])
        assert isinstance(fig, go.Figure)
        assert "No Exceptions Data" in fig.layout.title.text


class TestSupplierScorecard:

    def test_supplier_scorecard_scatter(self):
        suppliers = [
            {"supplier_id": "SUP-01", "name": "Apex", "fill_rate": 95.0, "quality_reject_rate": 1.5, "quoted_lead_time_days": 14, "tier": "PREFERRED", "reliability_score": 92.0},
            {"supplier_id": "SUP-02", "name": "Beta", "fill_rate": 82.0, "quality_reject_rate": 5.0, "quoted_lead_time_days": 21, "tier": "HIGH_RISK", "reliability_score": 68.0},
        ]
        fig = plot_supplier_scorecard(suppliers, scorecard_type="scatter")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert "Supplier Risk Matrix" in fig.layout.title.text

    def test_supplier_scorecard_ranking(self):
        suppliers = [
            {"supplier_id": "SUP-01", "name": "Apex", "composite_score": 92.0, "tier": "PREFERRED"},
            {"supplier_id": "SUP-02", "name": "Beta", "composite_score": 68.0, "tier": "HIGH_RISK"},
        ]
        fig = plot_supplier_scorecard(suppliers, scorecard_type="ranking")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert "Supplier Performance & Risk Tier Ranking" in fig.layout.title.text

    def test_supplier_scorecard_empty(self):
        fig = plot_supplier_scorecard([])
        assert isinstance(fig, go.Figure)
        assert "No Supplier Data" in fig.layout.title.text

