"""Unit tests for Phase 8.2 — Sensitivity Analysis Experiment (experiments/E03_sensitivity)."""

import os
from pathlib import Path
import pytest
import pandas as pd

from experiments.E03_sensitivity.run_experiment import (
    run_capacity_tightness_study,
    run_demand_variability_study,
    run_lead_time_variability_study,
    run_service_level_target_study,
)


class TestSensitivityAnalysisExperiment:

    def test_capacity_tightness_execution(self):
        df = run_capacity_tightness_study(utilization_targets=[0.70, 0.95], replications=1, seed_base=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "mean_utilization_pct" in df.columns
        assert "total_capacity_friction_cost" in df.columns
        # Tighter capacity (95%) must have higher or equal friction cost than 70% slack
        cost_70 = df[df["target_utilization"] == 0.70]["total_capacity_friction_cost"].iloc[0]
        cost_95 = df[df["target_utilization"] == 0.95]["total_capacity_friction_cost"].iloc[0]
        assert cost_95 >= cost_70

    def test_demand_variability_execution(self):
        df = run_demand_variability_study(demand_cvs=[0.10, 0.30], replications=1, seed_base=60)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "nervousness_index" in df.columns
        assert "mean_dynamic_safety_stock_units" in df.columns
        # Higher CV must yield higher dynamic safety stock
        ss_10 = df[df["demand_cv"] == 0.10]["mean_dynamic_safety_stock_units"].iloc[0]
        ss_30 = df[df["demand_cv"] == 0.30]["mean_dynamic_safety_stock_units"].iloc[0]
        assert ss_30 > ss_10

    def test_lead_time_variability_execution(self):
        df = run_lead_time_variability_study(lead_time_stddevs=[0.0, 5.0], replications=1, seed_base=70)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "mean_lead_time_buffer_days" in df.columns
        assert "estimated_on_time_delivery_pct" in df.columns
        buf_0 = df[df["lead_time_std_days"] == 0.0]["mean_lead_time_buffer_days"].iloc[0]
        buf_5 = df[df["lead_time_std_days"] == 5.0]["mean_lead_time_buffer_days"].iloc[0]
        assert buf_0 == 0.0
        assert buf_5 > 0.0

    def test_service_level_target_execution(self):
        df = run_service_level_target_study(service_level_targets=[0.90, 0.99], replications=1, seed_base=80)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "total_safety_stock_investment_dollar" in df.columns
        inv_90 = df[df["service_level_target"] == 0.90]["total_safety_stock_investment_dollar"].iloc[0]
        inv_99 = df[df["service_level_target"] == 0.99]["total_safety_stock_investment_dollar"].iloc[0]
        assert inv_99 > inv_90

    def test_generated_artifacts_exist(self):
        base_dir = Path("experiments/E03_sensitivity")
        assert (base_dir / "e03_capacity_tightness.csv").exists()
        assert (base_dir / "e03_demand_variability.csv").exists()
        assert (base_dir / "e03_lead_time_variability.csv").exists()
        assert (base_dir / "e03_service_level_target.csv").exists()
        assert Path("reports/E03_sensitivity_analysis_results.md").exists()

