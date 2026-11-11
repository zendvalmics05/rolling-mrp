"""Unit tests for Phase 8.1 — Cost of Nervousness Experiment (experiments/E02_cost_of_nervousness)."""

import json
import os
from pathlib import Path
import pytest
import pandas as pd

from experiments.E02_cost_of_nervousness.run_experiment import run_nervousness_cost_simulation


class TestCostOfNervousnessExperiment:

    def test_single_replication_simulation_runs(self):
        rec = run_nervousness_cost_simulation(
            frozen_length=2,
            demand_cv=0.15,
            seed=42,
            num_weeks=4,
        )
        assert isinstance(rec, dict)
        assert "nervousness_index" in rec
        assert "total_operational_cost" in rec
        assert "order_churn" in rec
        assert "timing_churn" in rec
        assert "cancellation_rate" in rec
        assert rec["nervousness_index"] >= 0.0
        assert rec["total_operational_cost"] > 0.0

    def test_frozen_window_suppresses_nervousness(self):
        # 0-week frozen window vs 2-week frozen window under identical demand volatility
        rec_free = run_nervousness_cost_simulation(frozen_length=0, demand_cv=0.25, seed=100, num_weeks=4)
        rec_frozen = run_nervousness_cost_simulation(frozen_length=2, demand_cv=0.25, seed=100, num_weeks=4)

        # A non-zero frozen window should strictly dampen or stabilize plan nervousness
        assert rec_frozen["nervousness_index"] <= rec_free["nervousness_index"] + 0.05
        assert rec_frozen["expedite_cost"] <= rec_free["expedite_cost"]

    def test_generated_experiment_artifacts_exist(self):
        base_dir = Path("experiments/E02_cost_of_nervousness")
        csv_results = base_dir / "e02_simulation_results.csv"
        csv_decomp = base_dir / "e02_nervousness_decomposition.csv"
        json_quant = base_dir / "e02_cost_quantification.json"
        md_report = Path("reports/E02_cost_of_nervousness_results.md")

        assert csv_results.exists()
        assert csv_decomp.exists()
        assert json_quant.exists()
        assert md_report.exists()

        # Check content validity
        df = pd.read_csv(csv_results)
        assert len(df) >= 30
        assert "nervousness_index" in df.columns
        assert "total_operational_cost" in df.columns

        with open(json_quant) as f:
            quant = json.load(f)
        assert "regression_total_cost" in quant
        assert "marginal_cost_per_0_01_ni" in quant["regression_total_cost"]
        assert quant["regression_total_cost"]["marginal_cost_per_0_01_ni"] > 0

