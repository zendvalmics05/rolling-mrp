"""Unit tests for Phase 7.2 — What-If Scenario Engine (app/scenario_engine.py)."""

import os
import json
import pytest
import pandas as pd

from app.scenario_engine import (
    ScenarioEngine,
    ScenarioParameters,
    ScenarioKPIs,
    ScenarioRunResult,
)
from src.integration.data_generator import generate_synthetic_data


@pytest.fixture
def synthetic_factory_setup(tmp_path):
    data = generate_synthetic_data(seed=42)
    items_df = data["items"]
    bom_df = data["bom"]
    routings_df = data["routings"]
    wc_df = data["work_centers"]
    inv_df = data["inventory"]

    merged_items = items_df.merge(inv_df, on="item_id", how="left")
    merged_items["quantity_on_hand"] = merged_items["quantity_on_hand"].fillna(0.0)
    merged_items["quantity_allocated"] = merged_items["quantity_allocated"].fillna(0.0)
    merged_items["lead_time"] = (merged_items["lead_time_days"] / 7.0).round().astype(int).clip(lower=1)

    periods = list(range(1, 7))  # 6 periods for fast testing
    fg_items = merged_items[merged_items["type"] == "finished_good"]["item_id"].tolist()

    # Create baseline MPS
    base_mps = {fg: [30.0] * len(periods) for fg in fg_items}

    storage_dir = tmp_path / "scenarios"

    engine = ScenarioEngine(
        base_items_df=merged_items,
        base_bom_df=bom_df,
        base_routings_df=routings_df,
        base_wc_df=wc_df,
        base_mps_requirements=base_mps,
        periods=periods,
        storage_dir=str(storage_dir),
    )
    return engine


class TestScenarioParameters:

    def test_parameter_round_trip(self):
        p = ScenarioParameters(
            scenario_id="SCEN-01",
            name="Demand Spike",
            demand_pct_change=25.0,
            lead_time_days_delta=7,
            lead_time_period_delta=1,
            safety_stock_pct_change=-10.0,
            capacity_hours_delta=-15.0,
            capacity_pct_change=-20.0,
        )
        d = p.to_dict()
        restored = ScenarioParameters.from_dict(d)
        assert restored.scenario_id == p.scenario_id
        assert restored.demand_pct_change == 25.0
        assert restored.capacity_hours_delta == -15.0


class TestScenarioPerturbations:

    def test_apply_demand_shift(self, synthetic_factory_setup):
        engine = synthetic_factory_setup
        params = ScenarioParameters(
            scenario_id="S1",
            name="Demand +50%",
            demand_pct_change=50.0,
        )
        _, _, mps_mod = engine.apply_parameters(params)
        for fg, reqs in mps_mod.items():
            # Base was 30 -> 30 * 1.5 = 45.0
            assert all(r == pytest.approx(45.0) for r in reqs)

    def test_apply_sku_specific_demand_shift(self, synthetic_factory_setup):
        engine = synthetic_factory_setup
        params = ScenarioParameters(
            scenario_id="S2",
            name="Selective Demand",
            demand_pct_change=0.0,
            item_specific_demand_pct={"FG-01": 100.0},
        )
        _, _, mps_mod = engine.apply_parameters(params)
        assert all(r == pytest.approx(60.0) for r in mps_mod["FG-01"])
        # Other items remain at 30.0
        assert all(r == pytest.approx(30.0) for r in mps_mod["FG-02"])

    def test_apply_capacity_shift(self, synthetic_factory_setup):
        engine = synthetic_factory_setup
        base_cap = engine.base_wc_df.set_index("work_center_id")["capacity_hours_per_week"].to_dict()
        params = ScenarioParameters(
            scenario_id="S3",
            name="Overtime +20%",
            capacity_pct_change=20.0,
        )
        _, wc_mod, _ = engine.apply_parameters(params)
        mod_cap = wc_mod.set_index("work_center_id")["capacity_hours_per_week"].to_dict()
        for wcid in base_cap:
            assert mod_cap[wcid] == pytest.approx(base_cap[wcid] * 1.2)


class TestScenarioExecutionAndComparison:

    def test_run_baseline_scenario(self, synthetic_factory_setup):
        engine = synthetic_factory_setup
        base_res = engine.get_base_scenario_result()
        assert isinstance(base_res, ScenarioRunResult)
        assert base_res.kpis.total_planned_orders > 0
        assert base_res.kpis.total_planned_units > 0
        assert base_res.kpis.max_wc_utilization_pct >= 0.0

    def test_run_demand_spike_scenario(self, synthetic_factory_setup):
        engine = synthetic_factory_setup
        base_res = engine.get_base_scenario_result()

        spike_params = ScenarioParameters(
            scenario_id="SPIKE_30",
            name="Demand Spike +30%",
            demand_pct_change=30.0,
        )
        spike_res = engine.run_scenario(spike_params)

        assert spike_res.kpis.total_planned_units > base_res.kpis.total_planned_units
        assert spike_res.kpis.total_inventory_cost > base_res.kpis.total_inventory_cost

    def test_compare_scenarios_view(self, synthetic_factory_setup):
        engine = synthetic_factory_setup
        base_res = engine.get_base_scenario_result()
        spike_params = ScenarioParameters(
            scenario_id="SPIKE_20",
            name="Demand Spike +20%",
            demand_pct_change=20.0,
        )
        spike_res = engine.run_scenario(spike_params)

        df_comp = engine.compare_scenarios(spike_res, base_res)
        assert isinstance(df_comp, pd.DataFrame)
        assert "KPI Metric" in df_comp.columns
        assert "Baseline" in df_comp.columns
        assert "Delta" in df_comp.columns
        assert "Delta %" in df_comp.columns

        # Verify rows present
        metrics = df_comp["KPI Metric"].tolist()
        assert "Total Planned Orders" in metrics
        assert "Total Planned Units" in metrics
        assert "Total Inventory Cost ($)" in metrics
        assert "Peak Work Center Utilization (%)" in metrics


class TestPersistenceAndExport:

    def test_save_load_delete_scenario(self, synthetic_factory_setup):
        engine = synthetic_factory_setup
        params = ScenarioParameters(
            scenario_id="TEST_SAVED",
            name="Saved Test Scenario",
            demand_pct_change=15.0,
            safety_stock_pct_change=10.0,
        )
        saved_path = engine.save_scenario(params)
        assert os.path.exists(saved_path)

        loaded = engine.load_scenario("TEST_SAVED")
        assert loaded is not None
        assert loaded.scenario_id == "TEST_SAVED"
        assert loaded.demand_pct_change == 15.0

        all_saved = engine.list_saved_scenarios()
        assert any(s.scenario_id == "TEST_SAVED" for s in all_saved)

        deleted = engine.delete_scenario("TEST_SAVED")
        assert deleted is True
        assert not os.path.exists(saved_path)
        assert engine.load_scenario("TEST_SAVED") is None

    def test_export_scenario_results_csv(self, synthetic_factory_setup, tmp_path):
        engine = synthetic_factory_setup
        base_res = engine.get_base_scenario_result()

        csv_path = tmp_path / "base_export.csv"
        content = engine.export_scenario_results(base_res, output_format="csv", output_path=str(csv_path))
        assert os.path.exists(csv_path)
        assert "SCENARIO KPIS" in content
        assert "PLANNED ORDERS" in content

    def test_export_comparison_view_csv(self, synthetic_factory_setup, tmp_path):
        engine = synthetic_factory_setup
        base_res = engine.get_base_scenario_result()
        comp_df = engine.compare_scenarios(base_res, base_res)

        csv_path = tmp_path / "comparison.csv"
        content = engine.export_comparison_view(comp_df, output_format="csv", output_path=str(csv_path))
        assert os.path.exists(csv_path)
        assert "KPI Metric" in content
        assert "Baseline" in content

    def test_additional_coverage_perturbations_and_export(self, synthetic_factory_setup, tmp_path):
        engine = synthetic_factory_setup
        params = ScenarioParameters(
            scenario_id="S_FULL",
            name="Full Perturbations",
            lead_time_days_delta=5,
            lead_time_period_delta=1,
            safety_stock_pct_change=25.0,
            capacity_hours_delta=-5.0,
            wc_specific_capacity_delta={"WC-100": 10.0},
        )
        items_mod, wc_mod, _ = engine.apply_parameters(params)
        assert items_mod["lead_time_days"].iloc[0] > 0

        res = engine.run_scenario(params)
        assert res.kpis.scenario_id == "S_FULL"

        # Cached hit test
        cached_res = engine.run_scenario(params, use_cache=True)
        assert cached_res == res

        # Export Excel fallback test
        excel_path = tmp_path / "scenario.xlsx"
        bytes_out = engine.export_scenario_results(res, output_format="excel", output_path=str(excel_path))
        assert os.path.exists(excel_path)
        assert len(bytes_out) > 0

        comp_df = engine.compare_scenarios(res)
        comp_excel_path = tmp_path / "comparison.xlsx"
        comp_bytes = engine.export_comparison_view(comp_df, output_format="excel", output_path=str(comp_excel_path))
        assert os.path.exists(comp_excel_path)
        assert len(comp_bytes) > 0

        # Bad format test
        with pytest.raises(ValueError):
            engine.export_scenario_results(res, output_format="pdf")
        with pytest.raises(ValueError):
            engine.export_comparison_view(comp_df, output_format="pdf")
