"""Unit tests for Phase 5.1 & 5.2: Supplier Reliability Scoring and Dynamic Safety Stock."""

import pytest
import numpy as np
import pandas as pd

from src.risk.supplier import (
    SupplierReliabilityScorer,
    SupplierMetrics,
    SupplierTier,
    DualSourcingManager,
    DualSourcingSplit,
)
from src.risk.safety_stock import (
    DynamicSafetyStockEngine,
    SafetyStockRecommendation,
)


def test_supplier_reliability_scoring_composite():
    """Verify composite reliability score calculation and weights."""
    scorer = SupplierReliabilityScorer()

    # Perfect Supplier: OTD 100%, Fill 100%, Quality 100% (0% rejects), Stddev = 0
    perf = scorer.calculate_score_from_summary(
        supplier_id="SUP-PERFECT",
        name="Perfect Supplier",
        on_time_delivery_rate=1.0,
        fill_rate=1.0,
        quality_reject_rate=0.0,
        actual_lead_time_stddev=0.0,
    )
    assert perf.composite_score == 100.0
    assert perf.tier == SupplierTier.PREFERRED.value

    # Hand-calculated test case:
    # OTD = 80%, Fill = 90%, Reject = 4% (Quality = 96%), Stddev = 2.5 days (out of 5.0 max -> 50% stability)
    # Weights: 30% OTD, 25% Fill, 25% Quality, 20% LT Variability
    # Score = 0.30*80 + 0.25*90 + 0.25*96 + 0.20*50
    #       = 24.0 + 22.5 + 24.0 + 10.0 = 80.5
    res = scorer.calculate_score_from_summary(
        supplier_id="SUP-TEST",
        name="Test Supplier",
        on_time_delivery_rate=0.80,
        fill_rate=0.90,
        quality_reject_rate=0.04,
        actual_lead_time_stddev=2.5,
    )
    assert pytest.approx(res.composite_score, abs=0.1) == 80.5
    assert res.tier == SupplierTier.APPROVED.value


def test_supplier_scoring_from_history():
    """Verify score calculation from purchase order history records."""
    po_data = pd.DataFrame([
        {"supplier_id": "SUP-01", "quoted_lead_time_days": 7, "actual_lead_time_days": 6, "ordered_quantity": 100, "received_quantity": 100, "defective_quantity": 0},
        {"supplier_id": "SUP-01", "quoted_lead_time_days": 7, "actual_lead_time_days": 7, "ordered_quantity": 200, "received_quantity": 200, "defective_quantity": 2},
        {"supplier_id": "SUP-01", "quoted_lead_time_days": 7, "actual_lead_time_days": 8, "ordered_quantity": 100, "received_quantity": 90, "defective_quantity": 5},
    ])

    scorer = SupplierReliabilityScorer()
    res = scorer.calculate_score_from_history("SUP-01", "Supplier 1", po_data)

    assert res.sample_order_count == 3
    # 2 out of 3 on time -> 66.7%
    assert pytest.approx(res.on_time_delivery_pct, abs=0.5) == 66.67
    # Received 390 / 400 -> 97.5%
    assert pytest.approx(res.fill_rate_pct, abs=0.1) == 97.5
    assert 0.0 <= res.composite_score <= 100.0


def test_dynamic_safety_stock_risk_multiplier():
    """Verify risk multiplier increases safety stock for high-risk suppliers."""
    engine = DynamicSafetyStockEngine(max_risk_multiplier=2.0, min_risk_multiplier=1.0)

    # Score 100 -> Multiplier 1.0
    assert engine.calculate_supplier_risk_multiplier(100.0) == 1.0
    # Score 0 -> Multiplier 2.0
    assert engine.calculate_supplier_risk_multiplier(0.0) == 2.0
    # Score 50 -> Multiplier 1.5
    assert engine.calculate_supplier_risk_multiplier(50.0) == 1.5

    demand = [100.0, 120.0, 90.0, 110.0, 105.0]

    # Reliable supplier (score 95)
    rec_reliable = engine.calculate_dynamic_safety_stock(
        item_id="RM-001",
        demand_series=demand,
        lead_time_periods=2.0,
        supplier_score=95.0,
    )

    # High-risk supplier (score 50)
    rec_risky = engine.calculate_dynamic_safety_stock(
        item_id="RM-001",
        demand_series=demand,
        lead_time_periods=2.0,
        supplier_score=50.0,
    )

    assert rec_risky.dynamic_safety_stock > rec_reliable.dynamic_safety_stock
    # Risky safety stock should be approximately 1.5 / 1.05 times reliable safety stock
    ratio = rec_risky.dynamic_safety_stock / rec_reliable.dynamic_safety_stock
    assert pytest.approx(ratio, abs=0.1) == (1.5 / 1.05)


def test_lead_time_buffering():
    """Verify lead time buffer addition based on supplier delivery variability."""
    engine = DynamicSafetyStockEngine(buffer_k_factor=1.5)

    rec = engine.calculate_dynamic_safety_stock(
        item_id="RM-002",
        demand_series=[50.0, 55.0, 48.0],
        lead_time_periods=1.0,
        quoted_lead_time_days=10,
        actual_lead_time_stddev_days=2.4,
    )

    # Buffer: ceil(1.5 * 2.4) = ceil(3.6) = 4 days
    assert rec.lead_time_buffer_days == 4
    # Effective lead time: 10 + 4 = 14 days
    assert rec.effective_lead_time_days == 14


def test_supplier_portfolio_and_item_recommendations():
    """Verify end-to-end portfolio scoring and item safety stock recommendations on synthetic factory data."""
    from src.integration.data_generator import generate_synthetic_data

    raw = generate_synthetic_data(seed=42)
    scorer = SupplierReliabilityScorer()
    scorecards = scorer.evaluate_supplier_portfolio(raw["suppliers"])

    assert len(scorecards) == 4
    # SUP-04 should be the lowest scoring (budget raw materials)
    assert scorecards[-1].supplier_id == "SUP-04"
    assert scorecards[-1].tier == SupplierTier.HIGH_RISK.value

    scorecard_map = {s.supplier_id: s for s in scorecards}
    ss_engine = DynamicSafetyStockEngine()
    recs = ss_engine.evaluate_item_portfolio(raw["items"], scorecard_map)

    assert len(recs) == len(raw["items"])
    # Verify risk multiplier increases safety stock for items supplied by SUP-04 vs SUP-01
    sup4_recs = [r for r in recs if r.supplier_id == "SUP-04"]
    sup1_recs = [r for r in recs if r.supplier_id == "SUP-01"]
    if sup4_recs and sup1_recs:
        assert sup4_recs[0].risk_multiplier > sup1_recs[0].risk_multiplier


def test_dual_sourcing_order_splitting_and_cost_premium():
    """Verify dual sourcing allocation splits, risk reduction, and cost premium."""
    scorer = SupplierReliabilityScorer()

    sup_primary = scorer.calculate_score_from_summary(
        "SUP-01", "Primary Partner", on_time_delivery_rate=0.98, fill_rate=0.99, quality_reject_rate=0.01, actual_lead_time_stddev=0.6
    )
    sup_backup = scorer.calculate_score_from_summary(
        "SUP-02", "Backup Partner", on_time_delivery_rate=0.85, fill_rate=0.90, quality_reject_rate=0.05, actual_lead_time_stddev=2.5
    )

    manager = DualSourcingManager(min_split_pct=20.0, max_primary_pct=80.0)

    split = manager.calculate_order_split(
        item_id="RM-001",
        total_quantity=1000.0,
        primary_metrics=sup_primary,
        backup_metrics=sup_backup,
        primary_unit_cost=10.0,
        backup_unit_cost=12.0,
    )

    # Verify split bounds
    assert 20.0 <= split.backup_share_pct <= 80.0
    assert 20.0 <= split.primary_share_pct <= 80.0
    assert split.primary_share_pct > split.backup_share_pct  # Primary has higher reliability score
    assert pytest.approx(split.primary_quantity + split.backup_quantity) == 1000.0

    # Verify cost calculations
    assert split.single_source_cost == 10000.0  # 1000 * $10
    assert split.dual_source_cost > split.single_source_cost  # Premium incurred due to backup vendor
    assert split.cost_premium > 0.0
    assert split.is_diversified is True


