"""Integration test for 10-Week Rolling Horizon Simulation (Phase 2)."""

import pytest
from src.integration.data_generator import generate_synthetic_data
from src.integration.load import clean_and_validate_dataset, initialize_inventory_status
from src.core.explosion import BOMExplosionEngine
from src.rolling.horizon import RollingHorizonManager, DemandEvolutionModel
from src.rolling.frozen_window import PlanComparator, TimeFencePolicy


def test_10_week_rolling_horizon_simulation_with_frozen_fences():
    """Simulate a 10-week rolling horizon with multi-level MRP and frozen window enforcement."""
    raw_data = generate_synthetic_data(seed=42)
    cleaned_data, _ = clean_and_validate_dataset(raw_data)
    inv_df = initialize_inventory_status(cleaned_data["inventory"], cleaned_data["items"])

    engine = BOMExplosionEngine(cleaned_data["bom"], cleaned_data["items"])
    comparator = PlanComparator(TimeFencePolicy(frozen_periods=2, slushy_periods=2))

    # Finished Goods rolling managers
    fg_items = cleaned_data["items"][cleaned_data["items"]["type"] == "finished_good"]["item_id"].tolist()
    fg_managers = {}
    for fg_id in fg_items:
        row = inv_df[inv_df["item_id"] == fg_id].iloc[0]
        fg_managers[fg_id] = RollingHorizonManager(
            item_id=fg_id,
            window_length=12,
            firm_window_length=2,
            initial_on_hand=float(row["quantity_on_hand"]),
            initial_allocated=float(row["quantity_allocated"]),
            safety_stock=float(row["safety_stock"]),
            lead_time_periods=1,
            seed=hash(fg_id) % 10000,
        )

    previous_full_plan = []
    simulation_audit = []

    # Run for 10 consecutive rolling weeks
    for week in range(1, 11):
        # 1. Build current 12-week window demand for each finished good
        current_periods = fg_managers[fg_items[0]].get_window_periods()
        mps_gross = {}
        for fg_id in fg_items:
            window_demands = fg_managers[fg_id].get_window_demand()
            mps_gross[fg_id] = [d.quantity for d in window_demands]

        # 2. Build current inventory state dictionary
        inv_dict = {}
        for _, row in inv_df.iterrows():
            iid = str(row["item_id"])
            if iid in fg_managers:
                inv_dict[iid] = {
                    "on_hand": fg_managers[iid].on_hand,
                    "allocated": fg_managers[iid].allocated,
                    "safety_stock": fg_managers[iid].safety_stock,
                    "lead_time_days": int(row["lead_time_days"]),
                }
            else:
                inv_dict[iid] = {
                    "on_hand": float(row["quantity_on_hand"]),
                    "allocated": float(row["quantity_allocated"]),
                    "safety_stock": float(row["safety_stock"]),
                    "lead_time_days": int(row["lead_time_days"]),
                }

        # 3. Execute multi-level MRP
        mrp_results, _ = engine.run_multi_level_mrp(
            mps_gross_requirements=mps_gross,
            periods=current_periods,
            inventory_status=inv_dict,
        )

        # 4. Flatten new plan planned orders
        new_plan = []
        for itm, res in mrp_results.items():
            for po in res.planned_orders:
                new_plan.append({
                    "item_id": po.item_id,
                    "due_period": po.due_period,
                    "release_period": po.release_period,
                    "quantity": po.quantity,
                    "status": po.status,
                })

        # 5. Enforce Frozen Window Time Fences against previous plan
        enforced_plan, changes = comparator.enforce_time_fences(
            new_plan=new_plan,
            previous_plan=previous_full_plan,
            current_period=week,
            allow_slushy_changes=True,
        )

        # Verify: no BLOCKED (frozen) change survives in enforced plan
        blocked_changes = [c for c in changes if c.status == "BLOCKED"]
        for b_change in blocked_changes:
            # The enforced plan must maintain the old quantity/period, not the unauthorized change
            matching_order = [
                o for o in enforced_plan if o["item_id"] == b_change.item_id and o["due_period"] == b_change.old_period
            ]
            if matching_order:
                assert matching_order[0]["quantity"] == b_change.old_quantity

        # 6. Advance simulation period for each Finished Good
        for fg_id in fg_items:
            # Check if any planned order for this item is due to arrive next period
            fg_orders = [o for o in enforced_plan if o["item_id"] == fg_id and o["due_period"] == week + 1]
            for ord_arrival in fg_orders:
                fg_managers[fg_id].record_scheduled_receipt(week + 1, ord_arrival["quantity"])

            step_log = fg_managers[fg_id].advance_period(active_planned_orders=fg_orders)
            simulation_audit.append(step_log)

        previous_full_plan = enforced_plan

    # Exit Criteria checks:
    # 1. 10 weeks simulated across all 5 Finished Goods = 50 step logs
    assert len(simulation_audit) == 50
    # 2. Window has progressed to week 11
    assert fg_managers[fg_items[0]].current_period == 11
    # 3. Verified zero uncaught errors

