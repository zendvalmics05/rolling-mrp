"""Unit tests for Phase 7.1 — Exception Inbox (app/exception_inbox.py)."""

import os
import json
import tempfile
import pytest
import pandas as pd

from app.exception_inbox import (
    ActionStatus,
    AuditActionRecord,
    ExceptionInboxManager,
    ExceptionMessage,
    ExceptionPriority,
    ExceptionType,
)


@pytest.fixture
def inbox_manager():
    return ExceptionInboxManager(
        critical_threshold_dollars=10000.0,
        high_threshold_dollars=2500.0,
        medium_threshold_dollars=500.0,
    )


@pytest.fixture
def populated_inbox(inbox_manager):
    inbox_manager.create_and_add_exception(
        exception_type=ExceptionType.EXPEDITE,
        item_id="FG-01",
        period=1,
        quantity=150.0,
        unit_cost=100.0,  # $15,000 -> CRITICAL
        description="Expedite past due order",
        recommended_action="Expedite line run",
    )
    inbox_manager.create_and_add_exception(
        exception_type=ExceptionType.DEFER,
        item_id="COMP-A",
        period=3,
        quantity=50.0,
        unit_cost=60.0,  # $3,000 -> HIGH
        description="Defer inbound component batch",
        recommended_action="Reschedule supplier shipment",
    )
    inbox_manager.create_and_add_exception(
        exception_type=ExceptionType.SAFETY_STOCK_BREACH,
        item_id="RAW-01",
        period=2,
        quantity=200.0,
        unit_cost=5.0,  # $1,000 -> MEDIUM
        description="Safety stock breach",
        recommended_action="Issue replenishment purchase order",
    )
    inbox_manager.create_and_add_exception(
        exception_type=ExceptionType.CANCEL,
        item_id="RAW-02",
        period=4,
        quantity=10.0,
        unit_cost=20.0,  # $200 -> LOW
        description="Cancel excess order",
        recommended_action="Cancel PO with supplier",
    )
    return inbox_manager


class TestExceptionPriorityAndCreation:

    def test_priority_tier_assignment(self, inbox_manager):
        assert inbox_manager.calculate_priority(15000.0) == ExceptionPriority.CRITICAL
        assert inbox_manager.calculate_priority(5000.0) == ExceptionPriority.HIGH
        assert inbox_manager.calculate_priority(1000.0) == ExceptionPriority.MEDIUM
        assert inbox_manager.calculate_priority(200.0) == ExceptionPriority.LOW

    def test_priority_colors(self, populated_inbox):
        msgs = populated_inbox.list_exceptions(sort_by="dollar_impact")
        assert msgs[0].priority == ExceptionPriority.CRITICAL
        assert msgs[0].priority_color == "#dc3545"
        assert msgs[1].priority == ExceptionPriority.HIGH
        assert msgs[1].priority_color == "#fd7e14"
        assert msgs[2].priority == ExceptionPriority.MEDIUM
        assert msgs[2].priority_color == "#ffc107"
        assert msgs[3].priority == ExceptionPriority.LOW
        assert msgs[3].priority_color == "#0d6efd"

    def test_to_dict_and_from_dict_round_trip(self, populated_inbox):
        msg = populated_inbox.list_exceptions()[0]
        d = msg.to_dict()
        restored = ExceptionMessage.from_dict(d)
        assert restored.exception_id == msg.exception_id
        assert restored.item_id == msg.item_id
        assert restored.dollar_impact == msg.dollar_impact
        assert restored.priority == msg.priority
        assert restored.status == msg.status


class TestFilteringAndSorting:

    def test_filter_by_priority(self, populated_inbox):
        critical_msgs = populated_inbox.list_exceptions(priority=ExceptionPriority.CRITICAL)
        assert len(critical_msgs) == 1
        assert critical_msgs[0].item_id == "FG-01"

    def test_filter_by_type(self, populated_inbox):
        expedites = populated_inbox.list_exceptions(exception_type=ExceptionType.EXPEDITE)
        assert len(expedites) == 1
        assert expedites[0].exception_type == ExceptionType.EXPEDITE

    def test_filter_by_min_dollar_impact(self, populated_inbox):
        high_value = populated_inbox.list_exceptions(min_dollar_impact=2000.0)
        assert len(high_value) == 2  # $15,000 and $3,000

    def test_sort_by_period(self, populated_inbox):
        sorted_msgs = populated_inbox.list_exceptions(sort_by="period", ascending=True)
        assert [m.period for m in sorted_msgs] == [1, 2, 3, 4]

    def test_to_dataframe(self, populated_inbox):
        df = populated_inbox.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "exception_id" in df.columns
        assert "dollar_impact" in df.columns
        assert "priority" in df.columns


class TestActionsAndAuditTrail:

    def test_approve_action(self, populated_inbox):
        msg = populated_inbox.approve("EXC-0001", user="john_planner", comment="Approved for expedite")
        assert msg.status == ActionStatus.APPROVED
        assert len(populated_inbox.audit_trail) == 1
        audit = populated_inbox.audit_trail[0]
        assert audit.exception_id == "EXC-0001"
        assert audit.action == ActionStatus.APPROVED
        assert audit.user == "john_planner"
        assert audit.comment == "Approved for expedite"

    def test_reject_action(self, populated_inbox):
        msg = populated_inbox.reject("EXC-0002", user="jane_lead", comment="Too costly")
        assert msg.status == ActionStatus.REJECTED
        assert populated_inbox.audit_trail[-1].action == ActionStatus.REJECTED

    def test_defer_action(self, populated_inbox):
        msg = populated_inbox.defer("EXC-0003", user="john_planner", comment="Wait for supplier confirmation")
        assert msg.status == ActionStatus.DEFERRED
        assert populated_inbox.audit_trail[-1].action == ActionStatus.DEFERRED

    def test_batch_action(self, populated_inbox):
        updated = populated_inbox.batch_action(
            exception_ids=["EXC-0001", "EXC-0002"],
            action=ActionStatus.APPROVED,
            user="batch_agent",
            comment="Bulk approved",
        )
        assert len(updated) == 2
        assert all(m.status == ActionStatus.APPROVED for m in updated)
        assert len(populated_inbox.audit_trail) == 2

    def test_audit_persistence_json_lines(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        mgr = ExceptionInboxManager(audit_log_path=str(log_file))
        mgr.create_and_add_exception(ExceptionType.EXPEDITE, "X", 1, 10.0, 10.0)
        mgr.approve("EXC-0001", user="alice", comment="approved")
        
        assert os.path.exists(log_file)
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["exception_id"] == "EXC-0001"
        assert record["action"] == "APPROVED"
        assert record["user"] == "alice"


class TestEngineIngestion:

    def test_ingest_mrp_netting_exceptions(self, inbox_manager):
        raw_netting_exceptions = [
            {"item_id": "FG-01", "period_due": 1, "release_period": -1, "type": "expedite", "quantity": 100.0, "description": "Past due release"},
            {"item_id": "FG-02", "period_due": 2, "release_period": 0, "type": "expedite", "quantity": 50.0, "description": "Period 0 past due"},
        ]
        created = inbox_manager.ingest_mrp_netting_exceptions(
            raw_netting_exceptions,
            unit_costs={"FG-01": 25.0, "FG-02": 50.0},
        )
        assert len(created) == 2
        assert created[0].dollar_impact == 2500.0
        assert created[0].priority == ExceptionPriority.HIGH
        assert created[1].dollar_impact == 2500.0

    def test_ingest_capacity_exceptions(self, inbox_manager):
        class MockCapacityException:
            def __init__(self, wc, p, hrs, action, items):
                self.work_center_id = wc
                self.period = p
                self.overload_hours = hrs
                self.suggested_action = action
                self.impacted_items = items

        cap_exceptions = [
            MockCapacityException("WC-300", 6, 40.0, "Pre-build early", ["FG-01", "FG-02"]),
        ]
        created = inbox_manager.ingest_capacity_exceptions(cap_exceptions, hourly_rate=100.0)
        assert len(created) == 1
        assert created[0].exception_type == ExceptionType.CAPACITY_OVERLOAD
        assert created[0].dollar_impact == 4000.0  # 40 hrs * $100
        assert created[0].work_center_id == "WC-300"
        assert created[0].priority == ExceptionPriority.HIGH

    def test_summary_metrics(self, populated_inbox):
        metrics = populated_inbox.get_summary_metrics()
        assert metrics["total_exceptions"] == 4
        assert metrics["pending_count"] == 4
        assert metrics["critical_pending_count"] == 1
        assert metrics["total_pending_dollar_impact"] == 19200.0  # 15000 + 3000 + 1000 + 200
        assert metrics["exceptions_by_type"]["EXPEDITE"] == 1

