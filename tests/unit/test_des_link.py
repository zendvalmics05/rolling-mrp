"""Unit tests for Phase 6.2 — DES Link (src/integration/des_link.py)."""

import json
import os
import tempfile

import pytest

from src.integration.des_link import (
    DeviationAnalyzer,
    DESExporter,
    DESImporter,
    ExecutionRecord,
    OrderStatus,
    PlannedOrderExport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_planned_orders():
    return [
        PlannedOrderExport(
            order_id="PO-FG01-3",
            item_id="FG-01",
            release_period=2,
            due_period=3,
            quantity=50.0,
            status=OrderStatus.PLANNED,
            lot_sizing_rule="EOQ",
        ),
        PlannedOrderExport(
            order_id="PO-FG02-5",
            item_id="FG-02",
            release_period=4,
            due_period=5,
            quantity=30.0,
            status=OrderStatus.PLANNED,
            lot_sizing_rule="LFL",
        ),
        PlannedOrderExport(
            order_id="PO-FG03-7",
            item_id="FG-03",
            release_period=6,
            due_period=7,
            quantity=20.0,
            status=OrderStatus.PLANNED,
            lot_sizing_rule="FPQ",
            work_center_id="WC-100",
            routing_hours=4.5,
        ),
    ]


@pytest.fixture
def sample_execution_records():
    return [
        ExecutionRecord(
            order_id="PO-FG01-3",
            item_id="FG-01",
            planned_quantity=50.0,
            actual_quantity=50.0,  # on-time, full
            planned_period=3,
            actual_period=3,
            status=OrderStatus.COMPLETED,
        ),
        ExecutionRecord(
            order_id="PO-FG02-5",
            item_id="FG-02",
            planned_quantity=30.0,
            actual_quantity=25.0,  # partial delivery, 1 period late
            planned_period=5,
            actual_period=6,
            status=OrderStatus.PARTIAL,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: PlannedOrderExport serialisation
# ---------------------------------------------------------------------------

class TestPlannedOrderExport:

    def test_to_dict_round_trip(self, sample_planned_orders):
        """to_dict / from_dict round-trip preserves all fields."""
        for po in sample_planned_orders:
            d = po.to_dict()
            restored = PlannedOrderExport.from_dict(d)
            assert restored.order_id == po.order_id
            assert restored.item_id == po.item_id
            assert restored.release_period == po.release_period
            assert restored.due_period == po.due_period
            assert restored.quantity == po.quantity
            assert restored.status == po.status
            assert restored.lot_sizing_rule == po.lot_sizing_rule

    def test_from_dict_handles_string_metadata(self):
        """from_dict handles metadata as a JSON-encoded string (from CSV)."""
        data = {
            "order_id": "PO-X",
            "item_id": "X",
            "release_period": "1",
            "due_period": "2",
            "quantity": "10.0",
            "metadata": '{"source": "auto"}',
        }
        po = PlannedOrderExport.from_dict(data)
        assert po.metadata == {"source": "auto"}

    def test_routing_hours_optional_none(self):
        """routing_hours should be None when not supplied."""
        data = {
            "order_id": "PO-Y",
            "item_id": "Y",
            "release_period": 1,
            "due_period": 2,
            "quantity": 5.0,
            "routing_hours": None,
        }
        po = PlannedOrderExport.from_dict(data)
        assert po.routing_hours is None


# ---------------------------------------------------------------------------
# Tests: JSON Export / Import Round-Trip
# ---------------------------------------------------------------------------

class TestDESExporterJSON:

    def test_json_export_structure(self, sample_planned_orders):
        """JSON export contains expected envelope fields."""
        content = DESExporter.export_planned_orders(sample_planned_orders, fmt="json")
        data = json.loads(content)
        assert data["format"] == "ROLLINGMRP_DES_EXPORT_v1"
        assert data["count"] == 3
        assert len(data["orders"]) == 3

    def test_json_export_order_fields(self, sample_planned_orders):
        """Each order in JSON has all required fields."""
        content = DESExporter.export_planned_orders(sample_planned_orders, fmt="json")
        data = json.loads(content)
        for order_dict in data["orders"]:
            assert "order_id" in order_dict
            assert "item_id" in order_dict
            assert "quantity" in order_dict
            assert "release_period" in order_dict
            assert "due_period" in order_dict
            assert "status" in order_dict

    def test_json_export_to_file_and_reimport(self, sample_planned_orders, tmp_path):
        """JSON export written to file can be re-parsed back to PlannedOrderExport objects."""
        filepath = str(tmp_path / "export.json")
        DESExporter.export_planned_orders(sample_planned_orders, fmt="json", filepath=filepath)
        assert os.path.exists(filepath)

        with open(filepath, "r") as fh:
            data = json.load(fh)
        restored = [PlannedOrderExport.from_dict(o) for o in data["orders"]]
        assert len(restored) == len(sample_planned_orders)
        assert restored[0].order_id == sample_planned_orders[0].order_id
        assert restored[2].routing_hours == pytest.approx(4.5)

    def test_json_export_empty_list(self):
        """Exporting empty list produces valid JSON with count 0."""
        content = DESExporter.export_planned_orders([], fmt="json")
        data = json.loads(content)
        assert data["count"] == 0
        assert data["orders"] == []

    def test_invalid_format_raises(self, sample_planned_orders):
        """Unsupported format string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            DESExporter.export_planned_orders(sample_planned_orders, fmt="xml")


# ---------------------------------------------------------------------------
# Tests: CSV Export / Import Round-Trip
# ---------------------------------------------------------------------------

class TestDESExporterCSV:

    def test_csv_export_round_trip(self, sample_planned_orders, tmp_path):
        """CSV export can be re-read line by line."""
        import csv as _csv

        content = DESExporter.export_planned_orders(sample_planned_orders, fmt="csv")
        lines = content.strip().splitlines()
        # Header + 3 data rows
        assert len(lines) == 4

        reader = _csv.DictReader(lines)
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["order_id"] == "PO-FG01-3"

    def test_csv_export_empty_list(self):
        """CSV export of empty list returns empty string."""
        content = DESExporter.export_planned_orders([], fmt="csv")
        assert content == ""

    def test_csv_case_insensitive_format(self, sample_planned_orders):
        """fmt='CSV' (uppercase) is handled correctly."""
        content = DESExporter.export_planned_orders(sample_planned_orders, fmt="CSV")
        assert "order_id" in content


# ---------------------------------------------------------------------------
# Tests: Status Transition
# ---------------------------------------------------------------------------

class TestStatusTransition:

    def test_planned_to_released(self, sample_planned_orders):
        updated = DESExporter.transition_status(sample_planned_orders, OrderStatus.RELEASED)
        for o in updated:
            assert o.status == OrderStatus.RELEASED

    def test_released_to_in_progress(self, sample_planned_orders):
        released = DESExporter.transition_status(sample_planned_orders, OrderStatus.RELEASED)
        in_progress = DESExporter.transition_status(released, OrderStatus.IN_PROGRESS)
        for o in in_progress:
            assert o.status == OrderStatus.IN_PROGRESS

    def test_in_progress_to_completed(self, sample_planned_orders):
        released = DESExporter.transition_status(sample_planned_orders, OrderStatus.RELEASED)
        in_prog = DESExporter.transition_status(released, OrderStatus.IN_PROGRESS)
        completed = DESExporter.transition_status(in_prog, OrderStatus.COMPLETED)
        for o in completed:
            assert o.status == OrderStatus.COMPLETED

    def test_invalid_transition_keeps_status(self, sample_planned_orders):
        """Transitioning PLANNED → COMPLETED directly is not valid; status should be unchanged."""
        unchanged = DESExporter.transition_status(sample_planned_orders, OrderStatus.COMPLETED)
        for o in unchanged:
            assert o.status == OrderStatus.PLANNED  # unchanged because COMPLETED not in PLANNED's valid_next

    def test_planned_to_cancelled(self, sample_planned_orders):
        cancelled = DESExporter.transition_status(sample_planned_orders, OrderStatus.CANCELLED)
        for o in cancelled:
            assert o.status == OrderStatus.CANCELLED

    def test_full_lifecycle(self):
        """PLANNED → RELEASED → IN_PROGRESS → SCRAPPED."""
        orders = [PlannedOrderExport("O1", "X", 1, 2, 10.0)]
        o = DESExporter.transition_status(orders, OrderStatus.RELEASED)
        o = DESExporter.transition_status(o, OrderStatus.IN_PROGRESS)
        o = DESExporter.transition_status(o, OrderStatus.SCRAPPED)
        assert o[0].status == OrderStatus.SCRAPPED


# ---------------------------------------------------------------------------
# Tests: DESImporter
# ---------------------------------------------------------------------------

class TestDESImporter:

    def test_import_from_list_of_dicts(self, sample_execution_records):
        raw = [r.to_dict() for r in sample_execution_records]
        imported = DESImporter.import_execution_results(raw)
        assert len(imported) == 2
        assert imported[0].order_id == "PO-FG01-3"

    def test_import_from_json_string(self, sample_execution_records):
        payload = json.dumps([r.to_dict() for r in sample_execution_records])
        imported = DESImporter.import_execution_results(payload)
        assert len(imported) == 2

    def test_import_from_json_envelope(self, sample_execution_records):
        payload = json.dumps({"records": [r.to_dict() for r in sample_execution_records]})
        imported = DESImporter.import_execution_results(payload)
        assert len(imported) == 2

    def test_import_from_json_file(self, sample_execution_records, tmp_path):
        filepath = str(tmp_path / "results.json")
        data = [r.to_dict() for r in sample_execution_records]
        with open(filepath, "w") as fh:
            json.dump(data, fh)
        imported = DESImporter.import_execution_results(filepath)
        assert len(imported) == 2
        assert imported[1].actual_quantity == pytest.approx(25.0)

    def test_import_bad_json_string_raises(self):
        with pytest.raises(ValueError):
            DESImporter.import_execution_results("not_a_file_or_json")

    def test_import_bad_type_raises(self):
        with pytest.raises(ValueError):
            DESImporter.import_execution_results(12345)  # type: ignore

    def test_delay_auto_computed(self):
        """delay_periods should be auto-computed if not supplied."""
        rec = ExecutionRecord(
            order_id="X", item_id="A",
            planned_quantity=10.0, actual_quantity=10.0,
            planned_period=5, actual_period=7,
        )
        assert rec.delay_periods == 2

    def test_early_completion_negative_delay(self):
        rec = ExecutionRecord(
            order_id="Y", item_id="B",
            planned_quantity=10.0, actual_quantity=10.0,
            planned_period=8, actual_period=6,
        )
        assert rec.delay_periods == -2


# ---------------------------------------------------------------------------
# Tests: DeviationAnalyzer
# ---------------------------------------------------------------------------

class TestDeviationAnalyzer:

    def test_on_time_full_delivery(self, sample_planned_orders, sample_execution_records):
        # Only FG-01 is on-time/full; FG-02 is partial/late; FG-03 has no execution (cancelled)
        report, per_order = DeviationAnalyzer.detect_deviations(
            sample_planned_orders, sample_execution_records
        )
        assert report.on_time_count == 1
        assert report.late_count == 1
        assert report.cancelled_count == 1

    def test_fill_rate_calculation(self, sample_planned_orders, sample_execution_records):
        """fill_rate = sum(actual) / sum(planned) = 75/100 = 0.75."""
        report, _ = DeviationAnalyzer.detect_deviations(
            sample_planned_orders, sample_execution_records
        )
        # planned = 50 + 30 + 20 = 100; actual = 50 + 25 + 0(cancelled) = 75
        assert report.fill_rate == pytest.approx(0.75, abs=1e-3)

    def test_partial_count(self, sample_planned_orders, sample_execution_records):
        report, _ = DeviationAnalyzer.detect_deviations(
            sample_planned_orders, sample_execution_records
        )
        assert report.partial_count == 1  # FG-02 is partial

    def test_scrapped_order_counted(self):
        planned = [PlannedOrderExport("O1", "FG-X", 1, 2, 100.0)]
        actuals = [ExecutionRecord("O1", "FG-X", 100.0, 0.0, 2, 2, scrap_quantity=100.0, status=OrderStatus.SCRAPPED)]
        report, _ = DeviationAnalyzer.detect_deviations(planned, actuals)
        assert report.scrapped_count == 1

    def test_all_on_time_full_fill_rate_one(self):
        planned = [PlannedOrderExport("O1", "A", 1, 2, 10.0), PlannedOrderExport("O2", "B", 2, 3, 20.0)]
        actuals = [
            ExecutionRecord("O1", "A", 10.0, 10.0, 2, 2),
            ExecutionRecord("O2", "B", 20.0, 20.0, 3, 3),
        ]
        report, _ = DeviationAnalyzer.detect_deviations(planned, actuals)
        assert report.fill_rate == pytest.approx(1.0)
        assert report.on_time_count == 2
        assert report.late_count == 0
        assert report.cancelled_count == 0

    def test_timing_deviation_max(self):
        planned = [
            PlannedOrderExport("O1", "A", 1, 3, 10.0),
            PlannedOrderExport("O2", "B", 2, 5, 20.0),
        ]
        actuals = [
            ExecutionRecord("O1", "A", 10.0, 10.0, 3, 3),         # on time
            ExecutionRecord("O2", "B", 20.0, 20.0, 5, 8, delay_periods=3),  # 3 periods late
        ]
        report, _ = DeviationAnalyzer.detect_deviations(planned, actuals)
        assert report.timing_deviation_max == 3

    def test_per_order_deviation_detail(self, sample_planned_orders, sample_execution_records):
        _, per_order = DeviationAnalyzer.detect_deviations(
            sample_planned_orders, sample_execution_records
        )
        by_id = {d["order_id"]: d for d in per_order}
        # FG-01: exact match → 0 deviation, 100% fill
        assert by_id["PO-FG01-3"]["quantity_deviation"] == pytest.approx(0.0)
        assert by_id["PO-FG01-3"]["fill_pct"] == pytest.approx(100.0)
        # FG-02: 25 actual vs 30 planned, 1 period late
        assert by_id["PO-FG02-5"]["quantity_deviation"] == pytest.approx(-5.0)
        assert by_id["PO-FG02-5"]["delay_periods"] == 1

    def test_update_order_statuses(self, sample_planned_orders, sample_execution_records):
        updated = DeviationAnalyzer.update_order_statuses(sample_planned_orders, sample_execution_records)
        status_map = {o.order_id: o.status for o in updated}
        assert status_map["PO-FG01-3"] == OrderStatus.COMPLETED
        assert status_map["PO-FG02-5"] == OrderStatus.PARTIAL
        # FG-03 has no execution record → stays PLANNED
        assert status_map["PO-FG03-7"] == OrderStatus.PLANNED

    def test_empty_planned_and_actuals(self):
        report, per_order = DeviationAnalyzer.detect_deviations([], [])
        assert report.total_orders == 0
        assert report.fill_rate == pytest.approx(1.0)
        assert per_order == []

    def test_item_level_summary_populated(self, sample_planned_orders, sample_execution_records):
        report, _ = DeviationAnalyzer.detect_deviations(
            sample_planned_orders, sample_execution_records
        )
        assert "FG-01" in report.item_level_summary
        assert "FG-02" in report.item_level_summary
        assert report.item_level_summary["FG-01"]["fill_rate"] == pytest.approx(1.0)

    def test_early_delivery_counted_as_early(self):
        planned = [PlannedOrderExport("O1", "X", 3, 5, 10.0)]
        actuals = [ExecutionRecord("O1", "X", 10.0, 10.0, 5, 3, delay_periods=-2)]
        report, _ = DeviationAnalyzer.detect_deviations(planned, actuals)
        assert report.early_count == 1
        assert report.late_count == 0

