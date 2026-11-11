"""DES (Discrete Event Simulation) Link Module for ROLLINGMRP.

Implements Phase 6.2:
- Export Format: Planned orders serialized as JSON or CSV for SimPy / AnyLogic / other DES engines.
- Import Format: Execution results (actual completions, delays, scrap) ingested from DES output.
- Status Mapping: PLANNED → RELEASED → IN_PROGRESS → COMPLETED / PARTIAL / SCRAPPED
- Deviation Detection: Actual vs. planned quantity and timing deviations with fill-rate reporting.
"""

import csv
import io
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    """Lifecycle states for a manufacturing / procurement order."""
    PLANNED = "PLANNED"
    RELEASED = "RELEASED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"        # Completed with less than planned quantity
    SCRAPPED = "SCRAPPED"      # Order voided; output discarded
    CANCELLED = "CANCELLED"    # Cancelled before release


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PlannedOrderExport:
    """Planned order serialized for DES consumption.

    Attributes:
        order_id: Unique identifier (auto-generated if not provided).
        item_id: Item/SKU reference.
        release_period: Period when the order should be released to the floor.
        due_period: Period when output is expected.
        quantity: Planned quantity (units).
        status: Current lifecycle status (default PLANNED).
        bom_level: BOM depth level (0 = finished good).
        work_center_id: Primary work centre for routing (optional).
        routing_hours: Total estimated run + setup hours (optional).
        lot_sizing_rule: Lot-sizing rule that generated this order (e.g. EOQ, LFL).
        metadata: Arbitrary key-value pairs for DES-specific annotations.
    """
    order_id: str
    item_id: str
    release_period: int
    due_period: int
    quantity: float
    status: str = OrderStatus.PLANNED
    bom_level: int = 0
    work_center_id: Optional[str] = None
    routing_hours: Optional[float] = None
    lot_sizing_rule: str = "LFL"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten metadata into the dict for CSV serialization
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedOrderExport":
        """Reconstruct from a plain dict (e.g. from JSON deserialization)."""
        # Handle metadata field that may be a JSON string in CSV
        meta = data.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        return cls(
            order_id=data["order_id"],
            item_id=data["item_id"],
            release_period=int(data["release_period"]),
            due_period=int(data["due_period"]),
            quantity=float(data["quantity"]),
            status=data.get("status", OrderStatus.PLANNED),
            bom_level=int(data.get("bom_level", 0)),
            work_center_id=data.get("work_center_id"),
            routing_hours=float(data["routing_hours"]) if data.get("routing_hours") is not None else None,
            lot_sizing_rule=data.get("lot_sizing_rule", "LFL"),
            metadata=meta,
        )


@dataclass
class ExecutionRecord:
    """Actual execution result imported from a DES simulation or shop-floor system.

    Attributes:
        order_id: Reference to the originating PlannedOrderExport.
        item_id: Item/SKU reference.
        planned_quantity: Quantity that was originally planned.
        actual_quantity: Quantity actually produced / received.
        planned_period: Period the order was due.
        actual_period: Period when the order actually completed.
        scrap_quantity: Units scrapped / rejected during execution.
        status: Final lifecycle status of the order.
        delay_periods: Signed delay (actual_period - planned_period); negative = early.
        notes: Free-text execution notes from the DES or operator.
    """
    order_id: str
    item_id: str
    planned_quantity: float
    actual_quantity: float
    planned_period: int
    actual_period: int
    scrap_quantity: float = 0.0
    status: str = OrderStatus.COMPLETED
    delay_periods: int = 0  # Computed on import if not provided
    notes: str = ""

    def __post_init__(self) -> None:
        # Auto-compute delay_periods if default (0) and periods differ
        if self.delay_periods == 0 and self.actual_period != self.planned_period:
            self.delay_periods = self.actual_period - self.planned_period

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        """Reconstruct from a plain dict."""
        actual_period = int(data["actual_period"])
        planned_period = int(data["planned_period"])
        delay = int(data.get("delay_periods", actual_period - planned_period))
        return cls(
            order_id=data["order_id"],
            item_id=data["item_id"],
            planned_quantity=float(data["planned_quantity"]),
            actual_quantity=float(data["actual_quantity"]),
            planned_period=planned_period,
            actual_period=actual_period,
            scrap_quantity=float(data.get("scrap_quantity", 0.0)),
            status=data.get("status", OrderStatus.COMPLETED),
            delay_periods=delay,
            notes=data.get("notes", ""),
        )


@dataclass
class DeviationReport:
    """Aggregated deviation analysis for a set of planned vs. actual order pairs.

    Attributes:
        total_orders: Total planned orders analysed.
        on_time_count: Orders completed in the planned period (delay == 0).
        late_count: Orders completed after the planned period (delay > 0).
        early_count: Orders completed before the planned period (delay < 0).
        partial_count: Orders where actual_quantity < planned_quantity.
        scrapped_count: Orders with status SCRAPPED.
        cancelled_count: Orders with status CANCELLED (no execution record).
        fill_rate: Σ(actual_quantity) / Σ(planned_quantity) — capped at 1.0.
        quantity_deviation_pct: Mean absolute % deviation in quantity across orders.
        timing_deviation_mean: Mean absolute delay in periods across completed orders.
        timing_deviation_max: Maximum absolute delay observed.
        item_level_summary: Per-item fill-rate and delay summary.
    """
    total_orders: int
    on_time_count: int
    late_count: int
    early_count: int
    partial_count: int
    scrapped_count: int
    cancelled_count: int
    fill_rate: float
    quantity_deviation_pct: float
    timing_deviation_mean: float
    timing_deviation_max: int
    item_level_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DES Exporter
# ---------------------------------------------------------------------------

class DESExporter:
    """Serializes planned orders into DES-compatible JSON or CSV format."""

    @staticmethod
    def export_planned_orders(
        orders: List[PlannedOrderExport],
        fmt: str = "json",
        filepath: Optional[str] = None,
    ) -> str:
        """Export a list of PlannedOrderExport objects to JSON or CSV.

        Args:
            orders: List of planned orders to export.
            fmt: Output format — ``"json"`` (default) or ``"csv"``.
            filepath: Optional file path to write the export. If None, returns
                      the serialized string without writing to disk.

        Returns:
            Serialized string (JSON or CSV).

        Raises:
            ValueError: If ``fmt`` is not ``"json"`` or ``"csv"``.
        """
        fmt = fmt.lower()
        if fmt == "json":
            payload = {
                "format": "ROLLINGMRP_DES_EXPORT_v1",
                "count": len(orders),
                "orders": [o.to_dict() for o in orders],
            }
            content = json.dumps(payload, indent=2, default=str)

        elif fmt == "csv":
            if not orders:
                content = ""
            else:
                fields = list(asdict(orders[0]).keys())
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=fields)
                writer.writeheader()
                for o in orders:
                    row = o.to_dict()
                    # Serialize nested dict fields to JSON strings for CSV
                    if isinstance(row.get("metadata"), dict):
                        row["metadata"] = json.dumps(row["metadata"])
                    writer.writerow(row)
                content = buf.getvalue()
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Use 'json' or 'csv'.")

        if filepath:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8", newline="") as fh:
                fh.write(content)

        return content

    @staticmethod
    def transition_status(
        orders: List[PlannedOrderExport],
        target_status: str,
    ) -> List[PlannedOrderExport]:
        """Apply a status transition to all orders, respecting the lifecycle flow.

        Valid transitions (sequential):
            PLANNED → RELEASED → IN_PROGRESS → COMPLETED / PARTIAL / SCRAPPED

        Args:
            orders: List of planned orders to update.
            target_status: Target ``OrderStatus`` value.

        Returns:
            New list of orders with updated status (originals unchanged).
        """
        _valid_next: Dict[str, List[str]] = {
            OrderStatus.PLANNED: [OrderStatus.RELEASED, OrderStatus.CANCELLED],
            OrderStatus.RELEASED: [OrderStatus.IN_PROGRESS, OrderStatus.CANCELLED],
            OrderStatus.IN_PROGRESS: [
                OrderStatus.COMPLETED,
                OrderStatus.PARTIAL,
                OrderStatus.SCRAPPED,
            ],
            OrderStatus.COMPLETED: [],
            OrderStatus.PARTIAL: [],
            OrderStatus.SCRAPPED: [],
            OrderStatus.CANCELLED: [],
        }
        updated: List[PlannedOrderExport] = []
        for o in orders:
            current = o.status
            if target_status in _valid_next.get(current, []):
                import dataclasses
                updated.append(dataclasses.replace(o, status=target_status))
            else:
                # Keep original if transition is not valid
                updated.append(o)
        return updated


# ---------------------------------------------------------------------------
# DES Importer
# ---------------------------------------------------------------------------

class DESImporter:
    """Ingests execution results from a DES simulation or shop-floor MES."""

    @staticmethod
    def import_execution_results(
        source: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[ExecutionRecord]:
        """Parse execution results from JSON filepath, JSON string, dict, or list of dicts.

        Args:
            source: One of:
                - ``str`` path to a JSON file containing execution records.
                - ``str`` raw JSON string with a list or ``{"records": [...]}`` envelope.
                - ``dict`` with a ``"records"`` key.
                - ``list`` of raw record dicts.

        Returns:
            List of :class:`ExecutionRecord` objects.

        Raises:
            ValueError: If the source format cannot be parsed.
            FileNotFoundError: If a filepath is provided but does not exist.
        """
        raw_records: List[Dict[str, Any]] = []

        if isinstance(source, list):
            raw_records = source

        elif isinstance(source, dict):
            raw_records = source.get("records", [source])  # single record or envelope

        elif isinstance(source, str):
            # Could be filepath or raw JSON string
            if os.path.exists(source):
                with open(source, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    raw_records = data
                elif isinstance(data, dict):
                    raw_records = data.get("records", [])
                else:
                    raise ValueError(f"Unexpected top-level JSON type in file '{source}'.")
            else:
                # Attempt to parse as raw JSON string
                try:
                    data = json.loads(source)
                    if isinstance(data, list):
                        raw_records = data
                    elif isinstance(data, dict):
                        raw_records = data.get("records", [])
                    else:
                        raise ValueError("JSON root must be a list or dict.")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Cannot parse source as filepath or JSON string: {exc}") from exc
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        return [ExecutionRecord.from_dict(r) for r in raw_records]

    @staticmethod
    def import_from_csv(filepath: str) -> List[ExecutionRecord]:
        """Parse execution records from a CSV file.

        Args:
            filepath: Path to CSV file with column headers matching ExecutionRecord fields.

        Returns:
            List of :class:`ExecutionRecord` objects.
        """
        records: List[ExecutionRecord] = []
        with open(filepath, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(ExecutionRecord.from_dict(dict(row)))
        return records


# ---------------------------------------------------------------------------
# Deviation Analyzer
# ---------------------------------------------------------------------------

class DeviationAnalyzer:
    """Compares planned orders against execution results and computes deviation metrics."""

    @staticmethod
    def detect_deviations(
        planned: List[PlannedOrderExport],
        actuals: List[ExecutionRecord],
    ) -> Tuple[DeviationReport, List[Dict[str, Any]]]:
        """Align planned orders with execution records and compute deviation metrics.

        Alignment is performed on ``order_id``. Planned orders with no matching
        execution record are treated as cancelled/missing.

        Args:
            planned: List of exported planned orders (pre-execution).
            actuals: List of execution records returned by the DES.

        Returns:
            Tuple of:
            - :class:`DeviationReport` with aggregate metrics.
            - List of per-order deviation dicts with fields:
              ``order_id``, ``item_id``, ``quantity_deviation``,
              ``quantity_deviation_pct``, ``delay_periods``, ``fill_pct``,
              ``status``.
        """
        # Build lookup maps
        planned_map: Dict[str, PlannedOrderExport] = {o.order_id: o for o in planned}
        actual_map: Dict[str, ExecutionRecord] = {r.order_id: r for r in actuals}

        total_planned_qty = sum(o.quantity for o in planned)
        total_actual_qty = 0.0
        sum_qty_dev_pct = 0.0
        sum_abs_delay = 0
        max_delay = 0

        on_time = late = early = partial = scrapped = cancelled = 0
        per_order: List[Dict[str, Any]] = []

        # --- Orders with execution records ---
        for order_id, rec in actual_map.items():
            po = planned_map.get(order_id)
            planned_qty = po.quantity if po else rec.planned_quantity
            actual_qty = rec.actual_quantity
            scrap_qty = rec.scrap_quantity

            total_actual_qty += actual_qty
            qty_dev = actual_qty - planned_qty
            qty_dev_pct = abs(qty_dev) / planned_qty * 100.0 if planned_qty > 0 else 0.0
            sum_qty_dev_pct += qty_dev_pct

            delay = rec.delay_periods
            abs_delay = abs(delay)
            sum_abs_delay += abs_delay
            if abs_delay > max_delay:
                max_delay = abs_delay

            if delay == 0:
                on_time += 1
            elif delay > 0:
                late += 1
            else:
                early += 1

            if rec.status == OrderStatus.SCRAPPED:
                scrapped += 1
            elif actual_qty < planned_qty - 1e-6:
                partial += 1

            fill_pct = min(1.0, actual_qty / planned_qty) * 100.0 if planned_qty > 0 else 100.0

            per_order.append({
                "order_id": order_id,
                "item_id": rec.item_id,
                "quantity_deviation": round(qty_dev, 4),
                "quantity_deviation_pct": round(qty_dev_pct, 2),
                "delay_periods": delay,
                "fill_pct": round(fill_pct, 2),
                "status": rec.status,
            })

        # --- Planned orders with NO execution record → cancelled / missing ---
        for order_id, po in planned_map.items():
            if order_id not in actual_map:
                cancelled += 1
                per_order.append({
                    "order_id": order_id,
                    "item_id": po.item_id,
                    "quantity_deviation": -po.quantity,
                    "quantity_deviation_pct": 100.0,
                    "delay_periods": None,
                    "fill_pct": 0.0,
                    "status": OrderStatus.CANCELLED,
                })

        n_executed = len(actual_map)
        fill_rate = (total_actual_qty / total_planned_qty) if total_planned_qty > 0 else 1.0
        fill_rate = min(1.0, fill_rate)

        mean_qty_dev_pct = sum_qty_dev_pct / n_executed if n_executed > 0 else 0.0
        mean_delay = sum_abs_delay / n_executed if n_executed > 0 else 0.0

        # --- Per-item summary ---
        item_summary: Dict[str, Dict[str, float]] = {}
        for d in per_order:
            iid = d["item_id"]
            if iid not in item_summary:
                item_summary[iid] = {
                    "total_orders": 0,
                    "total_planned": 0.0,
                    "total_actual": 0.0,
                    "total_delay": 0,
                    "fill_rate": 0.0,
                }
            item_summary[iid]["total_orders"] += 1
            # Find planned qty
            matching_planned = planned_map.get(d["order_id"])
            if matching_planned:
                item_summary[iid]["total_planned"] += matching_planned.quantity
            matching_actual = actual_map.get(d["order_id"])
            if matching_actual:
                item_summary[iid]["total_actual"] += matching_actual.actual_quantity
                item_summary[iid]["total_delay"] += abs(matching_actual.delay_periods)

        for iid, s in item_summary.items():
            tp = s["total_planned"]
            ta = s["total_actual"]
            s["fill_rate"] = round(min(1.0, ta / tp) if tp > 0 else 1.0, 4)

        report = DeviationReport(
            total_orders=len(planned),
            on_time_count=on_time,
            late_count=late,
            early_count=early,
            partial_count=partial,
            scrapped_count=scrapped,
            cancelled_count=cancelled,
            fill_rate=round(fill_rate, 4),
            quantity_deviation_pct=round(mean_qty_dev_pct, 2),
            timing_deviation_mean=round(mean_delay, 2),
            timing_deviation_max=max_delay,
            item_level_summary=item_summary,
        )

        return report, per_order

    @staticmethod
    def update_order_statuses(
        planned: List[PlannedOrderExport],
        actuals: List[ExecutionRecord],
    ) -> List[PlannedOrderExport]:
        """Reflect execution outcomes back onto planned order statuses.

        Args:
            planned: Original planned orders.
            actuals: Execution records with final statuses.

        Returns:
            New list of planned orders with statuses updated from actuals.
        """
        import dataclasses
        actual_map = {r.order_id: r for r in actuals}
        updated: List[PlannedOrderExport] = []
        for po in planned:
            rec = actual_map.get(po.order_id)
            if rec:
                updated.append(dataclasses.replace(po, status=rec.status))
            else:
                updated.append(po)
        return updated

