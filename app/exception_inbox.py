"""Exception Inbox & Planner Action Workbench for ROLLINGMRP.

Implements Phase 7.1:
- Message display: sortable, filterable table/records.
- Message types: EXPEDITE, DEFER, CANCEL, NEW_ORDER, SAFETY_STOCK_BREACH, CAPACITY_OVERLOAD.
- Priority indicators: Color-coded severity tiers calculated by dollar impact:
    - CRITICAL: >= $10,000 (Red / #dc3545)
    - HIGH: $2,500 - $9,999 (Orange / #fd7e14)
    - MEDIUM: $500 - $2,499 (Yellow / #ffc107)
    - LOW: < $500 (Blue / #0d6efd)
- Action management: APPROVE, REJECT, DEFER with planner comment.
- Batch operations: Multi-select bulk approve, reject, or defer.
- Audit trail: In-memory & JSON/CSV persistent logging with timestamps and user identification.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExceptionType(str, Enum):
    """Classification of planning exceptions."""
    EXPEDITE = "EXPEDITE"                    # Planned order release is past due or required earlier
    DEFER = "DEFER"                          # Scheduled receipt / order arrived before needed
    CANCEL = "CANCEL"                        # Order no longer required (excess supply)
    NEW_ORDER = "NEW_ORDER"                  # Unplanned demand trigger requiring order creation
    SAFETY_STOCK_BREACH = "SAFETY_STOCK"     # Projected balance drops below safety stock threshold
    CAPACITY_OVERLOAD = "CAPACITY_OVERLOAD"  # Work center capacity exceeds threshold limit


class ExceptionPriority(str, Enum):
    """Dollar-impact and operational severity tiers."""
    CRITICAL = "CRITICAL"  # >= $10,000 impact or severe past-due
    HIGH = "HIGH"          # $2,500 - $9,999
    MEDIUM = "MEDIUM"      # $500 - $2,499
    LOW = "LOW"            # < $500


class ActionStatus(str, Enum):
    """Lifecycle status of an exception action."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ExceptionMessage:
    """Standardized planner action message generated from MRP/Capacity engines.

    Attributes:
        exception_id: Unique identifier (e.g. 'EXC-0001').
        exception_type: Type of planning exception.
        item_id: Associated SKU / material.
        period: Due period or relevant period of the incident.
        quantity: Affected units.
        dollar_impact: Estimated total financial exposure ($).
        priority: Priority tier (CRITICAL, HIGH, MEDIUM, LOW).
        description: Human-readable explanation.
        recommended_action: Action suggested by MRP / RCCP logic.
        status: Action status (PENDING, APPROVED, REJECTED, DEFERRED).
        created_at: ISO-8601 generation timestamp.
        order_id: Associated planned or scheduled order reference (if any).
        work_center_id: Associated work center reference (for capacity overloads).
        metadata: Arbitrary additional attributes.
    """
    exception_id: str
    exception_type: ExceptionType
    item_id: str
    period: int
    quantity: float
    dollar_impact: float
    priority: ExceptionPriority
    description: str
    recommended_action: str
    status: ActionStatus = ActionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    order_id: Optional[str] = None
    work_center_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def priority_color(self) -> str:
        """Hex color code for UI rendering."""
        color_map = {
            ExceptionPriority.CRITICAL: "#dc3545",  # Red
            ExceptionPriority.HIGH: "#fd7e14",      # Orange
            ExceptionPriority.MEDIUM: "#ffc107",    # Yellow
            ExceptionPriority.LOW: "#0d6efd",       # Blue
        }
        return color_map.get(self.priority, "#6c757d")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exception_id": self.exception_id,
            "exception_type": self.exception_type.value,
            "item_id": self.item_id,
            "period": self.period,
            "quantity": self.quantity,
            "dollar_impact": self.dollar_impact,
            "priority": self.priority.value,
            "priority_color": self.priority_color,
            "description": self.description,
            "recommended_action": self.recommended_action,
            "status": self.status.value,
            "created_at": self.created_at,
            "order_id": self.order_id,
            "work_center_id": self.work_center_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExceptionMessage":
        return cls(
            exception_id=data["exception_id"],
            exception_type=ExceptionType(data["exception_type"]),
            item_id=data["item_id"],
            period=int(data["period"]),
            quantity=float(data["quantity"]),
            dollar_impact=float(data["dollar_impact"]),
            priority=ExceptionPriority(data["priority"]),
            description=data["description"],
            recommended_action=data["recommended_action"],
            status=ActionStatus(data.get("status", ActionStatus.PENDING)),
            created_at=data.get("created_at", datetime.datetime.now().isoformat()),
            order_id=data.get("order_id"),
            work_center_id=data.get("work_center_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AuditActionRecord:
    """Immutable audit trail record logging planner actions."""
    audit_id: str
    exception_id: str
    action: ActionStatus
    user: str
    timestamp: str
    comment: str = ""
    previous_status: ActionStatus = ActionStatus.PENDING
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "exception_id": self.exception_id,
            "action": self.action.value,
            "user": self.user,
            "timestamp": self.timestamp,
            "comment": self.comment,
            "previous_status": self.previous_status.value,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Exception Inbox Manager
# ---------------------------------------------------------------------------

class ExceptionInboxManager:
    """Manages the lifecycle, sorting, filtering, and action auditing of exceptions."""

    def __init__(
        self,
        audit_log_path: Optional[str] = None,
        critical_threshold_dollars: float = 10000.0,
        high_threshold_dollars: float = 2500.0,
        medium_threshold_dollars: float = 500.0,
    ):
        self._exceptions: Dict[str, ExceptionMessage] = {}
        self._audit_trail: List[AuditActionRecord] = []
        self.audit_log_path = audit_log_path
        self.critical_threshold_dollars = critical_threshold_dollars
        self.high_threshold_dollars = high_threshold_dollars
        self.medium_threshold_dollars = medium_threshold_dollars

    # ------------------------------------------------------------------
    # Message Ingestion & Factory Methods
    # ------------------------------------------------------------------

    def add_exception(self, exception: ExceptionMessage) -> None:
        """Register a single exception in the inbox."""
        self._exceptions[exception.exception_id] = exception

    def create_and_add_exception(
        self,
        exception_type: Union[ExceptionType, str],
        item_id: str,
        period: int,
        quantity: float,
        unit_cost: float = 0.0,
        description: str = "",
        recommended_action: str = "",
        order_id: Optional[str] = None,
        work_center_id: Optional[str] = None,
        custom_dollar_impact: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExceptionMessage:
        """Create a new ExceptionMessage with auto-calculated dollar impact and priority tier."""
        if isinstance(exception_type, str):
            exception_type = ExceptionType(exception_type.upper())

        dollar_impact = custom_dollar_impact if custom_dollar_impact is not None else (quantity * unit_cost)
        priority = self.calculate_priority(dollar_impact)
        
        idx = len(self._exceptions) + 1
        exception_id = f"EXC-{idx:04d}"

        msg = ExceptionMessage(
            exception_id=exception_id,
            exception_type=exception_type,
            item_id=item_id,
            period=period,
            quantity=quantity,
            dollar_impact=round(dollar_impact, 2),
            priority=priority,
            description=description,
            recommended_action=recommended_action,
            order_id=order_id,
            work_center_id=work_center_id,
            metadata=metadata or {},
        )
        self.add_exception(msg)
        return msg

    def calculate_priority(self, dollar_impact: float) -> ExceptionPriority:
        """Assign priority category based on calculated dollar impact."""
        impact = abs(dollar_impact)
        if impact >= self.critical_threshold_dollars:
            return ExceptionPriority.CRITICAL
        elif impact >= self.high_threshold_dollars:
            return ExceptionPriority.HIGH
        elif impact >= self.medium_threshold_dollars:
            return ExceptionPriority.MEDIUM
        else:
            return ExceptionPriority.LOW

    def ingest_mrp_netting_exceptions(
        self,
        netting_exceptions: List[Dict[str, Any]],
        unit_costs: Optional[Dict[str, float]] = None,
    ) -> List[ExceptionMessage]:
        """Ingest raw exceptions produced by NettingResult / calculate_net_requirements."""
        costs = unit_costs or {}
        created = []
        for exc in netting_exceptions:
            raw_type = exc.get("type", "expedite").lower()
            if "expedite" in raw_type:
                etype = ExceptionType.EXPEDITE
                action = f"Expedite supplier delivery or accelerate production release for period {exc.get('period_due')}."
            elif "defer" in raw_type:
                etype = ExceptionType.DEFER
                action = "Reschedule inbound purchase order to later date."
            elif "cancel" in raw_type:
                etype = ExceptionType.CANCEL
                action = "Cancel open order due to demand cancellation."
            else:
                etype = ExceptionType.EXPEDITE
                action = exc.get("description", "Expedite order.")

            item_id = exc.get("item_id", "UNKNOWN")
            qty = float(exc.get("quantity", 0.0))
            cost = costs.get(item_id, 10.0)

            msg = self.create_and_add_exception(
                exception_type=etype,
                item_id=item_id,
                period=int(exc.get("period_due", exc.get("period", 1))),
                quantity=qty,
                unit_cost=cost,
                description=exc.get("description", f"Netting exception for {item_id}"),
                recommended_action=action,
            )
            created.append(msg)
        return created

    def ingest_capacity_exceptions(
        self,
        capacity_exceptions: List[Any],
        hourly_rate: float = 75.0,
    ) -> List[ExceptionMessage]:
        """Ingest CapacityExceptionMessage instances from CapacityAwareMRPPipeline."""
        created = []
        for c_exc in capacity_exceptions:
            # Handle both dataclass and dict
            overload = getattr(c_exc, "overload_hours", 0.0) if hasattr(c_exc, "overload_hours") else c_exc.get("overload_hours", 0.0)
            wc_id = getattr(c_exc, "work_center_id", "") if hasattr(c_exc, "work_center_id") else c_exc.get("work_center_id", "")
            period = getattr(c_exc, "period", 1) if hasattr(c_exc, "period") else c_exc.get("period", 1)
            suggested = getattr(c_exc, "suggested_action", "") if hasattr(c_exc, "suggested_action") else c_exc.get("suggested_action", "")
            items = getattr(c_exc, "impacted_items", []) if hasattr(c_exc, "impacted_items") else c_exc.get("impacted_items", [])
            primary_item = items[0] if items else "MULTIPLE"

            impact = overload * hourly_rate

            msg = self.create_and_add_exception(
                exception_type=ExceptionType.CAPACITY_OVERLOAD,
                item_id=primary_item,
                period=period,
                quantity=overload,
                custom_dollar_impact=impact,
                work_center_id=wc_id,
                description=f"Capacity overload of {overload:.1f} hrs on {wc_id} in period {period}.",
                recommended_action=suggested or "Pre-build in earlier slack period or authorize overtime.",
                metadata={"impacted_items": items, "overload_hours": overload},
            )
            created.append(msg)
        return created

    # ------------------------------------------------------------------
    # Filtering, Sorting, and Presentation
    # ------------------------------------------------------------------

    def get_exception(self, exception_id: str) -> Optional[ExceptionMessage]:
        """Fetch exception by identifier."""
        return self._exceptions.get(exception_id)

    def list_exceptions(
        self,
        status: Optional[Union[ActionStatus, str]] = None,
        exception_type: Optional[Union[ExceptionType, str]] = None,
        priority: Optional[Union[ExceptionPriority, str]] = None,
        item_id: Optional[str] = None,
        work_center_id: Optional[str] = None,
        min_dollar_impact: Optional[float] = None,
        sort_by: str = "dollar_impact",
        ascending: bool = False,
    ) -> List[ExceptionMessage]:
        """Filter and sort inbox exceptions.

        Args:
            status: Filter by ActionStatus (PENDING, APPROVED, etc.).
            exception_type: Filter by ExceptionType.
            priority: Filter by Priority.
            item_id: Filter by SKU.
            work_center_id: Filter by work center.
            min_dollar_impact: Minimum dollar impact cutoff.
            sort_by: Field name to sort on ('dollar_impact', 'period', 'quantity', 'priority').
            ascending: Sort direction.
        """
        results = list(self._exceptions.values())

        if status:
            val = status.value if isinstance(status, ActionStatus) else str(status).upper()
            results = [e for e in results if e.status.value == val]

        if exception_type:
            val = exception_type.value if isinstance(exception_type, ExceptionType) else str(exception_type).upper()
            results = [e for e in results if e.exception_type.value == val]

        if priority:
            val = priority.value if isinstance(priority, ExceptionPriority) else str(priority).upper()
            results = [e for e in results if e.priority.value == val]

        if item_id:
            results = [e for e in results if e.item_id == item_id]

        if work_center_id:
            results = [e for e in results if e.work_center_id == work_center_id]

        if min_dollar_impact is not None:
            results = [e for e in results if e.dollar_impact >= min_dollar_impact]

        # Sorting
        if sort_by == "dollar_impact":
            results.sort(key=lambda x: x.dollar_impact, reverse=not ascending)
        elif sort_by == "priority":
            order = {ExceptionPriority.CRITICAL: 4, ExceptionPriority.HIGH: 3, ExceptionPriority.MEDIUM: 2, ExceptionPriority.LOW: 1}
            results.sort(key=lambda x: order.get(x.priority, 0), reverse=not ascending)
        elif sort_by == "period":
            results.sort(key=lambda x: x.period, reverse=not ascending)
        elif sort_by == "quantity":
            results.sort(key=lambda x: x.quantity, reverse=not ascending)
        elif sort_by == "created_at":
            results.sort(key=lambda x: x.created_at, reverse=not ascending)

        return results

    def to_dataframe(
        self,
        status: Optional[Union[ActionStatus, str]] = None,
        sort_by: str = "dollar_impact",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Return filtered exceptions as a pandas DataFrame."""
        msgs = self.list_exceptions(status=status, sort_by=sort_by, ascending=ascending)
        if not msgs:
            return pd.DataFrame(columns=[
                "exception_id", "priority", "exception_type", "item_id", "period",
                "quantity", "dollar_impact", "status", "description", "recommended_action",
            ])
        data = [m.to_dict() for m in msgs]
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Planner Actions & Batch Operations
    # ------------------------------------------------------------------

    def take_action(
        self,
        exception_id: str,
        action: Union[ActionStatus, str],
        user: str = "planner",
        comment: str = "",
    ) -> Optional[ExceptionMessage]:
        """Apply planner action (APPROVE, REJECT, DEFER) to an exception and record in audit log."""
        msg = self._exceptions.get(exception_id)
        if not msg:
            return None

        if isinstance(action, str):
            action = ActionStatus(action.upper())

        prev_status = msg.status
        msg.status = action

        # Create audit record
        audit_id = f"AUD-{len(self._audit_trail) + 1:05d}"
        audit_record = AuditActionRecord(
            audit_id=audit_id,
            exception_id=exception_id,
            action=action,
            user=user,
            timestamp=datetime.datetime.now().isoformat(),
            comment=comment,
            previous_status=prev_status,
            details={"item_id": msg.item_id, "dollar_impact": msg.dollar_impact},
        )
        self._audit_trail.append(audit_record)
        self._persist_audit_record(audit_record)

        return msg

    def approve(self, exception_id: str, user: str = "planner", comment: str = "") -> Optional[ExceptionMessage]:
        """Approve suggested exception resolution."""
        return self.take_action(exception_id, ActionStatus.APPROVED, user=user, comment=comment)

    def reject(self, exception_id: str, user: str = "planner", comment: str = "") -> Optional[ExceptionMessage]:
        """Reject suggested exception resolution."""
        return self.take_action(exception_id, ActionStatus.REJECTED, user=user, comment=comment)

    def defer(self, exception_id: str, user: str = "planner", comment: str = "") -> Optional[ExceptionMessage]:
        """Defer action on exception for later re-evaluation."""
        return self.take_action(exception_id, ActionStatus.DEFERRED, user=user, comment=comment)

    def batch_action(
        self,
        exception_ids: List[str],
        action: Union[ActionStatus, str],
        user: str = "planner",
        comment: str = "",
    ) -> List[ExceptionMessage]:
        """Bulk apply an action to multiple exceptions."""
        updated = []
        for exc_id in exception_ids:
            res = self.take_action(exc_id, action, user=user, comment=comment)
            if res:
                updated.append(res)
        return updated

    # ------------------------------------------------------------------
    # Audit Trail & Persistence
    # ------------------------------------------------------------------

    @property
    def audit_trail(self) -> List[AuditActionRecord]:
        """Return the complete audit trail."""
        return list(self._audit_trail)

    def get_audit_trail_dataframe(self) -> pd.DataFrame:
        """Return audit trail as a DataFrame."""
        if not self._audit_trail:
            return pd.DataFrame(columns=["audit_id", "exception_id", "action", "user", "timestamp", "comment"])
        return pd.DataFrame([r.to_dict() for r in self._audit_trail])

    def _persist_audit_record(self, record: AuditActionRecord) -> None:
        """Append record to persistent JSON/CSV audit log if path configured."""
        if not self.audit_log_path:
            return
        
        path = Path(self.audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if str(path).endswith(".csv"):
            df = pd.DataFrame([record.to_dict()])
            header = not path.exists()
            df.to_csv(path, mode="a", header=header, index=False)
        else:
            # Default to JSON lines
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # Summary Metrics
    # ------------------------------------------------------------------

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculate executive overview metrics for the dashboard inbox widget."""
        all_msgs = list(self._exceptions.values())
        total = len(all_msgs)
        pending = sum(1 for m in all_msgs if m.status == ActionStatus.PENDING)
        approved = sum(1 for m in all_msgs if m.status == ActionStatus.APPROVED)
        rejected = sum(1 for m in all_msgs if m.status == ActionStatus.REJECTED)
        deferred = sum(1 for m in all_msgs if m.status == ActionStatus.DEFERRED)

        critical_count = sum(1 for m in all_msgs if m.priority == ExceptionPriority.CRITICAL and m.status == ActionStatus.PENDING)
        total_pending_exposure = sum(m.dollar_impact for m in all_msgs if m.status == ActionStatus.PENDING)

        by_type: Dict[str, int] = {}
        for m in all_msgs:
            tname = m.exception_type.value
            by_type[tname] = by_type.get(tname, 0) + 1

        return {
            "total_exceptions": total,
            "pending_count": pending,
            "approved_count": approved,
            "rejected_count": rejected,
            "deferred_count": deferred,
            "critical_pending_count": critical_count,
            "total_pending_dollar_impact": round(total_pending_exposure, 2),
            "exceptions_by_type": by_type,
        }

