"""Pegging and Demand Traceability Module for ROLLINGMRP.

Provides:
- Bidirectional traceability:
  - Upward pegging: Component Planned Order -> Parent Planned Order -> MPS Demand Line
  - Downward pegging: MPS Demand Line -> Dependent Orders across all BOM tiers
- Pegging chain serialization for PLANNED_ORDER table storage (JSON)
- Tabular pegging report generation for production and procurement planners
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class PeggingLink:
    """Represents a single pegging link between an order requirement and its demand source."""
    child_item_id: str
    child_period: int
    child_quantity: float
    parent_item_id: Optional[str] = None
    parent_period: Optional[int] = None
    parent_order_id: Optional[str] = None
    mps_item_id: Optional[str] = None
    mps_period: Optional[int] = None
    chain: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PeggingTracker:
    """Manages full upward and downward demand traceability across MRP runs."""

    def __init__(self):
        self.links: List[PeggingLink] = []

    def clear(self) -> None:
        """Reset all tracked links."""
        self.links.clear()

    def record_link(
        self,
        child_item_id: str,
        child_period: int,
        child_quantity: float,
        parent_item_id: Optional[str] = None,
        parent_period: Optional[int] = None,
        parent_order_id: Optional[str] = None,
        mps_item_id: Optional[str] = None,
        mps_period: Optional[int] = None,
        parent_chain: Optional[List[Dict[str, Any]]] = None,
    ) -> PeggingLink:
        """Record a link from a dependent requirement up to its parent and root MPS demand."""
        chain = list(parent_chain) if parent_chain else []
        node_entry = {
            "item_id": child_item_id,
            "period": child_period,
            "quantity": child_quantity,
        }
        if parent_item_id:
            node_entry["parent_item_id"] = parent_item_id
            node_entry["parent_period"] = parent_period

        chain.append(node_entry)

        # Infer MPS root if not explicitly supplied but present in parent chain
        if not mps_item_id and parent_chain and len(parent_chain) > 0:
            root_node = parent_chain[0]
            mps_item_id = root_node.get("mps_item_id") or root_node.get("item_id")
            mps_period = root_node.get("mps_period") or root_node.get("period")

        link = PeggingLink(
            child_item_id=child_item_id,
            child_period=child_period,
            child_quantity=child_quantity,
            parent_item_id=parent_item_id,
            parent_period=parent_period,
            parent_order_id=parent_order_id,
            mps_item_id=mps_item_id,
            mps_period=mps_period,
            chain=chain,
        )
        self.links.append(link)
        return link

    def trace_up(self, item_id: str, period: Optional[int] = None) -> List[Dict[str, Any]]:
        """Trace upward from an item (and optional period) to find parent demands and driving MPS."""
        results = []
        for lk in self.links:
            if lk.child_item_id == item_id:
                if period is None or lk.child_period == period:
                    results.append(lk.to_dict())
        return results

    def trace_down(self, mps_item_id: str, mps_period: Optional[int] = None) -> List[Dict[str, Any]]:
        """Trace downward from an MPS demand line to find all dependent component orders."""
        results = []
        for lk in self.links:
            if lk.mps_item_id == mps_item_id:
                if mps_period is None or lk.mps_period == mps_period:
                    results.append(lk.to_dict())
        return results

    def generate_pegging_report(self) -> pd.DataFrame:
        """Generate a tabular pegging report suitable for planner review."""
        if not self.links:
            return pd.DataFrame(columns=[
                "child_item_id", "child_period", "child_quantity",
                "parent_item_id", "parent_period", "parent_order_id",
                "mps_item_id", "mps_period"
            ])

        records = [
            {
                "child_item_id": l.child_item_id,
                "child_period": l.child_period,
                "child_quantity": l.child_quantity,
                "parent_item_id": l.parent_item_id,
                "parent_period": l.parent_period,
                "parent_order_id": l.parent_order_id,
                "mps_item_id": l.mps_item_id,
                "mps_period": l.mps_period,
            }
            for l in self.links
        ]
        return pd.DataFrame(records)

