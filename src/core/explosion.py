"""Multi-Level Bill of Materials (BOM) Explosion Engine for ROLLINGMRP.

Implements:
- Phase 1.4: Multi-Level BOM Explosion:
  - Low-Level Coding (LLC) calculation to enforce topological processing order
  - Single-level explosion with scrap factor scaling: qty = (parent_qty * quantity_per) / (1 - scrap_factor)
  - Multi-level recursive explosion with depth tracking and memoization
  - Component commonality aggregation: aggregating demand across multiple parents
  - Phantom item handling: transparent pass-through of requirements
- Phase 1.5: Integrated Upward & Downward Pegging tracking
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from src.core.netting import NettingResult, PlannedOrder, calculate_net_requirements
from src.core.pegging import PeggingTracker


@dataclass
class BOMEdge:
    """Represents an edge in the BOM graph."""
    parent_item_id: str
    component_item_id: str
    quantity_per: float
    scrap_factor: float = 0.0
    is_phantom: bool = False


class BOMExplosionEngine:
    """BOM Explosion and Multi-Level MRP Execution Engine."""

    def __init__(
        self,
        bom_data: Union[List[Dict[str, Any]], pd.DataFrame],
        items_data: Union[List[Dict[str, Any]], pd.DataFrame],
    ):
        self.df_items = items_data if isinstance(items_data, pd.DataFrame) else pd.DataFrame(items_data)
        self.df_bom = bom_data if isinstance(bom_data, pd.DataFrame) else pd.DataFrame(bom_data)

        # Build items lookup
        self.items_by_id: Dict[str, Dict[str, Any]] = {}
        for _, row in self.df_items.iterrows():
            self.items_by_id[str(row["item_id"])] = dict(row)

        # Build BOM graph adjacency: parent -> list of BOMEdge
        self.adj: Dict[str, List[BOMEdge]] = defaultdict(list)
        self.parents_of: Dict[str, List[str]] = defaultdict(list)

        for _, row in self.df_bom.iterrows():
            p = str(row["parent_item_id"])
            c = str(row["component_item_id"])
            q_per = float(row["quantity_per"])
            scrap = float(row.get("scrap_factor", 0.0))
            # Check phantom status safely
            raw_ph = row.get("is_phantom", False)
            edge_is_ph = bool(raw_ph) if pd.notna(raw_ph) else False
            c_item = self.items_by_id.get(c, {})
            is_ph = edge_is_ph or (c_item.get("type") == "phantom")

            edge = BOMEdge(
                parent_item_id=p,
                component_item_id=c,
                quantity_per=q_per,
                scrap_factor=scrap,
                is_phantom=is_ph,
            )
            self.adj[p].append(edge)
            self.parents_of[c].append(p)

        # Memoization cache for recursive explosions
        self._memoized_explosions: Dict[Tuple[str, float], List[Tuple[str, float, int]]] = {}

        # Compute Low-Level Codes (LLC)
        self.low_level_codes = self._compute_low_level_codes()

    def _compute_low_level_codes(self) -> Dict[str, int]:
        """Compute Low-Level Code (LLC) for every item in the product structure.

        LLC is defined as the maximum path length (depth) from any root item down to the item.
        This guarantees an item is netted only after all parents driving it have completed netting.
        """
        all_items = set(self.items_by_id.keys())
        roots = [itm for itm in all_items if itm not in self.parents_of or len(self.parents_of[itm]) == 0]

        llc: Dict[str, int] = {itm: 0 for itm in all_items}

        # Topological traversal / longest path
        queue = deque([(root, 0) for root in roots])
        visited_depth: Dict[str, int] = defaultdict(int)

        while queue:
            curr, depth = queue.popleft()
            if depth > llc[curr]:
                llc[curr] = depth

            for edge in self.adj.get(curr, []):
                child = edge.component_item_id
                next_depth = depth + 1
                if next_depth > llc[child]:
                    llc[child] = next_depth
                    queue.append((child, next_depth))

        return llc

    def explode_single_level(self, parent_id: str, parent_quantity: float = 1.0) -> List[Dict[str, Any]]:
        """Perform a single-level explosion for a parent item with scrap scaling."""
        results = []
        for edge in self.adj.get(parent_id, []):
            denom = 1.0 - edge.scrap_factor
            if denom <= 0:
                raise ValueError(f"Invalid scrap factor {edge.scrap_factor} on edge {parent_id} -> {edge.component_item_id}")
            gross_needed = (parent_quantity * edge.quantity_per) / denom
            results.append({
                "parent_item_id": parent_id,
                "component_item_id": edge.component_item_id,
                "quantity_per": edge.quantity_per,
                "scrap_factor": edge.scrap_factor,
                "gross_quantity": gross_needed,
                "is_phantom": edge.is_phantom,
            })
        return results

    def explode_multi_level(
        self,
        parent_id: str,
        parent_quantity: float = 1.0,
        current_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Recursively explode BOM structure with depth tracking and memoization.

        Handles phantom items by seamlessly passing requirements directly to their children.
        """
        cache_key = (parent_id, parent_quantity)
        if cache_key in self._memoized_explosions:
            return [
                {"component_item_id": cid, "gross_quantity": qty, "depth": d}
                for cid, qty, d in self._memoized_explosions[cache_key]
            ]

        expanded: List[Dict[str, Any]] = []
        memo_list: List[Tuple[str, float, int]] = []

        direct_children = self.explode_single_level(parent_id, parent_quantity)
        for child in direct_children:
            cid = child["component_item_id"]
            c_qty = child["gross_quantity"]

            if child["is_phantom"]:
                # Phantom item: do not record phantom itself; directly explode its children
                sub_children = self.explode_multi_level(cid, c_qty, current_depth=current_depth)
                for sc in sub_children:
                    expanded.append(sc)
                    memo_list.append((sc["component_item_id"], sc["gross_quantity"], sc["depth"]))
            else:
                entry = {
                    "parent_item_id": parent_id,
                    "component_item_id": cid,
                    "gross_quantity": c_qty,
                    "depth": current_depth,
                }
                expanded.append(entry)
                memo_list.append((cid, c_qty, current_depth))

                # Recurse deeper for child's components
                sub_children = self.explode_multi_level(cid, c_qty, current_depth=current_depth + 1)
                for sc in sub_children:
                    expanded.append(sc)
                    memo_list.append((sc["component_item_id"], sc["gross_quantity"], sc["depth"]))

        self._memoized_explosions[cache_key] = memo_list
        return expanded

    def run_multi_level_mrp(
        self,
        mps_gross_requirements: Dict[str, List[float]],
        periods: List[int],
        inventory_status: Optional[Dict[str, Dict[str, float]]] = None,
        scheduled_receipts: Optional[Dict[str, List[float]]] = None,
        lot_sizing_callback: Optional[Callable[[float, str, Dict[str, Any]], float]] = None,
    ) -> Tuple[Dict[str, NettingResult], PeggingTracker]:
        """Execute complete multi-level MRP explosion and netting across all BOM levels.

        Processing Order:
        1. Groups items by Low-Level Code (LLC 0, 1, 2, ...).
        2. Aggregates gross requirements at Level 0 from MPS.
        3. Nets level by level; exploded child requirements are accumulated into lower levels.
        4. Maintains full upward and downward pegging chains.

        Returns:
            Tuple of (dict mapping item_id -> NettingResult, PeggingTracker instance).
        """
        pegging_tracker = PeggingTracker()
        results: Dict[str, NettingResult] = {}

        # Organize items by Low-Level Code
        items_by_llc: Dict[int, List[str]] = defaultdict(list)
        for itm_id, code in self.low_level_codes.items():
            items_by_llc[code].append(itm_id)

        # Dynamic table of gross requirements: item_id -> [gr_t1, gr_t2, ...]
        gross_requirements_table: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(periods))
        # Tracking pegging links for gross requirements: item_id -> period -> list of parent contributor dicts
        demand_contributors: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

        # Seed level 0 gross requirements from MPS
        for itm_id, reqs in mps_gross_requirements.items():
            for idx, p in enumerate(periods):
                qty = reqs[idx] if idx < len(reqs) else 0.0
                if qty > 0:
                    gross_requirements_table[itm_id][idx] += qty
                    demand_contributors[itm_id][p].append({
                        "parent_item_id": "MPS",
                        "parent_period": p,
                        "quantity": qty,
                        "mps_item_id": itm_id,
                        "mps_period": p,
                    })

        sorted_levels = sorted(items_by_llc.keys())

        for level in sorted_levels:
            current_items = items_by_llc[level]

            for itm_id in current_items:
                item_meta = self.items_by_id.get(itm_id, {})
                if item_meta.get("type") == "phantom":
                    # Phantoms do not have inventory netting
                    continue

                inv = (inventory_status or {}).get(itm_id, {})
                on_hand = inv.get("on_hand", 0.0)
                allocated = inv.get("allocated", 0.0)
                ss = inv.get("safety_stock", item_meta.get("safety_stock", 0.0))
                lt = int(inv.get("lead_time_days", item_meta.get("lead_time_days", 0)))
                # If lead time is in days and periods are weeks, convert or treat period LT as discrete period integer
                # By convention in benchmarks, lead_time in netting is integer periods:
                lt_periods = int(inv.get("lead_time_periods", item_meta.get("lead_time_periods", lt // 7 if lt >= 7 else lt)))
                lot_rule = item_meta.get("lot_sizing_rule", "L4L")

                sr = (scheduled_receipts or {}).get(itm_id, [0.0] * len(periods))
                gr = gross_requirements_table[itm_id]

                # Run Netting
                net_res = calculate_net_requirements(
                    item_id=itm_id,
                    periods=periods,
                    gross_requirements=gr,
                    on_hand=on_hand,
                    allocated=allocated,
                    safety_stock=ss,
                    lead_time=lt_periods,
                    scheduled_receipts=sr,
                    lot_sizing_rule=lot_rule,
                    lot_sizing_func=lot_sizing_callback,
                )
                results[itm_id] = net_res

                # Record pegging for orders of this item
                for po in net_res.planned_orders:
                    # Find driving contributors for this period
                    contributors = demand_contributors[itm_id].get(po.due_period, [])
                    if not contributors:
                        # Root finished good order pegged directly to its MPS requirement
                        pegging_tracker.record_link(
                            child_item_id=itm_id,
                            child_period=po.release_period,
                            child_quantity=po.quantity,
                            mps_item_id=itm_id,
                            mps_period=po.due_period,
                        )
                    else:
                        for contrib in contributors:
                            pegging_tracker.record_link(
                                child_item_id=itm_id,
                                child_period=po.release_period,
                                child_quantity=po.quantity,
                                parent_item_id=contrib.get("parent_item_id"),
                                parent_period=contrib.get("parent_period"),
                                mps_item_id=contrib.get("mps_item_id"),
                                mps_period=contrib.get("mps_period"),
                            )

                # Explode Planned Order Releases down to direct components
                for po in net_res.planned_orders:
                    rel_p = po.release_period
                    # Releases outside the active planning periods (e.g. past due <= 0)
                    # are commitments prior to the horizon and do not create future gross requirements
                    if rel_p not in periods:
                        continue

                    comp_p_idx = periods.index(rel_p)

                    # Direct explosion of this order's quantity
                    direct_comps = self.explode_single_level(itm_id, po.quantity)
                    for comp in direct_comps:
                        cid = comp["component_item_id"]
                        gross_qty = comp["gross_quantity"]

                        # Component Commonality Aggregation: add requirement
                        gross_requirements_table[cid][comp_p_idx] += gross_qty

                        demand_contributors[cid][rel_p].append({
                            "parent_item_id": itm_id,
                            "parent_period": rel_p,
                            "quantity": gross_qty,
                            "mps_item_id": itm_id if level == 0 else contributors[0].get("mps_item_id") if contributors else itm_id,
                            "mps_period": po.due_period if level == 0 else contributors[0].get("mps_period") if contributors else po.due_period,
                        })

        return results, pegging_tracker
