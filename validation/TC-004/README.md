# TC-004: Rough Cut Capacity Planning (RCCP) & Leveling Heuristic

## 1. Overview & Objective
Demonstrate the bridge between standard infinite-capacity MRP and finite-capacity MRP-II:
1. **Unconstrained RCCP**: Compute work center load profiles from planned orders and detect bottleneck overload periods ($Utilization > 100\%$).
2. **Capacity Leveling**: Apply backward-shifting leveling heuristics / optimization to pre-build inventory in earlier periods with surplus capacity, eliminating bottleneck overloads while minimizing inventory holding costs.

## 2. Work Center Master & Routing Parameters
### 2.1 Work Centers
- **WC-A (Machining)**: Available capacity = 40.0 hours / period.
- **WC-B (Assembly & Testing - Critical Bottleneck)**: Available capacity = 30.0 hours / period.

### 2.2 Routings
| Item | Work Center | Setup Time (hrs) | Run Time per Unit (hrs/unit) |
|------|-------------|------------------|------------------------------|
| **FG-X** | WC-A | 4.0 | 0.50 |
| **FG-X** | WC-B | 2.0 | 0.40 |
| **FG-Y** | WC-A | 5.0 | 0.60 |
| **FG-Y** | WC-B | 3.0 | 0.50 |

Total Work Center Load Formula:
$$\text{Load}(WC, t) = \sum_{i \in \text{Items}} \left( \text{Setup}_{i, WC} + \text{Run}_{i, WC} \times Q_{i, t} \right)$$

---

## 3. Unconstrained MRP Plan & Overload Analysis

### Production Demand by Period
- Period 1: FG-X = 20, FG-Y = 15
- Period 2: FG-X = 30, FG-Y = 25
- Period 3: FG-X = 40, FG-Y = 35  *(Peak demand shock)*
- Period 4: FG-X = 10, FG-Y = 10

### Unconstrained Load Profile
| Period | WC-A Load (hrs) | WC-A Util (%) | WC-B Load (hrs) | WC-B Util (%) | Status |
|--------|-----------------|---------------|-----------------|---------------|--------|
| **1** | 28.0 | 70.0% | 20.5 | 68.3% | Feasible (Slack: 12.0h / 9.5h) |
| **2** | 39.0 | 97.5% | 29.5 | 98.3% | Near capacity (Slack: 1.0h / 0.5h) |
| **3** | **50.0** | **125.0%** | **38.5** | **128.3%** | **CRITICAL OVERLOAD (+10.0h / +8.5h)** |
| **4** | 20.0 | 50.0% | 14.0 | 46.7% | Feasible (Slack: 20.0h / 16.0h) |

---

## 4. Leveled Production Plan (Pre-Build Smoothing)

To eliminate the Period 3 overload without missing customer deliveries, production is shifted backward into earlier periods with spare capacity:
- **Pre-build 20 units of FG-X in Period 1** (stored into inventory for Period 3 demand).
- **Pre-build 1 unit of FG-Y in Period 2** (stored into inventory for Period 3 demand).

### Leveled Production & Inventory
| Period | Leveled FG-X Prod | Leveled FG-Y Prod | End Inv FG-X | End Inv FG-Y | WC-A Load (Util) | WC-B Load (Util) |
|--------|-------------------|-------------------|--------------|--------------|------------------|------------------|
| **1** | 40 | 15 | 20 | 0 | 38.0h (95.0%) | 28.5h (95.0%) |
| **2** | 30 | 26 | 20 | 1 | 39.6h (99.0%) | 30.0h (100.0%) |
| **3** | 20 | 34 | 0 | 0 | 39.4h (98.5%) | 30.0h (100.0%) |
| **4** | 10 | 10 | 0 | 0 | 20.0h (50.0%) | 14.0h (46.7%) |

**Outcome**: Bottleneck overload in Period 3 is completely eliminated (WC-B utilization dropped from 128.3% to 100.0%), meeting all demand on time.

