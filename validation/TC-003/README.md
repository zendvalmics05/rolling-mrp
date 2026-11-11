# TC-003: Lot Sizing Comparison (L4L vs EOQ vs Wagner-Whitin)

## 1. Overview & Objective
Evaluate dynamic lot sizing rules across an 8-item, 8-period portfolio to verify:
- **Principle 3 (Implementation Plan §0)**: Complex lot-sizing rules must beat Lot-for-Lot (L4L) on total cost to justify complexity.
- Mathematical optimality of the **Wagner-Whitin (W-W)** dynamic programming algorithm.
- Conditions under which L4L is optimal vs. when batching produces significant cost savings (up to 56.5% savings).

## 2. Mathematical Formulations

### 2.1 Lot-for-Lot (L4L)
- Order in every period with non-zero net requirements: $Q_t = D_t$
- Total Setup Cost = $N_{\text{orders}} \times S$
- Holding Cost = $\$0.00$

### 2.2 Economic Order Quantity (EOQ)
$$EOQ = \sqrt{\frac{2 \bar{D} S}{h}}$$

### 2.3 Wagner-Whitin Dynamic Programming Formulation
Objective: Minimize total setup + inventory holding costs:
$$f(t) = \min_{1 \le j \le t} \left[ f(j-1) + S + \sum_{k=j}^t (k - j) \cdot h \cdot D_k \right]$$
where:
- $f(t)$ is the minimum cost to satisfy demand from period 1 through $t$.
- $j$ is the last period in which an order was placed.

## 3. Results Summary Across 8 Items (8 Periods)

| Item ID | Setup ($S$) | Hold ($h$) | Total Demand | L4L Total Cost | W-W Total Cost | W-W Cost Savings (%) |
|---------|-------------|------------|--------------|----------------|----------------|----------------------|
| **ITEM-01** | $100 | $2.0 | 500 | $800.00 | $740.00 | **+7.50%** |
| **ITEM-02** | $150 | $1.5 | 300 | $1,200.00 | $847.50 | **+29.38%** |
| **ITEM-03** | $80 | $3.0 | 800 | $640.00 | $640.00 | **0.00%** (L4L optimal) |
| **ITEM-04** | $200 | $1.0 | 170 | $1,600.00 | $695.00 | **+56.56%** |
| **ITEM-05** | $50 | $4.0 | 425 | $400.00 | $400.00 | **0.00%** (L4L optimal) |
| **ITEM-06** | $120 | $2.5 | 260 | $600.00 | $600.00 | **0.00%** (L4L optimal) |
| **ITEM-07** | $90 | $1.8 | 640 | $720.00 | $720.00 | **0.00%** (L4L optimal) |
| **ITEM-08** | $180 | $2.0 | 255 | $1,440.00 | $960.00 | **+33.33%** |

## 4. Analytical Insights
1. **High $S/h$ ratio (e.g. ITEM-04 with $S/h = 200$)**: Wagner-Whitin reduces order count from 8 to 2 batches, yielding massive 56.56% cost reduction.
2. **Low $S/h$ ratio (e.g. ITEM-05 with $S/h = 12.5$)**: Inventory carrying costs dominate; Wagner-Whitin naturally selects Lot-for-Lot as the exact cost-minimizing solution without holding unneeded buffer inventory.

