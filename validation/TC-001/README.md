# TC-001: Netting Logic Verification (Hand-Solved Benchmark)

## 1. Overview & Objective
Validate fundamental MRP netting arithmetic:
- Projected Available Balance (PAB): $PAB_t = PAB_{t-1} + SR_t - GR_t$
- Net Requirements: $NR_t = \max(0, SS - (PAB_{t-1} + SR_t - GR_t))$
- Planned Order Receipts: $PORec_t = NR_t$ (for L4L)
- Planned Order Releases: $PORel_{t-LT} = PORec_t$
- Exception message generation on past-due lead time offsets ($t - LT \le 0$).

## 2. Master Data
| Item ID | Description | Type | Lead Time (Periods) | Safety Stock | Initial On-Hand | Scheduled Receipts | Lot Sizing |
|---------|-------------|------|---------------------|--------------|-----------------|--------------------|------------|
| **FG-A** | Finished Good A | finished_good | 1 | 10 | 20 | None | L4L |
| **COMP-B** | Component B (qty: 2) | component | 1 | 15 | 30 | Period 1: 20 | L4L |
| **COMP-C** | Component C (qty: 1) | component | 2 | 5 | 10 | None | L4L |

## 3. Step-by-Step Hand Calculations

### 3.1 Finished Good: FG-A (LT = 1, SS = 10, On-Hand = 20)
| Period (t) | 0 | 1 | 2 | 3 | 4 |
|------------|---|---|---|---|---|
| **Gross Requirements** | - | 15 | 25 | 20 | 30 |
| **Scheduled Receipts** | - | 0 | 0 | 0 | 0 |
| **Projected Available Balance (PAB)** | 20 | 10 | 10 | 10 | 10 |
| **Net Requirements** | - | 5 | 25 | 20 | 30 |
| **Planned Order Receipts** | - | 5 | 25 | 20 | 30 |
| **Planned Order Releases** | 5 | 25 | 20 | 30 | - |

*Notes:*
- Period 1: $PAB_{pre} = 20 + 0 - 15 = 5$. Deficit below $SS (10) = 10 - 5 = 5$. Order 5. $PAB_1 = 10$.
  $PORel$ is in period 0 $\rightarrow$ triggers **ACTION / EXPEDITE EXCEPTION**.
- Period 2: $PAB_{pre} = 10 + 0 - 25 = -15$. Deficit below $SS (10) = 10 - (-15) = 25$. Order 25. $PORel$ in Period 1.
- Period 3: Net Req = 20. $PORel$ in Period 2.
- Period 4: Net Req = 30. $PORel$ in Period 3.

---

### 3.2 Component: COMP-B (LT = 1, SS = 15, On-Hand = 30, SR = 20 in P1)
*Gross Requirement = $2 \times \text{FG-A PORel}$*
- Period 1: $2 \times 25 = 50$
- Period 2: $2 \times 20 = 40$
- Period 3: $2 \times 30 = 60$
- Period 4: 0

| Period (t) | 0 | 1 | 2 | 3 | 4 |
|------------|---|---|---|---|---|
| **Gross Requirements** | - | 50 | 40 | 60 | 0 |
| **Scheduled Receipts** | - | 20 | 0 | 0 | 0 |
| **Projected Available Balance (PAB)** | 30 | 15 | 15 | 15 | 15 |
| **Net Requirements** | - | 15 | 40 | 60 | 0 |
| **Planned Order Receipts** | - | 15 | 40 | 60 | 0 |
| **Planned Order Releases** | 15 | 40 | 60 | 0 | - |

*Notes:*
- Period 1: $PAB_{pre} = 30 + 20 - 50 = 0$. Deficit below $SS (15) = 15$. Order 15. $PORel$ in Period 0 $\rightarrow$ Exception alert.
- Period 2: Net Req = 40. $PORel$ in Period 1.
- Period 3: Net Req = 60. $PORel$ in Period 2.
- Period 4: Gross Req = 0. No order needed.

---

### 3.3 Component: COMP-C (LT = 2, SS = 5, On-Hand = 10, SR = 0)
*Gross Requirement = $1 \times \text{FG-A PORel}$*
- Period 1: 25
- Period 2: 20
- Period 3: 30
- Period 4: 0

| Period (t) | -1 | 0 | 1 | 2 | 3 | 4 |
|------------|----|---|---|---|---|---|
| **Gross Requirements** | - | - | 25 | 20 | 30 | 0 |
| **Scheduled Receipts** | - | - | 0 | 0 | 0 | 0 |
| **Projected Available Balance (PAB)** | - | 10 | 5 | 5 | 5 | 5 |
| **Net Requirements** | - | - | 20 | 20 | 30 | 0 |
| **Planned Order Receipts** | - | - | 20 | 20 | 30 | 0 |
| **Planned Order Releases** | 20 | 20 | 30 | 0 | - | - |

*Notes:*
- Period 1: $PAB_{pre} = 10 - 25 = -15$. Deficit below $SS (5) = 20$. Order 20. With $LT=2$, $PORel$ is in period -1 $\rightarrow$ **CRITICAL EXPEDITE EXCEPTION**.
- Period 2: Net Req = 20. $PORel$ in Period 0 $\rightarrow$ Exception alert.
- Period 3: Net Req = 30. $PORel$ in Period 1.

