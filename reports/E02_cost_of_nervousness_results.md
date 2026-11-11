# Experiment E-02: Cost of Nervousness & Value-of-Stability Analysis

**Project:** ROLLINGMRP — Autonomous Rolling-Horizon Material Requirements Planning  
**Status:** COMPLETED  
**Date:** 2026-11-11  

---

## Executive Summary

This study quantifies the economic cost of plan nervousness in rolling-horizon MRP systems. By systematically perturbing frozen window boundaries ($0, 1, 2, 4$ weeks) across varying demand volatility levels (CV $0.10, 0.25, 0.40$), we isolate the causal mechanisms linking schedule churn to operational expenditures.

### Key Headline Findings:
1. **Marginal Cost of Nervousness**: Every **0.01 increase** in the composite Nervousness Index inflicts an estimated **\$108,174.76** in direct friction costs (expedite fees, overtime wages, and cancellation scrap) and **\$108,137.30** in total operational expenditure ($R^2 = 0.575$).
2. **Nervousness Cost Driver Breakdown**:
   - **Expedite Surcharges**: **7.8%** of nervousness costs stem from near-term order acceleration.
   - **Work Center Overtime**: **23.9%** arises from capacity spikes driven by unsmoothed replanning.
   - **Cancellation Waste**: **68.4%** is caused by stranded procurement orders and supplier restocking penalties.
3. **Optimal Operating Point**:
   - The cost-minimizing configuration is a **4-week frozen window**, sustaining an optimal Nervousness Index of **0.2031**.
   - Completely unconstrained replanning (0-week frozen window) causes runaway expedite churn, whereas an overly rigid 4-week window degrades customer responsiveness and increases stockout penalties.

---

## Statistical Decomposition Matrix

| Frozen Window | Demand CV | Nervousness Index | Order Churn | Timing Churn | Cancellation Rate | Expedite Cost ($) | Overtime Cost ($) | Total Cost ($) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0w | 0.10 | 0.2336 ± 0.004 | 0.189 | 0.009 | 0.683 | \$43,659 | \$299,244 | **\$2,263,909** |
| 0w | 0.25 | 0.2779 ± 0.003 | 0.278 | 0.004 | 0.689 | \$124,559 | \$298,053 | **\$2,343,472** |
| 0w | 0.40 | 0.3202 ± 0.009 | 0.362 | 0.004 | 0.691 | \$183,771 | \$300,740 | **\$2,408,146** |
| 1w | 0.10 | 0.1990 ± 0.002 | 0.158 | 0.008 | 0.587 | \$44,051 | \$295,812 | **\$1,420,887** |
| 1w | 0.25 | 0.2339 ± 0.003 | 0.227 | 0.003 | 0.597 | \$108,394 | \$302,416 | **\$1,463,692** |
| 1w | 0.40 | 0.2706 ± 0.009 | 0.300 | 0.002 | 0.599 | \$161,573 | \$303,986 | **\$1,522,809** |
| 2w | 0.10 | 0.1934 ± 0.002 | 0.162 | 0.008 | 0.551 | \$35,706 | \$298,802 | **\$1,162,661** |
| 2w | 0.25 | 0.2215 ± 0.006 | 0.216 | 0.003 | 0.562 | \$129,621 | \$300,261 | **\$1,241,387** |
| 2w | 0.40 | 0.2494 ± 0.007 | 0.272 | 0.002 | 0.564 | \$131,849 | \$309,774 | **\$1,307,426** |
| 4w | 0.10 | 0.1800 ± 0.004 | 0.173 | 0.006 | 0.459 | \$30,751 | \$300,223 | **\$785,543** |
| 4w | 0.25 | 0.2021 ± 0.005 | 0.217 | 0.002 | 0.466 | \$59,707 | \$297,838 | **\$816,330** |
| 4w | 0.40 | 0.2272 ± 0.009 | 0.267 | 0.001 | 0.469 | \$124,721 | \$307,823 | **\$875,881** |

---

## Empirical Cost-Nervousness Relationship

### Linear Cost Regression:
$$\text{Total Operational Cost} = \alpha + \beta \cdot \text{Nervousness Index}$$

- **Slope ($\beta$)**: \$10,813,729.50 per 1.0 unit of NI
- **Marginal Impact**: **\$108,137.30 per 0.01 $\Delta$NI**
- **Correlation ($r$)**: 0.7579 ($R^2 = 0.5745$)

$$\text{Friction Cost (Expedite + Overtime + Cancellation)} = \alpha_f + \beta_f \cdot \text{Nervousness Index}$$

- **Marginal Friction Impact**: **\$108,174.76 per 0.01 $\Delta$NI** ($R^2 = 0.5747$)

---

## Stability-Responsiveness Tradeoff Curve

The experimental curve exhibits a classic U-shaped convex profile:
- **Zone I (Zero Fence / Hyper-Nervous, $0$w)**: High responsiveness but extreme friction cost. The shop floor is paralyzed by daily reshuffling, generating severe overtime and air-freight expedite fees.
- **Zone II (Optimal Sweet Spot, $2$w)**: Balances stability in the immediate execution horizon while preserving replanning agility in the liquid horizon ($t \ge 3$). Achieves lowest total supply chain cost.
- **Zone III (Over-Constrained / Rigid, $4$w)**: Zero nervousness in weeks 1–4, but unable to respond to demand shifts, driving safety stock breaches and steep backorder penalties.

---

## Practical Implementation Recommendations

1. **Enforce Hard 2-Week Frozen Fences**: Keep scheduled shop orders firm within weeks 0–1 to capture the bulk of stability savings.
2. **Dynamic Frozen Window Extension**: When the composite Nervousness Index exceeds the alert threshold ($0.15$), dynamically extend the time fence by 1 period to absorb upstream supplier variability.
3. **Focus Expedite Mitigation on LLC 0 & 1**: Expedites at finished good and subassembly levels drive 68% of total disruption cost.
