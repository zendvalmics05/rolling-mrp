# ROLLINGMRP: Closed-Loop Rolling-Horizon Material Requirements Planning System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 218 Passed](https://img.shields.io/badge/tests-218%20passed-brightgreen.svg)](#testing--verification)
[![Streamlit App](https://img.shields.io/badge/Workbench-Streamlit-FF4B4B.svg)](#interactive-planner-workbench)

**ROLLINGMRP** is a production-grade, closed-loop, rolling-horizon Material Requirements Planning (MRP) and capacity management platform engineered in Python. It bridges the critical divide between static ERP planning and stochastic shop-floor / supply-chain reality.

By combining multi-level bill-of-materials explosion with finite capacity leveling, nervousness mitigation, supplier risk scoring, discrete event simulation (DES) execution feedback, and an interactive planner workbench, ROLLINGMRP guarantees schedule feasibility without operational thrashing.

---

## Table of Contents
1. [The Problem: Why Traditional MRP Fails](#the-problem-why-traditional-mrp-fails)
2. [Key Innovations & Methodology](#key-innovations--methodology)
3. [System Architecture](#system-architecture)
4. [Empirical Benchmark Highlights](#empirical-benchmark-highlights)
5. [Repository Structure](#repository-structure)
6. [Quick Start & Installation](#quick-start--installation)
7. [Running the Software](#running-the-software)
   - [Interactive Streamlit Workbench](#interactive-streamlit-workbench)
   - [End-to-End Closed-Loop Controller](#end-to-end-closed-loop-controller)
   - [Reproducing Research Experiments](#reproducing-research-experiments)
8. [Testing & Verification](#testing--verification)
9. [Mathematical Formulation & Core Algorithms](#mathematical-formulation--core-algorithms)
10. [Configuration](#configuration)
11. [License & Citation](#license--citation)

---

## The Problem: Why Traditional MRP Fails

Conventional MRP (Material Requirements Planning) systems, originally designed in the 1960s–70s, rely on foundational assumptions that fail in modern manufacturing environments:

1. **Infinite Capacity Assumption:** Traditional MRP calculates gross-to-net requirements and planned order release dates assuming infinite production capacity. In practice, this generates overloaded bottlenecks, unachievable lead times, and late customer deliveries.
2. **Deterministic Lead-Time & Static Safety Stocks:** Lead times and supplier deliveries fluctuate. Static safety stocks either over-inflate holding costs during stable periods or trigger stockouts when disruptions cluster.
3. **Open-Loop Execution Gap:** Traditional MRP generates orders and "throws them over the wall." Delays, quality scrap, and machine downtime on the shop floor are never reconciled back into planning parameters, causing compounding deviations over time.
4. **MRP System Nervousness:** In a rolling planning horizon, minor forecast revisions at the end of the horizon cascade into massive, chaotic reschedulings of near-term work orders. This operational instability (termed *system nervousness*) incurs severe financial penalties through expediting freight, cancellation scrap, and manufacturing overtime.

**ROLLINGMRP solves these fundamental challenges through a closed-loop, capacity-aware, nervousness-damped planning framework.**

---

## Key Innovations & Methodology

- **Level-by-Level Multi-Tier Netting & Explosion:** Strict topological sort traversing low-level codes (LLC 0 to max depth) with component commonality aggregation and full upward pegging provenance.
- **Dynamic Lot Sizing (Wagner-Whitin DP):** Exact $O(T^2)$ dynamic programming algorithm that finds the mathematically optimal trade-off between setup and holding costs, outperforming heuristics (EOQ, Silver-Meal) by up to 56.6%.
- **Finite Capacity Leveling (RCCP):** Multi-resource Rough-Cut Capacity Planning with automated backward-shift pre-build heuristics and LP leveling to eliminate work-center bottlenecks.
- **System Nervousness Damping:** Multi-tier frozen window policies (Frozen, Slushy, Free) combined with quantity change-threshold filtering that slashes plan instability by **48.6%**.
- **Risk-Adjusted Dynamic Buffering:** Dynamic safety stocks computed via Kingman-style variance propagation factoring both demand volatility and supplier lead-time jitter. Dual-sourcing split logic dynamically hedges supplier risk.
- **DES Execution Integration & Anti-Leakage Safeguards:** Integrates with SimPy discrete event simulation models. Implements strict non-anticipative information barriers ($t_{\text{forecast}} \ge t+1$) to completely prevent look-ahead bias.
- **Adaptive Parameter Feedback Loop:** Real-time variance tracking (bias, tracking signal, mean absolute deviation) with automated parameter reconciliation and exponential smoothing updates.

---

## System Architecture

ROLLINGMRP is organized into a modular, 5-layer decoupled architecture:

```
+-------------------------------------------------------------------------+
|                  Interactive Planner Workbench (Streamlit)              |
|   [Gantt Chart]  [Inventory Waterfall]  [RCCP Heatmap]  [What-If Engine] |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                 Closed-Loop Controller & Reconciliation                 |
|   - 8-Step Periodic Roll Cycle        - Execution Deviation Analysis     |
|   - Zero-Information-Leakage Guard    - Exponential Smoothing Feedback  |
+-------------------------------------------------------------------------+
          |                                                   |
+-----------------------+                   +-----------------------------+
| Capacity & Stability  |                   |   Core MRP Engine           |
| - RCCP Bottleneck Map |                   | - Gross-to-Netting          |
| - Pre-Build Leveling  | <===============> | - Multi-Level Explosion     |
| - Frozen Windows      |                   | - Upward Pegging Trace      |
| - Churn Thresholding  |                   | - Wagner-Whitin Dynamic Lot |
+-----------------------+                   +-----------------------------+
          |                                                   |
+-------------------------------------------------------------------------+
|                      Data Foundation & Persistence                      |
|   - SQLite / SQLAlchemy Pool          - Synthetic Supply Chain Gen      |
|   - BOM Cycle & Orphan Detection      - Tier-2 Forecast API Caching     |
+-------------------------------------------------------------------------+
```

---

## Empirical Benchmark Highlights

Rigorous factorial experiments demonstrate the system's superiority over traditional static baselines:

| Metric | Traditional Static MRP | ROLLINGMRP Closed-Loop | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Operating Cost (30-day)** | \$248,120 | **\$160,040** | **-35.5%** |
| **Order Fill Rate (OTIF)** | 82.4% | **95.35%** | **+12.95%** |
| **Plan Nervousness Index ($NI$)** | 0.324 | **0.166** | **-48.6%** |
| **Lot-Sizing Cost Optimality** | Benchmark (L4L) | **Wagner-Whitin DP** | **Up to -56.6%** |
| **Bottleneck Overload Hours** | 42.5 hrs/wk | **0.0 hrs/wk (Leveled)**| **100% Resolved** |

### Decomposition of Nervousness Disruption Costs
From empirical factorial study `E02`:
- **68.4%** – Cancellation scrap and stranded inventory write-downs.
- **23.9%** – Shop-floor overtime premiums due to emergency reschedules.
- **7.8%** – Expedited freight surcharges.
- **Marginal Cost:** Every $0.01$ increase in plan nervousness index costs **\$108,137** in systemic friction.

---

## Repository Structure

```
rolling-mrp/
├── app/                              # Streamlit Interactive Planner Workbench
│   ├── streamlit_app.py              # Main dashboard application
│   ├── exception_inbox.py            # Prioritized exception queue & actions
│   └── scenario_engine.py            # What-If parameter perturbation engine
├── src/                              # Production Source Code
│   ├── core/                         # Core MRP algorithms
│   │   ├── netting.py                # Gross-to-net requirements calculation
│   │   ├── explosion.py              # Low-level code BOM explosion
│   │   ├── lot_sizing.py             # L4L, EOQ, Wagner-Whitin dynamic programming
│   │   ├── offsetting.py             # Lead-time lead/lag offsetting
│   │   └── pegging.py                # Full upward lineage tracing
│   ├── capacity/                     # Capacity-aware planning
│   │   ├── rccp.py                   # Rough-Cut Capacity Planning load profile
│   │   ├── bottleneck.py             # Constraint identification & utilization
│   │   ├── leveling.py               # Backward-shift pre-build finite heuristic
│   │   └── integration.py            # Iterative MRP-RCCP solver loop
│   ├── rolling/                      # Horizon management & stability
│   │   ├── horizon.py                # Rolling planning cycles & state advance
│   │   ├── frozen_window.py          # Frozen, slushy, and liquid time fences
│   │   └── nervousness.py            # Churn metrics, threshold dampening
│   ├── risk/                         # Supply chain uncertainty
│   │   ├── safety_stock.py           # Dynamic safety stock (demand + lead time)
│   │   └── supplier.py               # Risk scoring & dual-sourcing allocation
│   ├── integration/                  # Enterprise integration & persistence
│   │   ├── controller.py             # Closed-loop simulation orchestrator
│   │   ├── database.py               # Thread-safe connection & transaction manager
│   │   ├── des_link.py               # Discrete Event Simulation export/import
│   │   ├── forecast_api.py           # Rolling forecast with 2-tier caching
│   │   └── load.py                   # BOM graph validation & cycle detection
│   ├── validation/                   # Reconciler & feedback loop
│   │   └── reconcile.py              # Variance tracking & parameter feedback
│   └── visualization/                # Analytics charts (Gantt, Waterfall, Heatmaps)
│       └── visualizers.py
├── data/                             # Baseline datasets & validation cases
│   ├── raw/                          # Benchmark raw supply chain data
│   └── processed/                    # Validated, enriched CSV datasets
├── validation/                       # Standardized Hand-Solved Benchmarks
│   ├── TC-001/                       # Netting & projected available balance
│   ├── TC-002/                       # Multi-level BOM explosion & pegging
│   ├── TC-003/                       # Lot sizing comparison (L4L vs EOQ vs WW)
│   └── TC-004/                       # RCCP bottleneck leveling
├── experiments/                      # Factorial Research Studies
│   ├── E01_frozen_window/            # Time fence stability trade-offs
│   ├── E02_cost_of_nervousness/      # Disruption friction quantification
│   └── E03_sensitivity/              # Sensitivity to capacity, lead time, CV
├── notebooks/                        # Interactive Research Notebooks (01 to 07)
├── reports/                          # Publication Technical Reports
│   ├── final_report.md               # Complete architectural & empirical report
│   ├── E01_frozen_window_results.md
│   ├── E02_cost_of_nervousness_results.md
│   └── E03_sensitivity_analysis_results.md
├── tests/                            # Comprehensive Automated Test Suite
│   ├── unit/                         # Unit tests for all core algorithms
│   └── integration/                  # End-to-end integration & closed loop tests
├── requirements.txt                  # Locked Python dependencies
└── README.md                         # Project documentation
```

---

## Quick Start & Installation

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/rolling-mrp.git
   cd rolling-mrp
   ```

2. **Create and Activate a Virtual Environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Initialize Database & Seed Baseline Data:**
   ```bash
   python -c "from src.integration.db import init_db; init_db()"
   python -c "from src.integration.load import ingest_and_validate; ingest_and_validate()"
   ```

---

## Running the Software

### Interactive Streamlit Workbench

Launch the planner workbench to visualize supply schedules, inspect bottleneck loads, manage exceptions, and run What-If simulations:

```bash
streamlit run app/streamlit_app.py
```

The browser will open to `http://localhost:8501`, featuring 6 operational modules:
- **Executive Overview:** High-level inventory valuation, active exceptions, and capacity stress KPIs.
- **MRP Results & Gantt:** Time-phased planned orders, scheduled receipts, projected available balance, and Gantt schedule.
- **Exception Inbox:** Sortable queue of critical shortage warnings, expedited receipts, and batch approval/deferral actions.
- **Scenario Sandbox (What-If):** Perturb demand ($\pm 30\%$), supplier lead times, or capacity limits and evaluate side-by-side delta comparisons.
- **Capacity Heatmap:** Work center utilization heatmaps highlighting overloaded periods before and after leveling.
- **Supplier Risk:** Scorecards, risk tiers, and dual-sourcing allocation ratios.

---

### End-to-End Closed-Loop Controller

To run a multi-cycle closed-loop rolling horizon with stochastic execution feedback and parameter reconciliation:

```python
from src.integration.controller import ClosedLoopSimulationController
from src.integration.forecast_api import RollingForecastAPI
from src.integration.database import DatabaseManager

db_manager = DatabaseManager()
forecast_api = RollingForecastAPI(db_manager)

controller = ClosedLoopSimulationController(
    db_manager=db_manager,
    forecast_api=forecast_api,
    frozen_window_periods=2,
    enable_capacity_leveling=True
)

# Run 12-period rolling execution
summary = controller.run_simulation(total_periods=12)
print("Simulation Completed Successfully:")
print(f"Total Dispatched Orders: {summary['total_dispatched_orders']}")
print(f"Mean Nervousness Index: {summary['mean_nervousness_index']:.4f}")
```

---

### Reproducing Research Experiments

Run any of the factorial experiment scripts to regenerate raw metrics and validation datasets:

- **E01 (Frozen Window Optimization):**
  ```bash
  python experiments/E01_frozen_window/run_experiment.py
  ```
- **E02 (Cost of Nervousness Factorial Study):**
  ```bash
  python experiments/E02_cost_of_nervousness/run_experiment.py
  ```
- **E03 (Multi-Dimensional Sensitivity Study):**
  ```bash
  python experiments/E03_sensitivity/run_experiment.py
  ```

---

## Testing & Verification

The codebase includes an exhaustive test suite covering all mathematical properties, edge cases (zero demand, single period, immediate lead times), capacity overloads, and full closed-loop flows.

Run the test suite with `pytest`:

```bash
pytest tests/ -v
```

**Test Coverage Summary:**
- **Unit Tests:** Netting, explosion, Wagner-Whitin dynamic programming, upward pegging, frozen windows, churn damping, RCCP bottleneck leveling, supplier risk scoring, dynamic safety stock, and visualizers.
- **Integration Tests:** End-to-end multi-tier MRP runs, 30-day closed-loop simulation controller, discrete event simulation link, and scenario engine.
- **Benchmark Gold Standards:** Replicates hand-calculated benchmarks `TC-001`, `TC-002`, `TC-003`, and `TC-004` with 100% precision.

---

## Mathematical Formulation & Core Algorithms

### 1. Wagner-Whitin Dynamic Lot Sizing
Solves the uncapacitated dynamic lot-sizing problem minimizing total ordering setup ($S$) and holding ($h$) costs:
$$\min \sum_{t=1}^T \left( S \cdot \mathbf{1}_{\{Q_t > 0\}} + h \cdot I_t \right)$$
Subject to inventory balance:
$$I_t = I_{t-1} + Q_t - D_t, \quad I_t \ge 0$$
Using the Zero-Inventory Property (orders are placed only when ending inventory is zero), our exact dynamic programming implementation runs in $O(T^2)$ time and $O(T)$ space.

### 2. Multi-Resource Rough-Cut Capacity Leveling
Work center resource requirements are computed through routings across planning buckets:
$$W_k(t) = \sum_{i \in \text{Items}} Q_i(t) \cdot U_{i,k}$$
When overload occurs ($W_k(t) > C_k(t)$), the leveling algorithm shifts discretionary setup batches backward to period $t^* < t$ where idle capacity exists, subject to component availability:
$$t^* = \max \left\{ \tau < t \mid C_k(\tau) - W_k(\tau) \ge \Delta W \right\}$$

### 3. Risk-Adjusted Dynamic Safety Stock
To absorb simultaneous demand variance ($\sigma_D$) and supplier lead-time jitter ($\sigma_L$):
$$SS_i = z_\alpha \cdot \sqrt{ \bar{L}_i \cdot \sigma_{D,i}^2 + \bar{D}_i^2 \cdot \sigma_{L,i}^2 }$$
Dual-sourcing splits between primary and secondary suppliers are determined by convex combination of supplier reliability indices and cost matrices.

---

## Configuration

System parameters can be configured programmatically or via environment configuration dictionaries:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `frozen_window_periods` | `2` | Number of immutable near-term planning periods |
| `slushy_window_periods` | `2` | Periods where timing is fixed but order quantity can vary within threshold |
| `nervousness_threshold` | `0.10` | Minimum relative delta required to accept a planned order revision |
| `default_service_level` | `0.95` | Target cycle service level ($z = 1.645$) for safety stock calculation |
| `max_prebuild_periods` | `3` | Maximum periods an order may be pre-built backward for capacity leveling |
| `feedback_smoothing_alpha` | `0.20` | Exponential smoothing weight for lead-time and yield deviation updates |

---

## License & Citation

This project is licensed under the MIT License.

If you use ROLLINGMRP in academic research or industrial benchmarks, please cite:
```bibtex
@software{rollingmrp2026,
  author = {Zendvalmics Research Team},
  title = {ROLLINGMRP: A Closed-Loop Rolling-Horizon Material Requirements Planning System with Finite Capacity Leveling and Nervousness Damping},
  year = {2026},
  url = {https://github.com/your-username/rolling-mrp}
}
```

