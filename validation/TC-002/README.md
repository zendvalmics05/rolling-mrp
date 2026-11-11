# TC-002: Multi-Level BOM Explosion & Pegging Traceability

## 1. Overview & Objective
Validate multi-level BOM explosion across 3 hierarchy levels (Level 0 $\rightarrow$ Level 1 $\rightarrow$ Level 2), component commonality aggregation (a raw material driven by multiple parent subassemblies), and full upward pegging traceability:
$$\text{Planned Order} \longrightarrow \text{Parent Subassembly Demand} \longrightarrow \text{Master Production Schedule (MPS)}$$

## 2. Product Hierarchy & Master Data
```
FG-1 (Level 0, LT=1, SS=0, On-Hand=0)
 ├── SA-1 (Level 1, qty=1, LT=1, SS=10, On-Hand=15)
 │    └── RM-1 (Level 2, qty=2, LT=2, SS=20, On-Hand=50)  <-- Common Component
 └── SA-2 (Level 1, qty=2, LT=1, SS=0, On-Hand=20)
      ├── RM-1 (Level 2, qty=1, LT=2, SS=20, On-Hand=50)  <-- Common Component
      └── RM-2 (Level 2, qty=3, LT=1, SS=10, On-Hand=30)
```

## 3. Step-by-Step Multi-Level Explosion

### 3.1 Level 0: FG-1 (MPS Demand: [0, 20, 0, 40, 0, 50])
- Net Requirements: [0, 20, 0, 40, 0, 50]
- Planned Order Receipts: [0, 20, 0, 40, 0, 50]
- Planned Order Releases (LT = 1):
  - Period 1: 20 (pegged to FG-1 MPS Period 2)
  - Period 3: 40 (pegged to FG-1 MPS Period 4)
  - Period 5: 50 (pegged to FG-1 MPS Period 6)

### 3.2 Level 1: Subassemblies
#### SA-1 (qty = 1 from FG-1 PORel, LT=1, SS=10, On-Hand=15)
- Gross Requirements: P1=20, P2=0, P3=40, P4=0, P5=50, P6=0
- Net Requirements: P1=15, P3=40, P5=50
- PORel (LT=1): P0=15, P2=40, P4=50

#### SA-2 (qty = 2 from FG-1 PORel, LT=1, SS=0, On-Hand=20)
- Gross Requirements: P1=40, P2=0, P3=80, P4=0, P5=100, P6=0
- Net Requirements: P1=20, P3=80, P5=100
- PORel (LT=1): P0=20, P2=80, P4=100

### 3.3 Level 2: Raw Materials & Component Commonality
#### RM-1 (Shared between SA-1 [qty 2] and SA-2 [qty 1], LT=2, SS=20, On-Hand=50)
- Aggregated Gross Requirements:
  - Period 2: $(2 \times 40) + (1 \times 80) = 80 + 80 = 160$
  - Period 4: $(2 \times 50) + (1 \times 100) = 100 + 100 = 200$
- Net Requirements:
  - Period 2: $20 - (50 - 160) = 130 \rightarrow PORec = 130 \rightarrow PORel$ in Period 0 (130 units)
  - Period 4: $20 - (20 - 200) = 200 \rightarrow PORec = 200 \rightarrow PORel$ in Period 2 (200 units)
- **Pegging Provenance**:
  - P2 Net Req of 130: Pegs 65 units to SA-1 (Period 2) and 65 units to SA-2 (Period 2), which both trace up to FG-1 Period 4 MPS.
  - P4 Net Req of 200: Pegs 100 units to SA-1 (Period 4) and 100 units to SA-2 (Period 4), tracing to FG-1 Period 6 MPS.

#### RM-2 (Used in SA-2 [qty 3], LT=1, SS=10, On-Hand=30)
- Gross Requirements:
  - Period 2: $3 \times 80 = 240$
  - Period 4: $3 \times 100 = 300$
- Net Requirements:
  - Period 2: $10 - (30 - 240) = 220 \rightarrow PORec = 220 \rightarrow PORel$ in Period 1 (220 units)
  - Period 4: $10 - (10 - 300) = 300 \rightarrow PORec = 300 \rightarrow PORel$ in Period 3 (300 units)

