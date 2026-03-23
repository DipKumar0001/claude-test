# CMP-N206 – Rational Taxi Agent: Full Explanation & Results

**Module:** CMP-N206 – Artificial Intelligence Coursework 1  
**Title:** A Rational Taxi Agent for Grid-London: Design, Implementation and Evaluation of Classical Search Strategies  
**Author:** Dipesh Kumar Yadav (Student ID: A00038176)  
**Date:** 12th March 2026  

> **About this file:** This document contains a complete description of `taxi_agent.ipynb`,
> including every section of the notebook, the actual outputs produced when it was run, and
> a full explanation of what the results mean.
> The executed notebook with embedded outputs is also saved as `taxi_agent_executed.ipynb`.

---

## Table of Contents

1. [What the Notebook Does](#1-what-the-notebook-does)
2. [Section 1 – Environment Setup & Imports](#2-section-1--environment-setup--imports)
3. [Section 2 – Grid-London Environment Model](#3-section-2--grid-london-environment-model)
4. [Section 3 – Agent Design & PEAS Framework](#4-section-3--agent-design--peas-framework)
5. [Section 4 – Search Algorithms](#5-section-4--search-algorithms)
6. [Section 5 – Evaluation Scenarios & Full Results](#6-section-5--evaluation-scenarios--full-results)
7. [Section 6 – Explainability](#7-section-6--explainability)
8. [Section 7 – Reflection: Rationality, Fairness & Sustainability](#8-section-7--reflection-rationality-fairness--sustainability)
9. [Section 8 – Conclusion & References](#9-section-8--conclusion--references)
10. [How to Run the Notebook](#10-how-to-run-the-notebook)

---

## 1. What the Notebook Does

`taxi_agent.ipynb` implements a **rational taxi agent** that navigates a 7×7 grid map of
Greater London. The agent picks up a passenger from one location and drops them off at a
destination, avoiding blocked cells that represent road closures or traffic jams.

Three classical AI search algorithms are implemented, tested and compared:

| Algorithm | Strategy | Optimal? |
|-----------|----------|----------|
| **BFS** (Breadth-First Search) | Layer-by-layer | ✅ Yes |
| **DFS** (Depth-First Search) | Deep-first | ❌ No |
| **A\*** (A-Star) | Heuristic-guided | ✅ Yes |

The notebook runs **3 scenarios** of increasing difficulty and records steps taken, final
score, and whether the goal was achieved for each algorithm.

---

## 2. Section 1 – Environment Setup & Imports

```python
from collections import deque   # FIFO queue for BFS
import heapq                    # min-heap / priority queue for A*
import matplotlib.pyplot as plt # grid visualisation
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd             # results comparison table
from datetime import datetime
```

**Actual output when run:**
```
All libraries imported successfully ✓
```

---

## 3. Section 2 – Grid-London Environment Model

### The `GridLondon` Class

The environment is a **7×7 grid** where every cell maps to a real London location.
The full 49-cell map:

```
Row 0: King's Cross | Islington    | Hackney       | Angel         | Bethnal Green | Bow         | Stratford
Row 1: Euston       | Bloomsbury   | Shoreditch    | Barbican      | Whitechapel   | Mile End    | West Ham
Row 2: Paddington   | Soho         | Covent Garden | Bank          | Liverpool St  | Stepney     | Plaistow
Row 3: Hammersmith  | Kensington   | Westminster   | Oxford Circus | The City      | Wapping     | Canary Wharf
Row 4: Chiswick     | Earls Court  | Chelsea       | Vauxhall      | Bermondsey    | Rotherhithe | Isle of Dogs
Row 5: Richmond     | Wimbledon    | Tooting       | Streatham     | Peckham       | New Cross   | Deptford
Row 6: Heathrow     | Kingston     | Morden        | Brixton       | Lewisham      | Blackheath  | Greenwich
```

**Key locations:**

| Grid (row, col) | London Zone |
|-----------------|-------------|
| (0, 0) | King's Cross (start in all scenarios) |
| (0, 3) | Angel |
| (0, 6) | Stratford |
| (2, 3) | Bank |
| (3, 0) | Hammersmith |
| (3, 3) | Oxford Circus |
| (3, 6) | Canary Wharf |
| (6, 0) | Heathrow |
| (6, 3) | Brixton |
| (6, 6) | Greenwich |

**Blocked cells** (dark cells on the visualisation) represent road closures. The taxi
cannot enter blocked cells and must route around them.

**Actual output when grid is demonstrated:**
```
Taxi at: King's Cross (0, 0)
Passenger at: Angel (0, 3)
Destination: Greenwich (6, 6)
Blocked (4 cells): Rotherhithe, Bloomsbury, Shoreditch, The City
```

### Environment Properties (REAS Classification)

| Property | Value | Reason |
|----------|-------|--------|
| Observable | **Partially** | Agent sees grid state but not future passengers |
| Deterministic | **Yes** | Actions always produce the same outcome |
| Episodic | **No (Sequential)** | Each step depends on the previous state |
| Static | **Yes** | Grid does not change while the agent runs |
| Discrete | **Yes** | Cells, not continuous space |
| Single-agent | **Yes** | One taxi only in this implementation |

### Key Methods

| Method | What it does |
|--------|-------------|
| `get_neighbours(cell)` | Returns valid adjacent cells (N/S/E/W), excluding blocked ones |
| `get_name(cell)` | Returns the London zone name for a grid position |
| `visualise(...)` | Renders the grid using matplotlib |

---

## 4. Section 3 – Agent Design & PEAS Framework

### PEAS Analysis

| Component | Description |
|-----------|-------------|
| **Performance** | +20 drop-off, +10 pickup, −1 per step, −5 blocked attempt, −10 timeout |
| **Environment** | 7×7 Grid-London; partially observable, deterministic, sequential |
| **Actuators** | move_north, move_south, move_east, move_west, pickup, dropoff |
| **Sensors** | current_location, passenger_location, destination, blocked_cells |

### Agent Type

The `TaxiAgent` is a **goal-based rational agent**. It:
1. **Perceives** its environment (`perceive()` returns current location, passenger status, etc.)
2. **Plans** using its chosen search strategy (BFS, DFS, or A\*)
3. **Executes** the computed path step-by-step
4. **Logs** every action with full explainability

The agent is **rational** because it maximises its utility function by taking the
mathematically optimal (or near-optimal) route given its percepts and prior knowledge.

### Utility (Scoring) Function

| Event | Score Change | Rationale |
|-------|-------------|-----------|
| Successful drop-off | **+20** | Primary goal — incentivises completion |
| Passenger pickup | **+10** | Sub-goal reward |
| Each step taken | **−1** | Fuel cost and driver time — encourages efficiency |
| Blocked cell attempt | **−5** | Traffic penalty — models reckless navigation |
| Exceeding 100 steps | **−10** | Timeout penalty — prevents infinite loops |

The maximum possible score for a journey is `+20 + 10 − (steps × 1)`.
A 12-step journey yields a score of **18** (= 30 − 12).

**Actual output when agent class is defined:**
```
TaxiAgent class defined (with corrected goal_achieved tracking)
```

---

## 5. Section 4 – Search Algorithms

### Algorithm Comparison Table

| Algorithm | Optimal? | Complete? | Time Complexity | Space Complexity | Notes |
|-----------|----------|-----------|----------------|-----------------|-------|
| **BFS** | ✅ Yes | ✅ Yes | O(b^d) | O(b^d) | Explores layer by layer |
| **DFS** | ❌ No | ❌ No* | O(b^m) | O(b·m) | May find very long paths |
| **A\*** | ✅ Yes | ✅ Yes | O(b^d) best | O(b^d) best | Guided by Manhattan heuristic |

*DFS uses a visited set to prevent infinite loops, but may still find non-optimal paths.

### BFS — Breadth-First Search

BFS explores every cell at distance 1, then every cell at distance 2, and so on.
It uses a **FIFO queue** (`deque`) to ensure the shallowest node is always expanded first.

- **Guarantees** the shortest path when all move costs are equal (unweighted grid).
- **Drawback:** Stores all frontier nodes in memory — can be slow on large grids.

### DFS — Depth-First Search

DFS follows one path as deep as possible before backtracking. It uses the **call stack**
(recursion) instead of an explicit queue.

- **Does NOT** guarantee the shortest path.
- **Advantage:** Low memory usage — only the current path needs to be stored.
- **In this project:** DFS consistently found paths much longer than optimal, causing it
  to hit the 100-step timeout.

### A* — A-Star Search

A\* combines BFS completeness with heuristic guidance. It expands nodes in order of:

```
f(n) = g(n) + h(n)
```

where:
- `g(n)` = actual cost from start to node n (steps taken so far)
- `h(n)` = estimated cost from n to goal (Manhattan distance heuristic)

**Manhattan Distance Heuristic:**
```
h(n) = |row_n − row_goal| + |col_n − col_goal|
```

This is **admissible** (never overestimates) because:
- The taxi can only move N/S/E/W — no diagonals.
- The Manhattan distance equals the minimum possible steps.
- Therefore `h(n) ≤ h*(n)` for all nodes — admissibility satisfied (Hart et al., 1968).

### Quick Test Results (Actual Output)

```
BFS Test: (0,0) → (0,3) | Path: [(0, 0), (0, 1), (0, 2), (0, 3)] | Length: 4 steps
BFS defined and tested ✓

DFS Test: (0,0) → (0,3) | Path: [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(6,1),(5,1),(4,1),
                                   (3,1),(2,1),(2,2),(3,2),(4,2),(5,2),(6,2),(6,3),(5,3),(4,3),
                                   (3,3),(2,3),(1,3),(0,3)] | Length: 24 steps
BFS path length: 4 | DFS path length: 24 — DFS may take longer!
DFS defined and tested ✓

A* Test:  (0,0) → (0,3) | Path: [(0, 0), (0, 1), (0, 2), (0, 3)] | Length: 4 steps

Algorithm comparison on test route (0,0) → (0,3):
  BFS path length: 4
  DFS path length: 24
  A*  path length: 4

All three search algorithms defined and tested ✓
```

**What this shows:** On a simple 4-cell route, DFS wandered all the way to the bottom of
the grid and back — a 24-step path — while BFS and A\* found the direct 4-step route.

---

## 6. Section 5 – Evaluation Scenarios & Full Results

Three scenarios of increasing difficulty are run with all three algorithms.

---

### Scenario 1 – Simple Route

| Setting | Value |
|---------|-------|
| Start | King's Cross `(0,0)` |
| Passenger | Bank `(2,3)` |
| Destination | Greenwich `(6,6)` |
| Blocked cells | 4 cells: Bloomsbury `(1,1)`, Shoreditch `(1,2)`, The City `(3,4)`, Rotherhithe `(4,5)` |

**Actual output:**
```
============================================================
SCENARIO 1 - Simple Route
King's Cross (0,0) -> Bank (2,3) -> Greenwich (6,6)
============================================================
  BFS  : Steps= 12 | Score=  18 | Goal=YES
  DFS  : Steps=100 | Score=-110 | Goal=NO
  ASTAR: Steps= 12 | Score=  18 | Goal=YES
```

**Full A\* Journey Log (Scenario 1) — Actual Output:**
```
================================================================================
  JOURNEY EXPLANATION — Strategy: ASTAR
================================================================================
  [INIT] Taxi starts at King's Cross (0, 0) | Strategy: ASTAR | Passenger at: Bank | Destination: Greenwich
  Step   1: Move East  | King's Cross (0, 0) -> Islington (0, 1)   | Heading to passenger at Bank (2,3) | Score: -1
  Step   2: Move East  | Islington (0, 1) -> Hackney (0, 2)        | Heading to passenger at Bank (2,3) | Score: -2
  Step   3: Move East  | Hackney (0, 2) -> Angel (0, 3)            | Heading to passenger at Bank (2,3) | Score: -3
  Step   4: Move South | Angel (0, 3) -> Barbican (1, 3)           | Heading to passenger at Bank (2,3) | Score: -4
  Step   5: Move South | Barbican (1, 3) -> Bank (2, 3)            | Heading to passenger at Bank (2,3) | Score: -5
            >>> PICKUP at Bank (2, 3) | Score +10 -> 5
  Step   6: Move East  | Bank (2, 3) -> Liverpool St (2, 4)        | Passenger aboard — to Greenwich (6,6) | Score: 4
  Step   7: Move East  | Liverpool St (2, 4) -> Stepney (2, 5)     | Passenger aboard — to Greenwich (6,6) | Score: 3
  Step   8: Move East  | Stepney (2, 5) -> Plaistow (2, 6)         | Passenger aboard — to Greenwich (6,6) | Score: 2
  Step   9: Move South | Plaistow (2, 6) -> Canary Wharf (3, 6)    | Passenger aboard — to Greenwich (6,6) | Score: 1
  Step  10: Move South | Canary Wharf (3, 6) -> Isle of Dogs (4, 6)| Passenger aboard — to Greenwich (6,6) | Score: 0
  Step  11: Move South | Isle of Dogs (4, 6) -> Deptford (5, 6)    | Passenger aboard — to Greenwich (6,6) | Score: -1
  Step  12: Move South | Deptford (5, 6) -> Greenwich (6, 6)       | Passenger aboard — to Greenwich (6,6) | Score: -2
            >>> DROPOFF at Greenwich (6, 6) | Score +20 -> 18 | GOAL ACHIEVED in 12 steps
--------------------------------------------------------------------------------
  Final Score    : 18
  Total Steps    : 12
  Goal Achieved  : YES
================================================================================
```

**Explanation of Scenario 1 results:**

- **BFS & A\* (12 steps, score 18):** Both algorithms found the direct optimal route. The taxi
  went east along row 0 to column 3, dropped south 2 rows to pick up the passenger at Bank,
  then went east to column 6 and south to Greenwich. Clean, minimal path.
- **DFS (100 steps, score −110):** DFS hit the timeout limit because it explored a winding
  path that went far from the goal. It never successfully picked up the passenger in 100 steps,
  so no pickup bonus was earned. Score = −1×100 steps − 10 timeout = −110.

---

### Scenario 2 – Blocked Route (Forced Detour)

| Setting | Value |
|---------|-------|
| Start | King's Cross `(0,0)` |
| Passenger | Oxford Circus `(3,3)` |
| Destination | Greenwich `(6,6)` |
| Blocked cells | 5 cells forming a barrier: Covent Garden `(2,2)`, Bank `(2,3)`, Liverpool St `(2,4)`, Westminster `(3,2)`, Chelsea `(4,2)` |

The blocked cells form a diagonal barrier that completely blocks the direct central route.

**Actual output:**
```
============================================================
SCENARIO 2 - Blocked Route (Forced Detour)
King's Cross (0,0) -> Oxford Circus (3,3) -> Greenwich (6,6)
Barrier forces agent to take a longer route!
============================================================
  BFS  : Steps= 16 | Score=  14 | Goal=YES
  DFS  : Steps=100 | Score=-110 | Goal=NO
  ASTAR: Steps= 16 | Score=  14 | Goal=YES
```

**Full BFS Journey Log (Scenario 2) — Actual Output:**
```
================================================================================
  JOURNEY EXPLANATION — Strategy: BFS
================================================================================
  [INIT] Taxi starts at King's Cross (0, 0) | Strategy: BFS | Passenger at: Oxford Circus | Destination: Greenwich
  Step   1: Move South | King's Cross (0, 0) -> Euston (1, 0)       | Heading to passenger at Oxford Circus (3,3) | Score: -1
  Step   2: Move South | Euston (1, 0) -> Paddington (2, 0)         | Heading to passenger at Oxford Circus (3,3) | Score: -2
  Step   3: Move South | Paddington (2, 0) -> Hammersmith (3, 0)    | Heading to passenger at Oxford Circus (3,3) | Score: -3
  Step   4: Move South | Hammersmith (3, 0) -> Chiswick (4, 0)      | Heading to passenger at Oxford Circus (3,3) | Score: -4
  Step   5: Move South | Chiswick (4, 0) -> Richmond (5, 0)         | Heading to passenger at Oxford Circus (3,3) | Score: -5
  Step   6: Move East  | Richmond (5, 0) -> Wimbledon (5, 1)        | Heading to passenger at Oxford Circus (3,3) | Score: -6
  Step   7: Move East  | Wimbledon (5, 1) -> Tooting (5, 2)         | Heading to passenger at Oxford Circus (3,3) | Score: -7
  Step   8: Move East  | Tooting (5, 2) -> Streatham (5, 3)         | Heading to passenger at Oxford Circus (3,3) | Score: -8
  Step   9: Move North | Streatham (5, 3) -> Vauxhall (4, 3)        | Heading to passenger at Oxford Circus (3,3) | Score: -9
  Step  10: Move North | Vauxhall (4, 3) -> Oxford Circus (3, 3)    | Heading to passenger at Oxford Circus (3,3) | Score: -10
            >>> PICKUP at Oxford Circus (3, 3) | Score +10 -> 0
  Step  11: Move South | Oxford Circus (3, 3) -> Vauxhall (4, 3)    | Passenger aboard — to Greenwich (6,6) | Score: -1
  Step  12: Move South | Vauxhall (4, 3) -> Streatham (5, 3)        | Passenger aboard — to Greenwich (6,6) | Score: -2
  Step  13: Move South | Streatham (5, 3) -> Brixton (6, 3)         | Passenger aboard — to Greenwich (6,6) | Score: -3
  Step  14: Move East  | Brixton (6, 3) -> Lewisham (6, 4)          | Passenger aboard — to Greenwich (6,6) | Score: -4
  Step  15: Move East  | Lewisham (6, 4) -> Blackheath (6, 5)       | Passenger aboard — to Greenwich (6,6) | Score: -5
  Step  16: Move East  | Blackheath (6, 5) -> Greenwich (6, 6)      | Passenger aboard — to Greenwich (6,6) | Score: -6
            >>> DROPOFF at Greenwich (6, 6) | Score +20 -> 14 | GOAL ACHIEVED in 16 steps
--------------------------------------------------------------------------------
  Final Score    : 14
  Total Steps    : 16
  Goal Achieved  : YES
================================================================================
```

**Explanation of Scenario 2 results:**

- **BFS & A\* (16 steps, score 14):** Both correctly detected the barrier and routed south
  all the way to row 5 before turning east and north to reach Oxford Circus. The extra 4 steps
  (compared to Scenario 1) directly reflect the real cost of road closures in London.
- **Why the detour?** Direct routes through the centre were blocked by 5 cells. Both BFS and
  A\* found the only viable path: south along column 0 → east along row 5 → north up column 3.
- **DFS (100 steps, score −110):** Again timed out without reaching the goal.

---

### Scenario 3 – Long Distance Stress Test

| Setting | Value |
|---------|-------|
| Start | King's Cross `(0,0)` |
| Passenger | Brixton `(6,3)` |
| Destination | Stratford `(0,6)` |
| Blocked cells | 6 cells: Euston `(1,0)`, Bloomsbury `(1,1)`, Shoreditch `(1,2)`, Wapping `(3,5)`, Rotherhithe `(4,5)`, New Cross `(5,5)` |

This is the hardest scenario — maximum diagonal distance from start to passenger to destination,
with blocked cells closing off the rightward corridor at column 5.

**Actual output:**
```
============================================================
SCENARIO 3 - Long Distance Stress Test
King's Cross (0,0) -> Brixton (6,3) -> Stratford (0,6)
Tests performance under long-distance travel with scattered blocks
============================================================
  BFS  : Steps= 18 | Score=  12 | Goal=YES
  DFS  : Steps=100 | Score=-110 | Goal=NO
  ASTAR: Steps= 18 | Score=  12 | Goal=YES
```

**Full A\* Journey Log (Scenario 3) — Actual Output:**
```
================================================================================
  JOURNEY EXPLANATION — Strategy: ASTAR
================================================================================
  [INIT] Taxi starts at King's Cross (0, 0) | Strategy: ASTAR | Passenger at: Brixton | Destination: Stratford

  Step   1: Move East  | King's Cross (0, 0) -> Islington (0, 1)      | Moving toward passenger at Brixton (6, 3) | Score: -1
  Step   2: Move East  | Islington (0, 1) -> Hackney (0, 2)           | Moving toward passenger at Brixton (6, 3) | Score: -2
  Step   3: Move East  | Hackney (0, 2) -> Angel (0, 3)               | Moving toward passenger at Brixton (6, 3) | Score: -3
  Step   4: Move South | Angel (0, 3) -> Barbican (1, 3)              | Moving toward passenger at Brixton (6, 3) | Score: -4
  Step   5: Move South | Barbican (1, 3) -> Bank (2, 3)               | Moving toward passenger at Brixton (6, 3) | Score: -5
  Step   6: Move South | Bank (2, 3) -> Oxford Circus (3, 3)          | Moving toward passenger at Brixton (6, 3) | Score: -6
  Step   7: Move South | Oxford Circus (3, 3) -> Vauxhall (4, 3)      | Moving toward passenger at Brixton (6, 3) | Score: -7
  Step   8: Move South | Vauxhall (4, 3) -> Streatham (5, 3)          | Moving toward passenger at Brixton (6, 3) | Score: -8
  Step   9: Move South | Streatham (5, 3) -> Brixton (6, 3)           | Moving toward passenger at Brixton (6, 3) | Score: -9
            >>> PICKUP at Brixton (6, 3) | Score +10 -> 1
  Step  10: Move North | Brixton (6, 3) -> Streatham (5, 3)           | Passenger aboard — driving to Stratford (0, 6) | Score: 0
  Step  11: Move North | Streatham (5, 3) -> Vauxhall (4, 3)          | Passenger aboard — driving to Stratford (0, 6) | Score: -1
  Step  12: Move North | Vauxhall (4, 3) -> Oxford Circus (3, 3)      | Passenger aboard — driving to Stratford (0, 6) | Score: -2
  Step  13: Move North | Oxford Circus (3, 3) -> Bank (2, 3)          | Passenger aboard — driving to Stratford (0, 6) | Score: -3
  Step  14: Move North | Bank (2, 3) -> Barbican (1, 3)               | Passenger aboard — driving to Stratford (0, 6) | Score: -4
  Step  15: Move North | Barbican (1, 3) -> Angel (0, 3)              | Passenger aboard — driving to Stratford (0, 6) | Score: -5
  Step  16: Move East  | Angel (0, 3) -> Bethnal Green (0, 4)         | Passenger aboard — driving to Stratford (0, 6) | Score: -6
  Step  17: Move East  | Bethnal Green (0, 4) -> Bow (0, 5)           | Passenger aboard — driving to Stratford (0, 6) | Score: -7
  Step  18: Move East  | Bow (0, 5) -> Stratford (0, 6)               | Passenger aboard — driving to Stratford (0, 6) | Score: -8
            >>> DROPOFF at Stratford (0, 6) | Score +20 -> 12 | GOAL ACHIEVED in 18 steps
--------------------------------------------------------------------------------
  Final Score    : 12
  Total Steps    : 18
  Goal Achieved  : YES
================================================================================
```

**Explanation of Scenario 3 results:**

- **A\* route logic:** The taxi went east 3 cells then south 6 cells to reach Brixton (9 steps
  to pickup). It then retraced north 6 cells back to row 0, then east 3 cells to Stratford
  (9 more steps). The total 18 steps is geometrically optimal given the blocked column-5
  corridor (Wapping, Rotherhithe, New Cross prevent the shorter rightward path).
- **Why the route looks "retracing"?** Going north along column 3 back to row 0 and then east
  is optimal when column 5 is blocked — there is no shorter unblocked path to Stratford.
- **DFS (100 steps, −110):** Failed again due to the 100-step timeout.

---

### Complete Results Table — All 9 Combinations (Actual Output)

```
================================================================================
FULL ALGORITHM x SCENARIO RESULTS TABLE
================================================================================
            Scenario Algorithm  Steps  Score Goal Achieved
 Scenario 1 (Simple)       BFS     12     18           YES
 Scenario 1 (Simple)       DFS    100   -110            NO
 Scenario 1 (Simple)     ASTAR     12     18           YES
Scenario 2 (Blocked)       BFS     16     14           YES
Scenario 2 (Blocked)       DFS    100   -110            NO
Scenario 2 (Blocked)     ASTAR     16     14           YES
   Scenario 3 (Long)       BFS     18     12           YES
   Scenario 3 (Long)       DFS    100   -110            NO
   Scenario 3 (Long)     ASTAR     18     12           YES

--- Summary: Average Steps per Algorithm ---
                 mean  min  max
Algorithm
ASTAR       15.333333   12   18
BFS         15.333333   12   18
DFS        100.000000  100  100

--- Summary: Average Score per Algorithm ---
                 mean  min  max
Algorithm
ASTAR       14.666667   12   18
BFS         14.666667   12   18
DFS       -110.000000 -110 -110

All 9 algorithm-scenario combinations completed successfully
```

### What the Results Mean

| Finding | Explanation |
|---------|-------------|
| **BFS and A\* are identical in steps and score** | In an unweighted grid, BFS already guarantees the shortest path. A\* finds the exact same route because its Manhattan heuristic is perfectly suited to grid navigation. A\* is more efficient internally (expands fewer nodes) but produces the same final path length. |
| **DFS failed all 3 scenarios** | DFS always hit the 100-step timeout. In a cyclical 7×7 grid, DFS wanders into arbitrarily long paths. Even with a visited set (preventing true infinite loops), the paths it explores are far longer than the 18–20 steps needed. The agent never successfully completes the mission. |
| **Score decreases as distance increases** | Scenario 1: 12 steps → score **18**. Scenario 2: 16 steps → score **14**. Scenario 3: 18 steps → score **12**. Each extra step costs −1, so longer journeys always yield lower scores even when the goal is achieved. |
| **Barrier in Scenario 2 added 4 extra steps** | The 5-cell barrier forced the taxi to go south to row 5, across, then north — adding 4 steps compared to the direct 12-step route of Scenario 1. This models real-world road closure impact on taxi efficiency. |
| **A\* is the recommended algorithm** | Same optimality as BFS, but with fewer internal node expansions in larger grids. A\*-style algorithms are used in real GPS systems (Google Maps, Waymo, etc.) for exactly this reason. |

---

## 7. Section 6 – Explainability

A key requirement for rational AI is **explainability** — the agent must be able to state
*why* it made each decision, building transparency and trust.

The `TaxiAgent` logs every action with five pieces of information:
1. The **step number**
2. The **direction** taken (North/South/East/West)
3. The **origin** and **destination** zone names
4. The **reason** for the decision (heading to passenger vs. heading to destination)
5. The **cumulative score** after the step

**Why the agent makes the decisions it does:**

**1. Rational Decision-Making**
At each step, the agent perceives its current location, passenger location, destination,
and blocked cells. It computes the optimal path using A\*, where every move minimises
`f(n) = g(n) + h(n)`. The agent never moves without a computed reason.

**2. How A\* Navigates Around Blocks**
The Manhattan heuristic guides the agent toward the goal at every step. When a blocked cell
appears on the computed path, A\* automatically reroutes — selecting the next best unblocked
path by re-evaluating `f(n)` for all frontier nodes.

**3. What "Rational" Means in This Context**
The agent always chooses the action that maximises its expected utility. Minimising steps
reduces the −1/step penalty and leads to a higher final score. A\* achieves this optimum
every time, while DFS cannot.

---

## 8. Section 7 – Reflection: Rationality, Fairness & Sustainability

### Rationality

A **rational agent** acts to maximise its expected performance measure given its percepts and
prior knowledge (Russell & Norvig, 2020). The TaxiAgent satisfies this definition:
- It perceives location, passenger, destination, and blocked cells
- It applies the optimal search strategy (A\*)
- It maximises score by always taking the shortest unblocked path

**Real-world limitations:**
- The grid is fully **static** — real roads change dynamically (accidents, roadworks, events)
- The agent has **no sensor noise** — real GPS has positioning uncertainty
- We assume a **single passenger** — real dispatching involves hundreds of concurrent requests
- No **traffic density** model — real routes have variable travel times

### Fairness & Algorithmic Bias

A critical ethical concern arises when blocked cells are not randomly distributed.
In real cities, **road closures and construction are often concentrated in deprived areas**.
If the taxi agent avoids these zones due to systematically blocked cells, it may
**refuse service to passengers there** — creating digital discrimination.

This mirrors documented real-world bias in ride-sharing platforms (e.g., Uber/Lyft in
low-income neighbourhoods). A genuinely fair agent should:
- Penalise algorithms that systematically avoid certain zones
- Implement a **fairness constraint**: serve all areas within equal expected wait times
- **Audit route patterns** across different grid zones for disparate impact

### Sustainability

The `−1 per step` cost directly models **real-world fuel consumption and carbon emissions**.
By finding the shortest path, A\* minimises the carbon footprint of each journey. This aligns
with industry goals for autonomous vehicles (Tesla Autopilot, Waymo) and the UK Government's
2035 zero-emission vehicle target.

**Sustainability improvements for future work:**
- Weight route costs by estimated CO₂ per road segment
- Prefer routes along electric vehicle charging corridors
- Batch multiple passengers into a single optimised multi-stop route

### Ethics

Should an autonomous taxi prioritise **speed** (efficiency) or **fairness** (equal service)?
This is a genuine tension in AI ethics. A purely rational agent maximises its own score —
but a **socially responsible** system must ensure equitable access for all citizens,
regardless of their socio-economic zone.

---

## 9. Section 8 – Conclusion & References

### Conclusion

This project implemented a rational taxi agent in a 7×7 Grid-London environment using the
PEAS framework. Three search algorithms were evaluated across three scenarios of increasing difficulty.

**Key findings:**
- **BFS** and **A\*** consistently found the shortest path in all 3 scenarios
- **DFS** failed all 3 scenarios by hitting the 100-step timeout
- **A\*** is the recommended algorithm for real-world taxi dispatch: optimal paths with
  efficient node exploration and a clear justification via the admissible Manhattan heuristic
- **Blocked cells directly affect scores** — a 5-cell barrier added 4 extra steps and
  reduced the score from 18 to 14 (Scenario 2 vs Scenario 1)

**Future work:**
- Multi-agent extension: 3–5 taxis sharing the grid
- Dynamic obstacle updating during the journey
- Reinforcement learning for adaptive route planning
- Fairness constraints to prevent zone-based service discrimination

### IEEE References

[1] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed.
    Hoboken, NJ: Pearson, 2020.

[2] P. E. Hart, N. J. Nilsson, and B. Raphael, "A formal basis for the heuristic
    determination of minimum cost paths," *IEEE Trans. Syst. Sci. Cybern.*, vol. 4,
    no. 2, pp. 100–107, Jul. 1968. doi: 10.1109/TSSC.1968.300136

[3] J. Pearl, *Heuristics: Intelligent Search Strategies for Computer Problem Solving*.
    Reading, MA: Addison-Wesley, 1984.

[4] J. Ziegler et al., "Making Bertha Drive — An Autonomous Journey on a Historic Route,"
    *IEEE Intell. Transp. Syst. Mag.*, vol. 6, no. 2, pp. 8–20, 2014.
    doi: 10.1109/MITS.2014.2306552

---

## 10. How to Run the Notebook

```bash
# 1. Install dependencies
pip install jupyter matplotlib numpy pandas

# 2. Run interactively in the browser
jupyter notebook taxi_agent.ipynb

# 3. Or execute all cells non-interactively and save outputs
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  taxi_agent.ipynb --output taxi_agent_executed.ipynb
```

The pre-executed notebook with all outputs already embedded is saved as
**`taxi_agent_executed.ipynb`** in this repository — open it in GitHub or Jupyter to see
all results and grid visualisations without needing to run anything.
