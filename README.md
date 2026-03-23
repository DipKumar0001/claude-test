# taxi_agent.ipynb — Description, Execution & Results

**Module:** CMP-N206 – Artificial Intelligence Coursework 1  
**Author:** Dipesh Kumar Yadav (A00038176)  
**Date:** 12th March 2026  

---

## What the Notebook Does

`taxi_agent.ipynb` implements a **rational taxi agent** that navigates a 7×7 grid map of London. The agent must pick up a passenger from one location and drop them off at a destination, avoiding blocked cells (representing road closures). Three classical AI search algorithms — **BFS**, **DFS**, and **A\*** — are implemented, compared, and evaluated across three test scenarios.

---

## Table of Contents

1. [Environment Setup & Imports](#section-1--environment-setup--imports)
2. [Grid-London Environment Model](#section-2--grid-london-environment-model)
3. [Agent Design & PEAS Framework](#section-3--agent-design--peas-framework)
4. [Search Algorithms](#section-4--search-algorithms)
5. [Evaluation Scenarios & Results](#section-5--evaluation-scenarios--results)
6. [Explainability](#section-6--explainability)
7. [Reflection: Rationality, Fairness & Sustainability](#section-7--reflection)

---

## Section 1 – Environment Setup & Imports

The notebook imports the following standard Python libraries:

| Library | Purpose |
|---------|---------|
| `collections.deque` | FIFO queue for BFS |
| `heapq` | Min-heap / priority queue for A* |
| `matplotlib` + `numpy` | Grid visualisation |
| `pandas` | Results comparison table |
| `datetime` | Timestamps |

**Output:** `All libraries imported successfully ✓`

---

## Section 2 – Grid-London Environment Model

### The `GridLondon` Class

A 7×7 grid where every cell maps to a real London location:

| Grid (row, col) | London Zone |
|-----------------|-------------|
| (0, 0) | King's Cross |
| (0, 3) | Angel |
| (0, 6) | Stratford |
| (2, 3) | Bank |
| (3, 0) | Hammersmith |
| (3, 3) | Oxford Circus |
| (3, 6) | Canary Wharf |
| (6, 0) | Heathrow |
| (6, 3) | Brixton |
| (6, 6) | Greenwich |

All 49 cells have real London zone names. **Blocked cells** (dark on the map) represent road closures or traffic jams that the taxi cannot enter.

### Environment Classification (REAS)

| Property | Value | Reason |
|----------|-------|--------|
| Observable | Partially | Agent sees grid state but not future passengers |
| Deterministic | Yes | Actions always produce the same outcome |
| Episodic | No (Sequential) | Each step depends on the previous state |
| Static | Yes | Grid does not change while the agent runs |
| Discrete | Yes | Cells, not continuous space |
| Single-agent | Yes | One taxi only |

The class provides:
- `get_neighbours(cell)` — returns valid adjacent cells (N/S/E/W, excluding blocked)
- `get_name(cell)` — returns the London zone name for a grid position
- `visualise(...)` — renders the grid with matplotlib (taxi = blue triangle, passenger = green circle, destination = red star, blocked = dark grey)

---

## Section 3 – Agent Design & PEAS Framework

### PEAS Analysis

| Component | Description |
|-----------|-------------|
| **Performance** | +20 for drop-off, +10 for pickup, −1 per step, −5 per blocked attempt, −10 per timeout |
| **Environment** | 7×7 Grid-London; partially observable, deterministic, sequential |
| **Actuators** | move_north, move_south, move_east, move_west, pickup, dropoff |
| **Sensors** | current_location, passenger_location, destination, blocked_cells |

### Agent Type

The `TaxiAgent` is a **goal-based rational agent**. It:
1. Perceives its environment (`perceive()`)
2. Selects a search strategy (BFS, DFS, or A*)
3. Follows the computed path step-by-step
4. Logs every action with full explainability

### Utility (Scoring) Function

| Event | Score Change |
|-------|-------------|
| Successful drop-off | +20 |
| Passenger pickup | +10 |
| Each step taken | −1 |
| Blocked cell attempt | −5 |
| Exceeding 100-step limit | Timeout, −110 total |

---

## Section 4 – Search Algorithms

Three algorithms are implemented and compared:

| Algorithm | Optimal? | Complete? | Space Complexity | Notes |
|-----------|----------|-----------|-----------------|-------|
| BFS | ✅ Yes | ✅ Yes | O(b^d) | Layer-by-layer exploration |
| DFS | ❌ No | ❌ No* | O(b·m) | Explores deeply first; may find very long paths |
| A* | ✅ Yes | ✅ Yes | O(b^d) best-case | Guided by Manhattan distance heuristic |

*DFS uses a visited set to prevent infinite loops in cyclic grids.

### Manhattan Distance Heuristic (A*)

```
h(n) = |row_n − row_goal| + |col_n − col_goal|
```

This heuristic is **admissible** because:
- The taxi can only move in 4 directions (no diagonals).
- The Manhattan distance is always ≤ the true path length.
- Therefore `h(n) ≤ h*(n)`, satisfying the admissibility condition (Hart et al., 1968).

### Quick Test Results

```
BFS Test: (0,0) → (0,3) | Path: [(0,0),(0,1),(0,2),(0,3)] | Length: 4 steps
DFS Test: (0,0) → (0,3) | Path: 24-step winding route      | Length: 24 steps
A*  Test: (0,0) → (0,3) | Path: [(0,0),(0,1),(0,2),(0,3)] | Length: 4 steps
```

The quick test immediately demonstrates DFS's core weakness: it found a **24-step path** where the optimal is just 4 steps.

---

## Section 5 – Evaluation Scenarios & Results

Three scenarios test increasing difficulty.

---

### Scenario 1 – Simple Route

**Route:** King's Cross `(0,0)` → Bank `(2,3)` → Greenwich `(6,6)`  
**Blocked cells:** 4 scattered cells (Bloomsbury, Shoreditch, The City, Rotherhithe)

```
============================================================
SCENARIO 1 - Simple Route
King's Cross (0,0) -> Bank (2,3) -> Greenwich (6,6)
============================================================
  BFS  : Steps= 12 | Score=  18 | Goal=YES
  DFS  : Steps=100 | Score=-110 | Goal=NO
  ASTAR: Steps= 12 | Score=  18 | Goal=YES
```

**A* Step-by-Step Journey (Scenario 1):**

| Step | Move | From → To | Notes |
|------|------|-----------|-------|
| 1 | East | King's Cross → Islington | Heading to passenger |
| 2 | East | Islington → Hackney | Heading to passenger |
| 3 | East | Hackney → Angel | Heading to passenger |
| 4 | South | Angel → Barbican | Heading to passenger |
| 5 | South | Barbican → Bank | **PICKUP** (+10, Score: 5) |
| 6 | East | Bank → Liverpool St | With passenger, to Greenwich |
| 7 | East | Liverpool St → Stepney | With passenger |
| 8 | East | Stepney → Plaistow | With passenger |
| 9 | South | Plaistow → Canary Wharf | With passenger |
| 10 | South | Canary Wharf → Isle of Dogs | With passenger |
| 11 | South | Isle of Dogs → Deptford | With passenger |
| 12 | South | Deptford → Greenwich | **DROPOFF** (+20, Final Score: 18) ✅ |

**Why DFS failed:** DFS explored 100 steps without finding the goal because it wanders into long dead-end paths. It hit the timeout limit (-1 × 100 steps = −100, plus no pickup bonus = −110).

---

### Scenario 2 – Blocked Route (Forced Detour)

**Route:** King's Cross `(0,0)` → Oxford Circus `(3,3)` → Greenwich `(6,6)`  
**Blocked cells:** 5 cells forming a barrier: Covent Garden `(2,2)`, Bank `(2,3)`, Liverpool St `(2,4)`, Westminster `(3,2)`, Chelsea `(4,2)`  
**Challenge:** The direct path through the middle of the grid is entirely blocked, forcing the agent south around the barrier.

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

**BFS Journey (Scenario 2 — 16 steps):**

The barrier forces the taxi to detour south around the blocked zone:
- Taxi goes: King's Cross → Euston → Paddington → Hammersmith → Chiswick → Richmond → Wimbledon → Tooting → Streatham → Vauxhall → **Oxford Circus** (PICKUP)
- Then continues: Vauxhall → Streatham → Brixton → Lewisham → Blackheath → **Greenwich** (DROPOFF)

Both BFS and A* find the same 16-step detour route. The scenario confirms both algorithms correctly navigate around barriers — the extra 4 steps (vs. Scenario 1) reflect the real cost of road closures.

---

### Scenario 3 – Long Distance Stress Test

**Route:** King's Cross `(0,0)` → Brixton `(6,3)` → Stratford `(0,6)`  
**Blocked cells:** 6 scattered cells: Euston `(1,0)`, Bloomsbury `(1,1)`, Shoreditch `(1,2)`, Wapping `(3,5)`, Rotherhithe `(4,5)`, New Cross `(5,5)`  
**Challenge:** Maximum diagonal distance across the grid, with blocks forcing the agent away from the direct column-5 corridor.

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

**Full A* Journey (Scenario 3 — 18 steps):**

```
Step  1:  East   King's Cross (0,0)  → Islington (0,1)     [To passenger]
Step  2:  East   Islington (0,1)     → Hackney (0,2)        [To passenger]
Step  3:  East   Hackney (0,2)       → Angel (0,3)          [To passenger]
Step  4:  South  Angel (0,3)         → Barbican (1,3)       [To passenger]
Step  5:  South  Barbican (1,3)      → Bank (2,3)           [To passenger]
Step  6:  South  Bank (2,3)          → Oxford Circus (3,3)  [To passenger]
Step  7:  South  Oxford Circus (3,3) → Vauxhall (4,3)       [To passenger]
Step  8:  South  Vauxhall (4,3)      → Streatham (5,3)      [To passenger]
Step  9:  South  Streatham (5,3)     → Brixton (6,3)        [PICKUP +10, Score: 1]
Step 10:  North  Brixton (6,3)       → Streatham (5,3)      [With passenger]
Step 11:  North  Streatham (5,3)     → Vauxhall (4,3)       [With passenger]
Step 12:  North  Vauxhall (4,3)      → Oxford Circus (3,3)  [With passenger]
Step 13:  North  Oxford Circus (3,3) → Bank (2,3)           [With passenger]
Step 14:  North  Bank (2,3)          → Barbican (1,3)       [With passenger]
Step 15:  North  Barbican (1,3)      → Angel (0,3)          [With passenger]
Step 16:  East   Angel (0,3)         → Bethnal Green (0,4)  [With passenger]
Step 17:  East   Bethnal Green (0,4) → Bow (0,5)            [With passenger]
Step 18:  East   Bow (0,5)           → Stratford (0,6)      [DROPOFF +20, Score: 12] ✅
```

**Note on the route:** A* correctly avoids the blocked column-5 corridor (Wapping, Rotherhithe, New Cross) and instead routes along column 3 north to row 0, then east to Stratford. The path mirrors the taxi literally retracing most of its steps — this is optimal given the grid geometry and the blocked cells.

---

### Full Results Summary

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
ASTAR         15.33   12   18
BFS           15.33   12   18
DFS          100.00  100  100

--- Summary: Average Score per Algorithm ---
               mean  min  max
Algorithm
ASTAR         14.67   12   18
BFS           14.67   12   18
DFS          -110.0 -110 -110
```

### What the Results Tell Us

| Finding | Explanation |
|---------|-------------|
| **BFS and A\* are identical in steps and score** | Both find the shortest path. In an unweighted grid, BFS is already optimal — A\* finds the same route using fewer node expansions thanks to the heuristic. |
| **DFS failed in all 3 scenarios** | DFS always hit the 100-step timeout. In a 7×7 grid with cyclical connections, DFS explores arbitrarily long paths. The visited-set prevents infinite loops but doesn't prevent wasted exploration. |
| **Score decreases with distance** | Scenario 1 (12 steps → score 18), Scenario 2 (16 steps → score 14), Scenario 3 (18 steps → score 12). The −1/step penalty means longer journeys always yield lower scores. |
| **The barrier in Scenario 2 added 4 extra steps** | Without the barrier the optimal route to Oxford Circus would be ~6 steps; the detour forces 10 steps to reach the passenger (4 extra). |

---

## Section 6 – Explainability

The `TaxiAgent` logs every action:
- Step number and direction taken
- Origin and destination zone names
- The reason for each decision (heading to passenger vs. heading to destination)
- Cumulative score after each step

This demonstrates **transparent AI**: any auditor can verify exactly why the taxi chose each route.

### Why the Agent Makes Its Choices

1. **Rational Decision-Making:** At each step, the agent perceives its location, passenger, destination, and blocked cells. It computes the optimal path using A* (where `f(n) = g(n) + h(n)`). Every move minimises the total estimated cost.

2. **How A\* Navigates Blocks:** The Manhattan heuristic guides the agent toward the goal. When a block appears in the computed path, A\* automatically reroutes around it by selecting the next best unblocked node.

3. **What "Rational" Means Here:** The agent maximises its utility. Minimising steps reduces the −1/step cost and leads to a higher final score. A\* guarantees this optimum every time.

---

## Section 7 – Reflection

### Rationality

The TaxiAgent is rational because it:
- Perceives its environment (location, passenger, destination, blocks)
- Applies the optimal search strategy
- Maximises score by always taking the shortest unblocked path

**Limitations in the real world:**
- The grid is static — real roads change dynamically (accidents, roadworks)
- No uncertainty about blocked cells — real GPS has sensing noise
- Single passenger — real dispatching involves concurrent requests

### Fairness & Algorithmic Bias

If blocked cells are concentrated in certain areas (e.g., deprived neighbourhoods), the agent will systematically avoid those zones — refusing service to passengers there. This mirrors documented real-world bias in ride-sharing platforms. A fair agent should:
- Penalise routes that systematically avoid certain zones
- Implement fairness constraints (equal expected wait times across areas)
- Audit route patterns for disparate impact

### Sustainability

The −1/step cost incentivises fuel efficiency, mirroring carbon-footprint considerations for real taxi fleets. A\*'s optimal routing minimises unnecessary travel and indirectly reduces emissions.

---

## How to Run the Notebook

```bash
# Install dependencies
pip install jupyter matplotlib numpy pandas

# Run the notebook interactively
jupyter notebook taxi_agent.ipynb

# Or execute non-interactively and save outputs
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  taxi_agent.ipynb --output taxi_agent_executed.ipynb
```

The executed notebook with all outputs is saved as `taxi_agent_executed.ipynb`.
