# 🚚 Multi-Constraint Last-Mile Delivery Optimizer
### DAA Experiential Learning Project — RV College of Engineering

---

## 📌 Problem Statement
Solve the **Vehicle Routing Problem with Time Windows (VRPTW)** — an NP-Hard
combinatorial optimisation problem directly applicable to hyperlocal delivery
platforms (Blinkit, Zepto, Swiggy Instamart).

Three constraints are handled simultaneously:
| Constraint | Description |
|---|---|
| ⏰ Time Windows | Each customer must be visited within their delivery slot |
| 📦 Vehicle Capacity | Each agent has a max weight limit |
| ⛽ Distance Cost | Minimise total travel distance across all agents |

---

## ⚙️ Algorithms Implemented

| # | Algorithm | Complexity | Notes |
|---|---|---|---|
| 1 | Greedy Nearest Neighbor | O(n²) | Fast, suboptimal |
| 2 | Branch & Bound | O(n!) worst case | Exact optimal, ≤12 nodes |
| 3 | Clarke-Wright Savings | O(n² log n) | Good approximation |
| 4 | **Clarke-Wright + 2-opt ★** | O(n² log n + n²) | Best quality-speed balance |

---

## 🗂️ Project Structure
```
vrp_project/
├── app.py             ← Main Dash dashboard (run this)
├── algorithms.py      ← All 4 VRPTW algorithms
├── data_generator.py  ← Random grid + Bengaluru locality data
├── requirements.txt   ← Python dependencies
└── README.md
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
python app.py
```

### Step 3 — Open in browser
```
http://127.0.0.1:8050
```

---

## 🖥️ Dashboard Features
- **Two data sources** — Random grid or real Bengaluru locality coordinates
- **Configurable parameters** — delivery points (5–20), agents (2–5), capacity
- **Algorithm selector** — run any combination of the 4 algorithms
- **Interactive route map** — switch between algorithms to compare routes visually
- **Comparison charts** — total distance and computation time side by side
- **Results panel** — per-algorithm summary with unserved customer count

---

## 👥 Team
| Role | Tasks |
|---|---|
| Jeevan R Chavan | Greedy NN · Branch & Bound · Complexity Analysis |
| Jeevith R Chavan | Clarke-Wright · 2-opt · Dashboard · Bengaluru Data |

---

## 📚 References
1. Dantzig, G. B., & Ramser, J. H. (1959). The Truck Dispatching Problem. *Management Science*.
2. Clarke, G., & Wright, J. W. (1964). Scheduling of Vehicles from a Central Depot. *Operations Research*.
3. Lin, S. (1965). Computer Solutions of the Traveling Salesman Problem. *Bell System Technical Journal*.
