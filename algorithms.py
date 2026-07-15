"""
algorithms.py
=============
Four algorithms for solving the Vehicle Routing Problem with Time Windows (VRPTW):
  1. Greedy Nearest Neighbor
  2. Branch and Bound  (exact, practical only for <= 12 nodes)
  3. Clarke-Wright Savings
  4. Clarke-Wright + 2-opt Local Search  (recommended hybrid)

All algorithms accept the same inputs and return a uniform result dictionary.
"""

import math
import time
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_dist_matrix(locations):
    n = len(locations)
    return [[euclidean(locations[i], locations[j]) for j in range(n)] for i in range(n)]


def route_distance(route, dist, depot=0):
    """Total distance of a route: depot -> c1 -> c2 -> ... -> depot"""
    if not route:
        return 0.0
    total = dist[depot][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][depot]
    return total


def route_time_feasible(route, dist, time_windows, speed=1.0):
    """Returns True if the route satisfies all time-window constraints."""
    current_time = 0.0
    prev = 0
    for node in route:
        current_time += dist[prev][node] / speed
        tw_open, tw_close = time_windows[node]
        if current_time > tw_close:
            return False
        if current_time < tw_open:
            current_time = tw_open          # wait until window opens
        prev = node
    return True


def build_result(algorithm_name, routes, dist, elapsed, unserved=0):
    total = sum(route_distance(r, dist) for r in routes)
    return {
        "algorithm":       algorithm_name,
        "routes":          routes,
        "total_distance":  round(total, 3),
        "time_ms":         round(elapsed * 1000, 3),
        "unserved":        unserved,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 1 — GREEDY NEAREST NEIGHBOR
# ─────────────────────────────────────────────────────────────────────────────

def greedy_nearest_neighbor(locations, demands, capacities, time_windows, num_vehicles):
    """
    At every step assign the nearest unvisited customer that fits
    capacity and time-window constraints. O(n^2) per vehicle.
    """
    dist = build_dist_matrix(locations)
    n = len(locations) - 1          # customers only (depot = index 0)
    unvisited = set(range(1, n + 1))
    routes = []

    t0 = time.time()
    for v in range(num_vehicles):
        if not unvisited:
            break
        route, load, curr, curr_time = [], 0, 0, 0.0
        cap = capacities[v]

        while unvisited:
            best, best_d = None, float("inf")
            for node in unvisited:
                if load + demands[node] > cap:
                    continue
                arrival = curr_time + dist[curr][node]
                if arrival > time_windows[node][1]:
                    continue
                if dist[curr][node] < best_d:
                    best_d, best = dist[curr][node], node

            if best is None:
                break

            arrival = curr_time + dist[curr][best]
            curr_time = max(arrival, time_windows[best][0])
            route.append(best)
            load += demands[best]
            unvisited.remove(best)
            curr = best

        routes.append(route)

    # pad remaining vehicles with empty routes
    while len(routes) < num_vehicles:
        routes.append([])

    elapsed = time.time() - t0
    return build_result("Greedy Nearest Neighbor", routes, dist, elapsed, len(unvisited))


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 2 — BRANCH AND BOUND
# ─────────────────────────────────────────────────────────────────────────────

BNB_NODE_LIMIT = 12        # Fall back to greedy beyond this size
BNB_TIME_LIMIT = 10.0      # seconds


def branch_and_bound(locations, demands, capacities, time_windows, num_vehicles):
    """
    Exact optimal solution via branch-and-bound pruning.
    Practical only for small instances (≤ 12 customers).
    Falls back to greedy for larger inputs.
    """
    n = len(locations) - 1
    if n > BNB_NODE_LIMIT:
        result = greedy_nearest_neighbor(locations, demands, capacities, time_windows, num_vehicles)
        result["algorithm"] = f"Branch & Bound (fell back — {n} nodes > {BNB_NODE_LIMIT} limit)"
        return result

    dist = build_dist_matrix(locations)
    customers = list(range(1, n + 1))
    best = {"cost": float("inf"), "routes": None}
    t0 = time.time()

    def lower_bound(remaining, partial_cost):
        if not remaining:
            return 0.0
        # LB = cheapest edge from depot to any remaining node
        return min(dist[0][c] for c in remaining)

    def backtrack(routes, cur_route, load, curr_time, curr_pos, remaining, v_idx):
        if time.time() - t0 > BNB_TIME_LIMIT:
            return

        if not remaining:
            cost = sum(route_distance(r, dist) for r in routes) + route_distance(cur_route, dist)
            if cost < best["cost"]:
                best["cost"] = cost
                best["routes"] = [r[:] for r in routes] + [cur_route[:]]
            return

        partial = sum(route_distance(r, dist) for r in routes) + route_distance(cur_route, dist)
        if partial + lower_bound(remaining, partial) >= best["cost"]:
            return  # prune

        for node in sorted(remaining, key=lambda x: dist[curr_pos][x]):
            arrival = curr_time + dist[curr_pos][node]
            if (load + demands[node] <= capacities[v_idx] and
                    arrival <= time_windows[node][1]):
                actual_time = max(arrival, time_windows[node][0])
                remaining.remove(node)
                cur_route.append(node)
                backtrack(routes, cur_route, load + demands[node],
                          actual_time, node, remaining, v_idx)
                cur_route.pop()
                remaining.add(node)

        # Start a new vehicle if vehicles remain
        if v_idx + 1 < num_vehicles and remaining:
            routes.append(cur_route[:])
            backtrack(routes, [], 0, 0.0, 0, remaining, v_idx + 1)
            routes.pop()

    backtrack([], [], 0, 0.0, 0, set(customers), 0)
    elapsed = time.time() - t0

    if best["routes"] is None:
        result = greedy_nearest_neighbor(locations, demands, capacities, time_windows, num_vehicles)
        result["algorithm"] = "Branch & Bound (no feasible solution — greedy used)"
        return result

    final_routes = [r for r in best["routes"] if r]
    while len(final_routes) < num_vehicles:
        final_routes.append([])
    return build_result("Branch & Bound", final_routes, dist, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 3 — CLARKE-WRIGHT SAVINGS
# ─────────────────────────────────────────────────────────────────────────────

def clarke_wright(locations, demands, capacities, time_windows, num_vehicles):
    """
    Clarke-Wright (1964) parallel savings algorithm.
    Savings: S(i,j) = d(0,i) + d(0,j) - d(i,j)
    Iteratively merges pairs of routes to maximise savings.
    """
    dist = build_dist_matrix(locations)
    n = len(locations) - 1
    t0 = time.time()

    # Start: one route per customer  {route_id: [customers]}
    routes  = {i: [i] for i in range(1, n + 1)}
    loads   = {i: demands[i] for i in range(1, n + 1)}
    head_of = {i: i for i in range(1, n + 1)}   # first customer of route
    tail_of = {i: i for i in range(1, n + 1)}   # last customer of route
    route_of = {i: i for i in range(1, n + 1)}  # customer -> route_id

    # Compute and sort savings descending
    savings = sorted(
        [(dist[0][i] + dist[0][j] - dist[i][j], i, j)
         for i in range(1, n + 1) for j in range(i + 1, n + 1)],
        reverse=True
    )

    min_cap = min(capacities[:num_vehicles])

    for s, i, j in savings:
        ri, rj = route_of.get(i), route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        # i must be tail of ri and j must be head of rj (or reversed)
        merge_order = None
        if tail_of[ri] == i and head_of[rj] == j:
            merge_order = (ri, rj, False)
        elif tail_of[rj] == j and head_of[ri] == i:
            merge_order = (rj, ri, False)
        elif tail_of[ri] == i and tail_of[rj] == j:
            merge_order = (ri, rj, True)    # reverse rj
        elif head_of[ri] == i and head_of[rj] == j:
            merge_order = (rj, ri, True)    # reverse ri (= prepend reversed)

        if merge_order is None:
            continue

        lead_id, follow_id, reverse_follow = merge_order

        # Capacity check
        if loads[lead_id] + loads[follow_id] > min_cap:
            continue

        follow_route = routes[follow_id]
        if reverse_follow:
            follow_route = follow_route[::-1]

        merged = routes[lead_id] + follow_route

        # Time-window feasibility check
        if not route_time_feasible(merged, dist, time_windows):
            continue

        # Commit merge
        for node in follow_route:
            route_of[node] = lead_id
        loads[lead_id] += loads[follow_id]
        routes[lead_id] = merged
        head_of[lead_id] = merged[0]
        tail_of[lead_id] = merged[-1]
        del routes[follow_id]
        del loads[follow_id]

    final_routes = list(routes.values())
    while len(final_routes) < num_vehicles:
        final_routes.append([])
    final_routes = final_routes[:num_vehicles]

    elapsed = time.time() - t0
    return build_result("Clarke-Wright Savings", final_routes, dist, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 4 — CLARKE-WRIGHT + 2-OPT LOCAL SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _two_opt(route, dist):
    """Improve a single route using 2-opt edge swaps."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                if route_distance(candidate, dist) < route_distance(best, dist):
                    best = candidate
                    improved = True
    return best


def clarke_wright_two_opt(locations, demands, capacities, time_windows, num_vehicles):
    """
    Run Clarke-Wright Savings, then refine every route with 2-opt.
    This is the recommended hybrid algorithm.
    """
    dist = build_dist_matrix(locations)
    t0 = time.time()

    cw = clarke_wright(locations, demands, capacities, time_windows, num_vehicles)
    improved_routes = [
        _two_opt(r, dist) if len(r) > 2 else r
        for r in cw["routes"]
    ]

    elapsed = time.time() - t0
    return build_result("Clarke-Wright + 2-opt", improved_routes, dist, elapsed)
