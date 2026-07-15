"""
data_generator.py
=================
Two data sources for the VRPTW solver:
  - Random grid  : synthetic delivery points mapped onto Bengaluru's lat/lng bounding box
  - Bengaluru    : delivery points centred around a real Bengaluru depot

Each returned dict contains both:
  locations  : (x, y) used by the algorithms
  lat_lngs   : (lat, lng) used by the Leaflet map
"""

import math
import random

# Bengaluru bounding box for projecting random grid data onto a real map
_LAT_MIN, _LAT_MAX = 12.850, 13.100
_LNG_MIN, _LNG_MAX = 77.450, 77.750

# Bengaluru depot and nearby extra depots
_DEPOT_COORDS = ("Pattanagere Depot", 12.9230, 77.5718)
_EXTRA_DEPOTS = [
    ("Pattanagere Hub A", 12.9236, 77.5732),
    ("Pattanagere Hub B", 12.9223, 77.5701),
    ("Pattanagere Hub C", 12.9241, 77.5690),
]
_PATTANAGERE_SPREAD = 0.0010


def _sample_houses(center_lat, center_lng, count, rng, spread=_PATTANAGERE_SPREAD):
    houses = []
    for _ in range(count):
        dlat = rng.uniform(-spread, spread)
        dlng = rng.uniform(-spread, spread)
        houses.append((center_lat + dlat, center_lng + dlng))
    return houses


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _lat_lng_to_xy(lat, lng, base_lat=12.9767, base_lng=77.5713, scale=111.0):
    x = (lng - base_lng) * scale * math.cos(math.radians(base_lat))
    y = (lat - base_lat) * scale
    return (round(x, 4), round(y, 4))


def _xy_to_latlng(x, y, grid_size=100):
    lat = _LAT_MIN + (y / grid_size) * (_LAT_MAX - _LAT_MIN)
    lng = _LNG_MIN + (x / grid_size) * (_LNG_MAX - _LNG_MIN)
    return (round(lat, 6), round(lng, 6))


def _latlng_to_xy(lat, lng, grid_size=100):
    x = ((lng - _LNG_MIN) / (_LNG_MAX - _LNG_MIN)) * grid_size
    y = ((lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)) * grid_size
    return (round(x, 4), round(y, 4))


def _make_rng(seed=None):
    if seed is None:
        return random.Random()
    return random.Random(int(seed))


def _make_time_windows(locations, num_customers, rng, spread_factor=0.6):
    depot = locations[0]
    time_windows = [(0, 1e6)]
    for i in range(1, num_customers + 1):
        d = _euclidean(depot, locations[i])
        jitter = rng.uniform(-0.2, 0.2) * d
        tw_open = max(0.0, d * (1 - spread_factor) + jitter)
        tw_close = d * (1 + spread_factor * 2) + abs(jitter) * 2 + 1.0
        time_windows.append((round(tw_open, 2), round(tw_close, 2)))
    return time_windows


# Public API

def _resolve_center(start_coord=None, end_coord=None):
    if start_coord is not None:
        return start_coord
    if end_coord is not None:
        return end_coord

    _, center_lat, center_lng = _DEPOT_COORDS
    return center_lat, center_lng


def get_bengaluru_data(num_customers=10, num_vehicles=3, capacity=50, seed=None, start_coord=None, end_coord=None):
    num_customers = max(1, num_customers)
    rng = _make_rng(seed)

    depot_lat, depot_lng = _DEPOT_COORDS[1], _DEPOT_COORDS[2]
    depot_name = "Pattanagere Depot"

    if start_coord is not None:
        delivery_lat, delivery_lng = start_coord
    elif end_coord is not None:
        delivery_lat, delivery_lng = end_coord
    else:
        delivery_lat, delivery_lng = depot_lat, depot_lng

    spread = 0.0010
    house_points = _sample_houses(depot_lat, depot_lng, num_customers, rng, spread=spread)

    if (delivery_lat, delivery_lng) != (depot_lat, depot_lng):
        house_points = [(delivery_lat, delivery_lng)] + house_points[: max(0, num_customers - 1)]
    else:
        house_points = house_points[:num_customers]

    lat_lngs = [(depot_lat, depot_lng)] + house_points
    locations = [_lat_lng_to_xy(lat, lng) for lat, lng in lat_lngs]
    names = [depot_name] + [f"House {i + 1}" for i in range(num_customers)]
    demands = [0] + [rng.randint(1, 5) for _ in range(num_customers)]
    time_windows = _make_time_windows(locations, num_customers, rng)

    extra_depots = [
        {
            "name": f"{depot_name} Hub {i}",
            "lat_lng": (depot_lat + offset_lat, depot_lng + offset_lng),
            "location": _lat_lng_to_xy(depot_lat + offset_lat, depot_lng + offset_lng),
        }
        for i, (offset_lat, offset_lng) in enumerate([(0.0006, 0.0004), (-0.0004, 0.0002), (0.0002, -0.0005)], start=1)
    ]
    depots = [
        {"name": depot_name, "lat_lng": (depot_lat, depot_lng), "location": _lat_lng_to_xy(depot_lat, depot_lng)}
    ] + extra_depots

    return {
        "locations": locations,
        "lat_lngs": lat_lngs,
        "names": names,
        "demands": demands,
        "time_windows": time_windows,
        "capacities": [capacity] * num_vehicles,
        "num_vehicles": num_vehicles,
        "source": "Bengaluru",
        "depots": depots,
        "extra_depots": extra_depots,
    }


def get_random_data(num_customers=15, num_vehicles=3, capacity=50, seed=None, grid_size=100, center_coord=None):
    rng = _make_rng(seed)

    depot_xy = (grid_size // 2, grid_size // 2)
    depot_lat_lng = _xy_to_latlng(depot_xy[0], depot_xy[1], grid_size)

    cust_xys = []
    for _ in range(num_customers):
        x = rng.randint(max(5, depot_xy[0] - 12), min(grid_size - 5, depot_xy[0] + 12))
        y = rng.randint(max(5, depot_xy[1] - 12), min(grid_size - 5, depot_xy[1] + 12))
        cust_xys.append((x, y))

    if center_coord is not None:
        center_lat, center_lng = center_coord
        delivery_xy = _latlng_to_xy(center_lat, center_lng, grid_size)
        delivery_xy = (max(5, min(grid_size - 5, int(round(delivery_xy[0])))), max(5, min(grid_size - 5, int(round(delivery_xy[1])))))
        cust_xys = [delivery_xy] + cust_xys[: max(0, num_customers - 1)]
        delivery_lat_lng = (round(center_lat, 6), round(center_lng, 6))
    else:
        delivery_lat_lng = None

    all_xys = [depot_xy] + cust_xys

    locations = [(float(x), float(y)) for x, y in all_xys]
    lat_lngs = [depot_lat_lng] + [delivery_lat_lng if delivery_lat_lng is not None and idx == 0 else _xy_to_latlng(x, y, grid_size)
                                  for idx, (x, y) in enumerate(cust_xys)]
    names = ["Depot (Hub)"] + [f"Customer {i + 1}" for i in range(num_customers)]
    demands = [0] + [rng.randint(5, 20) for _ in range(num_customers)]
    time_windows = _make_time_windows(locations, num_customers, rng)

    extra_depots = []
    for i, (dx, dy) in enumerate([(-8, -8), (8, 8), (-8, 8)], start=1):
        x = grid_size // 2 + dx
        y = grid_size // 2 + dy
        extra_depots.append({
            "name": f"Hub {i}",
            "lat_lng": _xy_to_latlng(x, y, grid_size),
            "location": (float(x), float(y)),
        })
    depots = [
        {"name": "Depot (Hub)", "lat_lng": lat_lngs[0], "location": locations[0]}
    ] + extra_depots

    return {
        "locations": locations,
        "lat_lngs": lat_lngs,
        "names": names,
        "demands": demands,
        "time_windows": time_windows,
        "capacities": [capacity] * num_vehicles,
        "num_vehicles": num_vehicles,
        "source": "Random Grid",
        "depots": depots,
        "extra_depots": extra_depots,
    }
