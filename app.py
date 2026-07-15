"""
app.py
======
Dash dashboard with:
  - Real Leaflet.js map (OpenStreetMap tiles, no API key needed)
  - Routes drawn as coloured polylines
  - Delivery stops as circle markers with tooltips
  - Algorithm comparison charts (Plotly)

Run:  python app.py  →  open http://127.0.0.1:8050
"""

import json
import math
import os
import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

from algorithms import (
    greedy_nearest_neighbor, branch_and_bound,
    clarke_wright, clarke_wright_two_opt,
)
from data_generator import get_bengaluru_data, get_random_data

# ── Constants ─────────────────────────────────────────────────────────────────

VEHICLE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

ALGO_MAP = {
    "greedy": ("greedy",  "Greedy Nearest Neighbor",  greedy_nearest_neighbor),
    "bnb":    ("bnb",     "Branch & Bound",           branch_and_bound),
    "cw":     ("cw",      "Clarke-Wright Savings",    clarke_wright),
    "cw2opt": ("cw2opt",  "Clarke-Wright + 2-opt ★",  clarke_wright_two_opt),
}

COMPLEXITY_MAP = {
    "greedy": "O(n²)",
    "bnb": "O(2^n)",
    "cw": "O(n² log n)",
    "cw2opt": "O(n²)",
}

MAP_CENTER  = [12.9767, 77.5713]
MAP_ZOOM    = 12

CARD_STYLE   = {"borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}
HEADER_STYLE = {
    "backgroundColor": "#2c3e50", "color": "white",
    "borderRadius": "10px 10px 0 0", "padding": "10px 16px",
}

# ── App init ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="VRP Optimizer — DAA EL Project",
)

LAST_PAYLOAD = None


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _assign_customers_to_nearest_depot(data):
    depots = data.get("depots") or []
    if not depots:
        return [], []

    assignments = [[] for _ in depots]
    for cust_idx in range(1, len(data["locations"])):
        cust_loc = data["locations"][cust_idx]
        nearest = min(range(len(depots)), key=lambda d: _euclidean(cust_loc, depots[d]["location"]))
        assignments[nearest].append(cust_idx)
    return depots, assignments


def _allocate_vehicles_to_depots(customer_counts, num_vehicles):
    alloc = [1 if count > 0 else 0 for count in customer_counts]
    active_depots = sum(1 for count in customer_counts if count > 0)
    if active_depots == 0:
        return [0] * len(customer_counts)

    if num_vehicles < active_depots:
        alloc = [0] * len(customer_counts)
        sorted_depots = sorted(range(len(customer_counts)), key=lambda d: customer_counts[d], reverse=True)
        for idx in sorted_depots[:num_vehicles]:
            alloc[idx] = 1
        return alloc

    remaining = num_vehicles - sum(alloc)
    while remaining > 0:
        best = max(range(len(customer_counts)), key=lambda d: customer_counts[d] / max(1, alloc[d]))
        alloc[best] += 1
        remaining -= 1
    return alloc


# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = dbc.Container(fluid=True,
    style={"backgroundColor": "#f0f2f5", "minHeight": "100vh", "paddingBottom": "30px"},
    children=[
        dbc.Row(dbc.Col(html.Div([
            html.H3("🚚  Multi-Constraint Last-Mile Delivery Optimizer",
                    style={"color": "#2c3e50", "fontWeight": "800", "marginBottom": "2px"}),
            html.Small(
                "VRPTW Solver  ·  DAA Experiential Learning Project  ·  RV College of Engineering",
                style={"color": "#7f8c8d", "fontSize": "13px"},
            ),
        ], style={"padding": "22px 10px 12px"}))),

        dbc.Row([
            dbc.Col(width=3, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.Div("⚙️  Configuration", style=HEADER_STYLE),
                    html.Div(style={"padding": "16px"}, children=[
                        html.Label("Delivery Area", className="fw-bold mt-1"),
                        dbc.RadioItems(
                            id="data-source",
                            options=[
                                {"label": " 🎲 Random Grid", "value": "random"},
                                {"label": " 🏙️ Bengaluru", "value": "bengaluru"},
                            ],
                            value="random", inline=True, className="mb-3",
                        ),

                        html.Label("Delivery Coordinate", className="fw-bold"),
                        dbc.Row([
                            dbc.Col([html.Small("Lat", className="text-muted"),
                                     dbc.Input(id="delivery-lat", type="number", value=12.9230, step=0.0001, className="mb-2")], width=6),
                            dbc.Col([html.Small("Lng", className="text-muted"),
                                     dbc.Input(id="delivery-lng", type="number", value=77.5718, step=0.0001, className="mb-2")], width=6),
                        ], className="g-2 mb-3"),

                        html.Label("Delivery Points (N)", className="fw-bold"),
                        dcc.Slider(id="num-customers", min=5, max=20, step=1, value=10,
                                   marks={5:"5",10:"10",15:"15",20:"20"},
                                   tooltip={"placement":"bottom"}, className="mb-3"),

                        html.Label("Delivery Agents (K)", className="fw-bold"),
                        dcc.Slider(id="num-vehicles", min=2, max=5, step=1, value=3,
                                   marks={2:"2",3:"3",4:"4",5:"5"},
                                   tooltip={"placement":"bottom"}, className="mb-3"),

                        html.Label("Vehicle Capacity (kg)", className="fw-bold"),
                        dcc.Slider(id="capacity", min=30, max=100, step=10, value=50,
                                   marks={30:"30",50:"50",70:"70",100:"100"},
                                   tooltip={"placement":"bottom"}, className="mb-3"),

                        html.Label("Random Seed", className="fw-bold"),
                        dbc.Input(id="seed", type="number", value=42,
                                  min=0, max=9999, className="mb-3"),

                        html.Hr(),

                        html.Label("Select Algorithms", className="fw-bold"),
                        dbc.Checklist(
                            id="algorithms",
                            options=[
                                {"label": " Greedy Nearest Neighbor", "value": "greedy"},
                                {"label": " Branch & Bound (≤12 nodes)", "value": "bnb"},
                                {"label": " Clarke-Wright Savings", "value": "cw"},
                                {"label": " Clarke-Wright + 2-opt ★", "value": "cw2opt"},
                            ],
                            value=["greedy", "cw", "cw2opt"], className="mb-3",
                        ),

                        dbc.Button("▶  Run Optimization", id="run-btn",
                                   color="primary", className="w-100 mt-1",
                                   style={"fontWeight":"700","fontSize":"15px",
                                          "borderRadius":"8px"}),
                        html.Div(id="run-status", className="mt-2 text-muted small"),
                    ]),
                ]),
            ]),

            dbc.Col(width=6, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.Div(style={**HEADER_STYLE, "display":"flex",
                                   "justifyContent":"space-between","alignItems":"center"},
                        children=[
                            html.Span("🗺️  Route Map  (OpenStreetMap)"),
                            html.Div(id="map-algo-label",
                                     style={"fontSize":"12px","opacity":"0.85"}),
                        ],
                    ),
                    html.Div(style={"padding": "8px 16px 4px"}, children=[
                        html.Small("View routes from:", className="text-muted"),
                        html.Div(id="algo-btn-group", className="mt-1"),
                    ]),
                    html.Div(
                        id="leaflet-container",
                        children=[
                            html.Iframe(src="/local_map", id="leaflet-iframe",
                                        style={"height":"440px","width":"100%","border":"0","borderRadius":"0 0 10px 10px"}),
                        ],
                        style={"height": "440px", "width": "100%",
                               "borderRadius": "0 0 10px 10px", "overflow":"hidden"},
                    ),
                ]),
            ]),

            dbc.Col(width=3, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.Div("📊  Results Summary", style=HEADER_STYLE),
                    html.Div(id="results-table", style={"padding": "12px"}),
                ]),
                html.Div(style={**CARD_STYLE, "marginTop": "16px"}, children=[
                    html.Div("ℹ️  Algorithm Notes", style=HEADER_STYLE),
                    html.Div(style={"padding":"12px","fontSize":"12px",
                                    "color":"#555","lineHeight":"1.7"}, children=[
                        html.B("Greedy NN: "), "O(n²) · fast, suboptimal.", html.Br(),
                        html.B("Branch & Bound: "), "Exact optimal, exp. time.", html.Br(),
                        html.B("Clarke-Wright: "), "O(n² log n) · good approx.", html.Br(),
                        html.B("CW + 2-opt ★: "), "Best quality-speed balance.",
                    ]),
                ]),
            ]),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(width=6, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.Div("📏  Total Distance Comparison", style=HEADER_STYLE),
                    dcc.Loading(
                        dcc.Graph(id="dist-chart", style={"height":"240px"},
                                  config={"displayModeBar":False}),
                        type="dot", color="#2c3e50",
                    ),
                ]),
            ]),
            dbc.Col(width=6, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.Div("⏱️  Computation Time (ms)", style=HEADER_STYLE),
                    dcc.Loading(
                        dcc.Graph(id="time-chart", style={"height":"240px"},
                                  config={"displayModeBar":False}),
                        type="dot", color="#2c3e50",
                    ),
                ]),
            ]),
        ], className="g-3"),

        dcc.Store(id="results-store"),
        dcc.Store(id="map-preview-store", data={}),
        dcc.Store(id="selected-algo", data="cw2opt"),
        html.Link(rel="stylesheet", href="/static_map.css"),
        html.Script(src="/static_map.js"),
    ],
)


# ── Helper: build Leaflet route layers ────────────────────────────────────────

def build_leaflet_layers(result, lat_lngs, names, demands, time_windows, extra_depots=None):
    layers = []
    for v_idx, route in enumerate(result["routes"]):
        if not route:
            continue
        color = VEHICLE_COLORS[v_idx % len(VEHICLE_COLORS)]

        path      = [0] + route + [0]
        positions = [[lat_lngs[i][0], lat_lngs[i][1]] for i in path]
        layers.append(dl.Polyline(
            positions=positions,
            color=color,
            weight=4,
            opacity=0.85,
            dashArray=None,
        ))

        for node in route:
            lat, lng = lat_lngs[node]
            tooltip_text = (
                f"{names[node]} | "
                f"Demand: {demands[node]} kg | "
                f"Window: {time_windows[node][0]:.1f}–{time_windows[node][1]:.1f}"
            )
            layers.append(dl.CircleMarker(
                center=[lat, lng],
                radius=9,
                color="white",
                weight=2,
                fillColor=color,
                fillOpacity=0.9,
                children=dl.Tooltip(tooltip_text),
            ))

    depot_lat, depot_lng = lat_lngs[0]
    layers.append(dl.Marker(
        position=[depot_lat, depot_lng],
        children=[
            dl.Tooltip(f"🏭 Depot — {names[0]}"),
            dl.Popup(html.Div([
                html.B("📦 Depot / Warehouse"),
                html.Br(),
                html.Span(names[0]),
            ])),
        ],
    ))

    if extra_depots:
        for depot in extra_depots:
            lat, lng = depot["lat_lng"]
            layers.append(dl.Marker(
                position=[lat, lng],
                children=[
                    dl.Tooltip(f"🏬 {depot['name']}"),
                    dl.Popup(html.Div([
                        html.B("📍 Secondary Depot"),
                        html.Br(),
                        html.Span(depot["name"]),
                    ])),
                ],
            ))

    return layers


# ── Helper: bar chart ─────────────────────────────────────────────────────────

def make_bar_chart(results, key, ylabel, colors):
    names  = [r["algorithm"].replace(" ★", "") for r in results]
    values = [r[key] for r in results]
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors[:len(results)],
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title=ylabel, gridcolor="#eee"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        showlegend=False,
    )
    return fig


# ── Helper: results table ─────────────────────────────────────────────────────

def make_results_table(results, delivery_coord=None):
    rows = []
    for r in results:
        badge_color = "success" if "2-opt" in r["algorithm"] else "secondary"
        route_rows = []
        for idx, route in enumerate(r.get("routes", []), start=1):
            path = route.get("path", [])
            est_minutes = max(8.0, round((len(path) + 1) * 5.5, 1))
            route_rows.append(html.Div(
                style={"borderTop":"1px solid #eee","paddingTop":"8px","marginTop":"8px"},
                children=[
                    html.Div([
                        html.Span(f"Driver {idx}", style={"fontWeight":"700","fontSize":"12px"}),
                        html.Span(f" · {route.get('depot_name', 'Depot')}", style={"color":"#777","fontSize":"12px"}),
                    ]),
                    html.Div([
                        html.Span("Delivery: ", style={"color":"#888","fontSize":"11px"}),
                        html.Span(coord_text(delivery_coord), style={"fontSize":"11px"}),
                    ], className="mt-1"),
                    html.Div([
                        html.Span("Estimated time: ", style={"color":"#888","fontSize":"11px"}),
                        html.Span(f"{est_minutes:.1f} min", style={"fontWeight":"600","fontSize":"11px"}),
                    ]),
                    html.Div([
                        html.Span("Time complexity: ", style={"color":"#888","fontSize":"11px"}),
                        html.Span(COMPLEXITY_MAP.get(r["key"], "O(n)"), style={"fontWeight":"600","fontSize":"11px"}),
                    ]),
                    html.Div([
                        html.Span("Stops: ", style={"color":"#888","fontSize":"11px"}),
                        html.Code(str(path), style={"fontSize":"11px","backgroundColor":"#f6f6f6","padding":"2px 4px","borderRadius":"4px"}),
                    ], style={"marginTop":"4px"}),
                ],
            ))

        rows.append(html.Div(
            style={"borderBottom":"1px solid #eee","paddingBottom":"10px","marginBottom":"10px"},
            children=[
                dbc.Badge(r["algorithm"].replace(" ★",""), color=badge_color,
                          style={"fontSize":"11px","whiteSpace":"normal"}),
                html.Div([
                    html.Span("Distance: ", style={"color":"#888","fontSize":"12px"}),
                    html.Span(f"{r['total_distance']:.2f}",
                              style={"fontWeight":"700","fontSize":"13px"}),
                ], className="mt-1"),
                html.Div([
                    html.Span("Time: ", style={"color":"#888","fontSize":"12px"}),
                    html.Span(f"{r['time_ms']:.2f} ms",
                              style={"fontWeight":"600","fontSize":"12px"}),
                ]),
                html.Div([
                    html.Span("Unserved: ", style={"color":"#888","fontSize":"12px"}),
                    html.Span(str(r["unserved"]),
                              style={"fontWeight":"600","fontSize":"12px",
                                     "color":"#e74c3c" if r["unserved"] else "#2ecc71"}),
                ]),
                html.Div(route_rows, style={"marginTop":"6px"}),
            ],
        ))
    return rows


def coord_text(coord):
    if not coord:
        return "—"
    return f"{coord[0]:.4f}, {coord[1]:.4f}"


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("map-preview-store", "data"),
    Input("delivery-lat", "value"),
    Input("delivery-lng", "value"),
    State("data-source", "value"),
    State("num-customers", "value"),
    State("num-vehicles", "value"),
    State("capacity", "value"),
    State("seed", "value"),
)
def preview_location(delivery_lat, delivery_lng, _source, num_cust, num_veh, cap, seed):
    delivery_coord = (float(delivery_lat), float(delivery_lng)) if delivery_lat is not None and delivery_lng is not None else None
    if _source == "random":
        data = get_random_data(num_cust, num_veh, cap, seed or 42, center_coord=delivery_coord)
    else:
        data = get_bengaluru_data(num_cust, num_veh, cap, seed or 42, start_coord=delivery_coord, end_coord=delivery_coord)
    payload = {"results": [], "data": data, "delivery_coord": delivery_coord}
    global LAST_PAYLOAD
    LAST_PAYLOAD = payload
    return payload


@app.callback(
    Output("results-store", "data"),
    Output("run-status",    "children"),
    Input("run-btn", "n_clicks"),
    State("data-source",   "value"),
    State("num-customers", "value"),
    State("num-vehicles",  "value"),
    State("capacity",      "value"),
    State("seed",          "value"),
    State("algorithms",    "value"),
    State("delivery-lat", "value"),
    State("delivery-lng", "value"),
    prevent_initial_call=True,
)
def run_optimization(_, source, num_cust, num_veh, cap, seed, algo_keys, delivery_lat, delivery_lng):
    if not algo_keys:
        return None, "⚠️ Select at least one algorithm."

    delivery_coord = (float(delivery_lat), float(delivery_lng)) if delivery_lat is not None and delivery_lng is not None else None
    if source == "random":
        data = get_random_data(num_cust, num_veh, cap, seed or 42, center_coord=delivery_coord)
    else:
        data = get_bengaluru_data(num_cust, num_veh, cap, seed or 42, start_coord=delivery_coord, end_coord=delivery_coord)

    depots, assignments = _assign_customers_to_nearest_depot(data)
    if not depots:
        return None, "⚠️ No depot configuration available."

    customer_counts = [len(bucket) for bucket in assignments]
    vehicle_alloc   = _allocate_vehicles_to_depots(customer_counts, num_veh)

    results = []
    for key in algo_keys:
        _, label, fn = ALGO_MAP[key]
        merged_result = {
            "algorithm": label,
            "key": key,
            "routes": [],
            "total_distance": 0.0,
            "time_ms": 0.0,
            "unserved": 0,
        }

        for depot_index, customer_indices in enumerate(assignments):
            if not customer_indices or vehicle_alloc[depot_index] == 0:
                continue

            sub_locations = [depots[depot_index]["location"]] + [data["locations"][i] for i in customer_indices]
            sub_demands   = [0] + [data["demands"][i] for i in customer_indices]
            sub_time_wins = [(0, 1_000_000)] + [data["time_windows"][i] for i in customer_indices]
            sub_capacity  = [cap] * vehicle_alloc[depot_index]

            sub_res = fn(sub_locations, sub_demands, sub_capacity, sub_time_wins, vehicle_alloc[depot_index])
            for route in sub_res.get("routes", []):
                global_route = [customer_indices[node - 1] for node in route]
                merged_result["routes"].append({
                    "depot_name": depots[depot_index]["name"],
                    "depot_lat_lng": depots[depot_index]["lat_lng"],
                    "path": global_route,
                })
            merged_result["total_distance"] += sub_res.get("total_distance", 0.0)
            merged_result["time_ms"] += sub_res.get("time_ms", 0.0)
            merged_result["unserved"] += sub_res.get("unserved", 0)

        results.append(merged_result)

    payload = {"results": results, "data": data, "delivery_coord": delivery_coord}
    global LAST_PAYLOAD
    LAST_PAYLOAD = payload
    status  = f"✅ Done — {num_cust} points · {num_veh} agents · {source}"
    return json.dumps(payload), status


@app.callback(
    Output("algo-btn-group", "children"),
    Output("selected-algo",  "data"),
    Input("results-store", "data"),
    Input({"type": "algo-btn", "index": dash.ALL}, "n_clicks"),
    State("selected-algo", "data"),
)
def update_algo_buttons(store_json, _clicks, current_sel):
    if not store_json:
        return [], current_sel

    results = json.loads(store_json)["results"]
    keys    = [r["key"] for r in results]

    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "algo-btn":
        current_sel = triggered["index"]
    elif current_sel not in keys:
        current_sel = keys[-1]

    buttons = [
        dbc.Button(
            r["algorithm"],
            id={"type": "algo-btn", "index": r["key"]},
            size="sm",
            color="primary" if r["key"] == current_sel else "outline-secondary",
            style={"marginRight":"6px","marginBottom":"4px",
                   "fontSize":"12px","borderRadius":"6px"},
        )
        for r in results
    ]
    return buttons, current_sel


@app.callback(
    Output("dist-chart",    "figure"),
    Output("time-chart",    "figure"),
    Output("results-table", "children"),
    Input("results-store",  "data"),
)
def update_charts(store_json):
    empty = go.Figure()
    empty.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa",
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
    if not store_json:
        return empty, empty, html.P("Run optimisation to see results.",
                                    className="text-muted small")

    payload = json.loads(store_json)
    results = payload["results"]
    colors  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    return (
        make_bar_chart(results, "total_distance", "Distance (km)", colors),
        make_bar_chart(results, "time_ms",        "Time (ms)",     colors),
        make_results_table(results, payload.get("delivery_coord")),
    )


@app.server.route('/map_payload')
def _map_payload():
    from flask import Response
    if LAST_PAYLOAD is None:
        return Response(json.dumps({}), mimetype='application/json')
    return Response(json.dumps(LAST_PAYLOAD), mimetype='application/json')


@app.server.route('/static_map.js')
def _static_map_js():
    from flask import Response
    path = os.path.join(os.path.dirname(__file__), 'assets', 'map.js')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return Response(fh.read(), mimetype='application/javascript')
    except Exception:
        return Response('', mimetype='application/javascript', status=404)


@app.server.route('/static_map.css')
def _static_map_css():
    from flask import Response
    path = os.path.join(os.path.dirname(__file__), 'assets', 'map.css')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return Response(fh.read(), mimetype='text/css')
    except Exception:
        return Response('', mimetype='text/css', status=404)


@app.server.route('/local_map')
def _local_map():
    from flask import Response
    js_path = os.path.join(os.path.dirname(__file__), 'assets', 'map.js')
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'map.css')
    try:
        with open(js_path, 'r', encoding='utf-8') as fj:
            js = fj.read()
        with open(css_path, 'r', encoding='utf-8') as fc:
            css = fc.read()
    except Exception:
        return Response('<p>Map assets missing</p>', mimetype='text/html', status=500)

    html_page = f"""
    <!doctype html>
    <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <style>{css}</style>
        </head>
        <body>
            <div id="leaflet-map" style="width:100%;height:100vh;min-height:420px"></div>
            <script>{js}</script>
        </body>
    </html>
    """
    return Response(html_page, mimetype='text/html')


if __name__ == "__main__":
    app.run(debug=True, port=8050)
