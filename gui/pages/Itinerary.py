import streamlit as st
import json
import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import pydeck as pdk

st.set_page_config(page_title="Itinerary Tool", layout="wide")
st.title("CongestionAI — Route & Delay Estimator")

# ------------------------------------------------------
# Load precomputed graph (VERY IMPORTANT: already saved)
# ------------------------------------------------------
@st.cache_resource
def load_graph_and_edges():
    G = ox.load_graphml("data/berlin_drive.graphml")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    return G, edges

G, edges_gdf = load_graph_and_edges()


# Load forecast (same format as page 1)
@st.cache_data
def load_forecast():
    with open("data/forecast.json", "r") as f:
        return json.load(f)

forecast = load_forecast()

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

# Geocode using Mapbox API (should be free for our use) - Alternative: Photon, Nominatim
def geocode(query: str):
    token = st.secrets["MAPBOX_TOKEN"]
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
    # Berlin bounding box: [min_lon, min_lat, max_lon, max_lat]
    bbox = "13.0884,52.3383,13.7611,52.6755"
    params = {
        "access_token": token,
        "limit": 1,
        "bbox": bbox,   # restrict results to Berlin only
    }
    
    r = requests.get(url, params=params)
    r.raise_for_status()
    features = r.json().get("features", [])
    if not features:
        return None, None
    lon, lat = features[0]["center"]
    place_name = features[0]["place_name"]
    st.write(f"Resolved place: {place_name}")

    return float(lat), float(lon)


# Convert OSMnx route nodes → list of edges (u,v,key)
def route_to_edges(route_nodes):
    edges = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = G.get_edge_data(u, v)
        # pick first key if multiple parallel edges
        key = list(data.keys())[0]
        edges.append((u, v, key))
    return edges

# Compute travel times
def compute_travel_times(edges, hour):
    base_time = 0
    cong_time = 0

    # Convert forecast to df for fast indexing
    f_df = pd.DataFrame(forecast["data"]).T
    f_df.columns = range(25)

    # Free-flow speed assumption (in km/h) TODO: adjust based on road id
    FREE_FLOW = 50.0

    seg_results = []

    for (u, v, k) in edges:
        edge = G[u][v][k]
        length_m = edge.get("length", 1.0)
        length_km = length_m / 1000.0

        # Base travel time (in minutes)
        t_base = (length_km / FREE_FLOW) * 60
        base_time += t_base

        # Find corresponding road ID in forecast
        # Format must match your preprocessing: u_v_key
        rid = f"{u}_{v}_{k}"

        if rid in f_df.index:
            cong = f_df.loc[rid, hour]
        else:
            cong = None

        if cong is None or pd.isna(cong):
            speed = FREE_FLOW
        else:
            speed = FREE_FLOW * (1 - cong)
        
        speed = max(speed, 1.0)  # avoid zero speeds

        t_cong = (length_km / speed) * 60
        cong_time += t_cong

        seg_results.append({
            "u": u, "v": v, "key": k,
            "length_km": length_km,
            "congestion": cong,
            "speed": speed,
            "t_base": t_base,
            "t_cong": t_cong
        })

    return base_time, cong_time, seg_results

# Build PyDeck route layer
def build_route_layer(edges):
    records = []
    for (u, v, k) in edges:
        try:
            geom = edges_gdf.loc[(u, v, k), "geometry"]
            coords = list(geom.coords)
            records.append({"path": coords})
        except KeyError:
            print(f"Missing edge in edges_gdf: {(u, v, k)}")
            continue  # Skip missing edges
    return pdk.Layer(
        "PathLayer",
        data=records,
        get_path="path",
        get_color=[0, 100, 255, 200],
        width_scale=20,
        width_min_pixels=4,
        width_max_pixels=8
    )

def compute_view_state(lat1, lon1, lat2, lon2):
    # Center is the midpoint
    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2

    # Approximate zoom calculation (simple but effective)
    lat_diff = abs(lat1 - lat2)
    lon_diff = abs(lon1 - lon2)

    # Larger distance → lower zoom (zoom out)
    max_diff = max(lat_diff, lon_diff)

    # Simple heuristic:
    if max_diff < 0.005:
        zoom = 15
    elif max_diff < 0.01:
        zoom = 14
    elif max_diff < 0.02:
        zoom = 13
    elif max_diff < 0.05:
        zoom = 12
    elif max_diff < 0.1:
        zoom = 11
    else:
        zoom = 10

    return pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0
    )


# ------------------------------------------------------
# UI Inputs
# ------------------------------------------------------
st.subheader("Route Inputs")

col1, col2 = st.columns(2)

with col1:
    start_text = st.text_input("Starting point (address or coordinates)")
with col2:
    dest_text = st.text_input("Destination (address or coordinates)")

hour_display = st.slider("Departure in (hours ahead)", 0, 24, 0)
hour = hour_display  # directly use as index

compute_btn = st.button("Compute Route")

# ------------------------------------------------------
# Main logic
# ------------------------------------------------------
if compute_btn:

    st.write("Geocoding locations...")

    lat1, lon1 = geocode(start_text)
    lat2, lon2 = geocode(dest_text)

    if lat1 is None or lat2 is None:
        st.error("Could not geocode one of the locations.")
        st.stop()

    st.write(f"Start: {lat1}, {lon1}")
    st.write(f"Dest: {lat2}, {lon2}")

    # Nearest graph nodes
    start_node = ox.nearest_nodes(G, lon1, lat1)
    end_node   = ox.nearest_nodes(G, lon2, lat2)

    # Compute shortest path
    st.write("Computing optimal route...")
    route_nodes = nx.shortest_path(G, start_node, end_node, weight="length")

    edges = route_to_edges(route_nodes)

    # Compute travel times
    base_t, cong_t, segs = compute_travel_times(edges, hour)

    delay = cong_t - base_t

    st.subheader("Travel Summary")
    st.metric("Base travel time", f"{base_t:.1f} min")
    st.metric("Congestion-adjusted time", f"{cong_t:.1f} min")
    st.metric("Expected delay", f"+{delay:.1f} min")

    # Build map layer
    route_layer = build_route_layer(edges)

    view = compute_view_state(lat1, lon1, lat2, lon2)


    st.pydeck_chart(
        pdk.Deck(
            layers=[route_layer],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/light-v11",
            api_keys={"mapbox": st.secrets["MAPBOX_TOKEN"]} 
        )
    )
