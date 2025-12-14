import streamlit as st
import json
import gzip
import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import pydeck as pdk
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

st.set_page_config(page_title="Itinerary Tool", layout="wide")
st.title("CongestionAI — Route & Delay Estimator")

# Backend API URL (for Streamlit Cloud deployment)
BACKEND_API_URL = st.secrets.get("BACKEND_API_URL", None)
BACKEND_API_KEY = st.secrets.get("BACKEND_API_KEY", None)

# ------------------------------------------------------
# Route color scheme - only 2 colors needed for alternatives
# Selected route uses congestion coloring (green/yellow/red)
# ------------------------------------------------------
ALTERNATIVE_COLORS = [
    {"color": [59, 130, 246], "hex": "#3B82F6", "label": "Blue"},      # Medium blue
    {"color": [147, 51, 234], "hex": "#9333EA", "label": "Purple"},    # Purple
]

# ------------------------------------------------------
# Initialize session state for route persistence
# ------------------------------------------------------
if "routes_data" not in st.session_state:
    st.session_state.routes_data = None  # List of route data dicts
if "route_layers" not in st.session_state:
    st.session_state.route_layers = None
if "selected_route_idx" not in st.session_state:
    st.session_state.selected_route_idx = 0  # Index of selected route (0 = best)
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "locations" not in st.session_state:
    st.session_state.locations = None  # Store geocoded locations
if "hourly_route_cache" not in st.session_state:
    st.session_state.hourly_route_cache = {}  # Cache: {(start_node, end_node, route_type, hour): travel_time}

# ------------------------------------------------------
# Load precomputed graph
# ------------------------------------------------------
@st.cache_resource
def load_graph_and_edges():
    """Load graph from compressed or uncompressed file."""
    gz_path = Path("data/berlin_drive.graphml.gz")
    raw_path = Path("data/berlin_drive.graphml")
    
    if gz_path.exists():
        # Decompress to temp file (osmnx needs file path)
        import tempfile
        with gzip.open(gz_path, 'rb') as f_in:
            with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as f_out:
                f_out.write(f_in.read())
                temp_path = f_out.name
        G = ox.load_graphml(temp_path)
        Path(temp_path).unlink()  # Clean up
    elif raw_path.exists():
        G = ox.load_graphml(raw_path)
    else:
        st.error("❌ berlin_drive.graphml not found!")
        return None, None
    
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    return G, edges

G, edges_gdf = load_graph_and_edges()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_forecast():
    """Load forecast from API (Streamlit Cloud) or local file (VM)."""
    if BACKEND_API_URL:
        try:
            headers = {}
            if BACKEND_API_KEY:
                headers["X-API-Key"] = BACKEND_API_KEY
            response = requests.get(f"{BACKEND_API_URL}/forecast", headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.warning(f"⚠️ Could not fetch forecast: {e}")
    
    # Fallback to local file
    local_path = Path("data/forecast.json")
    if local_path.exists():
        with open(local_path, "r") as f:
            return json.load(f)
    
    return {"data": {}}

forecast = load_forecast()

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

def geocode(query: str) -> Tuple[Optional[float], Optional[float], str]:
    """Geocode an address query to lat/lon coordinates."""
    if not query.strip():
        return None, None, "Please enter an address or coordinates."
    
    token = st.secrets["MAPBOX_TOKEN"]
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
    bbox = "13.0884,52.3383,13.7611,52.6755"  # Berlin bounding box
    params = {
        "access_token": token,
        "limit": 1,
        "bbox": bbox,
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            return None, None, f"Address not found: '{query}'. Try a more specific Berlin address."
        lon, lat = features[0]["center"]
        place_name = features[0]["place_name"]
        return float(lat), float(lon), place_name
    except requests.exceptions.Timeout:
        return None, None, "Geocoding request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, None, f"Geocoding error: {str(e)}"


def find_k_shortest_paths(G: nx.MultiDiGraph, start_node: int, end_node: int, 
                          k: int = 3, weight: str = "travel_time") -> List[List[Tuple[int, int, int]]]:
    """
    Find k shortest paths using OSMnx's built-in k_shortest_paths.
    Uses cascading similarity thresholds to maximize route diversity.
    """
    try:
        # Get more candidates than needed to filter for diversity
        paths_generator = ox.k_shortest_paths(G, start_node, end_node, k=k*5, weight=weight)
        
        # Convert generator to list of edge-tuple paths
        all_edge_paths = []
        for path_nodes in paths_generator:
            edges = []
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    best_key = min(edge_data.keys(), key=lambda k: edge_data[k].get(weight, float('inf')))
                    edges.append((u, v, best_key))
            if edges:
                all_edge_paths.append(edges)
        
        if not all_edge_paths:
            # Fallback to single shortest path
            path_nodes = nx.shortest_path(G, start_node, end_node, weight=weight)
            edges = []
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    best_key = min(edge_data.keys(), key=lambda k: edge_data[k].get(weight, float('inf')))
                    edges.append((u, v, best_key))
            return [edges] if edges else []
        
        # Cascading thresholds: try strict first, relax if needed
        thresholds = [0.5, 0.7, 0.85, 0.95, 1.0]
        
        for threshold in thresholds:
            diverse_paths = [all_edge_paths[0]]  # Always include best path
            
            for path in all_edge_paths[1:]:
                if not is_similar_edge_path(path, diverse_paths, similarity_threshold=threshold):
                    diverse_paths.append(path)
                if len(diverse_paths) >= k:
                    break
            
            if len(diverse_paths) >= k:
                return diverse_paths[:k]
        
        # If still not enough, return what we have
        return diverse_paths[:k] if diverse_paths else all_edge_paths[:k]
        
    except Exception as e:
        print(f"Warning: k_shortest_paths failed: {e}")
        try:
            path_nodes = nx.shortest_path(G, start_node, end_node, weight=weight)
            edges = []
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    best_key = list(edge_data.keys())[0]
                    edges.append((u, v, best_key))
            return [edges] if edges else []
        except nx.NetworkXNoPath:
            return []


def is_similar_edge_path(new_path: List[Tuple[int, int, int]], 
                         existing_paths: List[List[Tuple[int, int, int]]], 
                         similarity_threshold: float = 0.75) -> bool:
    """Check if new_path is too similar to any existing path based on shared edges."""
    if not existing_paths:
        return False
    
    new_set = set(new_path)
    
    for existing in existing_paths:
        existing_set = set(existing)
        intersection = len(new_set & existing_set)
        union = len(new_set | existing_set)
        
        if union > 0 and (intersection / union) > similarity_threshold:
            return True
    
    return False


def compute_travel_times_from_edges(edges: List[Tuple[int, int, int]], hour: int) -> Tuple[float, float, List[Dict]]:
    """
    Compute base and congestion-adjusted travel times for a route.
    
    Args:
        edges: List of (u, v, key) edge tuples
        hour: Departure hour for congestion lookup
    
    Returns:
        base_time: Travel time at free-flow speed (minutes)
        cong_time: Travel time with congestion (minutes)
        seg_results: List of segment details
    """
    base_time = 0
    cong_time = 0

    f_df = pd.DataFrame(forecast["data"]).T
    f_df.columns = range(25)

    FREE_FLOW = 50.0  # km/h
    seg_results = []

    for (u, v, k) in edges:
        edge = G[u][v][k]
        length_m = edge.get("length", 1.0)
        length_km = length_m / 1000.0

        t_base = (length_km / FREE_FLOW) * 60
        base_time += t_base

        rid = f"{u}_{v}_{k}"

        if rid in f_df.index:
            cong = f_df.loc[rid, hour]
        else:
            cong = None

        if cong is None or pd.isna(cong):
            speed = FREE_FLOW
            cong = 0.0
        else:
            speed = FREE_FLOW * (1 - cong)
        
        speed = max(speed, 1.0)

        t_cong = (length_km / speed) * 60
        cong_time += t_cong

        seg_results.append({
            "u": u, "v": v, "key": k,
            "length_km": length_km,
            "congestion": cong if cong is not None else 0.0,
            "speed": speed,
            "t_base": t_base,
            "t_cong": t_cong
        })

    return base_time, cong_time, seg_results


def congestion_to_color(cong: float) -> List[int]:
    """Convert congestion value (0-1) to RGB color."""
    if cong is None or pd.isna(cong):
        return [100, 100, 100, 200]
    
    cong = max(0, min(1, cong))
    
    if cong < 0.3:
        # Green to yellow
        ratio = cong / 0.3
        r = int(50 + ratio * 205)
        g = int(205)
        b = int(50)
    elif cong < 0.6:
        # Yellow to orange
        ratio = (cong - 0.3) / 0.3
        r = int(255)
        g = int(205 - ratio * 80)
        b = int(50)
    else:
        # Orange to red
        ratio = (cong - 0.6) / 0.4
        r = int(255)
        g = int(125 - ratio * 125)
        b = int(50)
    
    return [r, g, b, 220]


def build_route_layer_congestion(edges: List[Tuple], seg_results: List[Dict], width: int = 6) -> pdk.Layer:
    """Build a PathLayer with congestion-based coloring (for selected route)."""
    records = []
    seg_dict = {(s["u"], s["v"], s["key"]): s for s in seg_results}
    
    for (u, v, k) in edges:
        try:
            geom = edges_gdf.loc[(u, v, k), "geometry"]
            coords = list(geom.coords)
            
            seg = seg_dict.get((u, v, k), {})
            cong = seg.get("congestion", 0)
            color = congestion_to_color(cong)
            
            records.append({
                "path": coords,
                "color": color,
                "congestion": f"{cong*100:.0f}%"
            })
        except KeyError:
            continue
    
    return pdk.Layer(
        "PathLayer",
        data=records,
        get_path="path",
        get_color="color",
        width_scale=20,
        width_min_pixels=width,
        width_max_pixels=width + 4,
        pickable=True
    )


def build_route_layer_static(edges: List[Tuple], color: List[int], width: int = 3, opacity: int = 150) -> pdk.Layer:
    """Build a PathLayer with static color (for alternative routes)."""
    records = []
    color_with_opacity = color + [opacity]
    
    for (u, v, k) in edges:
        try:
            geom = edges_gdf.loc[(u, v, k), "geometry"]
            coords = list(geom.coords)
            
            records.append({
                "path": coords,
                "color": color_with_opacity
            })
        except KeyError:
            continue
    
    return pdk.Layer(
        "PathLayer",
        data=records,
        get_path="path",
        get_color="color",
        width_scale=15,
        width_min_pixels=width,
        width_max_pixels=width + 2,
        pickable=False
    )


def compute_view_state(lat1: float, lon1: float, lat2: float, lon2: float) -> pdk.ViewState:
    """Compute appropriate view state to show both start and end points."""
    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2

    lat_diff = abs(lat1 - lat2)
    lon_diff = abs(lon1 - lon2)
    max_diff = max(lat_diff, lon_diff)

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


def prepare_graph_with_travel_times(hour: int) -> None:
    """Add travel_time attribute to all edges based on congestion forecast."""
    f_df = pd.DataFrame(forecast["data"]).T
    f_df.columns = range(25)
    FREE_FLOW = 50.0
    
    for u, v, k, data in G.edges(keys=True, data=True):
        rid = f"{u}_{v}_{k}"
        length_m = data.get("length", 1.0)
        length_km = length_m / 1000.0
        
        if rid in f_df.index:
            cong = f_df.loc[rid, hour]
        else:
            cong = None
        
        if cong is None or pd.isna(cong):
            speed = FREE_FLOW
        else:
            speed = FREE_FLOW * (1 - cong)
        speed = max(speed, 1.0)
        
        data["travel_time"] = (length_km / speed) * 60


def get_route_display_info(routes_data: List[Dict], selected_idx: int) -> List[Dict]:
    """
    Get display info for each route, assigning colors dynamically.
    Selected route gets congestion colors (shown as gradient icon).
    Alternatives get blue/purple colors that rotate based on selection.
    """
    display_info = []
    alt_color_idx = 0
    
    for rd in routes_data:
        idx = rd["idx"]
        is_selected = (idx == selected_idx)
        
        if is_selected:
            # Selected route - uses congestion coloring
            display_info.append({
                **rd,
                "is_selected": True,
                "display_color": None,
                "color_hex": None,
                "color_label": "Congestion"
            })
        else:
            # Alternative route - use rotating blue/purple
            alt_color = ALTERNATIVE_COLORS[alt_color_idx % len(ALTERNATIVE_COLORS)]
            display_info.append({
                **rd,
                "is_selected": False,
                "display_color": alt_color["color"],
                "color_hex": alt_color["hex"],
                "color_label": alt_color["label"]
            })
            alt_color_idx += 1
    
    return display_info


def compute_routes(start_text: str, dest_text: str, hour: int, route_type: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Compute top 3 routes and return data dict, or None on error.
    """
    # Geocode start
    lat1, lon1, start_result = geocode(start_text)
    if lat1 is None:
        return None, f"❌ Start location error: {start_result}"
    
    # Geocode destination
    lat2, lon2, dest_result = geocode(dest_text)
    if lat2 is None:
        return None, f"❌ Destination error: {dest_result}"
    
    try:
        start_node = ox.nearest_nodes(G, lon1, lat1)
        end_node = ox.nearest_nodes(G, lon2, lat2)
        
        if start_node == end_node:
            return None, "Start and destination are too close (same network node)."
        
        # Prepare graph with travel times for the specified hour
        prepare_graph_with_travel_times(hour)
        
        # Choose weight based on route type
        weight = "travel_time" if route_type == "fastest" else "length"
        
        # Find k=3 shortest paths using OSMnx
        edge_paths = find_k_shortest_paths(G, start_node, end_node, k=3, weight=weight)
        
        if not edge_paths:
            return None, "❌ No route found between these locations. They may not be connected by roads."
        
        # Process each route
        routes_data = []
        for idx, edges in enumerate(edge_paths):
            base_t, cong_t, segs = compute_travel_times_from_edges(edges, hour)
            delay = cong_t - base_t
            total_dist = sum(s["length_km"] for s in segs)
            
            # Find top congested segments
            sorted_segs = sorted(segs, key=lambda x: x["congestion"], reverse=True)
            top_congested = sorted_segs[:5]
            
            routes_data.append({
                "idx": idx,
                "edges": edges,
                "segs": segs,
                "base_t": base_t,
                "cong_t": cong_t,
                "delay": delay,
                "total_dist": total_dist,
                "top_congested": top_congested,
            })
        
        # Sort by congestion time (fastest first)
        routes_data.sort(key=lambda x: x["cong_t"])
        
        # Reassign indices after sorting
        for i, rd in enumerate(routes_data):
            rd["idx"] = i
        
        locations = {
            "start_result": start_result,
            "dest_result": dest_result,
            "lat1": lat1, "lon1": lon1,
            "lat2": lat2, "lon2": lon2,
            "hour": hour,
            "route_type": route_type
        }
        
        return {"routes": routes_data, "locations": locations}, None
        
    except nx.NetworkXNoPath:
        return None, "❌ No route found between these locations. They may not be connected by roads."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error computing route: {str(e)}"


def build_map_layers(routes_data: List[Dict], selected_idx: int) -> List[pdk.Layer]:
    """
    Build pydeck layers for all routes.
    Selected route gets congestion coloring; others get blue/purple.
    """
    layers = []
    display_info = get_route_display_info(routes_data, selected_idx)
    
    # First, add alternative routes (behind selected route)
    for rd in display_info:
        if not rd["is_selected"]:
            layer = build_route_layer_static(
                rd["edges"],
                rd["display_color"],
                width=3,
                opacity=160
            )
            layers.append(layer)
    
    # Then add selected route on top
    selected_rd = next(rd for rd in display_info if rd["is_selected"])
    selected_layer = build_route_layer_congestion(
        selected_rd["edges"],
        selected_rd["segs"],
        width=6
    )
    layers.append(selected_layer)
    
    return layers


# ------------------------------------------------------
# Route selection panel (replaces the CSS + render_route_card)
# ------------------------------------------------------

def get_route_label(index: int, is_best: bool) -> str:
    """Get semantic label for route."""
    if is_best:
        return "Best (fastest)"
    else:
        # Alternative A, B, C...
        alt_letter = chr(ord('A') + index - 1)  # 1->A, 2->B, etc.
        return f"Alternative {alt_letter}"


def render_route_selector(routes: List[Dict], selected_idx: int) -> None:
    """Render route selection cards using Streamlit components."""
    
    for i, rd in enumerate(routes):
        is_selected = (rd["idx"] == selected_idx)
        is_best = (i == 0)
        route_label = get_route_label(i, is_best)
        
        # Determine styling
        if is_selected:
            # Selected route - green border container
            with st.container(border=True):
                cols = st.columns([1, 5, 4])  # Wider last column for SELECTED label
                with cols[0]:
                    st.markdown("🌈")  # Gradient indicator
                with cols[1]:
                    st.markdown(f"**{route_label}**")
                with cols[2]:
                    st.markdown("✅ **SELECTED**")  # Use markdown instead of st.success
                
                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.caption(f"⏱️ **{rd['cong_t']:.1f}** min")
                with stat_cols[1]:
                    st.caption(f"📏 **{rd['total_dist']:.1f}** km")
                with stat_cols[2]:
                    st.caption(f"⏳ **+{rd['delay']:.1f}** min")
        else:
            # Alternative route - clickable
            alt_color_idx = sum(1 for j in range(i) if j != selected_idx)
            alt_color = ALTERNATIVE_COLORS[alt_color_idx % len(ALTERNATIVE_COLORS)]
            
            # Use actual colored circles instead of shortcodes
            if alt_color['label'] == "Blue":
                color_emoji = "🔵"
            else:  # Purple
                color_emoji = "🟣"
            
            with st.container(border=True):
                cols = st.columns([1, 5, 4])
                with cols[0]:
                    st.markdown(color_emoji)
                with cols[1]:
                    st.markdown(f"**{route_label}**")
                with cols[2]:
                    if is_best:
                        st.markdown("⭐ **BEST**")
                
                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.caption(f"⏱️ {rd['cong_t']:.1f} min")
                with stat_cols[1]:
                    st.caption(f"📏 {rd['total_dist']:.1f} km")
                with stat_cols[2]:
                    st.caption(f"⏳ +{rd['delay']:.1f} min")
                
                if st.button(f"Select", key=f"sel_{rd['idx']}", use_container_width=True):
                    st.session_state.selected_route_idx = rd['idx']
                    st.rerun()


# ------------------------------------------------------
# UI Layout - Two Columns
# ------------------------------------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📍 Route Inputs")
    
    start_text = st.text_input("Starting point", placeholder="e.g., Alexanderplatz, Berlin")
    
    # Show resolved start address preview
    if start_text.strip():
        with st.spinner(""):
            lat, lon, result = geocode(start_text)
            if lat is not None:
                st.caption(f"📍 _{result}_")
            else:
                st.caption(f"⚠️ _{result}_")
    
    dest_text = st.text_input("Destination", placeholder="e.g., Europaplatz, Berlin")
    
    # Show resolved destination address preview
    if dest_text.strip():
        with st.spinner(""):
            lat, lon, result = geocode(dest_text)
            if lat is not None:
                st.caption(f"📍 _{result}_")
            else:
                st.caption(f"⚠️ _{result}_")
    
    st.divider()
    
    hour = st.slider("Departure in (hours ahead)", 0, 24, 0)
    
    st.divider()
    
    compute_btn = st.button("🚗 Compute Routes", use_container_width=True, type="primary")
    
    st.divider()
    
    # Route type toggle
    route_type = st.radio(
        "Route optimization",
        options=["fastest", "shortest"],
        format_func=lambda x: "⚡ Fastest (using congestion forecast)" if x == "fastest" else "🛣️ Shortest (by distance)",
        horizontal=True
    )
    
    # Route selection panel (only show when routes exist)
    # Moved BEFORE the divider and legend, and show unconditionally when data exists
    if st.session_state.routes_data is not None:
        st.divider()
        st.subheader("🛤️ Choose Route")
        st.caption("_In dense urban areas, alternative routes may partially overlap due to road network constraints._")
        render_route_selector(st.session_state.routes_data, st.session_state.selected_route_idx)
    
    st.divider()
    
    # Congestion legend
    st.caption("**Selected Route Colors:**")
    st.markdown("""
    <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
        <div style="width: 60px; height: 12px; border-radius: 6px; background: linear-gradient(90deg, #32CD32, #FFD700, #FF4500);"></div>
        <span style="font-size: 12px;">Congestion level</span>
    </div>
    """, unsafe_allow_html=True)
    
    legend_cols = st.columns(3)
    with legend_cols[0]:
        st.markdown("🟢 Low")
    with legend_cols[1]:
        st.markdown("🟡 Medium")
    with legend_cols[2]:
        st.markdown("🔴 High")
    
    st.caption("_ℹ️ Congestion = predicted traffic slowdown vs. free-flow speed._")

# ------------------------------------------------------
# Main logic - Compute on button OR on toggle change
# ------------------------------------------------------

should_compute = False
error_msg = None

if compute_btn:
    should_compute = True
elif st.session_state.last_inputs is not None:
    # Check if route_type or hour changed
    last = st.session_state.last_inputs
    if last["route_type"] != route_type or last["hour"] != hour:
        should_compute = True

# Compute routes BEFORE rendering right column content
if should_compute:
    # Use stored addresses if just toggling, otherwise use current input
    if compute_btn:
        use_start = start_text
        use_dest = dest_text
    else:
        use_start = st.session_state.last_inputs["start_text"]
        use_dest = st.session_state.last_inputs["dest_text"]
    
    with st.spinner("Computing routes..."):
        result, error_msg = compute_routes(use_start, use_dest, hour, route_type)
        
        if result is not None:
            st.session_state.routes_data = result["routes"]
            st.session_state.locations = result["locations"]
            st.session_state.selected_route_idx = 0  # Reset to best route
            st.session_state.last_inputs = {
                "start_text": use_start,
                "dest_text": use_dest,
                "hour": hour,
                "route_type": route_type
            }
            st.rerun()  # Rerun to show route selector immediately
        else:
            if compute_btn:  # Only clear on explicit compute failure
                st.session_state.routes_data = None
                st.session_state.locations = None
                st.session_state.last_inputs = None

with right_col:
    # Show error if any
    if error_msg:
        st.error(error_msg)
    
    # Display routes from session state
    if st.session_state.routes_data is not None and st.session_state.locations is not None:
        routes = st.session_state.routes_data
        locs = st.session_state.locations
        selected_idx = st.session_state.selected_route_idx
        selected_route = routes[selected_idx]
        
        # Location info
        st.success(f"📍 **From:** {locs['start_result']}")
        st.success(f"📍 **To:** {locs['dest_result']}")
        
        # Map FIRST (above travel summary)
        route_label = "⚡ Fastest Routes" if locs['route_type'] == "fastest" else "🛣️ Shortest Routes"
        st.subheader(f"🗺️ Route Map — {route_label}")
        
        # Build layers
        layers = build_map_layers(routes, selected_idx)
        view = compute_view_state(locs['lat1'], locs['lon1'], locs['lat2'], locs['lon2'])
        
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view,
                map_style="mapbox://styles/mapbox/light-v11",
                api_keys={"mapbox": st.secrets["MAPBOX_TOKEN"]},
                tooltip={"text": "Congestion: {congestion}"}
            )
        )
        
        st.caption(f"_Showing {len(routes)} routes • Departure: +{locs['hour']}h from now_")
        
        # Selected route summary (below map)
        st.subheader(f"📊 Route {selected_idx + 1} — Travel Summary")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Distance", f"{selected_route['total_dist']:.1f} km")
        with metric_cols[1]:
            st.metric("Base time", f"{selected_route['base_t']:.1f} min")
        with metric_cols[2]:
            st.metric("With congestion", f"{selected_route['cong_t']:.1f} min")
        with metric_cols[3]:
            delta_color = "inverse" if selected_route['delay'] > 0 else "off"
            st.metric(
                "Delay", 
                f"+{selected_route['delay']:.1f} min", 
                delta=f"{selected_route['delay']:.1f} min", 
                delta_color=delta_color
            )
        
        # Comparison with other routes
        if len(routes) > 1:
            st.caption("**Comparison with alternatives:**")
            comp_cols = st.columns(len(routes))
            for i, rd in enumerate(routes):
                with comp_cols[i]:
                    diff = rd['cong_t'] - selected_route['cong_t']
                    if rd['idx'] == selected_idx:
                        st.markdown(f"**Route {i+1}** ✓")
                        st.markdown(f"_{rd['cong_t']:.1f} min_")
                    else:
                        st.markdown(f"Route {i+1}")
                        if diff > 0:
                            st.markdown(f"_{rd['cong_t']:.1f} min (+{diff:.1f})_")
                        else:
                            st.markdown(f"_{rd['cong_t']:.1f} min ({diff:.1f})_")
        
        # Two expanders side by side using columns
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            with st.expander("⏱️ Time Breakdown", expanded=False):
                # Calculate time spent in each congestion level
                segs = selected_route['segs']
                
                time_low = sum(s['t_cong'] for s in segs if s['congestion'] < 0.3)
                time_med = sum(s['t_cong'] for s in segs if 0.3 <= s['congestion'] < 0.6)
                time_high = sum(s['t_cong'] for s in segs if s['congestion'] >= 0.6)
                total_time = selected_route['cong_t']
                
                # Display as progress bars
                st.caption("Time spent by congestion level:")
                
                if total_time > 0:
                    # Low congestion
                    pct_low = (time_low / total_time) * 100
                    st.markdown(f"🟢 **Low** ({time_low:.1f} min)")
                    st.progress(pct_low / 100)
                    
                    # Medium congestion
                    pct_med = (time_med / total_time) * 100
                    st.markdown(f"🟡 **Medium** ({time_med:.1f} min)")
                    st.progress(pct_med / 100)
                    
                    # High congestion
                    pct_high = (time_high / total_time) * 100
                    st.markdown(f"🔴 **High** ({time_high:.1f} min)")
                    st.progress(pct_high / 100)
                    
                    # Summary
                    st.divider()
                    if pct_high > 30:
                        st.warning(f"⚠️ {pct_high:.0f}% of your trip is in heavy traffic")
                    elif pct_low > 70:
                        st.success(f"✅ {pct_low:.0f}% of your trip is free-flowing")
                    else:
                        st.info(f"ℹ️ Mixed traffic conditions")
        
        with exp_col2:
            with st.expander("🕐 Best Departure Time", expanded=False):
                st.caption("Optimal route travel time by hour:")
                
                current_hour = locs['hour']
                departure_options = []
                
                # Get start/end nodes for route recomputation
                start_node = ox.nearest_nodes(G, locs['lon1'], locs['lat1'])
                end_node = ox.nearest_nodes(G, locs['lon2'], locs['lat2'])
                
                # Cache key base (same for all hours with same endpoints and route type)
                cache_key_base = (start_node, end_node, locs['route_type'])
                
                # Check -3 to +3 hours around selected time (within 0-24 bounds)
                start_offset = max(0, current_hour - 3)
                end_offset = min(25, current_hour + 4)  # +4 because range is exclusive
                
                for test_hour in range(start_offset, end_offset):
                    cache_key = (*cache_key_base, test_hour)
                    
                    # Check cache first
                    if cache_key in st.session_state.hourly_route_cache:
                        test_cong_t = st.session_state.hourly_route_cache[cache_key]
                    else:
                        # Compute and cache
                        prepare_graph_with_travel_times(test_hour)
                        
                        try:
                            # Find best route for this departure time
                            weight = "travel_time" if locs['route_type'] == "fastest" else "length"
                            path_nodes = nx.shortest_path(G, start_node, end_node, weight=weight)
                            
                            # Convert to edges
                            test_edges = []
                            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                                edge_data = G.get_edge_data(u, v)
                                if edge_data:
                                    key = list(edge_data.keys())[0]
                                    test_edges.append((u, v, key))
                            
                            # Compute travel time for this optimal route at this hour
                            _, test_cong_t, _ = compute_travel_times_from_edges(test_edges, test_hour)
                            
                            # Store in cache
                            st.session_state.hourly_route_cache[cache_key] = test_cong_t
                        except:
                            continue
                    
                    departure_options.append({
                        "hour": test_hour,
                        "time": test_cong_t,
                        "diff": test_cong_t - selected_route['cong_t']
                    })
                
                # Restore graph to current hour
                prepare_graph_with_travel_times(current_hour)

                if departure_options:
                    # Find best and worst times
                    best = min(departure_options, key=lambda x: x["time"])
                    worst = max(departure_options, key=lambda x: x["time"])
                    
                    # Create a simple bar chart using markdown
                    max_time = worst["time"]
                    min_time = best["time"]
                    time_range = max_time - min_time if max_time != min_time else 1
                    
                    for opt in departure_options:
                        # Calculate bar width (normalized)
                        bar_pct = ((opt["time"] - min_time) / time_range) * 100 if time_range > 0 else 50
                        bar_pct = max(10, min(100, 30 + bar_pct * 0.7))  # Scale for visibility
                        
                        # Label - show offset from now
                        if opt["hour"] == 0:
                            label = "Now"
                        else:
                            label = f"+{opt['hour']}h"
                        
                        # Mark selected departure time
                        is_selected_time = (opt["hour"] == current_hour)
                        
                        # Color based on relative time
                        if opt["hour"] == best["hour"]:
                            color = "#22c55e"  # Green
                            suffix = " 🏆"
                        elif opt["hour"] == worst["hour"] and len(departure_options) > 2:
                            color = "#ef4444"  # Red
                            suffix = ""
                        else:
                            color = "#3b82f6"  # Blue
                            suffix = ""
                        
                        # Add marker for currently selected time
                        if is_selected_time:
                            suffix += " ← selected"
                        
                        # Render bar
                        st.markdown(
                            f"**{label}**: {opt['time']:.0f} min{suffix}"
                        )
                        st.markdown(
                            f'<div style="background: {color}; height: 8px; width: {bar_pct}%; '
                            f'border-radius: 4px; margin-bottom: 8px;"></div>',
                            unsafe_allow_html=True
                        )
                    
                    st.divider()
                    
                    # Recommendation
                    if best["hour"] != current_hour and (selected_route['cong_t'] - best["time"]) > 1:
                        savings = selected_route['cong_t'] - best["time"]
                        if best["hour"] == 0:
                            st.success(f"💡 **Leave now** to save **{savings:.0f} min**")
                        else:
                            st.success(f"💡 **Leave in +{best['hour']}h** to save **{savings:.0f} min**")
                    elif best["hour"] == current_hour:
                        st.success("✅ **Your selected time is optimal!**")
                    else:
                        st.info("ℹ️ Travel time is similar across hours")
    else:
        # No routes computed yet
        st.subheader("🗺️ Route Map")
        st.info("👈 Enter start and destination addresses, then click **Compute Routes** to see the top 3 alternatives.")
        
        st.pydeck_chart(
            pdk.Deck(
                layers=[],
                initial_view_state=pdk.ViewState(
                    latitude=52.52,
                    longitude=13.405,
                    zoom=11,
                    pitch=0
                ),
                map_style="mapbox://styles/mapbox/light-v11",
                api_keys={"mapbox": st.secrets["MAPBOX_TOKEN"]}
            )
        )
