import streamlit as st
import json
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import requests
from pathlib import Path


mapbox_key = st.secrets["MAPBOX_TOKEN"]

# Backend API URL (for Streamlit Cloud deployment)
# Set this in .streamlit/secrets.toml: BACKEND_API_URL = "http://your-vm-ip:8000"
BACKEND_API_URL = st.secrets.get("BACKEND_API_URL", None)
BACKEND_API_KEY = st.secrets.get("BACKEND_API_KEY", None)

st.set_page_config(page_title="CongestionAI", layout="wide")

st.title("CongestionAI — Live & Forecast Dashboard")

# ---------------------------------------------
# Load data (roads, forecast with weather)
# ---------------------------------------------
@st.cache_data
def load_roads():
    gdf = gpd.read_file("data/berlin_roads.geojson")
    if "road_id" not in gdf.columns:
        print("road_id not found! exiting.")
        exit(1)
    return gdf

@st.cache_data(ttl=300)  # Cache for 5 minutes, then refetch
def load_forecast():
    """
    Load forecast from backend API (remote) or local file (VM deployment).
    
    For Streamlit Cloud: Set BACKEND_API_URL and BACKEND_API_KEY in secrets.toml
    For local/VM: Falls back to local file if no API URL configured
    """
    # Try remote API first (for Streamlit Cloud)
    if BACKEND_API_URL:
        try:
            headers = {}
            if BACKEND_API_KEY:
                headers["X-API-Key"] = BACKEND_API_KEY
            response = requests.get(f"{BACKEND_API_URL}/forecast", headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.warning(f"⚠️ Could not fetch from backend API: {e}")
            # Fall through to local file
    
    # Fallback to local file (VM deployment or development)
    local_path = Path("data/forecast.json")
    if local_path.exists():
        with open(local_path, "r") as f:
            return json.load(f)
    
    # No data available
    st.error("❌ No forecast data available. Backend may be starting up.")
    return {"data": {}, "weather": None}

roads = load_roads()
forecast = load_forecast()

# Extract weather from forecast (backend now includes it)
if "weather" in forecast:
    weather = forecast["weather"]
else:
    # Fallback for old format or missing weather
    weather = {
        "current": {"temp": 10, "precip": 0, "description": "unavailable"},
        "hourly": [{"hour": h, "temp": 10, "precip": 0} for h in range(25)]
    }

# ---------------------------------------------
# Create placeholder for weather (will fill later)
# ---------------------------------------------
weather_placeholder = st.empty()

# ---------------------------------------------
# Select forecast hour (visually below weather)
# ---------------------------------------------
hour = st.slider("Forecast hour ahead", 0, 24, 0)
st.write(f"Showing forecast for **+{hour} hours**")

# ---------------------------------------------
# Fill weather section using the placeholder
# ---------------------------------------------
hourly_df = pd.DataFrame(weather["hourly"])
if hour == 0:
    selected_temp = weather["current"]["temp"]
    selected_precip = weather["current"]["precip"]
    selected_desc = weather["current"].get("description", "Current conditions")
else:
    selected = hourly_df[hourly_df["hour"] == hour].iloc[0] if hour in hourly_df["hour"].values else hourly_df.iloc[hour]
    selected_temp = selected["temp"]
    selected_precip = selected["precip"]
    selected_desc = selected.get("condition", f"Forecast (hour +{hour})")

with weather_placeholder.expander("🌤️ Weather at selected hour", expanded=True):
    if hour == 0:
        st.subheader("Current Weather")
    else:
        st.subheader(f"Weather in {hour} hours")

    col1, col2 = st.columns(2)
    col1.metric("Temperature", f"{selected_temp} °C")
    col2.metric("Precipitation", f"{selected_precip} mm")
    st.markdown(f"**Conditions:** {selected_desc}")
    
    # Show forecast age if available
    if "updated_at" in weather:
        st.caption(f"Weather updated: {weather['updated_at']}")

    st.write("### Full 24h Forecast")
    st.line_chart(hourly_df.set_index("hour")[["temp", "precip"]])

# ---------------------------------------------
# Prepare congestion data for selected hour
# ---------------------------------------------
roads = roads.set_index("road_id")

forecast_df = pd.DataFrame(forecast["data"]).T
forecast_df.columns = range(25)

roads["congestion"] = forecast_df[hour]

FREE_FLOW_SPEED = 50
roads["estimated_speed"] = FREE_FLOW_SPEED * (1 - roads["congestion"])

def congestion_to_color(x):
    if pd.isna(x): return [128, 128, 128, 180]
    if x < 0.3: return [0, 255, 0, 180]
    if x < 0.6: return [255, 165, 0, 180]
    return [255, 0, 0, 180]

roads["color"] = roads["congestion"].map(congestion_to_color)
roads["path"] = roads["geometry"].apply(lambda geom: list(geom.coords))
roads = roads.reset_index()

pydeck_data = roads.assign(
    cong=roads["congestion"].round(2),
    est_speed=roads["estimated_speed"].round(1)
)[["road_id", "path", "color", "cong", "est_speed"]].to_dict("records")

# ---------------------------------------------
# Render map with PyDeck
# ---------------------------------------------
layer = pdk.Layer(
    "PathLayer",
    data=pydeck_data,
    get_path="path",
    get_color="color",
    width_scale=20,
    width_min_pixels=3,  
    width_max_pixels=10, 
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=52.52,
    longitude=13.405,
    zoom=11,
    min_zoom=9,
    max_zoom=16,
    pitch=0,
    bearing=0,
)

tooltip = {
    "html": (
        "<b>Road:</b> {road_id}<br/>"
        "<b>Congestion:</b> {cong}<br/>"
        "<b>Estimated speed:</b> {est_speed} km/h<br/>"
    ),
    "style": {
        "backgroundColor": "rgba(0,0,0,0.75)",
        "color": "white",
        "fontSize": "12px"
    }
}

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v11",
    api_keys={"mapbox": mapbox_key},
    tooltip=tooltip,
))

