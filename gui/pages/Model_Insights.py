import streamlit as st
import json
import pandas as pd
import plotly.express as px

st.title("Model Evaluation & Performance")
st.caption(
    "Evaluation performed offline on held-out validation data. "
    "Metrics reflect detector-level forecasts before road interpolation."
)

# Section A: Overall metrics
with open("gui/data/metrics_overall.json") as f:
    metrics = json.load(f)
cols = st.columns(5)
cols[0].metric("MAE", f"{metrics['mae']:.3f}")
cols[1].metric("RMSE", f"{metrics['rmse']:.3f}")
cols[2].metric("R²", f"{metrics['r2']:.3f}")
cols[3].metric("Within 0.10", f"{metrics['within_010']*100:.1f}%")
cols[4].metric("Within 0.15", f"{metrics['within_015']*100:.1f}%")

# Section B: Horizon degradation
with open("gui/data/metrics_by_horizon.json") as f:
    horizon = json.load(f)
fig = px.line(x=horizon["horizon"], y=horizon["mae"], labels={"x": "Forecast Horizon (h)", "y": "MAE"})
st.plotly_chart(fig, use_container_width=True)
st.caption("Error increases with horizon as expected; steepest increase after ~12h.")

# Section C: Temporal slice
with open("gui/data/metrics_by_hour.json") as f:
    by_hour = json.load(f)
fig = px.bar(x=by_hour["hour"], y=by_hour["mae"], labels={"x": "Hour of Day", "y": "MAE"})
st.plotly_chart(fig, use_container_width=True)
st.caption("Rush hour windows show higher error due to rapid regime changes.")

with open("gui/data/metrics_by_dow.json") as f:
    by_dow = json.load(f)
fig = px.bar(x=by_dow["dow"], y=by_dow["mae"], labels={"x": "Day of Week", "y": "MAE"})
st.plotly_chart(fig, use_container_width=True)
st.caption("Weekends are more predictable / less variable.")

# Section D: Error distribution
with open("gui/data/error_distribution.json") as f:
    err_dist = json.load(f)
fig = px.bar(x=err_dist["bins"][:-1], y=err_dist["counts"], labels={"x": "Absolute Error", "y": "Count"})
st.plotly_chart(fig, use_container_width=True)
st.caption("Error distribution: most predictions are well-calibrated.")

# Section E: (Optional) Spatial
try:
    with open("gui/data/metrics_by_detector.json") as f:
        by_det = pd.DataFrame(json.load(f))
    st.map(by_det[["lat", "lon"]])
    st.dataframe(by_det.sort_values('mae', ascending=False).head(10))
except Exception:
    st.info("Spatial metrics not available.")

st.markdown("""
**Insights:**
- Model achieves strong overall accuracy and is robust across time and space.
- Error increases with forecast horizon, especially after 12h.
- Rush hours are harder to predict due to rapid changes.
- Backend runs hourly; prediction distribution and out-of-range checks are logged.
""")

# Footer
st.markdown(
    '''
    <hr style="margin-top:2em;margin-bottom:0.5em;">
    <div style="font-size:0.95em; color: #888;">
        Built by Adib — Traffic Forecasting & ML Engineering<br>
        📧 <a href="mailto:adib.mellah.projets@gmail.com">adib.mellah.projets@gmail.com</a> ·
        <a href="https://github.com/Axidix" target="_blank">GitHub</a> ·
        Last update: 2025-12-15
    </div>
    ''',
    unsafe_allow_html=True
)