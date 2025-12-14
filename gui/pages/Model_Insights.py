"""
📊 Model Performance & Insights Page
=====================================

TODO: Implement this page to showcase model quality for portfolio.

## Required Data Files (to generate from trained model):
---------------------------------------------------------
All files go in gui/data/

1. metrics_overall.json
   {"mae": 0.08, "rmse": 0.12, "r2": 0.85, "spike_f1": 0.45}

2. metrics_by_horizon.json
   {"horizon": [1,2,...,24], "mae": [0.05, 0.06, ..., 0.12]}

3. metrics_by_hour.json
   {"hour": [0,1,...,23], "mae": [0.09, 0.08, ..., 0.11]}

4. metrics_by_dow.json
   {"dow": [0,1,...,6], "mae": [0.08, 0.09, ..., 0.07]}
   # 0=Monday, 6=Sunday

5. metrics_by_detector.json
   {"detector_id": [...], "lat": [...], "lon": [...], "mae": [...]}

6. ablation_results.json
   {
     "baseline": {"mae": 0.08, "description": "Full model"},
     "no_weather": {"mae": 0.09, "description": "Without weather features"},
     "no_lags": {"mae": 0.11, "description": "Without congestion lags"},
     "no_calendar": {"mae": 0.10, "description": "Without calendar features"}
   }


## Page Structure:
------------------

### Header
- Title: "📊 Model Performance"
- Subtitle: "TCN forecaster trained on 8 years of Berlin traffic data"

### Section 1: Overall Metrics (4 columns)
- MAE: X.XX
- RMSE: X.XX  
- R²: X.XX
- Spike F1: X.XX

### Section 2: Tabs

[Tab 1: Forecast Horizon]
- Line chart: MAE vs forecast hour (1-24)
- Caption: "Error increases with forecast horizon, as expected"

[Tab 2: Temporal Patterns]
- Heatmap: hour (y) × day_of_week (x) colored by MAE
- Or two bar charts: MAE by hour, MAE by day
- Insight: "Rush hours (8-9am, 5-7pm) are harder to predict"

[Tab 3: Spatial Performance]
- Pydeck map: detectors as circles, color = MAE (green=low, red=high)
- Table below: "Top 10 Hardest Detectors" with detector_id, location, MAE
- Insight: "Highway interchanges show highest prediction error"

[Tab 4: Feature Importance]
- Bar chart: ablation results showing delta MAE from baseline
- Bars: "No weather (+X%)", "No lags (+X%)", "No calendar (+X%)"
- Insight: "Congestion lag features provide the largest improvement"


## Evaluation Script to Create Data:
------------------------------------
Create: scripts/generate_model_insights.py

```python
# Pseudocode
1. Load trained model and validation data
2. Run predictions on validation set
3. Compute metrics:
   - Overall: MAE, RMSE, R²
   - By horizon: group by forecast step, compute MAE
   - By hour: group by hour-of-day, compute MAE
   - By DOW: group by day-of-week, compute MAE
   - By detector: group by detector_id, compute MAE
4. For ablation: need to retrain with feature subsets (or load pre-saved results)
5. Save all as JSON to gui/data/
```


## Libraries Needed:
--------------------
- streamlit (already have)
- plotly (for charts)
- pydeck (for map, already have)
- pandas, numpy (already have)

"""

import streamlit as st

st.set_page_config(page_title="Model Insights", layout="wide")

st.title("📊 Model Performance")
st.caption("_TCN forecaster trained on 8 years of Berlin traffic data_")

st.divider()

st.warning("""
⚠️ **This page is a placeholder.**

To complete this page:
1. Run the evaluation script to generate metrics JSON files
2. Implement the visualizations described in the comments above

See the docstring at the top of this file for the full implementation plan.
""")

# Placeholder metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("MAE", "—")
with col2:
    st.metric("RMSE", "—")
with col3:
    st.metric("R²", "—")
with col4:
    st.metric("Spike F1", "—")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast Horizon", 
    "🕐 Temporal Patterns", 
    "🗺️ Spatial Performance",
    "🧪 Feature Ablation"
])

with tab1:
    st.info("TODO: Line chart showing MAE by forecast horizon (t+1 to t+24)")
    
with tab2:
    st.info("TODO: Heatmap or bar charts showing MAE by hour of day and day of week")
    
with tab3:
    st.info("TODO: Map with detectors colored by prediction error + table of hardest detectors")
    
with tab4:
    st.info("TODO: Bar chart showing ablation results (baseline vs no_weather vs no_lags vs no_calendar)")
