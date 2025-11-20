# CongestionAI
AI tool forecasting traffic congestion


get_data_dwd.ipynb — DWD Weather Data Collection
In this notebook, we set up an automated pipeline to retrieve hourly weather data for Berlin using the BrightSky API, selecting the DWD station 00403 (Berlin-Tegel) for consistency across the entire 2015–2025 period. We looped over each day from 2015-01-01 to 2025-01-01, requested the corresponding weather data, and aggregated all responses into a single DataFrame. Only the relevant meteorological variables were kept—temperature, dew point, precipitation, wind speed, wind direction, visibility, cloud cover, and weather conditions. The result is a unified, continuous hourly weather dataset ready to be merged with the traffic data.

get_data_wiz_api.ipynb — Traffic Data Download
This notebook builds a fully automated downloader for the Berlin TEU traffic detector archives. Because the API inconsistently uses two URL patterns depending on the year, we implemented a dual-pattern retrieval system with fallback and year-level flagging for auditing. The script systematically loops through all months from 2015 to 2025, downloads every .csv.gz file, organizes them into annual directories, and confirms consistency across base and alternative URL patterns. The outcome is a complete, structured local archive of all TEU detector data for a 10-year period, without any manual clicking.

clean_traffic_data.ipynb — Traffic Data Cleaning
In this notebook, we loaded the full 10-year traffic dataset, created a clean working copy, and standardized all column names to English. We applied a multi-stage cleaning pipeline: removed duplicates, filtered out low-quality rows based on the provided quality indicator, and discarded physically impossible speed or volume values. We then diagnosed detector reliability, identifying “broken” sensors with constant speeds or massive missing-hour gaps, and excluded them entirely. Additional diagnostics flagged detectors with very high missingness, which were also removed according to our threshold. The end result is a high-quality, reliable dataset of traffic speeds and volumes from functioning detectors, ready for preprocessing and integration with weather and spatial data.

preprocess_traffic_data.ipynb — Traffic Data pre-processing

- hour, day of week, is_weekend, month, season, holiday indicator, rush hour, school_holiday (and make sure to not repeat multiple times date, time stamps, hours as it may already exist in some form)
- Free-Flow Speed Estimation (per detector) and congestion index. However, there are no limits in germany for autoroutes, how to tackle that ?
- weather data will be merged at the very end, dont consider it for now
- spatial enrichment (everything available in the metadata)
