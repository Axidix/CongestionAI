"""
Detector ↔ Road Mapping
=======================

Maps between detector IDs and road network geometry.

Functions:
----------
- load_detector_metadata() -> pd.DataFrame
    Load detector info: id, lat, lon, road_name, direction

- get_detector_locations() -> dict
    Return {detector_id: (lat, lon)} for all detectors

- detector_to_road_segments(detector_id) -> list
    Map detector to OSM road segment IDs (for GUI routing)

- road_segment_to_detectors(segment_id) -> list
    Reverse mapping: which detectors cover this road segment

- get_nearest_detector(lat, lon) -> str
    Find closest detector to a given point

Data Files:
-----------
- detector_metadata.csv: detector_id, lat, lon, name, road_type
- detector_road_mapping.json: {detector_id: [road_segment_ids]}

Used By:
--------
- GUI: To show forecasts on map
- Service: To map predictions to road network
- Features: To get lat/lon for each detector

Notes:
------
- Mapping was established during data preprocessing
- Some road segments may not have detector coverage
- For uncovered segments, use nearest detector or interpolation
"""

# TODO: Implement
#
# def load_detector_metadata() -> pd.DataFrame:
#     """
#     Load from prepared_data or backend/data/
#     """
#     pass
#
# def get_detector_locations() -> dict:
#     """
#     Return {detector_id: {"lat": x, "lon": y, "name": "..."}}
#     """
#     pass
#
# def build_road_segment_mapping(G, detectors_df) -> dict:
#     """
#     For each detector, find nearby road segments in OSM graph.
#     Save mapping for fast lookup.
#     """
#     pass
#
# def get_forecast_for_road_segment(segment_id: str, forecast_dict: dict) -> float:
#     """
#     Given a road segment, return the predicted congestion.
#     Uses detector mapping or interpolation.
#     """
#     pass
