"""
Detector ↔ Road Mapping
=======================

Maps between detector IDs and road network geometry.

The model predicts congestion for ~380 detectors, but the GUI displays
~73,000 road segments. This module maps each road segment to its nearest
detector so we can display forecasts on the full road network.

Functions:
----------
- load_detector_locations() -> pd.DataFrame
    Load detector id, lat, lon from training data

- build_road_to_detector_mapping(roads_gdf, detectors_df) -> dict
    Spatial join: each road_id -> nearest detector_id

- load_road_to_detector_mapping() -> dict
    Load cached mapping from disk

- expand_detector_forecast_to_roads(detector_forecast: dict) -> dict
    Convert {detector_id: [24 values]} -> {road_id: [24 values]}

Data Files:
-----------
- road_to_detector_mapping.json: {road_id: detector_id}
- detector_locations.parquet: detector_id, lon, lat
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path(__file__).parent
MAPPING_FILE = DATA_DIR / "road_to_detector_mapping.json"
DETECTOR_LOCATIONS_FILE = DATA_DIR / "detector_to_wfs_segment.parquet"

# Cached mapping
_road_to_detector: Optional[Dict[str, str]] = None


def load_detector_locations() -> pd.DataFrame:
    """
    Load detector locations from the parquet file.
    
    Returns:
        DataFrame with columns: detector_id, lon, lat
    """
    if not DETECTOR_LOCATIONS_FILE.exists():
        raise FileNotFoundError(f"Detector locations not found: {DETECTOR_LOCATIONS_FILE}")
    
    df = pd.read_parquet(DETECTOR_LOCATIONS_FILE)
    # Get unique detector locations
    return df[["detector_id", "lon", "lat"]].drop_duplicates("detector_id")


def build_road_to_detector_mapping(
    roads_gdf,  # GeoDataFrame with road_id and geometry
    detectors_df: pd.DataFrame,  # DataFrame with detector_id, lon, lat
    max_distance_km: float = 5.0,
) -> Dict[str, str]:
    """
    Build mapping from road segments to nearest detector.
    
    Uses BallTree for efficient nearest-neighbor search.
    
    Args:
        roads_gdf: GeoDataFrame of road network with 'road_id' and 'geometry'
        detectors_df: DataFrame with 'detector_id', 'lon', 'lat'
        max_distance_km: Maximum distance to consider (roads further get None)
    
    Returns:
        Dict mapping road_id -> detector_id
    """
    from sklearn.neighbors import BallTree
    
    logger.info(f"Building road-to-detector mapping for {len(roads_gdf)} roads...")
    
    # Get detector coordinates in radians for haversine distance
    det_coords = np.radians(detectors_df[["lat", "lon"]].values)
    det_ids = detectors_df["detector_id"].astype(str).values
    
    # Build BallTree for fast nearest-neighbor lookup
    tree = BallTree(det_coords, metric="haversine")
    
    # Get road centroids
    centroids = roads_gdf.geometry.centroid
    road_coords = np.radians(np.column_stack([centroids.y, centroids.x]))
    road_ids = roads_gdf["road_id"].values
    
    # Find nearest detector for each road
    # Returns distance in radians (earth radius ~ 6371 km)
    distances, indices = tree.query(road_coords, k=1)
    distances_km = distances.flatten() * 6371  # Convert to km
    
    # Build mapping
    mapping = {}
    for i, road_id in enumerate(road_ids):
        if distances_km[i] <= max_distance_km:
            mapping[road_id] = det_ids[indices[i, 0]]
        else:
            # Road too far from any detector - use closest anyway
            mapping[road_id] = det_ids[indices[i, 0]]
            logger.debug(f"Road {road_id} is {distances_km[i]:.1f}km from nearest detector")
    
    logger.info(f"Mapped {len(mapping)} roads to {len(set(mapping.values()))} unique detectors")
    
    return mapping


def save_road_to_detector_mapping(mapping: Dict[str, str], path: Optional[Path] = None) -> None:
    """Save mapping to JSON file."""
    path = path or MAPPING_FILE
    with open(path, "w") as f:
        json.dump(mapping, f)
    logger.info(f"Saved road-to-detector mapping to {path}")


def load_road_to_detector_mapping() -> Dict[str, str]:
    """
    Load cached road-to-detector mapping.
    
    Returns:
        Dict mapping road_id -> detector_id
    
    Raises:
        FileNotFoundError if mapping doesn't exist (need to run build first)
    """
    global _road_to_detector
    
    if _road_to_detector is not None:
        return _road_to_detector
    
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Road-to-detector mapping not found: {MAPPING_FILE}. "
            f"Run: python -m backend.data.mapping --build"
        )
    
    with open(MAPPING_FILE) as f:
        _road_to_detector = json.load(f)
    
    logger.info(f"Loaded road-to-detector mapping: {len(_road_to_detector)} roads")
    return _road_to_detector


def expand_detector_forecast_to_roads(
    detector_forecast: Dict[str, List[float]],
    default_value: float = 0.3,
) -> Dict[str, List[float]]:
    """
    Expand detector-level forecasts to road-level forecasts.
    
    Takes predictions for ~380 detectors and expands to ~73,000 roads
    by mapping each road to its nearest detector.
    
    Args:
        detector_forecast: {detector_id: [24 congestion values]}
        default_value: Value for roads with no matching detector
    
    Returns:
        {road_id: [24 congestion values]}
    """
    mapping = load_road_to_detector_mapping()
    
    # Determine forecast horizon from first detector
    if detector_forecast:
        first_det = next(iter(detector_forecast.values()))
        horizon = len(first_det)
    else:
        horizon = 25  # 0-24 hours
    
    default_forecast = [default_value] * horizon
    
    road_forecast = {}
    missing_count = 0
    
    for road_id, detector_id in mapping.items():
        if detector_id in detector_forecast:
            road_forecast[road_id] = detector_forecast[detector_id]
        else:
            # Try string conversion (detector_id might be int in forecast)
            det_str = str(detector_id)
            if det_str in detector_forecast:
                road_forecast[road_id] = detector_forecast[det_str]
            else:
                road_forecast[road_id] = default_forecast
                missing_count += 1
    
    if missing_count > 0:
        logger.warning(
            f"{missing_count}/{len(mapping)} roads mapped to detectors without forecasts. "
            f"Using default value {default_value}."
        )
    
    return road_forecast


def main():
    """CLI for building the mapping."""
    import argparse
    import geopandas as gpd
    
    parser = argparse.ArgumentParser(description="Build road-to-detector mapping")
    parser.add_argument("--build", action="store_true", help="Build and save mapping")
    parser.add_argument(
        "--roads",
        default="gui/data/berlin_roads.geojson",
        help="Path to roads GeoJSON"
    )
    args = parser.parse_args()
    
    if args.build:
        # Load data
        print(f"Loading roads from {args.roads}...")
        roads = gpd.read_file(args.roads)
        
        print("Loading detector locations...")
        detectors = load_detector_locations()
        
        # Build mapping
        mapping = build_road_to_detector_mapping(roads, detectors)
        
        # Save
        save_road_to_detector_mapping(mapping)
        print(f"Done! Mapping saved to {MAPPING_FILE}")
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
