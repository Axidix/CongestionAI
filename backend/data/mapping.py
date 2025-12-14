"""
Detector ↔ Road Mapping
=======================

Maps between detector IDs and road network geometry.

The model predicts congestion for ~380 detectors, but the GUI displays
~73,000 road segments. This module maps each road segment to its k nearest
detectors and uses Inverse Distance Weighting (IDW) interpolation to 
produce smooth congestion values across the road network.

Interpolation Method:
--------------------
For each road segment, we find the k=5 nearest detectors and compute:
    congestion = Σ(w_i * c_i) / Σ(w_i)
    where w_i = 1 / d_i^2 (inverse square distance weight)

This produces smooth gradients between detectors rather than hard boundaries.

Functions:
----------
- load_detector_locations() -> pd.DataFrame
    Load detector id, lat, lon from training data

- build_road_to_detector_mapping(roads_gdf, detectors_df, k) -> dict
    Spatial mapping: each road_id -> [(detector_id, weight), ...]

- load_road_to_detector_mapping() -> dict
    Load cached mapping from disk

- expand_detector_forecast_to_roads(detector_forecast: dict) -> dict
    Convert {detector_id: [24 values]} -> {road_id: [24 interpolated values]}

Data Files:
-----------
- road_to_detector_mapping.json: {road_id: [[det_id, weight], ...]}
- detector_locations.parquet: detector_id, lon, lat
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path(__file__).parent
MAPPING_FILE = DATA_DIR / "road_to_detector_mapping.json"
DETECTOR_LOCATIONS_FILE = DATA_DIR / "detector_to_wfs_segment.parquet"

# Cached mapping
_road_to_detector: Optional[Dict[str, List[Tuple[str, float]]]] = None

# IDW parameters
DEFAULT_K_NEIGHBORS = 5
DEFAULT_POWER = 2  # inverse square weighting


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
    k: int = DEFAULT_K_NEIGHBORS,
    power: float = DEFAULT_POWER,
) -> Dict[str, List[List]]:
    """
    Build IDW mapping from road segments to k nearest detectors with weights.
    
    Uses BallTree for efficient k-nearest-neighbor search, then computes
    inverse distance weights for interpolation.
    
    Args:
        roads_gdf: GeoDataFrame of road network with 'road_id' and 'geometry'
        detectors_df: DataFrame with 'detector_id', 'lon', 'lat'
        k: Number of nearest detectors to use for interpolation
        power: Power for inverse distance weighting (2 = inverse square)
    
    Returns:
        Dict mapping road_id -> [[detector_id, weight], ...]
        Weights are normalized to sum to 1.0 for each road.
    """
    from sklearn.neighbors import BallTree
    
    logger.info(f"Building IDW road-to-detector mapping (k={k}) for {len(roads_gdf)} roads...")
    
    # Get detector coordinates in radians for haversine distance
    det_coords = np.radians(detectors_df[["lat", "lon"]].values)
    det_ids = detectors_df["detector_id"].astype(str).values
    
    # Build BallTree for fast k-NN lookup
    tree = BallTree(det_coords, metric="haversine")
    
    # Get road centroids
    centroids = roads_gdf.geometry.centroid
    road_coords = np.radians(np.column_stack([centroids.y, centroids.x]))
    road_ids = roads_gdf["road_id"].values
    
    # Find k nearest detectors for each road
    # Returns distance in radians (earth radius ~ 6371 km)
    distances, indices = tree.query(road_coords, k=min(k, len(det_ids)))
    distances_km = distances * 6371  # Convert to km
    
    # Build mapping with IDW weights
    mapping = {}
    for i, road_id in enumerate(road_ids):
        dists = distances_km[i]
        idxs = indices[i]
        
        # Compute inverse distance weights
        # Add small epsilon to avoid division by zero for exact matches
        weights = 1.0 / (dists + 0.001) ** power
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        # Store as [[detector_id, weight], ...]
        mapping[road_id] = [
            [det_ids[idx], round(float(w), 6)]
            for idx, w in zip(idxs, weights)
        ]
    
    # Log stats
    unique_detectors = set()
    for entries in mapping.values():
        for det_id, _ in entries:
            unique_detectors.add(det_id)
    
    logger.info(f"Mapped {len(mapping)} roads using {len(unique_detectors)} unique detectors")
    logger.info(f"Average detectors per road: {k}, using IDW interpolation (power={power})")
    
    return mapping


def save_road_to_detector_mapping(mapping: Dict[str, str], path: Optional[Path] = None) -> None:
    """Save mapping to JSON file."""
    path = path or MAPPING_FILE
    with open(path, "w") as f:
        json.dump(mapping, f)
    logger.info(f"Saved road-to-detector mapping to {path}")


def load_road_to_detector_mapping() -> Dict[str, List[List]]:
    """
    Load cached road-to-detector IDW mapping.
    
    Returns:
        Dict mapping road_id -> [[detector_id, weight], ...]
    
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
    
    logger.info(f"Loaded road-to-detector IDW mapping: {len(_road_to_detector)} roads")
    return _road_to_detector


def expand_detector_forecast_to_roads(
    detector_forecast: Dict[str, List[float]],
    default_value: float = 0.3,
) -> Dict[str, List[float]]:
    """
    Expand detector-level forecasts to road-level forecasts using IDW interpolation.
    
    For each road, computes weighted average of forecasts from k nearest detectors:
        road_forecast[h] = Σ(weight_i * detector_forecast_i[h])
    
    This produces smooth spatial gradients rather than hard detector boundaries.
    
    Args:
        detector_forecast: {detector_id: [24 congestion values]}
        default_value: Value for roads with no matching detectors
    
    Returns:
        {road_id: [24 interpolated congestion values]}
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
    partial_coverage_count = 0
    
    for road_id, detector_weights in mapping.items():
        # Collect forecasts and weights for available detectors
        available_forecasts = []
        available_weights = []
        
        for det_id, weight in detector_weights:
            # Try to find detector forecast (handle string/int ID mismatch)
            forecast = None
            if det_id in detector_forecast:
                forecast = detector_forecast[det_id]
            elif str(det_id) in detector_forecast:
                forecast = detector_forecast[str(det_id)]
            
            if forecast is not None:
                available_forecasts.append(forecast)
                available_weights.append(weight)
        
        if not available_forecasts:
            # No detectors have forecasts - use default
            road_forecast[road_id] = default_forecast
            partial_coverage_count += 1
        else:
            # IDW interpolation: weighted average
            weights_arr = np.array(available_weights)
            weights_arr = weights_arr / weights_arr.sum()  # Re-normalize for available detectors
            
            forecasts_arr = np.array(available_forecasts)  # Shape: (n_available, horizon)
            interpolated = np.average(forecasts_arr, axis=0, weights=weights_arr)
            
            road_forecast[road_id] = [round(float(v), 4) for v in interpolated]
    
    if partial_coverage_count > 0:
        logger.warning(
            f"{partial_coverage_count}/{len(mapping)} roads have no detector coverage. "
            f"Using default value {default_value}."
        )
    
    return road_forecast


def main():
    """CLI for building the mapping."""
    import argparse
    import geopandas as gpd
    
    parser = argparse.ArgumentParser(description="Build road-to-detector IDW mapping")
    parser.add_argument("--build", action="store_true", help="Build and save mapping")
    parser.add_argument(
        "--roads",
        default="gui/data/berlin_roads.geojson",
        help="Path to roads GeoJSON"
    )
    parser.add_argument(
        "-k", "--neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help=f"Number of nearest detectors for IDW (default: {DEFAULT_K_NEIGHBORS})"
    )
    args = parser.parse_args()
    
    if args.build:
        # Load data
        print(f"Loading roads from {args.roads}...")
        roads = gpd.read_file(args.roads)
        
        print("Loading detector locations...")
        detectors = load_detector_locations()
        
        # Build mapping with IDW
        print(f"Building IDW mapping with k={args.neighbors} neighbors...")
        mapping = build_road_to_detector_mapping(roads, detectors, k=args.neighbors)
        
        # Save
        save_road_to_detector_mapping(mapping)
        print(f"Done! IDW mapping saved to {MAPPING_FILE}")
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
