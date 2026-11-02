"""Shared paths, fuel type encoding and province lookup for the pipeline scripts."""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

# ----------------------------
# PATHS
# ----------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = DATA_DIR / "cache"
OSM_CACHE_DIR = CACHE_DIR / "osm_boundaries"
MODEL_DIR = ROOT / "model"

# Raw enriched inputs
RAW_FILES_ALL = [
    DATA_DIR / "combined_ignition_enriched.csv",
    DATA_DIR / "combined_nonignition_enriched.csv",
    DATA_DIR / "combined_ignition_enriched_14_to_18.csv",
    DATA_DIR / "combined_nonignition_enriched_14_to_18.csv",
]
RAW_FILES_14_TO_18 = [
    DATA_DIR / "combined_ignition_enriched_14_to_18.csv",
    DATA_DIR / "combined_nonignition_enriched_14_to_18.csv",
]

# Feature engineering outputs
TRAIN_AB_CSV = PROCESSED_DIR / "train_alberta_osm.csv"
TRAIN_BC_CSV = PROCESSED_DIR / "train_bc_osm.csv"

# Trained model
MODEL_PATH = MODEL_DIR / "xgb_mac_v2_Mar15_AB.joblib"

# Inference / post-processing outputs
PREDICTIONS_CSV = OUTPUT_DIR / "predictions_AB.csv"
UNCERTAIN_PREDICTIONS_CSV = OUTPUT_DIR / "uncertain_predictions_AB.csv"
METRICS_CSV = OUTPUT_DIR / "metrics.csv"
UNCERTAIN_METRICS_CSV = OUTPUT_DIR / "uncertain_metrics.csv"
LOGIC_PLOTS_DIR = OUTPUT_DIR / "logic_plots"
UI_INPUT_CSV = OUTPUT_DIR / "ShowCaseUIInput.csv"

# ----------------------------
# FUEL TYPE ENCODING
# ----------------------------
# CFFDRS fuel type IDs mapped to the integer codes used during training
FUEL_TYPE_IDS = [1, 2, 3, 4, 5, 7, 11, 13, 31, 101, 102, 105, 415, 625, 650, 675]
FUEL_TYPE_TO_CODE = {fid: code for code, fid in enumerate(FUEL_TYPE_IDS)}


def encode_fuel_type(series):
    """Map raw fuel IDs to model codes. Unknown or missing IDs become -1."""
    ids = pd.to_numeric(series, errors="coerce")
    return ids.map(FUEL_TYPE_TO_CODE).fillna(-1).astype(int)


# ----------------------------
# OSM PROVINCE ASSIGNER
# ----------------------------
NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"


class OSMProvinceAssigner:
    """Assigns points to Canadian provinces using Natural Earth / OSM boundaries"""

    def __init__(self, cache_dir=OSM_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print("[OSM] Loading Canadian provincial boundaries...")
        cache_file = self.cache_dir / "canada_provinces.geojson"

        if cache_file.exists():
            print("[OSM]   Using cached boundaries")
            self.gdf = gpd.read_file(cache_file)
        else:
            print("[OSM]   Downloading boundaries (first time only)...")
            self._download_boundaries(cache_file)

        self.alberta = self._get_province_geometry("Alberta")
        self.bc = self._get_province_geometry("British Columbia")

        for name, geom in [("Alberta", self.alberta), ("British Columbia", self.bc)]:
            if geom is None:
                print(f"[OSM] ERROR: Could not find {name} in boundary data")
            else:
                shapely.prepare(geom)
                print(f"[OSM] {name} boundary loaded")

    def _download_boundaries(self, cache_file):
        try:
            self.gdf = gpd.read_file(NATURAL_EARTH_URL)
            self.gdf = self.gdf[self.gdf["admin"] == "Canada"].copy()
            self.gdf.to_file(cache_file, driver="GeoJSON")
            print(f"[OSM]   Cached to {cache_file}")
        except Exception as e:
            print(f"[OSM] Error downloading: {e}")
            raise RuntimeError("Could not download province boundaries. Check internet connection.")

    def _get_province_geometry(self, province_name):
        for col in ["name", "NAME", "NAME_1", "admin_1"]:
            if col in self.gdf.columns:
                mask = self.gdf[col].str.contains(province_name, case=False, na=False)
                if mask.any():
                    return self.gdf[mask].union_all()
        return None

    def assign(self, lat, lon):
        """Return 'Alberta', 'British Columbia' or None for each point."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        out = np.full(lat.shape, None, dtype=object)
        if self.alberta is None or self.bc is None:
            return out

        valid = ~(np.isnan(lat) | np.isnan(lon))
        in_ab = np.zeros(lat.shape, dtype=bool)
        in_bc = np.zeros(lat.shape, dtype=bool)
        in_ab[valid] = shapely.contains_xy(self.alberta, lon[valid], lat[valid])
        in_bc[valid] = shapely.contains_xy(self.bc, lon[valid], lat[valid]) & ~in_ab[valid]

        out[in_ab] = "Alberta"
        out[in_bc] = "British Columbia"
        return out

    def is_in_alberta(self, lat, lon):
        return self.assign(lat, lon) == "Alberta"
