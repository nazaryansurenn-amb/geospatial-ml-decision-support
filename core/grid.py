from __future__ import annotations
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer

GRID_M = 250
UTM_EPSG = "EPSG:32638"  # Echmiadzin area

def load_coords(coords_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(coords_csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if not {"cell_id", "lat", "lon"}.issubset(df.columns):
        raise ValueError("Coords CSV must have columns: cell_id, lat, lon")
    df["cell_id"] = df["cell_id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df[["cell_id", "lat", "lon"]].copy()

def build_grid(coords: pd.DataFrame) -> gpd.GeoDataFrame:
    to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    to_wgs = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)

    half = GRID_M / 2.0
    polys = []
    for _, r in coords.iterrows():
        x, y = to_utm.transform(float(r["lon"]), float(r["lat"]))
        sq_utm = box(x - half, y - half, x + half, y + half)
        minx, miny, maxx, maxy = sq_utm.bounds
        lon1, lat1 = to_wgs.transform(minx, miny)
        lon2, lat2 = to_wgs.transform(maxx, maxy)
        polys.append(box(lon1, lat1, lon2, lat2))

    gdf = gpd.GeoDataFrame(coords[["cell_id"]].copy(), geometry=polys, crs="EPSG:4326")
    return gdf

def load_boundary(boundary_shp: Path) -> gpd.GeoDataFrame:
    if not boundary_shp.exists():
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(boundary_shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf