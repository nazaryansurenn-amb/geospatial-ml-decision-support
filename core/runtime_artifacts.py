from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from core.crop import build_stable_crop_map, classify_crop_year
from core.features import build_reference_monthly_baselines, build_temporal_features
from core.io import deduplicate_cell_weeks


RUNTIME_SCHEMA_VERSION = 1
DEFAULT_CROP_THRESHOLDS = {
    "ndvi_green_thr": 0.30,
    "perennial_green_weeks": 10,
    "perennial_p25": 0.25,
}
REFERENCE_COLUMNS = [
    "cell_id",
    "week_start",
    "week_end",
    "year",
    "month",
    "doy",
    "NDVI_med",
    "NDMI_med",
    "dNDMI",
    "Rain_mm_7d",
    "Temp_C_7d",
    "valid_s2",
    "veg_flag",
]


def artifact_paths(runtime_dir: Path) -> dict[str, Path]:
    return {
        "manifest": runtime_dir / "manifest.json",
        "historical_reference": runtime_dir / "historical_reference.parquet",
        "monthly_baselines": runtime_dir / "monthly_baselines.parquet",
        "stable_crop_map": runtime_dir / "stable_crop_map.parquet",
    }


def historical_features_path(runtime_dir: Path, year: int) -> Path:
    return runtime_dir / f"historical_features_{int(year)}.parquet"


def historical_crop_map_path(runtime_dir: Path, year: int) -> Path:
    return runtime_dir / f"historical_crop_map_{int(year)}.parquet"


def load_runtime_manifest(runtime_dir: Path) -> dict:
    path = artifact_paths(runtime_dir)["manifest"]
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return manifest if manifest.get("schema_version") == RUNTIME_SCHEMA_VERSION else {}


def _write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)


def build_runtime_artifacts(
    exports: pd.DataFrame,
    runtime_dir: Path,
    *,
    source_signature: list[dict] | None = None,
) -> dict:
    started = perf_counter()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(runtime_dir)

    clean = deduplicate_cell_weeks(exports.dropna(subset=["cell_id", "week_start"]).copy())
    clean["cell_id"] = clean["cell_id"].astype(str)
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean = clean.dropna(subset=["year"])
    clean["year"] = clean["year"].astype(int)
    years = sorted(clean["year"].unique().tolist())

    reference = clean[[column for column in REFERENCE_COLUMNS if column in clean.columns]].copy()
    _write_parquet(reference, paths["historical_reference"])

    baselines = build_reference_monthly_baselines(reference)
    _write_parquet(baselines, paths["monthly_baselines"])

    stable_crop_map = build_stable_crop_map(reference, **DEFAULT_CROP_THRESHOLDS)
    _write_parquet(stable_crop_map, paths["stable_crop_map"])

    featured = build_temporal_features(clean)
    drop_columns = [
        column
        for column in ("system:index", ".geo", "_src", "_src_mtime_ns", "source_file")
        if column in featured.columns
    ]
    if drop_columns:
        featured = featured.drop(columns=drop_columns)

    yearly_rows = {}
    for year in years:
        year_frame = featured[featured["year"].astype(int).eq(int(year))].copy()
        _write_parquet(year_frame, historical_features_path(runtime_dir, year))
        year_crop_map = classify_crop_year(year_frame, **DEFAULT_CROP_THRESHOLDS)
        _write_parquet(year_crop_map, historical_crop_map_path(runtime_dir, year))
        yearly_rows[str(year)] = int(len(year_frame))

    artifact_files = [
        paths["historical_reference"],
        paths["monthly_baselines"],
        paths["stable_crop_map"],
        *[historical_features_path(runtime_dir, year) for year in years],
        *[historical_crop_map_path(runtime_dir, year) for year in years],
    ]
    manifest = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": years,
        "historical_rows": int(len(clean)),
        "historical_rows_by_year": yearly_rows,
        "crop_thresholds": DEFAULT_CROP_THRESHOLDS,
        "source_signature": source_signature or [],
        "artifacts": {
            path.name: {"bytes": int(path.stat().st_size)}
            for path in artifact_files
        },
        "total_artifact_bytes": int(sum(path.stat().st_size for path in artifact_files)),
        "build_seconds": round(perf_counter() - started, 3),
    }
    temporary_manifest = paths["manifest"].with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(paths["manifest"])
    return manifest
