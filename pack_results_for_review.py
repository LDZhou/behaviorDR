#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pack necessary outputs for ChatGPT review.

Run from the project root directory, for example:

    python pack_results_for_review.py

It will create:

    review_package_behavior_dr.zip

Upload that zip file back to ChatGPT.
"""

from pathlib import Path
import zipfile
import json
import os
import platform
from datetime import datetime

PROJECT_ROOT = Path(".").resolve()
ZIP_NAME = "review_package_behavior_dr.zip"

# ============================================================
# Files / folders to include
# ============================================================

INCLUDE_PATHS = [
    # Core session-level dataset metadata / sample
    # "dr_sessions.csv",

    # Paper 1: setback / opt-out response
    "paper1_out/paper1_results.txt",
    "paper1_out/f1_setback_bins.csv",
    "paper1_out/f2_precool.csv",
    "paper1_out/f3_state_agg.csv",
    "paper1_out/f4_persistence.csv",
    "paper1_out/figs",

    # Paper 1 robustness
    "paper1_out/robustness",
    "paper1_out/robustness/figs",

    # Behavior-aware reliability model
    "behavior_model_out/behavior_model_report.txt",
    "behavior_model_out",
    "behavior_model_out/figs",

    # Remaining validation experiments
    "remaining_experiments_out",
    "remaining_experiments_out/figs",

    # Persistence / causality sanity checks
    "persistence_sanity_out",
    "persistence_sanity_out/figs",

    # Optional exploratory summary, if exists
    "findings.csv",
]

# Extensions worth including from output folders
ALLOW_EXT = {
    ".txt",
    ".csv",
    ".json",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".npz",
}

# Avoid huge / unnecessary model binaries and caches
EXCLUDE_NAMES = {
    "__pycache__",
    ".git",
    ".DS_Store",
}

EXCLUDE_EXT = {
    ".pkl",
    ".joblib",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".bin",
}

# Avoid enormous raw data folders
EXCLUDE_DIR_PATTERNS = [
    "dyd_data",
    "weather_cache",
    "wx_by_grid",
]

# Max size per file. Keeps upload manageable.
# Change this if needed.
MAX_FILE_MB = 80


def should_exclude_path(path: Path) -> bool:
    parts = set(path.parts)

    if any(part in EXCLUDE_NAMES for part in path.parts):
        return True

    for pat in EXCLUDE_DIR_PATTERNS:
        if pat in parts:
            return True

    if path.suffix.lower() in EXCLUDE_EXT:
        return True

    return False


def should_include_file(path: Path) -> bool:
    if should_exclude_path(path):
        return False

    if path.suffix.lower() not in ALLOW_EXT:
        return False

    try:
        size_mb = path.stat().st_size / 1024 / 1024
    except FileNotFoundError:
        return False

    if size_mb > MAX_FILE_MB:
        return False

    return True


def add_path_to_zip(zf: zipfile.ZipFile, path: Path, added: set, manifest: list):
    if not path.exists():
        return

    if path.is_file():
        files = [path]
    else:
        files = [p for p in path.rglob("*") if p.is_file()]

    for f in files:
        if not should_include_file(f):
            continue

        arcname = f.relative_to(PROJECT_ROOT).as_posix()
        if arcname in added:
            continue

        zf.write(f, arcname)
        added.add(arcname)

        stat = f.stat()
        manifest.append({
            "path": arcname,
            "size_mb": round(stat.st_size / 1024 / 1024, 3),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })


def write_session_preview(manifest_dir: Path):
    """
    Create a small preview of dr_sessions.csv so review can inspect columns,
    sample size, and basic outcome rates without needing the full raw data.
    """
    try:
        import pandas as pd
    except Exception:
        return None

    csv_path = PROJECT_ROOT / "dr_sessions.csv"
    if not csv_path.exists():
        return None

    out_path = manifest_dir / "dr_sessions_preview.json"

    # Read enough for metadata safely.
    df_head = pd.read_csv(csv_path, nrows=20)
    cols = list(df_head.columns)

    # Read selected useful columns only, if present.
    useful_cols = [
        "Identifier",
        "Session_Start",
        "HvacMode",
        "Setback_Amplitude_Mean",
        "OptOut_Immediate",
        "Opted_Out",
        "OptOut_Hold_Only",
        "OptOut_StateChange",
        "Cool_Reduction_Frac",
        "Delivered_Flex_Observed",
        "Tout_onset",
        "CDH_during",
        "Duration_Min",
        "province_state",
        "DR_Type",
    ]
    available = [c for c in useful_cols if c in cols]

    summary = {
        "file": "dr_sessions.csv",
        "columns": cols,
        "first_20_rows": df_head.to_dict(orient="records"),
    }

    try:
        df = pd.read_csv(csv_path, usecols=available) if available else pd.read_csv(csv_path)
        summary["n_rows"] = int(len(df))

        if "Identifier" in df.columns:
            summary["n_users"] = int(df["Identifier"].nunique())

        for y in ["OptOut_Immediate", "Opted_Out", "OptOut_Hold_Only", "OptOut_StateChange"]:
            if y in df.columns:
                yy = pd.to_numeric(df[y], errors="coerce")
                summary[f"{y}_mean"] = float(yy.mean())
                summary[f"{y}_nonmissing"] = int(yy.notna().sum())

        if "HvacMode" in df.columns:
            summary["HvacMode_counts"] = df["HvacMode"].astype(str).value_counts(dropna=False).head(20).to_dict()

        if "province_state" in df.columns:
            summary["province_state_counts_top20"] = df["province_state"].astype(str).value_counts(dropna=False).head(20).to_dict()

        if "Setback_Amplitude_Mean" in df.columns:
            sb = pd.to_numeric(df["Setback_Amplitude_Mean"], errors="coerce")
            summary["Setback_Amplitude_Mean_describe"] = sb.describe().to_dict()

        if "Cool_Reduction_Frac" in df.columns:
            cr = pd.to_numeric(df["Cool_Reduction_Frac"], errors="coerce")
            summary["Cool_Reduction_Frac_describe"] = cr.describe().to_dict()

    except Exception as e:
        summary["full_summary_error"] = repr(e)

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main():
    package_meta_dir = PROJECT_ROOT / "_review_package_meta"
    package_meta_dir.mkdir(exist_ok=True)

    session_preview = write_session_preview(package_meta_dir)

    manifest = []
    added = set()

    zip_path = PROJECT_ROOT / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add requested output paths
        for rel in INCLUDE_PATHS:
            add_path_to_zip(zf, PROJECT_ROOT / rel, added, manifest)

        # Add generated preview metadata
        if session_preview is not None:
            add_path_to_zip(zf, session_preview, added, manifest)

        # Add manifest last
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(PROJECT_ROOT),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "zip_name": ZIP_NAME,
            "max_file_mb": MAX_FILE_MB,
            "included_file_count": len(manifest),
            "included_files": manifest,
            "notes": [
                "Raw thermostat files, weather cache, parquet caches, and model binaries are excluded.",
                "CSV/TXT/PNG/JSON/PDF/NPZ outputs are included when under the size limit.",
                "dr_sessions_preview.json contains a lightweight summary of dr_sessions.csv.",
            ],
        }

        manifest_path = package_meta_dir / "manifest.json"
        manifest_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        zf.write(manifest_path, "_review_package_meta/manifest.json")

    size_mb = zip_path.stat().st_size / 1024 / 1024

    print("=" * 78)
    print("Review package created")
    print("=" * 78)
    print(f"Output: {zip_path}")
    print(f"Size:   {size_mb:.2f} MB")
    print(f"Files:  {len(manifest)}")
    print()
    print("Upload this zip file back to ChatGPT:")
    print(f"    {ZIP_NAME}")
    print()

    if size_mb > 200:
        print("WARNING: zip is fairly large.")
        print("If upload fails, lower MAX_FILE_MB or remove dr_sessions.csv from INCLUDE_PATHS.")


if __name__ == "__main__":
    main()