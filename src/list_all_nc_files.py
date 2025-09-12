import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional

import boto3

BUCKET = "noaa-oar-mlwp-data"
MODEL_ROOTS = [
    "AURO_v100_GFS",
    "AURO_v100_IFS",
    "FOUR_v100_GFS",
    "FOUR_v200_GFS",
    "FOUR_v200_IFS",
    "GRAP_v100_GFS",
    "GRAP_v100_IFS",
    "PANG_v100_GFS",
    "PANG_v100_IFS",
]

OUTPUT_CSV = Path("output/all_nc_files.csv")

# Example filename: AURO_v100_GFS_2025012212_f000_f240_06.nc
FILENAME_DT_REGEX = re.compile(r"_(\d{10})_f")  # captures YYYYMMDDHH before _f


def list_all_for_model(model: str, s3_client, year_filter: Optional[set]) -> List[Dict[str, str]]:
    prefix = f"{model}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    rows: List[Dict[str, str]] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".nc"):
                continue
            # key pattern: model/YYYY/MMDD/filename
            parts = key.split("/")
            if len(parts) < 4:
                continue
            year = parts[1]
            if year_filter and year not in year_filter:
                continue
            m = FILENAME_DT_REGEX.search(key)
            ymdh = m.group(1) if m else ""
            rows.append(
                {
                    "model": model,
                    "year": year,
                    "key": key,
                    "ymdh": ymdh,
                    "size": str(obj.get("Size")),
                }
            )
    return rows


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    only_models = os.getenv("ONLY_MODELS")  # comma separated
    year_filter_env = os.getenv("YEAR_FILTER")
    models = MODEL_ROOTS
    if only_models:
        requested = {m.strip() for m in only_models.split(",") if m.strip()}
        models = [m for m in MODEL_ROOTS if m in requested]
    year_filter = None
    if year_filter_env:
        year_filter = {y.strip() for y in year_filter_env.split(",") if y.strip().isdigit()}

    session = boto3.session.Session()
    s3 = session.client("s3", config=boto3.session.Config(signature_version="unsigned"))

    all_rows: List[Dict[str, str]] = []
    for model in models:
        print(f"Listing model {model} ...")
        try:
            model_rows = list_all_for_model(model, s3, year_filter)
            print(f"  {len(model_rows)} files")
            all_rows.extend(model_rows)
        except Exception as e:
            all_rows.append(
                {"model": model, "year": "", "key": "", "ymdh": "", "size": "", "error": str(e)}
            )

    fieldnames = ["model", "year", "key", "ymdh", "size", "error"]
    with OUTPUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            if "error" not in r:
                r["error"] = ""
            w.writerow(r)
    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
