import csv
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set

import boto3
import pandas as pd


BUCKET = "noaa-oar-mlwp-data"
PREFIXES = [
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


def load_dates(csv_path: str) -> List[datetime]:
    df = pd.read_csv(csv_path, usecols=["datetime"], parse_dates=["datetime"])
    # Normalize to hourly resolution (truncate microseconds)
    df["datetime"] = df["datetime"].dt.floor("h")
    return sorted(df["datetime"].unique())


def group_dates_by_day(dates: List[datetime]) -> Dict[str, List[datetime]]:
    grouped: Dict[str, List[datetime]] = defaultdict(list)
    for dt in dates:
        day_key = dt.strftime("%Y%m%d")
        grouped[day_key].append(dt)
    return grouped


def list_nc_for_day(s3_client, model_prefix: str, day_key: str) -> List[dict]:
    """List .nc files for a given model prefix and day (YYYYMMDD)."""
    year = day_key[:4]
    # Observed folder pattern: PREFIX/YYYY/MMDD/filename.nc where MMDD are month+day without separator
    mmdd = day_key[4:]
    prefix = f"{model_prefix}/{year}/{mmdd}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    items: List[dict] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if key.endswith(".nc"):
                items.append(obj)
    return items


def list_day_directories(s3_client, model_prefix: str) -> List[str]:
    """Return list of available day keys (YYYYMMDD) for a model prefix by traversing S3 'directories'."""
    day_keys: List[str] = []
    # List year 'directories'
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{model_prefix}/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            year_prefix = cp.get("Prefix")  # e.g. MODEL/2025/
            if not year_prefix:
                continue
            parts = year_prefix.split("/")
            if len(parts) < 2:
                continue
            year = parts[1]
            if not (year.isdigit() and len(year) == 4):
                continue
            # List day subdirectories inside the year
            for sub_page in paginator.paginate(Bucket=BUCKET, Prefix=year_prefix, Delimiter="/"):
                for sub_cp in sub_page.get("CommonPrefixes", []):
                    day_prefix = sub_cp.get("Prefix")  # e.g. MODEL/2025/0122/
                    if not day_prefix:
                        continue
                    segs = day_prefix.split("/")
                    if len(segs) < 3:
                        continue
                    mmdd = segs[2]
                    if len(mmdd) == 4 and mmdd.isdigit():
                        day_keys.append(f"{year}{mmdd}")
    return sorted(set(day_keys))


def list_nc_for_day(s3_client, model_prefix: str, day_key: str) -> List[dict]:
    year = day_key[:4]
    mmdd = day_key[4:]
    prefix = f"{model_prefix}/{year}/{mmdd}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    items: List[dict] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            if key.endswith(".nc"):
                items.append(obj)
    return items


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CSV of available .nc files matching input track dates"
    )
    parser.add_argument(
        "--input",
        default="input/western_pacific_typhoons_superfast.csv",
        help="Input CSV path with 'datetime' column",
    )
    parser.add_argument("--output", default="output/nc_file_urls.csv", help="Output CSV path")
    parser.add_argument(
        "--prefixes", nargs="*", default=PREFIXES, help="Subset of model prefixes to search"
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Limit number of distinct days processed (for testing)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N day folders per model",
    )
    args = parser.parse_args()

    dates = load_dates(args.input)
    grouped = group_dates_by_day(dates)
    day_keys = sorted(grouped.keys())
    if args.max_days is not None:
        day_keys = day_keys[: args.max_days]

    from botocore import UNSIGNED
    from botocore.config import Config

    session = boto3.session.Session()
    s3 = session.client("s3", config=Config(signature_version=UNSIGNED))

    rows = []
    csv_day_keys: Set[str] = set(grouped.keys())

    rows = []
    for model_prefix in args.prefixes:
        try:
            available_day_keys = list_day_directories(s3, model_prefix)
        except Exception as e:
            print(f"[WARN] Failed to list days for {model_prefix}: {e}", file=sys.stderr)
            continue
        target_days = [d for d in available_day_keys if d in csv_day_keys]
        if args.max_days is not None:
            target_days = target_days[: args.max_days]
        print(f"[INFO] {model_prefix}: {len(target_days)} matching day folders")
        for idx, day_key in enumerate(target_days, 1):
            if idx % max(1, args.progress_every) == 0:
                print(
                    f"[PROGRESS] {model_prefix}: {idx}/{len(target_days)} days processed, rows={len(rows)}"
                )
            try:
                objs = list_nc_for_day(s3, model_prefix, day_key)
            except Exception as e:
                print(f"[WARN] Failed list objects {model_prefix} {day_key}: {e}", file=sys.stderr)
                continue
            for obj in objs:
                key = obj.get("Key", "")
                filename = os.path.basename(key)
                init_time = ""
                try:
                    for p in filename.split("_"):
                        if len(p) == 10 and p.isdigit():
                            init_time = datetime.strptime(p, "%Y%m%d%H").isoformat()
                            break
                except Exception:
                    pass
                lm = obj.get("LastModified")
                lm_iso = lm.isoformat() if lm else ""
                rows.append(
                    {
                        "day": day_key.strip(),
                        "model_prefix": model_prefix.strip(),
                        "key": key.strip(),
                        "s3_url": f"s3://{BUCKET}/{key.strip()}",
                        "size_bytes": obj.get("Size"),
                        "last_modified": lm_iso.strip(),
                        "init_time": init_time.strip(),
                    }
                )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    columns = ["day", "model_prefix", "key", "s3_url", "size_bytes", "last_modified", "init_time"]
    if rows:
        df_out = pd.DataFrame(rows, columns=columns)
        df_out.to_csv(args.output, index=False)
    else:
        with open(args.output, "w", newline="\n") as f:
            f.write(",".join(columns) + "\n")
    print(f"[DONE] Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
