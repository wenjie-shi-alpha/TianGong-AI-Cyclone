"""Compatibility facade for the refactored initial tracker package.

The historical monolithic implementation has been split into the modular
package located in ``src/initial_tracker``.  This file keeps the public API
stable for callers such as ``extractSyst.py`` while delegating all heavy
lifting to the new modules.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from initial_tracker import (
    NoEyeException,
    Tracker,
    _DsAdapter,
    _Metadata,
    _SimpleBatch,
    _build_batch_from_ds,
    _build_batch_from_ds_fast,
    _load_all_points,
    _load_initial_points,
    _safe_get,
    _select_initials_for_time,
    _to_0360,
    extrapolate,
    get_box,
    get_closest_min,
    havdist,
    track_file_with_initials,
)
from initial_tracker import __all__ as _PKG_ALL

logger = logging.getLogger(__name__)

__all__ = list(_PKG_ALL) + ["main"]


def main(args: Optional[List[str]] = None) -> None:
    """Command line entry-point that mirrors the legacy behaviour."""
    parser = argparse.ArgumentParser(description="基于初始点与 NetCDF 文件的热带气旋逐步追踪")
    parser.add_argument(
        "--initials_csv",
        default=str(Path("input") / "western_pacific_typhoons_superfast.csv"),
        help="包含每个气旋起始点的 CSV 路径",
    )
    parser.add_argument(
        "--nc_dir",
        default=str(Path("data") / "nc_files"),
        help="AWS 下载的 NetCDF 文件目录",
    )
    parser.add_argument(
        "--limit_storms",
        type=int,
        default=None,
        help="最多处理的气旋数量 (按 CSV 首行去重后的 storm_id)",
    )
    parser.add_argument(
        "--limit_files",
        type=int,
        default=None,
        help="最多处理的 NetCDF 文件数量",
    )
    parser.add_argument(
        "--time_window_hours",
        type=int,
        default=6,
        help="匹配气旋初始点的时间窗口(小时), 仅在该窗口内存在记录的气旋会被追踪",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path("track_output")),
        help="输出气旋追踪路径 CSV 目录",
    )

    parsed = parser.parse_args(args=args)

    all_points = _load_all_points(Path(parsed.initials_csv))
    nc_dir = Path(parsed.nc_dir)
    nc_files = sorted([p for p in nc_dir.glob("*.nc") if p.is_file()])
    if parsed.limit_files is not None:
        nc_files = nc_files[: parsed.limit_files]
    output_dir = Path(parsed.output_dir)

    logger.info("CSV 总记录数: %s | NetCDF 文件数: %s", len(all_points), len(nc_files))
    total_written: List[Path] = []
    for nc in nc_files:
        try:
            written = track_file_with_initials(
                nc,
                all_points,
                output_dir,
                max_storms=parsed.limit_storms,
                time_window_hours=parsed.time_window_hours,
            )
            total_written.extend(written)
            logger.info("%s: 写入 %s 条路径", nc.name, len(written))
        except Exception as exc:
            logger.exception("处理 %s 失败: %s", nc.name, exc)

    print(f"完成: 共写入 {len(total_written)} 条路径到 {output_dir}")


if __name__ == "__main__":
    main()
