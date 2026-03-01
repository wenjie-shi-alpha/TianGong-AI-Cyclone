"""Command-line entry point for the environment extraction workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .deps import ensure_available
from .pipeline import process_nc_files, streaming_from_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一体化: 下载->追踪->环境分析")
    parser.add_argument("--csv", default="output/nc_file_urls.csv", help="含s3_url的列表CSV")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理前N个NC文件 (默认处理全部)",
    )
    parser.add_argument("--nc", default=None, help="直接指定单个NC文件 (跳过下载与追踪)")
    parser.add_argument(
        "--tracks",
        default=None,
        help="直接指定轨迹CSV (跳过追踪)\n若与--nc同时给出则只做环境分析",
    )
    parser.add_argument("--no-clean", action="store_true", help="分析后不删除NC")
    parser.add_argument("--keep-nc", action="store_true", help="同 --no-clean (兼容)")
    parser.add_argument("--auto", action="store_true", help="无轨迹则自动运行追踪")
    parser.add_argument("--search-range", type=float, default=3.0, help="追踪搜索范围")
    parser.add_argument("--memory", type=int, default=3, help="追踪记忆时间步")
    parser.add_argument(
        "--initials",
        default=str(Path("input") / "matched_cyclone_tracks_2021onwards.csv"),
        help="initialTracker 初始点CSV",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="使用旧的批量模式: 先全部下载+追踪, 再统一做环境分析",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="并行运行的进程数 (>=1)。最大并行任务数与进程数一致",
    )
    parser.add_argument(
        "--concise-log",
        action="store_true",
        help="启用精简日志模式，仅输出文件完成情况",
    )
    return parser


def _prepare_batch_targets(
    csv_path: Path, limit: int | None, initials_csv: Path, concise_log: bool = False
) -> list[Path]:
    import pandas as pd

    from initialTracker import track_file_with_initials as it_track_file_with_initials
    from initialTracker import _load_all_points as it_load_all_points

    from .workflow_utils import (
        combine_initial_tracker_outputs,
        download_s3_public,
        extract_forecast_tag,
        sanitize_filename,
    )

    def detail(message: str) -> None:
        if not concise_log:
            print(message)

    def summary(message: str) -> None:
        print(message)

    if not csv_path.exists():
        summary(f"❌ CSV不存在: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required_cols = {"s3_url", "model_prefix", "init_time"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        summary(f"❌ CSV缺少必要列: {missing}")
        sys.exit(1)

    if limit is not None:
        df = df.head(limit)

    persist_dir = Path("data/nc_files")
    persist_dir.mkdir(parents=True, exist_ok=True)
    track_dir = Path("track_single")
    track_dir.mkdir(exist_ok=True)

    if initials_csv.exists():
        initials_path = initials_csv
    else:
        fallback = Path("input/matched_cyclone_tracks_2021onwards.csv")
        if fallback.exists():
            summary(f"⚠️ 指定初始点文件不存在, 使用默认: {fallback}")
            initials_path = fallback
        else:
            summary(f"❌ 找不到初始点CSV: {initials_csv}")
            sys.exit(1)
    initials_df = it_load_all_points(initials_path)

    prepared: list[Path] = []

    def remove_nc_file(path: Path, reason: str) -> None:
        """删除无法用于后续分析的 NC 文件。"""
        try:
            path.unlink()
            detail(f"🧹 已删除NC ({reason})")
        except FileNotFoundError:
            pass
        except Exception as exc:
            summary(f"⚠️ 删除NC失败({reason}): {exc}")
    detail(f"⬇️ [批量模式] 逐项下载与追踪 (limit={limit})")
    for idx, row in df.iterrows():
        s3_url = row["s3_url"]
        model_prefix = row["model_prefix"]
        init_time = row["init_time"]
        fname = Path(s3_url).name
        forecast_tag = extract_forecast_tag(fname)
        safe_prefix = sanitize_filename(model_prefix)
        safe_init = sanitize_filename(init_time.replace(":", "").replace("-", ""))
        combined_track_csv = track_dir / f"tracks_{safe_prefix}_{safe_init}_{forecast_tag}.csv"
        nc_local = persist_dir / fname
        nc_stem = nc_local.stem

        detail(f"\n[{idx+1}/{len(df)}] ▶️ 处理: {fname}")

        if not nc_local.exists():
            try:
                detail(f"⬇️  下载NC: {s3_url}")
                download_s3_public(s3_url, nc_local)
            except Exception as exc:
                summary(f"❌ 下载失败: {exc}")
                continue
        else:
            detail("📦 已存在NC文件, 复用")

        track_csv: Path | None = None

        if combined_track_csv.exists():
            track_csv = combined_track_csv
            detail("🗺️  已存在轨迹CSV, 跳过追踪")
        else:
            single_candidates = sorted(track_dir.glob(f"track_*_{nc_stem}.csv"))
            if len(single_candidates) == 1:
                try:
                    combined = combine_initial_tracker_outputs(single_candidates, nc_local)
                    if combined is not None and not combined.empty:
                        combined.to_csv(single_candidates[0], index=False)
                    track_csv = single_candidates[0]
                    detail("🗺️  发现单条轨迹文件, 已更新后直接使用")
                except Exception as exc:
                    summary(f"⚠️ 单轨迹文件格式更新失败: {exc}")
            elif len(single_candidates) > 1:
                try:
                    combined = combine_initial_tracker_outputs(single_candidates, nc_local)
                    if combined is not None and not combined.empty:
                        combined.to_csv(combined_track_csv, index=False)
                        track_csv = combined_track_csv
                        detail(
                            f"🗺️  发现多条单独轨迹文件, 已合并生成 {combined_track_csv.name}"
                        )
                except Exception as exc:
                    summary(f"⚠️ 合并已有轨迹失败: {exc}")

        if track_csv is not None:
            prepared.append(nc_local)
            continue

        try:
            per_storm = it_track_file_with_initials(nc_local, initials_df, track_dir)
            if not per_storm:
                detail("⚠️ 无有效轨迹 -> 删除NC")
                remove_nc_file(nc_local, "无轨迹")
                continue
            combined = combine_initial_tracker_outputs(per_storm, nc_local)
            if combined is None or combined.empty:
                detail("⚠️ 合并轨迹失败 -> 删除NC")
                remove_nc_file(nc_local, "无轨迹")
                continue

            if combined["particle"].nunique() == 1:
                single_path = Path(per_storm[0])
                combined.to_csv(single_path, index=False)
                track_csv = single_path
                detail(f"💾 保存单条轨迹: {single_path.name}")
                if combined_track_csv.exists():
                    try:
                        combined_track_csv.unlink()
                    except Exception:
                        pass
            else:
                combined.to_csv(combined_track_csv, index=False)
                track_csv = combined_track_csv
                detail(
                    f"💾 合并保存轨迹: {combined_track_csv.name} (含 {combined['particle'].nunique()} 条路径)"
                )
        except Exception as exc:
            summary(f"❌ 追踪失败: {exc}")
            remove_nc_file(nc_local, "追踪失败")
            continue

        if track_csv is None:
            remove_nc_file(nc_local, "无轨迹")
            continue

        prepared.append(nc_local)

    if not prepared:
        summary("❌ 未成功准备任何NC文件")
        sys.exit(1)

    return prepared


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    ensure_available()

    def detail(message: str) -> None:
        if not args.concise_log:
            print(message)

    def summary(message: str) -> None:
        print(message)

    logs_root = Path("final_single_output") / "logs"

    detail("🌀 一体化热带气旋分析流程启动")
    detail("=" * 60)

    if args.nc:
        nc_path = Path(args.nc)
        if not nc_path.exists():
            summary(f"❌ 指定NC不存在: {nc_path}")
            sys.exit(1)
        target_nc_files = [nc_path]
        detail("📦 单文件分析模式")
    else:
        if args.batch:
            target_nc_files = _prepare_batch_targets(
                Path(args.csv), args.limit, Path(args.initials), args.concise_log
            )
            detail(f"📦 待环境分析NC数量: {len(target_nc_files)}")
        else:
            detail("🚚 启用流式顺序处理: 每个NC独立完成(下载->追踪->环境分析->清理)")
            streaming_from_csv(
                csv_path=Path(args.csv),
                limit=args.limit,
                search_range=args.search_range,
                memory=args.memory,
                keep_nc=(args.no_clean or args.keep_nc),
                initials_csv=Path(args.initials) if args.initials else None,
                processes=max(1, args.processes),
                concise_log=args.concise_log,
                logs_root=logs_root,
            )
            detail("🎯 流式处理完成 (无需进入批量后处理循环)")
            return

    process_nc_files(
        target_nc_files,
        args,
        concise_log=args.concise_log,
        logs_root=logs_root,
    )


__all__ = ["main", "build_parser"]
