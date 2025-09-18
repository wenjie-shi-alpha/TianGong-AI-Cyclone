"""Pipeline orchestration for the tropical cyclone environment extractor."""

from __future__ import annotations

from pathlib import Path

from .extractor import TCEnvironmentalSystemsExtractor
from .workflow_utils import (
    combine_initial_tracker_outputs,
    download_s3_public,
    extract_forecast_tag,
    sanitize_filename,
)


# ================= 新增: 流式顺序处理函数 =================
def streaming_from_csv(
    csv_path: Path,
    limit: int | None = None,
    search_range: float = 3.0,
    memory: int = 3,
    keep_nc: bool = False,
    initials_csv: Path | None = None,
):
    """逐行读取CSV, 每个NC文件执行: 下载 -> 追踪 -> 环境分析 -> (可选删除)

    与原批量模式最大区别: 不预先下载全部; 每个文件完成后即可释放磁盘。
    """
    if not csv_path.exists():
        print(f"❌ CSV不存在: {csv_path}")
        return
    import pandas as pd, traceback
    from initialTracker import track_file_with_initials as it_track_file_with_initials
    from initialTracker import _load_all_points as it_load_initial_points

    df = pd.read_csv(csv_path)
    required_cols = {"s3_url", "model_prefix", "init_time"}
    if not required_cols.issubset(df.columns):
        print(f"❌ CSV缺少必要列: {required_cols - set(df.columns)}")
        return
    if limit is not None:
        df = df.head(limit)
    print(f"📄 流式待处理数量: {len(df)} (limit={limit})")

    persist_dir = Path("data/nc_files")
    persist_dir.mkdir(parents=True, exist_ok=True)
    track_dir = Path("track_single")
    track_dir.mkdir(exist_ok=True)
    final_dir = Path("final_single_output")
    final_dir.mkdir(exist_ok=True)

    processed = 0
    skipped = 0
    for idx, row in df.iterrows():
        s3_url = row["s3_url"]
        model_prefix = row["model_prefix"]
        init_time = row["init_time"]
        fname = Path(s3_url).name
        forecast_tag = extract_forecast_tag(fname)
        safe_prefix = sanitize_filename(model_prefix)
        safe_init = sanitize_filename(init_time.replace(":", "").replace("-", ""))
        track_csv = track_dir / f"tracks_{safe_prefix}_{safe_init}_{forecast_tag}.csv"
        nc_local = persist_dir / fname

        print(f"\n[{idx+1}/{len(df)}] ▶️ 处理: {fname}")

        existing_json = list(final_dir.glob(f"{Path(fname).stem}_TC_Analysis_*.json"))
        if existing_json:
            non_empty = [p for p in existing_json if p.stat().st_size > 10]
            if non_empty:
                print(f"⏭️  已存在最终JSON({len(non_empty)}) -> 跳过")
                skipped += 1
                continue

        if not nc_local.exists():
            try:
                print(f"⬇️  下载NC: {s3_url}")
                download_s3_public(s3_url, nc_local)
            except Exception as e:
                print(f"❌ 下载失败, 跳过: {e}")
                skipped += 1
                continue
        else:
            print("📦 已存在NC文件, 复用")

        if not track_csv.exists():
            try:
                print("🧭 使用 initialTracker 执行追踪...")
                initials_path = initials_csv or Path("input/western_pacific_typhoons_superfast.csv")
                initials_df = it_load_initial_points(initials_path)
                per_storm_csvs = it_track_file_with_initials(Path(nc_local), initials_df, track_dir)
                if not per_storm_csvs:
                    print("⚠️ 无有效轨迹 -> 跳过环境分析")
                    if not keep_nc:
                        try:
                            nc_local.unlink()
                            print("🧹 已删除NC (无轨迹)")
                        except Exception:
                            pass
                    skipped += 1
                    continue

                combined = combine_initial_tracker_outputs(per_storm_csvs, nc_local)
                if combined is None or combined.empty:
                    print("⚠️ 无法合并轨迹输出 -> 跳过环境分析")
                    if not keep_nc:
                        try:
                            nc_local.unlink()
                            print("🧹 已删除NC (无轨迹)")
                        except Exception:
                            pass
                    skipped += 1
                    continue
                combined.to_csv(track_csv, index=False)
                print(
                    f"💾 合并保存轨迹: {track_csv.name} (含 {combined['particle'].nunique()} 条路径)"
                )
            except Exception as e:
                print(f"❌ 追踪失败: {e}")
                traceback.print_exc()
                if not keep_nc:
                    try:
                        nc_local.unlink()
                        print("🧹 已删除NC (追踪失败)")
                    except Exception:
                        pass
                skipped += 1
                continue
        else:
            print("🗺️  已存在轨迹CSV, 直接环境分析")

        try:
            extractor = TCEnvironmentalSystemsExtractor(str(nc_local), str(track_csv))
            extractor.analyze_and_export_as_json("final_single_output")
            processed += 1
        except Exception as e:
            print(f"❌ 环境分析失败: {e}")
        finally:
            if not keep_nc:
                try:
                    nc_local.unlink()
                    print("🧹 已删除NC文件")
                except Exception as ee:
                    print(f"⚠️ 删除NC失败: {ee}")

    print("\n📊 流式处理结果:")
    print(f"  ✅ 完成: {processed}")
    print(f"  ⏭️ 跳过: {skipped}")
    print(f"  📁 输出目录: final_single_output")


def process_nc_files(target_nc_files, args):
    """处理已准备好的 NC 文件列表，保持 legacy 行为不变。"""
    import pandas as pd

    final_output_dir = Path("final_single_output")
    final_output_dir.mkdir(exist_ok=True)

    processed = 0
    skipped = 0
    for idx, nc_file in enumerate(target_nc_files, start=1):
        import re

        nc_stem = nc_file.stem
        print(f"\n[{idx}/{len(target_nc_files)}] ▶️ 处理 NC: {nc_file.name}")
        existing = list(final_output_dir.glob(f"{nc_stem}_TC_Analysis_*.json"))
        non_empty = [p for p in existing if p.stat().st_size > 10]
        if non_empty:
            print(f"⏭️  已存在分析结果 ({len(non_empty)}) -> 跳过 {nc_stem}")
            skipped += 1
            continue

        track_file = None
        if args.tracks:
            t = Path(args.tracks)
            if t.exists():
                track_file = t
        if track_file is None:
            tdir = Path("track_single")
            if tdir.exists():
                forecast_tag_match = re.search(r"(f\d{3}_f\d{3}_\d{2})", nc_stem)
                potential = []
                if forecast_tag_match:
                    tag = forecast_tag_match.group(1)
                    potential = list(tdir.glob(f"tracks_*_{tag}.csv"))
                tracks_all = sorted(tdir.glob("tracks_*.csv"))
                if potential:
                    track_file = potential[0]
                elif tracks_all:
                    track_file = tracks_all[0]
                    print(f"⚠️ 未精确匹配 forecast_tag, 使用 {track_file.name}")
        if track_file is None:
            if args.auto:
                from initialTracker import track_file_with_initials as it_track_file_with_initials
                from initialTracker import _load_all_points as it_load_initial_points

                print("🔄 使用 initialTracker 自动追踪当前NC以生成轨迹...")
                try:
                    initials_path = (
                        Path(args.initials)
                        if args.initials
                        else Path("input/western_pacific_typhoons_superfast.csv")
                    )
                    initials_df = it_load_initial_points(initials_path)
                    out_dir = Path("track_single")
                    out_dir.mkdir(exist_ok=True)
                    per_storm = it_track_file_with_initials(Path(nc_file), initials_df, out_dir)
                    if not per_storm:
                        print("⚠️ 无轨迹 -> 跳过该NC")
                        skipped += 1
                        continue
                    combined = combine_initial_tracker_outputs(per_storm, nc_file)
                    if combined is None or combined.empty:
                        print("⚠️ 自动追踪无有效轨迹 -> 跳过该NC")
                        skipped += 1
                        continue
                    first_time = (
                        combined.iloc[0]["time"] if "time" in combined.columns else None
                    )
                    ts0 = (
                        pd.to_datetime(first_time).strftime("%Y%m%d%H")
                        if pd.notnull(first_time)
                        else "T000"
                    )
                    track_file = out_dir / f"tracks_auto_{nc_stem}_{ts0}.csv"
                    combined.to_csv(track_file, index=False)
                    print(
                        f"💾 自动轨迹文件: {track_file.name} (含 {combined['particle'].nunique()} 条路径)"
                    )
                except Exception as e:
                    print(f"❌ 自动追踪失败: {e}")
                    skipped += 1
                    continue
            else:
                print("⚠️ 未找到对应轨迹且未启用 --auto, 跳过")
                skipped += 1
                continue

        print(f"✅ 使用轨迹文件: {track_file}")
        try:
            extractor = TCEnvironmentalSystemsExtractor(str(nc_file), str(track_file))
            extractor.analyze_and_export_as_json("final_single_output")
            processed += 1
        except Exception as e:
            print(f"❌ 分析失败 {nc_file.name}: {e}")
            continue

        if not (args.no_clean or args.keep_nc):
            try:
                nc_file.unlink()
                print(f"🧹 已删除 NC: {nc_file.name}")
            except Exception as e:
                print(f"⚠️ 删除NC失败: {e}")
        else:
            print("ℹ️ 按参数保留NC文件")

    print("\n🎉 多文件环境分析完成. 统计:")
    print(f"  ✅ 已分析: {processed}")
    print(f"  ⏭️ 跳过(已有结果/无轨迹): {skipped}")
    print(f"  📦 总计遍历: {len(target_nc_files)}")
    print("结果目录: final_single_output")

    return processed, skipped
