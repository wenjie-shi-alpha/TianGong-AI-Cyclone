"""Pipeline orchestration for the tropical cyclone environment extractor."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from .extractor import TCEnvironmentalSystemsExtractor
from .workflow_utils import (
    combine_initial_tracker_outputs,
    download_s3_public,
    extract_forecast_tag,
    sanitize_filename,
)


def _run_environment_analysis(
    nc_path: str,
    track_csv: str,
    output_dir: str,
    keep_nc: bool,
    log_file: str | None = None,
) -> tuple[bool, str | None]:
    """Worker helper executed in a child process for 环境分析."""

    success = False
    error_message: str | None = None
    nc_name = Path(nc_path).name

    def _execute() -> None:
        nonlocal success, error_message

        print(f"[{datetime.utcnow().isoformat()}] ▶️ 环境分析开始: {nc_name}")
        try:
            extractor = TCEnvironmentalSystemsExtractor(nc_path, track_csv)
            extractor.analyze_and_export_as_json(output_dir)
            success = True
            print(f"[{datetime.utcnow().isoformat()}] ✅ 环境分析完成: {nc_name}")
        except Exception as exc:  # pragma: no cover - worker side error path
            error_message = str(exc)
            print(f"[{datetime.utcnow().isoformat()}] ❌ 环境分析失败: {nc_name} -> {error_message}")
        finally:
            if not keep_nc:
                try:
                    Path(nc_path).unlink()
                    print(f"[{datetime.utcnow().isoformat()}] 🧹 已删除NC文件: {nc_name}")
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    if success:
                        success = False
                        error_message = f"删除NC失败: {exc}"
                    print(
                        f"[{datetime.utcnow().isoformat()}] ⚠️ 删除NC失败 ({nc_name}): {exc}"
                    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(f"日志文件: {nc_name}\n")
            fh.flush()
            with redirect_stdout(fh), redirect_stderr(fh):
                _execute()
    else:
        _execute()

    return success, error_message


# ================= 新增: 流式顺序处理函数 =================
def streaming_from_csv(
    csv_path: Path,
    limit: int | None = None,
    search_range: float = 3.0,
    memory: int = 3,
    keep_nc: bool = False,
    initials_csv: Path | None = None,
    processes: int = 1,
    max_in_flight: int = 2,
    concise_log: bool = False,
    logs_root: Path | None = None,
): 
    """逐行读取CSV, 每个NC文件执行: 下载 -> 追踪 -> 环境分析 -> (可选删除)

    与原批量模式最大区别: 不预先下载全部; 每个文件完成后即可释放磁盘。
    """
    def detail(message: str) -> None:
        if not concise_log:
            print(message)

    def summary(message: str) -> None:
        print(message)

    if not csv_path.exists():
        summary(f"❌ CSV不存在: {csv_path}")
        return
    import pandas as pd, traceback
    from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
    from initialTracker import track_file_with_initials as it_track_file_with_initials
    from initialTracker import _load_all_points as it_load_initial_points

    df = pd.read_csv(csv_path)
    required_cols = {"s3_url", "model_prefix", "init_time"}
    if not required_cols.issubset(df.columns):
        summary(f"❌ CSV缺少必要列: {required_cols - set(df.columns)}")
        return
    if limit is not None:
        df = df.head(limit)

    processes = max(1, int(processes))
    max_in_flight = processes

    detail(f"📄 流式待处理数量: {len(df)} (limit={limit})")

    persist_dir = Path("data/nc_files")
    persist_dir.mkdir(parents=True, exist_ok=True)
    track_dir = Path("track_single")
    track_dir.mkdir(exist_ok=True)
    final_dir = Path("final_single_output")
    final_dir.mkdir(exist_ok=True)

    parallel = processes > 1
    executor: ProcessPoolExecutor | None = None
    active_futures: dict[Future, dict[str, str]] = {}
    logs_dir: Path | None = None
    if logs_root is not None and parallel:
        logs_dir = logs_root
        logs_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    if parallel:
        detail(
            f"⚙️ 已启用并行环境分析: 进程数={processes}, 每次最多并行{max_in_flight}个文件"
        )
        executor = ProcessPoolExecutor(max_workers=processes)

    def drain_completed(block: bool) -> None:
        nonlocal processed
        if not parallel or not active_futures:
            return

        futures = list(active_futures.keys())
        if block:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
        else:
            done = {f for f in futures if f.done()}
            if not done:
                return

        for fut in done:
            meta = active_futures.pop(fut, {})
            label = meta.get("label", "未知文件")
            try:
                success, error_msg = fut.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                success = False
                error_msg = str(exc)
            if success:
                processed += 1
                summary(f"✅ 环境分析完成: {label}")
            else:
                log_hint = meta.get("log")
                extra = f" -> {error_msg}" if error_msg else ""
                if log_hint:
                    summary(
                        f"❌ 环境分析失败: {label}{extra} (详见 {log_hint})"
                    )
                else:
                    summary(f"❌ 环境分析失败: {label}{extra}")

    def ensure_capacity() -> None:
        if not parallel:
            return
        while len(active_futures) >= max_in_flight:
            drain_completed(block=True)

    try:
        for idx, row in df.iterrows():
            if parallel:
                drain_completed(block=False)
                ensure_capacity()

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

            existing_json = list(final_dir.glob(f"{Path(fname).stem}_TC_Analysis_*.json"))
            if existing_json:
                non_empty = [p for p in existing_json if p.stat().st_size > 10]
                if non_empty:
                    detail(f"⏭️  已存在最终JSON({len(non_empty)}) -> 跳过")
                    skipped += 1
                    continue

            if not nc_local.exists():
                try:
                    detail(f"⬇️  下载NC: {s3_url}")
                    download_s3_public(s3_url, nc_local)
                except Exception as e:
                    summary(f"❌ 下载失败, 跳过: {e}")
                    skipped += 1
                    continue
            else:
                detail("📦 已存在NC文件, 复用")

            track_csv: Path | None = None

            if combined_track_csv.exists():
                track_csv = combined_track_csv
                detail("🗺️  已存在轨迹CSV, 直接环境分析")
            else:
                single_candidates = sorted(track_dir.glob(f"track_*_{nc_stem}.csv"))
                if len(single_candidates) == 1:
                    try:
                        combined = combine_initial_tracker_outputs(single_candidates, nc_local)
                        if combined is not None and not combined.empty:
                            combined.to_csv(single_candidates[0], index=False)
                        track_csv = single_candidates[0]
                        detail("🗺️  发现单条轨迹文件, 已更新后直接使用")
                    except Exception as e:
                        summary(f"⚠️ 单轨迹文件格式更新失败: {e}")
                elif len(single_candidates) > 1:
                    try:
                        combined = combine_initial_tracker_outputs(single_candidates, nc_local)
                        if combined is not None and not combined.empty:
                            combined.to_csv(combined_track_csv, index=False)
                            track_csv = combined_track_csv
                            detail(
                                f"🗺️  发现多条单独轨迹文件, 已合并生成 {combined_track_csv.name}"
                            )
                    except Exception as e:
                        summary(f"⚠️ 合并已有轨迹失败: {e}")

            if track_csv is None:
                try:
                    detail("🧭 使用 initialTracker 执行追踪...")
                    initials_path = initials_csv or Path("input/western_pacific_typhoons_superfast.csv")
                    initials_df = it_load_initial_points(initials_path)
                    per_storm_csvs = it_track_file_with_initials(
                        Path(nc_local), initials_df, track_dir
                    )
                    if not per_storm_csvs:
                        detail("⚠️ 无有效轨迹 -> 跳过环境分析")
                        if not keep_nc:
                            try:
                                nc_local.unlink()
                                detail("🧹 已删除NC (无轨迹)")
                            except Exception:
                                pass
                        skipped += 1
                        continue

                    combined = combine_initial_tracker_outputs(per_storm_csvs, nc_local)
                    if combined is None or combined.empty:
                        detail("⚠️ 无法合并轨迹输出 -> 跳过环境分析")
                        if not keep_nc:
                            try:
                                nc_local.unlink()
                                detail("🧹 已删除NC (无轨迹)")
                            except Exception:
                                pass
                        skipped += 1
                        continue

                    if combined["particle"].nunique() == 1:
                        single_path = Path(per_storm_csvs[0])
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
                except Exception as e:
                    summary(f"❌ 追踪失败: {e}")
                    if not concise_log:
                        traceback.print_exc()
                    if not keep_nc:
                        try:
                            nc_local.unlink()
                            detail("🧹 已删除NC (追踪失败)")
                        except Exception:
                            pass
                    skipped += 1
                    continue

            if track_csv is None:
                detail("⚠️ 未能生成有效轨迹 -> 跳过环境分析")
                if not keep_nc:
                    try:
                        nc_local.unlink()
                        detail("🧹 已删除NC (无轨迹)")
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        summary(f"⚠️ 删除NC失败: {exc}")
                skipped += 1
                continue

            if parallel and executor:
                detail("🧮 已提交环境分析任务 (并行)")
                log_file = (
                    str((logs_dir / f"{nc_local.stem}.log").resolve())
                    if logs_dir is not None
                    else None
                )
                future = executor.submit(
                    _run_environment_analysis,
                    str(nc_local),
                    str(track_csv),
                    "final_single_output",
                    keep_nc,
                    log_file,
                )
                active_futures[future] = {
                    "label": nc_local.name,
                    "log": log_file,
                }
            else:
                try:
                    success, error_msg = _run_environment_analysis(
                        str(nc_local),
                        str(track_csv),
                        "final_single_output",
                        keep_nc,
                        None,
                    )
                    if success:
                        processed += 1
                    elif error_msg:
                        summary(f"❌ 环境分析失败: {error_msg}")
                except Exception as e:
                    summary(f"❌ 环境分析失败: {e}")

    finally:
        if parallel and executor:
            while active_futures:
                drain_completed(block=True)
            executor.shutdown(wait=True)

    summary("\n📊 流式处理结果:")
    summary(f"  ✅ 完成: {processed}")
    summary(f"  ⏭️ 跳过: {skipped}")
    summary(f"  📁 输出目录: final_single_output")


def process_nc_files(
    target_nc_files,
    args,
    concise_log: bool = False,
    logs_root: Path | None = None,
):
    """处理已准备好的 NC 文件列表，保持 legacy 行为不变。"""
    import pandas as pd
    from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

    def detail(message: str) -> None:
        if not concise_log:
            print(message)

    def summary(message: str) -> None:
        print(message)

    final_output_dir = Path("final_single_output")
    final_output_dir.mkdir(exist_ok=True)

    processes = max(1, int(getattr(args, "processes", 1)))
    max_in_flight = processes
    parallel = processes > 1
    executor: ProcessPoolExecutor | None = None
    active_futures: dict[Future, dict[str, str]] = {}

    if parallel:
        detail(
            f"⚙️ 并行环境分析已启用 (进程数={processes}, 每次最多{max_in_flight}个文件)"
        )
        executor = ProcessPoolExecutor(max_workers=processes)

    keep_nc_flag = bool(getattr(args, "no_clean", False) or getattr(args, "keep_nc", False))

    def remove_nc_file(path: Path, reason: str) -> None:
        if keep_nc_flag:
            return
        try:
            path.unlink()
            detail(f"🧹 已删除 NC ({reason}): {path.name}")
        except FileNotFoundError:
            pass
        except Exception as exc:
            summary(f"⚠️ 删除NC失败({reason}): {exc}")

    def drain_completed(block: bool) -> None:
        nonlocal processed
        if not parallel or not active_futures:
            return

        futures = list(active_futures.keys())
        if block:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
        else:
            done = {f for f in futures if f.done()}
            if not done:
                return

        for fut in done:
            meta = active_futures.pop(fut, {})
            label = meta.get("label", "未知文件")
            try:
                success, error_msg = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                success = False
                error_msg = str(exc)
            if success:
                processed += 1
                summary(f"✅ 环境分析完成: {label}")
            else:
                log_hint = meta.get("log")
                extra = f" -> {error_msg}" if error_msg else ""
                if log_hint:
                    summary(
                        f"❌ 环境分析失败: {label}{extra} (详见 {log_hint})"
                    )
                else:
                    summary(f"❌ 环境分析失败: {label}{extra}")

    def ensure_capacity() -> None:
        if not parallel:
            return
        while len(active_futures) >= max_in_flight:
            drain_completed(block=True)

    processed = 0
    skipped = 0
    for idx, nc_file in enumerate(target_nc_files, start=1):
        import re

        if parallel:
            drain_completed(block=False)
            ensure_capacity()

        nc_stem = nc_file.stem
        detail(f"\n[{idx}/{len(target_nc_files)}] ▶️ 处理 NC: {nc_file.name}")
        existing = list(final_output_dir.glob(f"{nc_stem}_TC_Analysis_*.json"))
        non_empty = [p for p in existing if p.stat().st_size > 10]
        if non_empty:
            detail(f"⏭️  已存在分析结果 ({len(non_empty)}) -> 跳过 {nc_stem}")
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
                single_candidates = sorted(tdir.glob(f"track_*_{nc_stem}.csv"))
                if forecast_tag_match:
                    tag = forecast_tag_match.group(1)
                    potential = list(tdir.glob(f"tracks_*_{tag}.csv"))
                tracks_all = sorted(tdir.glob("tracks_*.csv"))
                if potential:
                    track_file = potential[0]
                elif len(single_candidates) == 1:
                    track_file = single_candidates[0]
                elif len(single_candidates) > 1:
                    summary(
                        "⚠️ 检测到多个单轨迹文件, 请确认后选择正确文件"
                    )
                elif tracks_all:
                    track_file = tracks_all[0]
                    summary(f"⚠️ 未精确匹配 forecast_tag, 使用 {track_file.name}")
        if track_file is None:
            if args.auto:
                from initialTracker import track_file_with_initials as it_track_file_with_initials
                from initialTracker import _load_all_points as it_load_initial_points

                detail("🔄 使用 initialTracker 自动追踪当前NC以生成轨迹...")
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
                        detail("⚠️ 无轨迹 -> 跳过该NC")
                        remove_nc_file(nc_file, "无轨迹")
                        skipped += 1
                        continue
                    combined = combine_initial_tracker_outputs(per_storm, nc_file)
                    if combined is None or combined.empty:
                        detail("⚠️ 自动追踪无有效轨迹 -> 跳过该NC")
                        remove_nc_file(nc_file, "无轨迹")
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
                    if combined["particle"].nunique() == 1:
                        track_file = Path(per_storm[0])
                        combined.to_csv(track_file, index=False)
                        detail(f"💾 自动轨迹文件: {track_file.name} (单条路径)")
                    else:
                        track_file = out_dir / f"tracks_auto_{nc_stem}_{ts0}.csv"
                        combined.to_csv(track_file, index=False)
                        detail(
                            f"💾 自动轨迹文件: {track_file.name} (含 {combined['particle'].nunique()} 条路径)"
                        )
                except Exception as e:
                    summary(f"❌ 自动追踪失败: {e}")
                    remove_nc_file(nc_file, "追踪失败")
                    skipped += 1
                    continue
            else:
                detail("⚠️ 未找到对应轨迹且未启用 --auto, 跳过")
                remove_nc_file(nc_file, "无轨迹")
                skipped += 1
                continue

        detail(f"✅ 使用轨迹文件: {track_file}")
        if parallel and executor:
            detail("🧮 已提交环境分析任务 (并行)")
            log_file = (
                str((logs_dir / f"{nc_file.stem}.log").resolve())
                if logs_dir is not None
                else None
            )
            future = executor.submit(
                _run_environment_analysis,
                str(nc_file),
                str(track_file),
                "final_single_output",
                keep_nc_flag,
                log_file,
            )
            active_futures[future] = {
                "label": nc_file.name,
                "log": log_file,
            }
        else:
            try:
                success, error_msg = _run_environment_analysis(
                    str(nc_file),
                    str(track_file),
                    "final_single_output",
                    keep_nc_flag,
                    None,
                )
                if success:
                    processed += 1
                elif error_msg:
                    summary(f"❌ 分析失败 {nc_file.name}: {error_msg}")
            except Exception as e:
                summary(f"❌ 分析失败 {nc_file.name}: {e}")
                continue

    if parallel and executor:
        while active_futures:
            drain_completed(block=True)
        executor.shutdown(wait=True)

    summary("\n🎉 多文件环境分析完成. 统计:")
    summary(f"  ✅ 已分析: {processed}")
    summary(f"  ⏭️ 跳过(已有结果/无轨迹): {skipped}")
    summary(f"  📦 总计遍历: {len(target_nc_files)}")
    summary("结果目录: final_single_output")

    return processed, skipped
