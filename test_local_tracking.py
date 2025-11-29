#!/usr/bin/env python
"""
本地测试脚本：完整气旋追踪流程
"""
import sys
sys.path.insert(0, '/root/TianGong-AI-Cyclone/src')

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict

print("=" * 70)
print("【本地追踪测试】WeatherBench2 + 气旋追踪")
print("=" * 70)

# ============================================================
# 第1步：加载GCS数据
# ============================================================
print("\n【步骤1】从GCS加载WeatherBench2数据...")
print("-" * 70)

try:
    import gcsfs
    
    DATASET_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
    TIME_RANGE = ("2020-07-25", "2020-08-05")
    LAT_RANGE = (-5.0, 45.0)
    LON_RANGE = (100.0, 180.0)  # 扩大到100°以包含110.5°
    
    rename_map = {
        "mean_sea_level_pressure": "msl",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "geopotential": "z",
        "land_sea_mask": "lsm",
    }
    
    print(f"打开数据源: {DATASET_URL}")
    ds_raw = xr.open_zarr(
        DATASET_URL,
        consolidated=True,
        storage_options={"token": "anon"},
    )
    
    # 检查哪些变量存在
    present = {src: dst for src, dst in rename_map.items() if src in ds_raw}
    
    # 首先应用坐标名称检测
    LAT_NAME = "latitude" if "latitude" in ds_raw.coords else "lat"
    LON_NAME = "longitude" if "longitude" in ds_raw.coords else "lon"
    
    # 定义切片函数
    def _lat_slice(coord, bounds):
        lower, upper = bounds
        if coord[0] > coord[-1]:
            lower, upper = upper, lower
        return slice(lower, upper)
    
    def _lon_slice(coord, bounds):
        lower, upper = bounds
        coord_min = float(coord.min())
        coord_max = float(coord.max())
        if coord_min >= 0:
            lower = lower % 360
            upper = upper % 360
        if lower > upper:
            raise ValueError("Longitude selection crosses dateline")
        lower = max(lower, coord_min)
        upper = min(upper, coord_max)
        return slice(lower, upper)
    
    lat_slice = _lat_slice(ds_raw[LAT_NAME].values, LAT_RANGE)
    lon_slice = _lon_slice(ds_raw[LON_NAME].values, LON_RANGE)
    
    # 先切片，再改名和处理
    ds_adapted = ds_raw[list(present.keys())].sel({
        LAT_NAME: lat_slice,
        LON_NAME: lon_slice,
        "time": slice(*TIME_RANGE)
    }).rename(present)
    
    if "z" in ds_adapted:
        ds_adapted["z"] = ds_adapted["z"] / 9.80665
    
    # 创建虚拟LSM（2D，全部为海洋 = 0）
    # LSM需要是 (lat, lon) 形状，不能包含时间维度
    msl_sample = ds_adapted["msl"].isel(time=0).values
    if msl_sample.ndim == 3:
        msl_sample = msl_sample[0]
    lsm_data = np.zeros_like(msl_sample)
    ds_adapted["lsm"] = xr.DataArray(
        lsm_data,
        dims=[LAT_NAME, LON_NAME],
        coords={LAT_NAME: ds_adapted[LAT_NAME], LON_NAME: ds_adapted[LON_NAME]}
    )
    
    ds_adapted = ds_adapted.chunk({"time": 1, "latitude": 181, "longitude": 361})
    ds_focus = ds_adapted
    
    print(f"✓ 数据加载成功!")
    print(f"  时间步: {len(ds_focus.time)}")
    print(f"  空间: {len(ds_focus[LAT_NAME])} × {len(ds_focus[LON_NAME])}")
    print(f"  变量: {list(ds_focus.data_vars)}")
    print(f"  内存: {ds_focus.nbytes / 1e9:.2f} GB")
    
except Exception as e:
    print(f"✗ GCS数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 第2步：加载初始气旋位置
# ============================================================
print("\n【步骤2】加载初始气旋位置...")
print("-" * 70)

try:
    from initial_tracker.initials import _load_all_points, _select_initials_for_time
    
    initials_csv = Path("/root/TianGong-AI-Cyclone/input/western_pacific_typhoons_superfast.csv")
    all_initials = _load_all_points(initials_csv)
    
    # 重命名列以匹配追踪器期望
    all_initials = all_initials.rename(columns={
        'latitude': 'init_lat',
        'longitude': 'init_lon'
    })
    
    print(f"✓ 已加载 {len(all_initials)} 条初始记录")
    
    # 选择特定时间附近的初始点
    start_time = pd.Timestamp("2020-08-01 00:00")
    init_candidates = _select_initials_for_time(all_initials, start_time, tol_hours=6)
    
    print(f"✓ {start_time.strftime('%Y-%m-%d %H:%M')} 附近的气旋: {len(init_candidates)}")
    if len(init_candidates) > 0:
        print(init_candidates[['storm_id', 'init_lat', 'init_lon', 'max_wind_usa', 'min_pressure_usa']].head())
    else:
        print("⚠ 该时间范围没有初始点，尝试使用全局初始点...")
        init_candidates = all_initials[all_initials['datetime'].dt.year == 2020].head(2)
        print(f"  使用 {len(init_candidates)} 条2020年的气旋")
        
except Exception as e:
    print(f"✗ 初始气旋加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 第3步：定义气旋追踪函数
# ============================================================
print("\n【步骤3】定义气旋追踪函数...")
print("-" * 70)

try:
    from initial_tracker.dataset_adapter import _DsAdapter, _build_batch_from_ds_fast
    from initial_tracker.tracker import Tracker
    from initial_tracker.exceptions import NoEyeException
    
    def _normalize_lon_for_grid(lon_value: float, lon_grid: np.ndarray) -> float:
        lon = float(lon_value)
        grid_min = float(lon_grid.min())
        grid_max = float(lon_grid.max())
        if grid_min >= 0 and lon < 0:
            lon = lon % 360
        if grid_max <= 180 and lon > 180:
            lon = ((lon + 180) % 360) - 180
        return lon
    
    def track_cyclones_from_dataset(
        ds: xr.Dataset,
        initials_df: pd.DataFrame,
        *,
        start_time: str | pd.Timestamp | None = None,
        max_storms: int | None = 2,
        max_steps: int | None = 40,
        output_dir: str | Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        adapter = _DsAdapter.build(ds)
        times = pd.Index(adapter.times)
        
        if len(times) == 0:
            raise ValueError("Dataset has no time dimension.")
        
        if start_time is None:
            start_idx = 0
        else:
            start_time = pd.Timestamp(start_time)
            start_idx = int(np.argmin(np.abs(times - start_time)))
        
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        time_lookup = {pd.Timestamp(t): idx for idx, t in enumerate(adapter.times)}
        tracks: dict[str, pd.DataFrame] = {}
        processed = 0
        
        for _, row in initials_df.sort_values("storm_id").iterrows():
            if max_storms is not None and processed >= max_storms:
                break
            
            storm_id = str(row["storm_id"])
            init_lat = float(row["init_lat"])
            init_lon = _normalize_lon_for_grid(float(row["init_lon"]), adapter.lons)
            init_wind = float(row["max_wind_usa"]) if pd.notna(row.get("max_wind_usa")) else None
            init_msl = float(row["min_pressure_usa"]) * 100.0 if pd.notna(row.get("min_pressure_usa")) else None
            
            print(f"\n  追踪气旋 {storm_id}:")
            print(f"    初始位置: ({init_lat:.2f}°, {init_lon:.2f}°)")
            print(f"    适配器范围: 纬 {adapter.lats[0]:.2f}-{adapter.lats[-1]:.2f}°, 经 {adapter.lons[0]:.2f}-{adapter.lons[-1]:.2f}°")
            print(f"    范围内? 纬{adapter.lats[0] <= init_lat <= adapter.lats[-1]} 经{adapter.lons[0] <= init_lon <= adapter.lons[-1]}")
            
            tracker = Tracker(
                init_lat=init_lat,
                init_lon=init_lon,
                init_time=times[start_idx],
                init_msl=init_msl,
                init_wind=init_wind,
            )
            
            last_idx = len(adapter.times) if max_steps is None else min(len(adapter.times), start_idx + max_steps)
            steps_taken = 0
            
            for time_idx in range(start_idx, last_idx):
                batch = _build_batch_from_ds_fast(adapter, time_idx)
                try:
                    tracker.step(batch)
                    steps_taken += 1
                except NoEyeException as exc:
                    if time_idx == start_idx:
                        print(f"    ✗ 第一步失败: {exc}")
                        tracker = None
                        break
                    print(f"    ⚠ 第 {time_idx - start_idx} 步跳过 -> {exc}")
                    continue
                
                if tracker.dissipated:
                    print(f"    ! 气旋消散于第 {steps_taken} 步")
                    break
            
            if tracker is None:
                print(f"    ✗ {storm_id} 追踪失败")
                continue
            
            df = tracker.results()
            df["storm_id"] = storm_id
            df["particle"] = storm_id
            df["time_idx"] = df["time"].map(time_lookup).astype("Int64")
            tracks[storm_id] = df
            
            print(f"    ✓ {storm_id} 追踪完成: {len(df)} 个点")
            
            if output_path:
                out_csv = output_path / f"track_{storm_id}.csv"
                df.to_csv(out_csv, index=False)
                print(f"      保存到: {out_csv}")
            
            processed += 1
        
        return tracks
    
    print("✓ 追踪函数定义成功")
    
except Exception as e:
    print(f"✗ 追踪函数定义失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 第4步：执行追踪
# ============================================================
print("\n【步骤4】执行气旋追踪...")
print("-" * 70)

try:
    output_dir = Path("/root/TianGong-AI-Cyclone/local_tracking_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracking_results = track_cyclones_from_dataset(
        ds_focus,
        init_candidates,
        start_time="2020-08-01 00:00",
        max_storms=1,  # 仅追踪第一个气旋
        max_steps=5,   # 仅追踪5个时间步进行测试
        output_dir=output_dir,
    )
    
    print(f"\n✓ 追踪完成!")
    print(f"  生成的追踪数: {len(tracking_results)}")
    
    for storm_id, track_df in tracking_results.items():
        print(f"\n  {storm_id}:")
        print(f"    数据点: {len(track_df)}")
        print(f"    时间: {track_df['time'].min()} 到 {track_df['time'].max()}")
        print(f"    坐标范围:")
        print(f"      纬度: {track_df['lat'].min():.2f}° 到 {track_df['lat'].max():.2f}°")
        print(f"      经度: {track_df['lon'].min():.2f}° 到 {track_df['lon'].max():.2f}°")
        if len(track_df) > 0:
            print(f"\n    首行数据:")
            print(track_df[['time', 'lat', 'lon', 'msl', 'wind']].head(3).to_string(index=False))
    
except Exception as e:
    print(f"✗ 追踪执行失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("【测试完成】")
print("=" * 70)
