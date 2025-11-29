#!/usr/bin/env python
"""
完整测试：气旋追踪 + 环境提取
"""
import sys
sys.path.insert(0, '/root/TianGong-AI-Cyclone/src')

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

print("=" * 70)
print("【完整测试】气旋追踪 + 天气系统环境提取")
print("=" * 70)

# ============================================================
# 第1-3步：加载数据、初始气旋、定义追踪函数（复用之前的代码）
# ============================================================
print("\n【步骤1-3】加载数据和追踪函数...")

DATASET_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
TIME_RANGE = ("2020-07-25", "2020-08-05")
LAT_RANGE = (-5.0, 45.0)
LON_RANGE = (100.0, 180.0)

rename_map = {
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "geopotential": "z",
}

ds_raw = xr.open_zarr(DATASET_URL, consolidated=True, storage_options={"token": "anon"})
present = {src: dst for src, dst in rename_map.items() if src in ds_raw}
ds_adapted = ds_raw[list(present.keys())].rename(present)
ds_adapted["z"] = ds_adapted["z"] / 9.80665

# 创建虚拟LSM
LAT_NAME = "latitude"
LON_NAME = "longitude"

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

lat_slice = _lat_slice(ds_adapted[LAT_NAME].values, LAT_RANGE)
lon_slice = _lon_slice(ds_adapted[LON_NAME].values, LON_RANGE)

ds_adapted = ds_adapted.sel({
    LAT_NAME: lat_slice,
    LON_NAME: lon_slice,
    "time": slice(*TIME_RANGE)
}).chunk({"time": 1, "latitude": 181, "longitude": 361})

msl_sample = ds_adapted["msl"].isel(time=0).values
if msl_sample.ndim == 3:
    msl_sample = msl_sample[0]
lsm_data = np.zeros_like(msl_sample)
ds_adapted["lsm"] = xr.DataArray(
    lsm_data,
    dims=[LAT_NAME, LON_NAME],
    coords={LAT_NAME: ds_adapted[LAT_NAME], LON_NAME: ds_adapted[LON_NAME]}
)

ds_focus = ds_adapted

print(f"✓ 数据加载: {ds_focus.nbytes / 1e9:.2f} GB")

# 导入追踪模块
from initial_tracker.dataset_adapter import _DsAdapter, _build_batch_from_ds_fast
from initial_tracker.initials import _load_all_points, _select_initials_for_time
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
    initials_csv: str | Path,
    *,
    start_time: str | pd.Timestamp | None = None,
    max_storms: int | None = 2,
    max_steps: int | None = 40,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    adapter = _DsAdapter.build(ds)
    times = pd.Index(adapter.times)
    
    if start_time is None:
        start_idx = 0
    else:
        start_time = pd.Timestamp(start_time)
        start_idx = int(np.argmin(np.abs(times - start_time)))
    
    all_initials = _load_all_points(Path(initials_csv))
    all_initials = all_initials.rename(columns={
        'latitude': 'init_lat',
        'longitude': 'init_lon'
    })
    init_candidates = _select_initials_for_time(all_initials, times[start_idx], tol_hours=6)
    
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    time_lookup = {pd.Timestamp(t): idx for idx, t in enumerate(adapter.times)}
    tracks: dict[str, pd.DataFrame] = {}
    processed = 0
    
    for _, row in init_candidates.sort_values("storm_id").iterrows():
        if max_storms is not None and processed >= max_storms:
            break
        
        storm_id = str(row["storm_id"])
        init_lat = float(row["init_lat"])
        init_lon = _normalize_lon_for_grid(float(row["init_lon"]), adapter.lons)
        init_wind = float(row["max_wind_usa"]) if pd.notna(row.get("max_wind_usa")) else None
        init_msl = float(row["min_pressure_usa"]) * 100.0 if pd.notna(row.get("min_pressure_usa")) else None
        
        tracker = Tracker(
            init_lat=init_lat,
            init_lon=init_lon,
            init_time=times[start_idx],
            init_msl=init_msl,
            init_wind=init_wind,
        )
        
        last_idx = len(adapter.times) if max_steps is None else min(len(adapter.times), start_idx + max_steps)
        
        for time_idx in range(start_idx, last_idx):
            batch = _build_batch_from_ds_fast(adapter, time_idx)
            try:
                tracker.step(batch)
            except NoEyeException:
                continue
            if tracker.dissipated:
                break
        
        df = tracker.results()
        df["storm_id"] = storm_id
        df["particle"] = storm_id
        df["time_idx"] = df["time"].map(time_lookup).astype("Int64")
        tracks[storm_id] = df
        
        if output_path:
            out_csv = output_path / f"track_{storm_id}.csv"
            df.to_csv(out_csv, index=False)
            print(f"  ✓ {storm_id}: {len(df)} 个追踪点")
        
        processed += 1
    
    return tracks

print("✓ 追踪函数定义完成")

# ============================================================
# 第4步：执行追踪
# ============================================================
print("\n【步骤4】执行气旋追踪...")
print("-" * 70)

initials_csv = Path("/root/TianGong-AI-Cyclone/input/western_pacific_typhoons_superfast.csv")
output_dir = Path("/root/TianGong-AI-Cyclone/local_tracking_output")

tracking_results = track_cyclones_from_dataset(
    ds_focus,
    initials_csv,
    start_time="2020-08-01 00:00",
    max_storms=2,
    max_steps=40,
    output_dir=output_dir,
)

print(f"\n✓ 追踪完成: {len(tracking_results)} 个气旋")

# ============================================================
# 第5步：环境提取
# ============================================================
print("\n【步骤5】环境系统提取...")
print("-" * 70)

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor

def subset_dataset_for_track(
    ds: xr.Dataset,
    track_df: pd.DataFrame,
    lat_pad: float = 8.0,
    lon_pad: float = 8.0,
    time_pad_steps: int = 2,
) -> xr.Dataset:
    """在追踪周围切割一个紧凑的数据立方体"""
    if track_df.empty:
        raise ValueError("Track DataFrame is empty.")
    
    lat_vals = track_df["lat"].astype(float)
    lon_vals = track_df["lon"].astype(float)
    lon_grid = ds[LON_NAME].values
    
    lon_is_0360 = float(lon_grid.min()) >= 0
    if lon_is_0360:
        lon_vals = lon_vals % 360
    
    lat_min = max(lat_vals.min() - lat_pad, float(ds[LAT_NAME].values.min()))
    lat_max = min(lat_vals.max() + lat_pad, float(ds[LAT_NAME].values.max()))
    lon_min = max(lon_vals.min() - lon_pad, float(lon_grid.min()))
    lon_max = min(lon_vals.max() + lon_pad, float(lon_grid.max()))
    
    lat_slice = _lat_slice(ds[LAT_NAME].values, (lat_min, lat_max))
    lon_slice = slice(lon_min, lon_max)
    
    times = pd.to_datetime(track_df["time"])
    delta = pd.Timedelta(hours=6 * time_pad_steps)
    time_slice = slice(times.min() - delta, times.max() + delta)
    
    return ds.sel({LAT_NAME: lat_slice, LON_NAME: lon_slice, "time": time_slice})

def persist_subset_to_netcdf(ds_subset: xr.Dataset, folder: Path, stem: str) -> Path:
    """将子集导出为NetCDF文件"""
    folder.mkdir(parents=True, exist_ok=True)
    nc_path = folder / f"{stem}.nc"
    encoding = {
        name: {"dtype": "float32", "zlib": True, "complevel": 4}
        for name, da in ds_subset.data_vars.items()
        if np.issubdtype(da.dtype, np.floating)
    }
    ds_subset.to_netcdf(nc_path, engine="netcdf4", encoding=encoding, compute=True)
    return nc_path

def run_environment_extraction(
    ds_subset: xr.Dataset,
    track_df: pd.DataFrame,
    storm_id: str,
    output_root: Path,
) -> dict:
    """运行环境提取"""
    output_root.mkdir(parents=True, exist_ok=True)
    tracks_dir = output_root / "tracks_for_extractor"
    nc_dir = output_root / "nc_subsets"
    analysis_dir = output_root / "analysis_json"
    
    tracks_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存追踪CSV
    track_csv = tracks_dir / f"{storm_id}_track.csv"
    track_df.to_csv(track_csv, index=False)
    
    # 导出NetCDF子集
    nc_path = persist_subset_to_netcdf(ds_subset, nc_dir, f"{storm_id}_subset")
    print(f"  ✓ 数据子集: {nc_path.stat().st_size / 1e6:.1f} MB")
    
    # 运行提取器
    print(f"  运行TCEnvironmentalSystemsExtractor...")
    with TCEnvironmentalSystemsExtractor(
        str(nc_path),
        str(track_csv),
        enable_detailed_shape_analysis=False,
    ) as extractor:
        result = extractor.analyze_and_export_as_json(output_dir=str(analysis_dir))
    
    print(f"  ✓ 提取完成，粒子数: {len(result)}")
    
    return {
        "json_dir": analysis_dir,
        "nc_path": nc_path,
        "track_csv": track_csv,
        "result": result,
    }

# 对每个追踪的气旋进行环境提取
extraction_results = {}

for storm_id, track_df in tracking_results.items():
    print(f"\n【{storm_id}】环境提取")
    print(f"  气旋追踪: {len(track_df)} 点，时间范围 {track_df['time'].min()} 到 {track_df['time'].max()}")
    
    # 切割数据立方体
    local_ds = subset_dataset_for_track(
        ds_focus,
        track_df,
        lat_pad=8.0,
        lon_pad=8.0,
        time_pad_steps=1,
    )
    
    print(f"  数据立方体: {local_ds.nbytes / 1e6:.1f} MB, 形状 time={local_ds.dims.get('time')} lat={local_ds.dims.get(LAT_NAME)} lon={local_ds.dims.get(LON_NAME)}")
    
    # 运行提取
    try:
        env_output = run_environment_extraction(
            local_ds,
            track_df,
            storm_id,
            output_root=Path("/root/TianGong-AI-Cyclone/colab_outputs_local"),
        )
        extraction_results[storm_id] = env_output
    except Exception as e:
        print(f"  ✗ 提取失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("【完整流程完成】")
print("=" * 70)

print(f"\n总结:")
print(f"  追踪气旋数: {len(tracking_results)}")
print(f"  成功提取数: {len(extraction_results)}")

for storm_id, result in extraction_results.items():
    print(f"\n【{storm_id}】输出文件:")
    print(f"  JSON目录: {result['json_dir']}")
    print(f"  NetCDF: {result['nc_path']}")
    print(f"  追踪CSV: {result['track_csv']}")
    print(f"  粒子数: {len(result['result'])}")
