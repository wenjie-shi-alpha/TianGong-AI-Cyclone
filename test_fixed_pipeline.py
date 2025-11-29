#!/usr/bin/env python
"""
验证修复后的完整管道：
1. 加载多高度风场和温度数据
2. 追踪气旋
3. 提取环境系统
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_PATH = Path(__file__).parent
sys.path.insert(0, str(PROJECT_PATH / "src"))

from initial_tracker.dataset_adapter import _DsAdapter, _build_batch_from_ds_fast
from initial_tracker.initials import _load_all_points, _select_initials_for_time
from initial_tracker.tracker import Tracker
from initial_tracker.exceptions import NoEyeException
from environment_extractor.extractor import TCEnvironmentalSystemsExtractor

# ============================================================================
# 配置
# ============================================================================
DATASET_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
TIME_RANGE = ("2020-07-25", "2020-08-05")
LAT_RANGE = (-5.0, 45.0)
LON_RANGE = (100.0, 180.0)  # 扩展以包含所有气旋候选

OUTPUT_DIR = PROJECT_PATH / "colab_outputs_local"
INITIALS_CSV = PROJECT_PATH / "input" / "western_pacific_typhoons_superfast.csv"

print("\n" + "="*70)
print("【完整测试】环境系统提取修复验证")
print("="*70)

# ============================================================================
# 1. 加载数据 - 包含多高度风场
# ============================================================================
print("\n📥 1. 加载 WeatherBench 2 数据（含多高度风场）...")
rename_map = {
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "u_component_of_wind": "u",        # ✅ 多高度风 (必需)
    "v_component_of_wind": "v",        # ✅ 多高度风 (必需)
    "temperature": "t",                # 多高度温度
    "specific_humidity": "q",
    "geopotential": "z",
    "land_sea_mask": "lsm",
    "2m_temperature": "t2m",
}

ds_raw = xr.open_zarr(
    DATASET_URL,
    consolidated=True,
    storage_options={"token": "anon"},
)

present = {src: dst for src, dst in rename_map.items() if src in ds_raw}
ds = ds_raw[list(present.keys())].rename(present)

# 单位转换
ds["z"] = ds["z"] / 9.80665  # m^2/s^2 -> geopotential height (m)

# 合成 LSM
if "lsm" not in ds:
    n_lat, n_lon = len(ds.latitude), len(ds.longitude)
    ds["lsm"] = xr.DataArray(
        np.zeros((n_lat, n_lon), dtype=np.float32),
        coords={"latitude": ds.latitude, "longitude": ds.longitude},
        dims=["latitude", "longitude"],
    )

# 空间-时间切片
ds = ds.sel(
    latitude=slice(-5, 45),
    longitude=slice(100, 180),
    time=slice(*TIME_RANGE),
).chunk({"time": 1, "latitude": 181, "longitude": 361})

print(f"✅ 数据加载完成")
print(f"   Shape: time={len(ds.time)}, lat={len(ds.latitude)}, lon={len(ds.longitude)}")
print(f"   Memory: {ds.nbytes/1e9:.2f} GB")
print(f"   Variables: {list(ds.data_vars)}")
if "u" in ds.data_vars:
    print(f"   ✅ 多高度风数据已加载 (levels: {sorted(ds.level.values)})")

# ============================================================================
# 2. 气旋追踪
# ============================================================================
print("\n🌪️ 2. 执行气旋追踪...")

def _normalize_lon_for_grid(lon_value: float, lon_grid: np.ndarray) -> float:
    lon = float(lon_value)
    grid_min = float(lon_grid.min())
    grid_max = float(lon_grid.max())
    if grid_min >= 0 and lon < 0:
        lon = lon % 360
    if grid_max <= 180 and lon > 180:
        lon = ((lon + 180) % 360) - 180
    return lon

adapter = _DsAdapter.build(ds)
times = pd.Index(adapter.times)
start_time = pd.Timestamp("2020-08-01 00:00")
start_idx = int(np.argmin(np.abs(times - start_time)))

all_initials = _load_all_points(INITIALS_CSV)
init_candidates = _select_initials_for_time(all_initials, times[start_idx], tol_hours=6)

print(f"✅ 发现 {len(init_candidates)} 个气旋候选")

tracks = {}
for _, row in init_candidates.sort_values("storm_id").head(2).iterrows():  # 只追踪前2个
    storm_id = str(row["storm_id"])
    init_lat = float(row["init_lat"])
    init_lon = _normalize_lon_for_grid(float(row["init_lon"]), adapter.lons)
    
    tracker = Tracker(
        init_lat=init_lat,
        init_lon=init_lon,
        init_time=times[start_idx],
        init_msl=None,
        init_wind=None,
    )
    
    for time_idx in range(start_idx, min(start_idx + 30, len(adapter.times))):
        batch = _build_batch_from_ds_fast(adapter, time_idx)
        try:
            tracker.step(batch)
        except NoEyeException:
            if time_idx == start_idx:
                tracker = None
                break
            continue
        if tracker.dissipated:
            break
    
    if tracker is not None:
        df = tracker.results()
        df["storm_id"] = storm_id
        tracks[storm_id] = df
        print(f"   ✅ {storm_id}: {len(df)} 个追踪点")

# ============================================================================
# 3. 环境提取
# ============================================================================
print("\n🌊 3. 环境系统提取...")

OUTPUT_DIR.mkdir(exist_ok=True)

for storm_id, track_df in tracks.items():
    print(f"\n   处理 {storm_id}...")
    
    # 提取子集
    lat_vals = track_df["lat"].astype(float)
    lon_vals = track_df["lon"].astype(float)
    lat_min = max(lat_vals.min() - 8, float(ds.latitude.values.min()))
    lat_max = min(lat_vals.max() + 8, float(ds.latitude.values.max()))
    lon_min = max(lon_vals.min() - 8, float(ds.longitude.values.min()))
    lon_max = min(lon_vals.max() + 8, float(ds.longitude.values.max()))
    
    times_track = pd.to_datetime(track_df["time"])
    time_slice = slice(times_track.min() - pd.Timedelta(hours=12), times_track.max() + pd.Timedelta(hours=12))
    
    ds_subset = ds.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
        time=time_slice,
    )
    
    # 保存到 NetCDF
    nc_path = OUTPUT_DIR / "nc_subsets" / f"{storm_id}_subset_fixed.nc"
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    
    encoding = {
        name: {"dtype": "float32", "zlib": True, "complevel": 4}
        for name in ds_subset.data_vars
        if np.issubdtype(ds_subset[name].dtype, np.floating)
    }
    ds_subset.to_netcdf(nc_path, engine="netcdf4", encoding=encoding, compute=True)
    
    print(f"     ✅ NC 子集已保存: {nc_path.name} ({nc_path.stat().st_size/1e6:.1f} MB)")
    print(f"        变量: {list(ds_subset.data_vars)}")
    if "u" in ds_subset.data_vars:
        print(f"        ✅ 多高度风包含在内 (levels: {len(ds_subset.level)})")
    
    # 保存追踪 CSV
    track_path = OUTPUT_DIR / "tracks_for_extractor" / f"{storm_id}_track_fixed.csv"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_df.to_csv(track_path, index=False)
    
    # 运行环境提取
    print(f"     🔧 启动环境提取器...")
    try:
        with TCEnvironmentalSystemsExtractor(
            str(nc_path),
            str(track_path),
            enable_detailed_shape_analysis=False,
        ) as extractor:
            result = extractor.analyze_and_export_as_json(
                output_dir=str(OUTPUT_DIR / "analysis_json")
            )
        
        # 检查结果
        json_files = list((OUTPUT_DIR / "analysis_json").glob(f"{storm_id}*.json"))
        if json_files:
            import json
            with open(json_files[0]) as f:
                data = json.load(f)
            
            systems_found = 0
            for ts in data.get("time_series", []):
                systems_found += len(ts.get("environmental_systems", []))
            
            print(f"     ✅ 提取成功！共找到 {systems_found} 个环境系统")
        else:
            print(f"     ❌ 未生成 JSON 文件")
            
    except Exception as e:
        print(f"     ❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("✅ 完整测试完成！")
print("="*70)
