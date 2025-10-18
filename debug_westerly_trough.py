#!/usr/bin/env python3
"""
简单调试西风槽提取
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor

# 测试FOUR案例（高纬度，最可能有西风槽）
nc_file = 'data/FOUR_v200_GFS_2020093012_f000_f240_06.nc'
track_file = 'data/test/tracks/track_2020270N17159_FOUR_v200_GFS_2020093012_f000_f240_06.csv'

print("创建提取器...")
extractor = TCEnvironmentalSystemsExtractor(nc_file, track_file)

# 测试第一个时间点
time_idx = 0
tc_lat = 41.2
tc_lon = 166.3

print(f"\n测试时间索引 {time_idx}, 台风位置: {tc_lat}°N, {tc_lon}°E\n")

# 获取500hPa高度场
z500 = extractor._get_data_at_level("z", 500, time_idx)
print(f"Z500 shape: {z500.shape}")
print(f"Z500 range: {np.nanmin(z500):.0f} - {np.nanmax(z500):.0f} gpm")

# 计算纬向平均和距平
z500_zonal_mean = np.nanmean(z500, axis=1, keepdims=True)
z500_anomaly = z500 - z500_zonal_mean

print(f"\nZ500距平 range: {np.nanmin(z500_anomaly):.0f} - {np.nanmax(z500_anomaly):.0f} gpm")

# 中纬度掩膜
mid_lat_mask = (extractor.lat >= 20) & (extractor.lat <= 60)
print(f"中纬度掩膜: {np.sum(mid_lat_mask)} 纬度点")

# 在中纬度区域的距平
z500_anomaly_mid = z500_anomaly.copy()
z500_anomaly_mid[~mid_lat_mask, :] = np.nan

print(f"中纬度Z500距平 range: {np.nanmin(z500_anomaly_mid):.0f} - {np.nanmax(z500_anomaly_mid):.0f} gpm")

# 负距平区域
negative_anomaly = z500_anomaly_mid < 0
print(f"负距平点数: {np.sum(negative_anomaly)}")

if np.any(negative_anomaly):
    neg_values = z500_anomaly_mid[negative_anomaly]
    print(f"负距平值范围: {np.min(neg_values):.0f} - {np.max(neg_values):.0f} gpm")
    print(f"负距平25分位数: {np.percentile(neg_values, 25):.0f} gpm")
    
    trough_threshold = np.percentile(neg_values, 25)
    trough_mask = (z500_anomaly < trough_threshold)
    print(f"槽掩膜点数（全局）: {np.sum(trough_mask)}")
    
    # 局部搜索区域
    lat_idx = np.abs(extractor.lat - tc_lat).argmin()
    lon_idx = np.abs(extractor.lon - tc_lon).argmin()
    radius_points = int(30 / extractor.lat_spacing)
    
    print(f"\n局部搜索:")
    print(f"  台风索引: lat={lat_idx}, lon={lon_idx}")
    print(f"  搜索半径: {radius_points} 点 (约30°)")
    
    lat_start = max(0, lat_idx - radius_points)
    lat_end = min(len(extractor.lat), lat_idx + radius_points + 1)
    lon_start = max(0, lon_idx - radius_points)
    lon_end = min(len(extractor.lon), lon_idx + radius_points + 1)
    
    print(f"  纬度范围: {lat_start}-{lat_end} ({extractor.lat[lat_start]:.1f}-{extractor.lat[lat_end-1]:.1f}°N)")
    print(f"  经度范围: {lon_start}-{lon_end} ({extractor.lon[lon_start]:.1f}-{extractor.lon[lon_end-1]:.1f}°E)")
    
    # 创建局部掩膜
    local_mask = np.zeros_like(z500_anomaly, dtype=bool)
    local_mask[lat_start:lat_end, lon_start:lon_end] = True
    local_mask = local_mask & mid_lat_mask[:, np.newaxis]
    
    print(f"  局部掩膜点数: {np.sum(local_mask)}")
    
    # 在局部区域内的槽
    trough_mask_local = (z500_anomaly < trough_threshold) & local_mask
    print(f"  局部槽掩膜点数: {np.sum(trough_mask_local)}")
    
    if np.sum(trough_mask_local) > 0:
        print("\n  ✅ 检测到槽区域!")
        
        # 检查能否形成轴线
        lon_indices = np.where(np.any(trough_mask_local, axis=0))[0]
        print(f"  槽跨越经度数: {len(lon_indices)}")
        
        if len(lon_indices) >= 2:
            print("  ✅ 可以形成槽轴线")
        else:
            print("  ❌ 槽太窄，无法形成轴线")
    else:
        print("\n  ❌ 局部区域内未检测到槽")
else:
    print("❌ 中纬度无负距平区域")

print("\n" + "="*60)
print("调用提取函数:")
result = extractor.extract_westerly_trough(time_idx, tc_lat, tc_lon)
if result:
    print("✅ 成功检测到西风槽")
    print(f"描述: {result['description']}")
else:
    print("❌ 未检测到西风槽")
