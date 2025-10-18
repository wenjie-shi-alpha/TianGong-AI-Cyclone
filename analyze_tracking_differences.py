#!/usr/bin/env python3
"""
分析两次追踪结果差异的原因
"""

import pandas as pd
import xarray as xr
from pathlib import Path

# 读取两个CSV文件
csv1 = "data/test/tracks/track_2020270N17159_FOUR_v200_GFS_2020093012_f000_f240_06.csv"
csv2 = "data/test/tracks/track_AUTO_FOUR_v200_GFS_2020093012_f000_f240_06_FOUR_v200_GFS_2020093012_f000_f240_06.csv"

print("=" * 80)
print("追踪结果差异分析")
print("=" * 80)

# 读取数据
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

print(f"\n文件1: {csv1}")
print(f"  - 行数: {len(df1)}")
print(f"  - 列: {list(df1.columns)}")
print(f"\n前3行:")
print(df1.head(3))

print(f"\n文件2: {csv2}")
print(f"  - 行数: {len(df2)}")
print(f"  - 列: {list(df2.columns)}")
print(f"\n前3行:")
print(df2.head(3))

# 分析初始点
print("\n" + "=" * 80)
print("初始点分析")
print("=" * 80)

print(f"\n文件1 初始点:")
print(f"  时间: {df1.iloc[0]['time']}")
print(f"  位置: ({df1.iloc[0]['lat']:.2f}°N, {df1.iloc[0]['lon']:.2f}°E)")
if 'msl' in df1.columns:
    print(f"  气压: {df1.iloc[0]['msl']:.1f} Pa ({df1.iloc[0]['msl']/100:.1f} hPa)")
if 'wind' in df1.columns:
    print(f"  风速: {df1.iloc[0]['wind']:.1f} m/s")

print(f"\n文件2 初始点:")
print(f"  时间: {df2.iloc[0]['time']}")
print(f"  位置: ({df2.iloc[0]['lat']:.2f}°N, {df2.iloc[0]['lon']:.2f}°E)")
if 'msl' in df2.columns:
    print(f"  气压: {df2.iloc[0]['msl']:.1f} Pa ({df2.iloc[0]['msl']/100:.1f} hPa)")
if 'wind' in df2.columns:
    print(f"  风速: {df2.iloc[0]['wind']:.1f} m/s")

# 计算初始点差异
lat_diff = abs(df1.iloc[0]['lat'] - df2.iloc[0]['lat'])
lon_diff = abs(df1.iloc[0]['lon'] - df2.iloc[0]['lon'])

print(f"\n初始点差异:")
print(f"  纬度差: {lat_diff:.2f}°")
print(f"  经度差: {lon_diff:.2f}°")
print(f"  距离差: ~{((lat_diff**2 + lon_diff**2)**0.5 * 111):.0f} km")

# 分析NC文件信息
print("\n" + "=" * 80)
print("NC文件分析")
print("=" * 80)

nc_file = "data/FOUR_v200_GFS_2020093012_f000_f240_06.nc"
ds = xr.open_dataset(nc_file)

print(f"\n文件: {nc_file}")
print(f"  时间范围: {ds.time.values[0]} 到 {ds.time.values[-1]}")
print(f"  时间步数: {len(ds.time)}")
print(f"  纬度范围: {float(ds.latitude.min().values):.1f}° 到 {float(ds.latitude.max().values):.1f}°")
print(f"  经度范围: {float(ds.longitude.min().values):.1f}° 到 {float(ds.longitude.max().values):.1f}°")

# 检查第一个时次的气压场
first_time = ds.time.values[0]
msl = ds['msl'].isel(time=0).values
lat = ds.latitude.values
lon = ds.longitude.values

print(f"\n第一个时次 ({first_time}):")

# 在文件1初始点附近的气压
lat1, lon1 = df1.iloc[0]['lat'], df1.iloc[0]['lon']
lat_idx1 = abs(lat - lat1).argmin()
lon_idx1 = abs(lon - lon1).argmin()
pressure1 = msl[lat_idx1, lon_idx1]

print(f"\n  文件1初始点 ({lat1:.2f}°N, {lon1:.2f}°E) 附近:")
print(f"    最近格点: ({lat[lat_idx1]:.2f}°N, {lon[lon_idx1]:.2f}°E)")
print(f"    气压: {pressure1:.1f} Pa ({pressure1/100:.1f} hPa)")

# 在文件2初始点附近的气压
lat2, lon2 = df2.iloc[0]['lat'], df2.iloc[0]['lon']
lat_idx2 = abs(lat - lat2).argmin()
lon_idx2 = abs(lon - lon2).argmin()
pressure2 = msl[lat_idx2, lon_idx2]

print(f"\n  文件2初始点 ({lat2:.2f}°N, {lon2:.2f}°E) 附近:")
print(f"    最近格点: ({lat[lat_idx2]:.2f}°N, {lon[lon_idx2]:.2f}°E)")
print(f"    气压: {pressure2:.1f} Pa ({pressure2/100:.1f} hPa)")

# 在热带地区找最低气压点
tropical_mask = (lat >= 5) & (lat <= 35)
lat_tropical = lat[tropical_mask]
msl_tropical = msl[tropical_mask, :]

min_idx = msl_tropical.argmin()
min_lat_idx, min_lon_idx = divmod(min_idx, msl_tropical.shape[1])
min_lat = lat_tropical[min_lat_idx]
min_lon = lon[min_lon_idx]
min_pressure = msl_tropical[min_lat_idx, min_lon_idx]

print(f"\n  热带地区 (5-35°N) 最低气压点:")
print(f"    位置: ({min_lat:.2f}°N, {min_lon:.2f}°E)")
print(f"    气压: {min_pressure:.1f} Pa ({min_pressure/100:.1f} hPa)")

ds.close()

# 分析真实台风数据
print("\n" + "=" * 80)
print("历史观测数据 (KUJIRA 2020270N17159)")
print("=" * 80)

# 从输入CSV读取KUJIRA台风的信息
input_csv = "input/western_pacific_typhoons_superfast.csv"
df_input = pd.read_csv(input_csv)
kujira = df_input[df_input['storm_id'] == '2020270N17159']

if len(kujira) > 0:
    first_record = kujira.iloc[0]
    print(f"\n第一个观测记录:")
    print(f"  时间: {first_record['datetime']}")
    print(f"  位置: ({first_record['latitude']:.2f}°N, {first_record['longitude']:.2f}°E)")
    if 'min_pressure_usa' in first_record and pd.notna(first_record['min_pressure_usa']):
        print(f"  气压: {first_record['min_pressure_usa']:.1f} hPa")
    if 'max_wind_usa' in first_record and pd.notna(first_record['max_wind_usa']):
        print(f"  风速: {first_record['max_wind_usa']:.1f} m/s")
    
    # 找到与NC文件第一个时次最接近的观测
    nc_first_time = pd.Timestamp('2020-09-30 12:00:00')
    kujira['datetime_dt'] = pd.to_datetime(kujira['datetime'])
    time_diff = abs(kujira['datetime_dt'] - nc_first_time)
    closest_idx = time_diff.idxmin()
    closest_record = kujira.loc[closest_idx]
    
    print(f"\n与NC文件第一个时次 ({nc_first_time}) 最接近的观测:")
    print(f"  时间: {closest_record['datetime']}")
    print(f"  位置: ({closest_record['latitude']:.2f}°N, {closest_record['longitude']:.2f}°E)")
    if 'min_pressure_usa' in closest_record and pd.notna(closest_record['min_pressure_usa']):
        print(f"  气压: {closest_record['min_pressure_usa']:.1f} hPa")
    if 'max_wind_usa' in closest_record and pd.notna(closest_record['max_wind_usa']):
        print(f"  风速: {closest_record['max_wind_usa']:.1f} m/s")
    
    print(f"\n  与文件1初始点的差异:")
    print(f"    纬度差: {abs(closest_record['latitude'] - lat1):.2f}°")
    print(f"    经度差: {abs(closest_record['longitude'] - lon1):.2f}°")
    
    print(f"\n  与文件2初始点的差异:")
    print(f"    纬度差: {abs(closest_record['latitude'] - lat2):.2f}°")
    print(f"    经度差: {abs(closest_record['longitude'] - lon2):.2f}°")

# 总结
print("\n" + "=" * 80)
print("差异原因分析总结")
print("=" * 80)

print("""
两次追踪结果不一致的原因:

1. **初始点不同**
   - 文件1 (track_2020270N17159_...): 使用了历史观测的真实台风初始点
     来自 western_pacific_typhoons_superfast.csv 中的 KUJIRA 台风记录
   
   - 文件2 (track_AUTO_...): 使用了自动搜索算法找到的初始点
     通过分析第一个时次的海平面气压场自动寻找低压中心

2. **初始位置差异导致追踪路径完全不同**
   - 由于初始点相差很远（可能在不同的低压系统上）
   - 追踪算法会跟踪不同的气旋系统
   - 导致整个追踪路径完全不同

3. **解决方案**
   - 如果要追踪特定的历史台风，应该使用历史观测的初始点
   - 如果要进行预报数据的追踪，可以使用自动搜索算法
   - 需要明确追踪的目标：历史验证 vs 自动预报

建议:
  - 对于历史台风验证，使用 western_pacific_typhoons_superfast.csv 中的初始点
  - 对于新的预报数据，使用自动搜索算法
  - 在文档中明确说明使用的初始点来源
""")

print("\n" + "=" * 80)
