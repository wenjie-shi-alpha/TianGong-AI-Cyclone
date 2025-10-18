#!/usr/bin/env python3
"""
可视化两次追踪结果的差异
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# 读取两个CSV文件
csv1 = "data/test/tracks/track_2020270N17159_FOUR_v200_GFS_2020093012_f000_f240_06.csv"
csv2 = "data/test/tracks/track_AUTO_FOUR_v200_GFS_2020093012_f000_f240_06_FOUR_v200_GFS_2020093012_f000_f240_06.csv"

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# 创建地图
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置地图范围（覆盖两条路径）
all_lats = list(df1['lat']) + list(df2['lat'])
all_lons = list(df1['lon']) + list(df2['lon'])
lat_min, lat_max = min(all_lats) - 5, max(all_lats) + 5
lon_min, lon_max = min(all_lons) - 5, max(all_lons) + 5

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# 添加地图要素
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

# 绘制路径1（真实台风KUJIRA）
ax.plot(df1['lon'], df1['lat'], 'ro-', markersize=6, linewidth=2, 
        label='真实台风KUJIRA (2020270N17159)', transform=ccrs.PlateCarree(), zorder=3)

# 标记路径1的初始点
ax.plot(df1.iloc[0]['lon'], df1.iloc[0]['lat'], 'r*', markersize=20, 
        label=f'KUJIRA初始点 ({df1.iloc[0]["lat"]:.1f}°N, {df1.iloc[0]["lon"]:.1f}°E)', 
        transform=ccrs.PlateCarree(), zorder=4)

# 绘制路径2（自动搜索）
ax.plot(df2['lon'], df2['lat'], 'bs-', markersize=6, linewidth=2, 
        label='自动搜索的气旋 (AUTO)', transform=ccrs.PlateCarree(), zorder=3)

# 标记路径2的初始点
ax.plot(df2.iloc[0]['lon'], df2.iloc[0]['lat'], 'b*', markersize=20, 
        label=f'AUTO初始点 ({df2.iloc[0]["lat"]:.1f}°N, {df2.iloc[0]["lon"]:.1f}°E)', 
        transform=ccrs.PlateCarree(), zorder=4)

# 添加连线显示初始点差异
ax.plot([df1.iloc[0]['lon'], df2.iloc[0]['lon']], 
        [df1.iloc[0]['lat'], df2.iloc[0]['lat']], 
        'k--', linewidth=1, alpha=0.5, label='初始点差异 (~3368 km)', 
        transform=ccrs.PlateCarree(), zorder=2)

# 添加时间标签（每隔几个点）
for i in range(0, len(df1), 5):
    ax.text(df1.iloc[i]['lon'], df1.iloc[i]['lat'], 
            df1.iloc[i]['time'].split()[0][-5:], 
            fontsize=7, color='red', transform=ccrs.PlateCarree())

for i in range(0, len(df2), 5):
    ax.text(df2.iloc[i]['lon'], df2.iloc[i]['lat'], 
            df2.iloc[i]['time'].split()[0][-5:], 
            fontsize=7, color='blue', transform=ccrs.PlateCarree())

ax.legend(loc='upper left', fontsize=10)
ax.set_title('气旋追踪结果对比\nFOUR_v200_GFS_2020093012_f000_f240_06.nc', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data/test/tracking_comparison.png', dpi=300, bbox_inches='tight')
print("\n可视化图表已保存到: data/test/tracking_comparison.png")

# 创建气压对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 路径1气压变化
if 'msl' in df1.columns:
    times1 = range(len(df1))
    ax1.plot(times1, df1['msl']/100, 'ro-', linewidth=2, markersize=6)
    ax1.set_xlabel('时间步', fontsize=12)
    ax1.set_ylabel('海平面气压 (hPa)', fontsize=12)
    ax1.set_title('KUJIRA (真实台风) - 气压变化', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

# 路径2气压变化
if 'msl' in df2.columns:
    times2 = range(len(df2))
    ax2.plot(times2, df2['msl']/100, 'bs-', linewidth=2, markersize=6)
    ax2.set_xlabel('时间步', fontsize=12)
    ax2.set_ylabel('海平面气压 (hPa)', fontsize=12)
    ax2.set_title('AUTO (自动搜索) - 气压变化', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/test/pressure_comparison.png', dpi=300, bbox_inches='tight')
print("气压对比图已保存到: data/test/pressure_comparison.png")

print("\n图表说明:")
print("1. 红色路径: 基于历史观测的真实台风KUJIRA追踪结果")
print("2. 蓝色路径: 基于自动搜索算法找到的气旋追踪结果")
print("3. 两条路径追踪的是完全不同的气旋系统")
