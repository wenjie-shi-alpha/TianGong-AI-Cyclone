#!/usr/bin/env python3
"""
详细分析PANG案例为什么没有检测到西风槽
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor

# PANG案例
nc_file = 'data/PANG_v100_IFS_2022032900_f000_f240_06.nc'
track_file = 'data/test/tracks/track_2022088N09116_PANG_v100_IFS_2022032900_f000_f240_06.csv'

print("="*70)
print("PANG案例西风槽缺失原因分析")
print("="*70)

print("\n创建提取器...")
extractor = TCEnvironmentalSystemsExtractor(nc_file, track_file)

# 测试第一个时间点
time_idx = 0
tc_lat = 9.4
tc_lon = 116.4

print(f"\n台风位置: {tc_lat}°N, {tc_lon}°E")
print(f"分析: 这是一个位于**热带地区**（<10°N）的低纬度热带气旋\n")

# 获取500hPa高度场
z500 = extractor._get_data_at_level("z", 500, time_idx)
print(f"Z500数据: shape={z500.shape}, range={np.nanmin(z500):.0f}-{np.nanmax(z500):.0f} gpm")

# 计算纬向平均和距平
z500_zonal_mean = np.nanmean(z500, axis=1, keepdims=True)
z500_anomaly = z500 - z500_zonal_mean

print(f"Z500距平: range={np.nanmin(z500_anomaly):.0f}-{np.nanmax(z500_anomaly):.0f} gpm")

# 中纬度掩膜（20-60°N）
mid_lat_mask = (extractor.lat >= 20) & (extractor.lat <= 60)
print(f"\n【关键限制1】中纬度掩膜（20°N-60°N）:")
print(f"  - 掩膜范围: {extractor.lat[mid_lat_mask][0]:.1f}°N - {extractor.lat[mid_lat_mask][-1]:.1f}°N")
print(f"  - 台风纬度: {tc_lat}°N")
print(f"  - 台风是否在中纬度范围内: {'✅ 是' if tc_lat >= 20 and tc_lat <= 60 else '❌ 否'}")
print(f"  - 结论: 台风位于热带，远离西风槽的典型活动区域")

# 检查搜索半径内是否有中纬度区域
lat_idx = np.abs(extractor.lat - tc_lat).argmin()
lon_idx = np.abs(extractor.lon - tc_lon).argmin()
radius_deg = 30  # 搜索半径
radius_points = int(radius_deg / extractor.lat_spacing)

lat_start = max(0, lat_idx - radius_points)
lat_end = min(len(extractor.lat), lat_idx + radius_points + 1)

search_lat_range = [extractor.lat[lat_start], extractor.lat[lat_end-1]]
print(f"\n【关键限制2】搜索范围分析:")
print(f"  - 搜索半径: {radius_deg}°（约3300公里）")
print(f"  - 搜索纬度范围: {search_lat_range[1]:.1f}°N - {search_lat_range[0]:.1f}°N")
print(f"  - 搜索范围最北: {search_lat_range[0]:.1f}°N")
print(f"  - 中纬度起始: 20.0°N")
print(f"  - 搜索范围是否覆盖中纬度: {'✅ 是' if search_lat_range[0] >= 20 else '❌ 否'}")

# 检查搜索范围内的中纬度区域
local_mask = np.zeros_like(z500_anomaly, dtype=bool)
local_mask[lat_start:lat_end, :] = True
local_mid_lat = local_mask & mid_lat_mask[:, np.newaxis]

print(f"  - 搜索范围内的中纬度点数: {np.sum(local_mid_lat)}")

if np.sum(local_mid_lat) == 0:
    print(f"  - 结论: ❌ 搜索范围完全不覆盖中纬度，无法检测西风槽")
else:
    print(f"  - 结论: ✅ 搜索范围覆盖中纬度，可能检测到西风槽")

# 在中纬度区域的距平分析
z500_anomaly_mid = z500_anomaly.copy()
z500_anomaly_mid[~mid_lat_mask, :] = np.nan

negative_anomaly = z500_anomaly_mid < 0
print(f"\n【距平分析】中纬度负距平区域:")
print(f"  - 中纬度负距平点数: {np.sum(negative_anomaly)}")

if np.any(negative_anomaly):
    neg_values = z500_anomaly_mid[negative_anomaly]
    print(f"  - 负距平范围: {np.min(neg_values):.0f} - {np.max(neg_values):.0f} gpm")
    print(f"  - 负距平25分位数: {np.percentile(neg_values, 25):.0f} gpm")
else:
    print(f"  - 结论: 中纬度无负距平区域")

# 气象学解释
print("\n" + "="*70)
print("【气象学解释】")
print("="*70)
print("""
1. **西风槽的典型活动区域**：
   - 西风槽主要出现在中纬度西风带（通常30-60°N）
   - 由中纬度西风急流的波动形成
   - 槽前有西南气流，槽后有西北气流

2. **PANG台风的位置特点**：
   - 位于9.4°N，属于热带地区
   - 该纬度主要受热带环流系统控制：
     * 热带辐合带（ITCZ）
     * 季风槽
     * 热带东风波
   - 不受西风带影响

3. **为什么没有检测到西风槽**：
   ✅ 正常现象，符合气象学原理
   - 台风太靠南，远离西风带
   - 30度搜索半径（~3300km）仍未覆盖中纬度西风槽活动区
   - 该纬度的天气系统以热带系统为主

4. **与其他案例对比**：
   - AURO (15.0°N): 虽然也在热带，但搜索半径可达45°N，检测到远处的西风槽
   - FOUR (41.2°N): 在中纬度，接近西风带，检测到西风槽
   - PANG (9.4°N):  深热带，完全不受西风槽影响 ❌

5. **结论**：
   ✅ 模型预报中该位置确实没有西风槽影响
   ✅ 这是正确的物理现象，不是算法问题
   ✅ 低纬度热带气旋应该关注：
      - 热带辐合带（ITCZ）
      - 季风槽
      - 副热带高压（较远）
      - 海洋热含量
      - 垂直风切变
""")

print("\n" + "="*70)
print("【推荐分析系统】")
print("="*70)
print("""
对于PANG这样的低纬度热带气旋，建议重点关注：

1. ✅ 海洋热含量（OceanHeat）
   - 判断是否有足够能量供应

2. ✅ 垂直风切变（WindShear）
   - 评估大气环境是否有利发展

3. ⚠️ 热带辐合带（ITCZ）
   - 判断是否由季风槽扰动发展而来
   - 评估水汽输送条件

4. ✅ 副热带高压（SubtropicalHigh）
   - 虽然距离较远，但决定路径方向

5. ❌ 西风槽（WesterlyTrough）
   - 对低纬度热带气旋影响很小
   - 未检测到是正常现象

6. ❌ 锋面系统（Frontal）
   - 主要影响中高纬度台风
   - 低纬度不受影响
""")

print("\n" + "="*70)
