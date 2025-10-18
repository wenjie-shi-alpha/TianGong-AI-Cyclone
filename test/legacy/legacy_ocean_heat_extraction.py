#!/usr/bin/env python3
"""
测试脚本：验证海洋热含量边界提取的局部化修复

基于真实台风路径CSV文件，测试修复后的海洋热含量提取功能：
1. 边界坐标是否在台风中心附近的合理范围内
2. 边界不再跨越全球范围
3. 边界坐标正确处理跨越日期变更线的情况
"""

import json
import sys
from pathlib import Path

import pandas as pd

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor


def test_ocean_heat_extraction(nc_file, csv_file, output_file):
    """
    测试海洋热含量提取
    
    Args:
        nc_file: NC数据文件路径
        csv_file: 台风路径CSV文件
        output_file: 输出JSON文件路径
    """
    print(f"\n{'='*70}")
    print(f"测试文件: {Path(nc_file).name}")
    print(f"追踪文件: {Path(csv_file).name}")
    print(f"{'='*70}\n")
    
    # 创建提取器
    with TCEnvironmentalSystemsExtractor(nc_file, csv_file) as extractor:
        # 读取第一个追踪点
        tracks = pd.read_csv(csv_file)
        tracks['time'] = pd.to_datetime(tracks['time'])
        
        first_point = tracks.iloc[0]
        tc_lat = first_point['lat']
        tc_lon = first_point['lon']
        time_idx = 0  # 使用第一个时间步
        
        print(f"台风中心位置: {tc_lat:.2f}°N, {tc_lon:.2f}°E")
        print(f"时间: {first_point['time']}")
        print(f"\n正在提取海洋热含量...\n")
        
        # 提取海洋热含量
        result = extractor.extract_ocean_heat_content(time_idx, tc_lat, tc_lon, radius_deg=2.0)
        
        if result is None:
            print("❌ 海洋热含量提取失败")
            return
        
        # 分析结果
        print("✅ 海洋热含量提取成功\n")
        print(f"描述: {result['description']}\n")
        print(f"SST平均值: {result['intensity']['value']}°C")
        print(f"等级: {result['intensity']['level']}")
        
        # 检查边界坐标
        boundary = result.get('shape', {}).get('warm_water_boundary_26.5C')
        
        if boundary is None or len(boundary) == 0:
            print("\n⚠️  未提取到26.5°C等值线边界")
            print("   （可能局部区域内不存在该等值线）")
        else:
            print(f"\n边界点数量: {len(boundary)}")
            
            # 计算边界范围
            lons = [p[0] for p in boundary]
            lats = [p[1] for p in boundary]
            
            lon_min, lon_max = min(lons), max(lons)
            lat_min, lat_max = min(lats), max(lats)
            lon_span = lon_max - lon_min
            lat_span = lat_max - lat_min
            
            print(f"\n边界范围分析:")
            print(f"  经度范围: {lon_min:.2f}° - {lon_max:.2f}° (跨度: {lon_span:.2f}°)")
            print(f"  纬度范围: {lat_min:.2f}° - {lat_max:.2f}° (跨度: {lat_span:.2f}°)")
            
            # 计算边界中心距台风中心的距离
            boundary_center_lon = sum(lons) / len(lons)
            boundary_center_lat = sum(lats) / len(lats)
            lon_diff = abs(boundary_center_lon - tc_lon)
            lat_diff = abs(boundary_center_lat - tc_lat)
            
            print(f"\n边界中心:")
            print(f"  位置: {boundary_center_lat:.2f}°N, {boundary_center_lon:.2f}°E")
            print(f"  距台风中心: Δlat={lat_diff:.2f}°, Δlon={lon_diff:.2f}°")
            
            # 验证边界是否在合理范围内
            print(f"\n验证结果:")
            
            # 由于提取半径是radius_deg*3 = 6度，边界应该在12度范围内
            expected_max_span = 12.0  # 6度半径 × 2
            
            if lon_span > 50:
                print(f"  ❌ 经度跨度过大({lon_span:.1f}°)，可能仍存在全局边界问题")
            elif lon_span > expected_max_span:
                print(f"  ⚠️  经度跨度({lon_span:.1f}°)超出预期({expected_max_span}°)，但在可接受范围")
            else:
                print(f"  ✅ 经度跨度({lon_span:.1f}°)在合理范围内(预期≤{expected_max_span}°)")
            
            if lat_span > 50:
                print(f"  ❌ 纬度跨度过大({lat_span:.1f}°)，可能仍存在全局边界问题")
            elif lat_span > expected_max_span:
                print(f"  ⚠️  纬度跨度({lat_span:.1f}°)超出预期({expected_max_span}°)，但在可接受范围")
            else:
                print(f"  ✅ 纬度跨度({lat_span:.1f}°)在合理范围内(预期≤{expected_max_span}°)")
            
            if lon_diff > 10:
                print(f"  ⚠️  边界中心距台风中心较远(Δlon={lon_diff:.1f}°)")
            else:
                print(f"  ✅ 边界中心经度接近台风中心(Δlon={lon_diff:.1f}°)")
            
            if lat_diff > 10:
                print(f"  ⚠️  边界中心距台风中心较远(Δlat={lat_diff:.1f}°)")
            else:
                print(f"  ✅ 边界中心纬度接近台风中心(Δlat={lat_diff:.1f}°)")
        
        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        result_serializable = convert_numpy_types(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")


def main():
    """运行所有测试"""
    # 定义测试用例
    test_cases = [
        {
            'name': 'AURO (2025)',
            'nc_file': 'data/AURO_v100_IFS_2025061000_f000_f240_06.nc',
            'csv_file': 'data/test/tracks/track_2025162N15114_AURO_v100_IFS_2025061000_f000_f240_06.csv',
            'output': 'data/testExtract/oceanHeat/test_ocean_heat_AURO.json'
        },
        {
            'name': 'FOUR (2020)',
            'nc_file': 'data/FOUR_v200_GFS_2020093012_f000_f240_06.nc',
            'csv_file': 'data/test/tracks/track_2020270N17159_FOUR_v200_GFS_2020093012_f000_f240_06.csv',
            'output': 'data/testExtract/oceanHeat/test_ocean_heat_FOUR.json'
        },
        {
            'name': 'PANG (2022)',
            'nc_file': 'data/PANG_v100_IFS_2022032900_f000_f240_06.nc',
            'csv_file': 'data/test/tracks/track_2022088N09116_PANG_v100_IFS_2022032900_f000_f240_06.csv',
            'output': 'data/testExtract/oceanHeat/test_ocean_heat_PANG.json'
        }
    ]
    
    print("\n" + "="*70)
    print("海洋热含量局部化提取测试")
    print("="*70)
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n[测试 {i}/{len(test_cases)}] {test_case['name']}")
            test_ocean_heat_extraction(
                test_case['nc_file'],
                test_case['csv_file'],
                test_case['output']
            )
            success_count += 1
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"测试完成: {success_count}/{len(test_cases)} 成功")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
