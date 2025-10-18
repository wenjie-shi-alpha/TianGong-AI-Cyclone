#!/usr/bin/env python3
"""
副热带高压提取测试脚本

测试副高系统提取的关键改进:
1. 区域化处理 - 使用局部子域而非全局场
2. 脊线位置提取 - 588线的东西端点
3. 中心位置计算
4. 相对于台风的方位

使用方法:
    python3 test_subtropical_high_extraction.py --nc_file data/AURO_v100_IFS_2025061000_f000_f240_06.nc \
        --tc_lat 15.0 --tc_lon 113.9 --time_idx 0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from environment_extractor import TCEnvironmentalSystemsExtractor
import xarray as xr


def test_subtropical_high_extraction(nc_file, tc_lat, tc_lon, time_idx=0, output_dir=None):
    """
    测试副热带高压提取功能
    
    验证要点:
    1. 是否成功识别副高系统
    2. 是否提取了脊线信息(东西端点)
    3. 中心位置是否合理
    4. 相对台风的方位描述是否准确
    5. 边界坐标是否局地化(不再是全球范围)
    
    Args:
        nc_file: NetCDF文件路径
        tc_lat: 台风中心纬度
        tc_lon: 台风中心经度  
        time_idx: 时次索引
        output_dir: 输出目录
    
    Returns:
        提取结果字典
    """
    print(f"\n{'='*70}")
    print(f"副热带高压提取测试")
    print(f"{'='*70}")
    print(f"数据文件: {nc_file}")
    print(f"台风位置: ({tc_lat}°N, {tc_lon}°E)")
    print(f"时次索引: {time_idx}")
    
    try:
        # 1. 打开数据集以获取时间/坐标信息（随后由提取器内部重新打开）
        ds = xr.open_dataset(nc_file)
        print(f"\n数据集信息:")
        print(f"  - 时间维度: {len(ds.time)} 个时次")

        # 兼容不同的坐标名称
        lat_coord = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_coord = 'longitude' if 'longitude' in ds.coords else 'lon'

        print(f"  - 纬度范围: {float(ds[lat_coord].min().values):.2f}° 到 {float(ds[lat_coord].max().values):.2f}°")
        print(f"  - 经度范围: {float(ds[lon_coord].min().values):.2f}° 到 {float(ds[lon_coord].max().values):.2f}°")

        # 2. 创建临时追踪CSV文件 (提取器需要路径形式的tracks CSV)
        import tempfile
        import pandas as pd

        first_time = pd.to_datetime(ds.time[0].values)
        ds.close()

        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_df = pd.DataFrame({
            'time': [first_time],
            'lat': [tc_lat],
            'lon': [tc_lon],
            'name': ['TEST_TC']
        })
        temp_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()

        # 3. 使用提取器（context manager 模式），按库要求传入文件路径
        print(f"\n初始化环境场提取器...")
        with TCEnvironmentalSystemsExtractor(nc_file, temp_csv.name) as extractor:
            # 4. 提取副热带高压系统
            print(f"\n开始提取副热带高压系统...")
            steering_result = extractor.extract_steering_system(time_idx, tc_lat, tc_lon)
        
        if steering_result is None:
            print("❌ 副热带高压提取失败 - 返回None")
            return None
        
        # 4. 分析提取结果
        print(f"\n✅ 副热带高压提取成功!")
        print(f"\n{'='*70}")
        print(f"提取结果分析")
        print(f"{'='*70}")
        
        # 4.1 检查系统类型
        if 'systems' in steering_result:
            systems = steering_result['systems']
            print(f"\n识别到 {len(systems)} 个系统:")
            for i, sys in enumerate(systems):
                print(f"\n  系统 {i+1}: {sys.get('type', 'unknown')}")
                if sys.get('type') == 'subtropical_high':
                    print(f"    - 这是副热带高压系统 ✓")
                    analyze_subtropical_high(sys, tc_lat, tc_lon)
        
        # 4.2 检查脊线信息
        if 'properties' in steering_result and 'ridge_line' in steering_result['properties']:
            ridge_info = steering_result['properties']['ridge_line']
            print(f"\n【脊线信息】")
            print(f"  - 东端: ({ridge_info['east_end']['latitude']}°N, {ridge_info['east_end']['longitude']}°E)")
            print(f"    相对台风: {ridge_info['east_end']['relative_position']}")
            print(f"  - 西端: ({ridge_info['west_end']['latitude']}°N, {ridge_info['west_end']['longitude']}°E)")
            print(f"    相对台风: {ridge_info['west_end']['relative_position']}")
            print(f"  - 描述: {ridge_info['description']}")
        
        # 4.3 检查引导气流
        if 'steering_flow' in steering_result:
            flow = steering_result['steering_flow']
            print(f"\n【引导气流】")
            print(f"  - 速度: {flow.get('speed', 'N/A')} m/s")
            print(f"  - 方向: {flow.get('direction', 'N/A')}°")
            print(f"  - 计算方法: {flow.get('method', 'N/A')}")
        
        # 5. 验证边界坐标范围
        if 'boundary_coordinates' in steering_result:
            coords = steering_result['boundary_coordinates']
            if coords and len(coords) > 0:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                lon_span = max(lons) - min(lons)
                lat_span = max(lats) - min(lats)
                
                print(f"\n【边界坐标验证】")
                print(f"  - 边界点数: {len(coords)}")
                print(f"  - 经度范围: {min(lons):.2f}° 到 {max(lons):.2f}° (跨度: {lon_span:.2f}°)")
                print(f"  - 纬度范围: {min(lats):.2f}° 到 {max(lats):.2f}° (跨度: {lat_span:.2f}°)")
                
                # 检查是否为局地化边界
                if lon_span > 50 or lat_span > 40:
                    print(f"  ⚠️ 警告: 边界跨度过大，可能仍是全球范围!")
                    print(f"      期望跨度: 经度<50°, 纬度<40°")
                else:
                    print(f"  ✓ 边界范围合理，已局地化")
        
        # 6. 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            nc_name = Path(nc_file).stem
            output_file = output_path / f"{nc_name}_subtropical_high_t{time_idx}.json"
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                import numpy as _np
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (_np.integer, _np.floating)):
                    return obj.item()
                elif isinstance(obj, _np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            steering_serializable = convert_numpy_types(steering_result)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(steering_serializable, f, ensure_ascii=False, indent=2)

            print(f"\n结果已保存到: {output_file}")
        
        return steering_result
        
    except Exception as e:
        print(f"\n❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_subtropical_high(system_info, tc_lat, tc_lon):
    """分析副热带高压系统的详细信息"""
    
    # 位置信息
    if 'position' in system_info:
        pos = system_info['position']
        if 'center_of_mass' in pos:
            center = pos['center_of_mass']
            print(f"    【中心位置】")
            print(f"      中心: ({center.get('lat', 'N/A')}°N, {center.get('lon', 'N/A')}°E)")
            
            # 计算与台风的距离
            if 'lat' in center and 'lon' in center:
                import math
                dlat = center['lat'] - tc_lat
                dlon = center['lon'] - tc_lon
                distance_deg = math.sqrt(dlat**2 + dlon**2)
                print(f"      距台风: {distance_deg:.2f}° (~{distance_deg*111:.0f} km)")
    
    # 强度信息
    if 'intensity' in system_info:
        intensity = system_info['intensity']
        print(f"    【强度】")
        print(f"      值: {intensity.get('value', 'N/A')} {intensity.get('unit', '')}")
    
    # 边界坐标
    if 'boundary_coordinates' in system_info:
        coords = system_info['boundary_coordinates']
        print(f"    【边界】")
        print(f"      边界点数: {len(coords) if coords else 0}")
        
        if coords and len(coords) > 0:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            print(f"      经度跨度: {max(lons)-min(lons):.2f}°")
            print(f"      纬度跨度: {max(lats)-min(lats):.2f}°")
    
    # 脊线信息
    if 'properties' in system_info and 'ridge_line' in system_info['properties']:
        ridge = system_info['properties']['ridge_line']
        print(f"    【脊线】")
        print(f"      东端: {ridge['east_end']['relative_position']}")
        print(f"      西端: {ridge['west_end']['relative_position']}")
    
    # 提取方法
    if 'extraction_info' in system_info:
        info = system_info['extraction_info']
        print(f"    【提取信息】")
        print(f"      方法: {info.get('method', 'N/A')}")
        if 'dynamic_threshold' in info:
            print(f"      动态阈值: {info['dynamic_threshold']} gpm")


def main():
    parser = argparse.ArgumentParser(description='测试副热带高压提取功能')
    parser.add_argument('--nc_file', required=True, help='NetCDF数据文件路径')
    parser.add_argument('--tc_lat', type=float, required=True, help='台风中心纬度')
    parser.add_argument('--tc_lon', type=float, required=True, help='台风中心经度')
    parser.add_argument('--time_idx', type=int, default=0, help='时次索引(默认0)')
    parser.add_argument('--output', default='data/testExtract/subtropicalHigh', 
                       help='输出目录(默认data/testExtract/subtropicalHigh)')
    
    args = parser.parse_args()
    
    # 运行测试
    result = test_subtropical_high_extraction(
        args.nc_file,
        args.tc_lat,
        args.tc_lon,
        args.time_idx,
        args.output
    )
    
    if result:
        print(f"\n{'='*70}")
        print(f"✅ 测试完成!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"❌ 测试失败")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == '__main__':
    main()
