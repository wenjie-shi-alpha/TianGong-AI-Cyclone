#!/usr/bin/env python3
"""
副热带高压和引导气流提取功能测试脚本

测试extract_steering_system函数，验证：
1. 副热带高压位置、强度、形态
2. 引导气流方向和速度
3. 脊线位置输出
4. 与台风中心的相对位置关系
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.environment_extractor.extractor import TCEnvironmentalSystemsExtractor


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_track_data(track_file):
    """加载台风路径数据"""
    df = pd.read_csv(track_file)
    print(f"\n=== 台风路径数据 ===")
    print(f"文件: {track_file}")
    print(f"数据点数: {len(df)}")
    print(f"时间范围: {df['time'].min()} -> {df['time'].max()}")
    return df


def test_steering_extraction(nc_file, track_file, output_file, time_idx=0):
    """测试副热带高压和引导气流提取"""
    
    print(f"\n{'='*60}")
    print(f"副热带高压和引导气流提取测试")
    print(f"{'='*60}")
    print(f"NC文件: {nc_file}")
    print(f"路径文件: {track_file}")
    print(f"时间索引: {time_idx}")
    
    # 加载数据
    track_df = load_track_data(track_file)
    
    # 获取指定时刻的台风位置
    if time_idx >= len(track_df):
        print(f"错误: time_idx {time_idx} 超出范围 (最大: {len(track_df)-1})")
        return
    
    tc_row = track_df.iloc[time_idx]
    tc_lat = float(tc_row['lat'])
    tc_lon = float(tc_row['lon'])
    tc_time = tc_row['time']
    tc_msl = float(tc_row['msl']) if 'msl' in tc_row and pd.notna(tc_row['msl']) else None
    tc_wind = float(tc_row['wind']) if 'wind' in tc_row and pd.notna(tc_row['wind']) else None
    
    print(f"\n=== 台风信息 (time_idx={time_idx}) ===")
    print(f"时间: {tc_time}")
    print(f"位置: {tc_lat:.2f}°N, {tc_lon:.2f}°E")
    if tc_msl:
        print(f"中心气压: {tc_msl:.1f} hPa")
    if tc_wind:
        print(f"最大风速: {tc_wind:.1f} m/s")
    
    # 初始化提取器
    extractor = TCEnvironmentalSystemsExtractor(nc_file, track_file)
    
    # 提取副热带高压和引导气流
    print(f"\n{'='*60}")
    print(f"开始提取副热带高压和引导气流...")
    print(f"{'='*60}")
    
    try:
        steering_result = extractor.extract_steering_system(time_idx, tc_lat, tc_lon)
        
        if steering_result:
            print(f"\n✓ 副热带高压提取成功!")
            
            # 打印关键信息
            print(f"\n=== 副热带高压信息 ===")
            position = steering_result.get('position', {})
            print(f"位置: {position.get('latitude', 'N/A')}°N, {position.get('longitude', 'N/A')}°E")
            print(f"描述: {position.get('description', 'N/A')}")
            
            properties = steering_result.get('properties', {})
            print(f"\n强度: {properties.get('intensity', 'N/A')} gpm")
            print(f"等级: {properties.get('intensity_level', 'N/A')}")
            
            shape = steering_result.get('shape', {})
            print(f"\n形态: {shape.get('shape_type', 'N/A')}")
            print(f"朝向: {shape.get('orientation', 'N/A')}")
            
            # 检查是否有脊线位置输出
            if 'ridge_line' in properties:
                ridge_line = properties['ridge_line']
                print(f"\n脊线位置:")
                print(f"  东端: {ridge_line.get('east_end', 'N/A')}")
                print(f"  西端: {ridge_line.get('west_end', 'N/A')}")
            
            # 引导气流
            if 'steering_flow' in steering_result:
                steering = steering_result['steering_flow']
                print(f"\n=== 引导气流 ===")
                print(f"速度: {steering.get('speed', 'N/A')} m/s")
                print(f"方向: {steering.get('direction', 'N/A')}° (气象惯例)")
                print(f"u分量: {steering.get('u_component', 'N/A')} m/s")
                print(f"v分量: {steering.get('v_component', 'N/A')} m/s")
                if 'calculation_method' in steering:
                    print(f"计算方法: {steering['calculation_method']}")
            
            print(f"\n影响分析: {steering_result.get('impact', 'N/A')}")
            
            # 检查边界坐标数量
            boundary = steering_result.get('boundary_coordinates', [])
            if boundary:
                print(f"\n边界坐标点数: {len(boundary)}")
                if len(boundary) > 0:
                    lons = [c[0] for c in boundary]
                    lats = [c[1] for c in boundary]
                    lon_range = max(lons) - min(lons)
                    lat_range = max(lats) - min(lats)
                    print(f"经度范围: {min(lons):.2f}° - {max(lons):.2f}° (跨度: {lon_range:.2f}°)")
                    print(f"纬度范围: {min(lats):.2f}° - {max(lats):.2f}° (跨度: {lat_range:.2f}°)")
                    
                    # 检查是否合理（不应该跨越全球）
                    if lon_range > 50 or lat_range > 50:
                        print(f"⚠️  警告: 边界范围过大，可能提取了全局系统而非局部副高")
            else:
                print(f"\n边界坐标: 无")
            
        else:
            print(f"\n✗ 未检测到副热带高压")
            steering_result = {
                "status": "not_detected",
                "message": "在当前时刻未检测到副热带高压系统"
            }
    
    except Exception as e:
        print(f"\n✗ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        steering_result = {
            "status": "error",
            "error": str(e)
        }
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "test_info": {
            "nc_file": str(nc_file),
            "track_file": str(track_file),
            "time_idx": time_idx,
            "test_time": tc_time,
        },
        "typhoon_position": {
            "latitude": tc_lat,
            "longitude": tc_lon,
            "msl": tc_msl,
            "wind": tc_wind,
        },
        "steering_system": convert_to_json_serializable(steering_result)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='测试副热带高压和引导气流提取功能')
    parser.add_argument('--nc_file', required=True, help='NC文件路径')
    parser.add_argument('--track_file', required=True, help='台风路径CSV文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--time_idx', type=int, default=0, help='时间索引 (默认: 0)')
    
    args = parser.parse_args()
    
    test_steering_extraction(
        args.nc_file,
        args.track_file,
        args.output,
        args.time_idx
    )


if __name__ == '__main__':
    main()
