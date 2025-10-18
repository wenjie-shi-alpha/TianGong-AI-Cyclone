#!/usr/bin/env python3
"""
西风槽提取测试脚本
根据真实台风追踪数据测试西风槽系统提取功能
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor


def load_track_data(track_file):
    """加载台风追踪数据"""
    df = pd.read_csv(track_file)
    return df


def test_westerly_trough_extraction(nc_file, track_file, output_dir, time_indices=None):
    """
    测试西风槽提取功能
    
    Args:
        nc_file: NC文件路径
        track_file: 台风追踪CSV文件路径
        output_dir: 输出目录
        time_indices: 要测试的时间索引列表，None表示测试所有时间点
    """
    print(f"\n{'='*60}")
    print(f"测试文件: {nc_file}")
    print(f"追踪数据: {track_file}")
    print(f"{'='*60}\n")
    
    # 创建提取器（直接传递文件路径）
    extractor = TCEnvironmentalSystemsExtractor(nc_file, track_file)
    
    # 加载追踪数据
    track_df = load_track_data(track_file)
    
    print(f"追踪数据包含 {len(track_df)} 个时间点")
    
    # 确定要测试的时间索引
    if time_indices is None:
        time_indices = range(min(len(track_df), len(extractor.ds.time)))
    
    results = []
    
    for idx in time_indices:
        if idx >= len(track_df) or idx >= len(extractor.ds.time):
            print(f"⚠️  时间索引 {idx} 超出范围，跳过")
            continue
            
        row = track_df.iloc[idx]
        tc_lat = row['lat']
        tc_lon = row['lon']
        
        print(f"\n--- 时间索引 {idx} ---")
        print(f"台风位置: {tc_lat:.2f}°N, {tc_lon:.2f}°E")
        
        # 提取西风槽
        westerly_trough = extractor.extract_westerly_trough(idx, tc_lat, tc_lon)
        
        if westerly_trough:
            print("✅ 检测到西风槽系统")
            print(f"   描述: {westerly_trough['description']}")
            
            # 检查输出内容
            if 'position' in westerly_trough:
                pos = westerly_trough['position']
                if 'center_of_mass' in pos:
                    print(f"   质心位置: {pos['center_of_mass']['lat']:.2f}°N, {pos['center_of_mass']['lon']:.2f}°E")
            
            if 'properties' in westerly_trough:
                props = westerly_trough['properties']
                if 'distance_to_tc_km' in props:
                    print(f"   距台风中心: {props['distance_to_tc_km']:.0f} km")
                if 'bearing_from_tc' in props:
                    print(f"   方位角: {props['bearing_from_tc']:.1f}°")
            
            if 'shape' in westerly_trough:
                shape = westerly_trough['shape']
                if 'coordinates' in shape:
                    coords = shape['coordinates']
                    if 'boundary' in coords:
                        print(f"   边界点数: {len(coords['boundary'])}")
                        # 检查边界范围
                        lons = [c[0] for c in coords['boundary']]
                        lats = [c[1] for c in coords['boundary']]
                        lon_span = max(lons) - min(lons)
                        lat_span = max(lats) - min(lats)
                        print(f"   边界范围: 经度跨度 {lon_span:.1f}°, 纬度跨度 {lat_span:.1f}°")
                        
                        # 检查是否为局地系统
                        if lon_span > 50 or lat_span > 50:
                            print(f"   ⚠️  边界范围异常大！可能提取了全局系统")
                    
                    # 检查槽轴线（改进后应该有）
                    if 'trough_axis' in coords:
                        print(f"   ✅ 包含槽轴线: {len(coords['trough_axis'])} 个点")
                    else:
                        print(f"   ⚠️  缺少槽轴线信息")
                    
                    # 检查槽底（改进后应该有）
                    if 'trough_bottom' in coords:
                        print(f"   ✅ 槽底位置: {coords['trough_bottom']}")
                    else:
                        print(f"   ⚠️  缺少槽底位置")
            
            # 检查新增字段
            if 'intensity' in westerly_trough:
                intensity = westerly_trough['intensity']
                if 'z500_anomaly' in intensity:
                    print(f"   ✅ 500hPa高度距平: {intensity['z500_anomaly']:.0f} gpm")
                if 'pv_gradient' in intensity:
                    print(f"   ✅ PV梯度: {intensity['pv_gradient']:.2e} PVU/km")
        else:
            print("❌ 未检测到西风槽系统")
        
        # 保存结果
        result = {
            'time_idx': int(idx),
            'tc_position': {
                'lat': float(tc_lat),
                'lon': float(tc_lon)
            },
            'westerly_trough': westerly_trough
        }
        results.append(result)
    
    # 保存结果到文件
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nc_name = Path(nc_file).stem
    output_file = output_dir / f"{nc_name}_westerly_trough.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*60}\n")
    
    # 统计摘要
    detected_count = sum(1 for r in results if r['westerly_trough'] is not None)
    print(f"\n统计摘要:")
    print(f"  总测试点数: {len(results)}")
    print(f"  检测到西风槽: {detected_count}")
    print(f"  检测率: {detected_count/len(results)*100:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试西风槽提取功能')
    parser.add_argument('--nc_file', type=str,
                       help='NC数据文件路径')
    parser.add_argument('--track_file', type=str,
                       help='台风追踪CSV文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='data/testExtract/westerlyTrough',
                       help='输出目录')
    parser.add_argument('--time_indices', type=str,
                       help='要测试的时间索引，用逗号分隔，如"0,1,2"')
    parser.add_argument('--all', action='store_true',
                       help='测试所有三个示例文件')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.all and (not args.nc_file or not args.track_file):
        parser.error('如果不使用 --all，必须提供 --nc_file 和 --track_file')
    
    if args.all:
        # 测试所有三个示例文件
        test_cases = [
            {
                'nc_file': 'data/AURO_v100_IFS_2025061000_f000_f240_06.nc',
                'track_file': 'data/test/tracks/track_2025162N15114_AURO_v100_IFS_2025061000_f000_f240_06.csv'
            },
            {
                'nc_file': 'data/FOUR_v200_GFS_2020093012_f000_f240_06.nc',
                'track_file': 'data/test/tracks/track_2020270N17159_FOUR_v200_GFS_2020093012_f000_f240_06.csv'
            },
            {
                'nc_file': 'data/PANG_v100_IFS_2022032900_f000_f240_06.nc',
                'track_file': 'data/test/tracks/track_2022088N09116_PANG_v100_IFS_2022032900_f000_f240_06.csv'
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'#'*60}")
            print(f"# 测试案例 {i}/{len(test_cases)}")
            print(f"{'#'*60}")
            try:
                test_westerly_trough_extraction(
                    case['nc_file'],
                    case['track_file'],
                    args.output_dir,
                    time_indices=[0]  # 只测试第一个时间点
                )
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        # 解析时间索引
        time_indices = None
        if args.time_indices:
            time_indices = [int(x.strip()) for x in args.time_indices.split(',')]
        
        test_westerly_trough_extraction(
            args.nc_file,
            args.track_file,
            args.output_dir,
            time_indices
        )


if __name__ == '__main__':
    main()
