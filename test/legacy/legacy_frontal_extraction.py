#!/usr/bin/env python3
"""
锋面系统提取测试脚本

用于测试改进后的锋面系统(Frontal System)提取功能。
支持单文件测试和批量测试。
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor


def numpy_to_python(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


def find_track_file(nc_file):
    """根据NC文件名查找对应的track文件"""
    nc_path = Path(nc_file)
    track_dir = Path("data/test/tracks")
    
    # 尝试匹配文件名
    nc_stem = nc_path.stem
    for track_file in track_dir.glob("*.csv"):
        if nc_stem in track_file.name:
            return str(track_file)
    
    return None


def test_single_file(nc_file, track_file=None, time_idx=0, output_dir="data/testExtract"):
    """测试单个NC文件的锋面系统提取"""
    
    # 自动查找track文件
    if track_file is None:
        track_file = find_track_file(nc_file)
        if track_file is None:
            print(f"❌ 找不到对应的track文件: {nc_file}")
            return None
    
    print(f"\n{'='*60}")
    print(f"测试文件: {nc_file}")
    print(f"追踪文件: {track_file}")
    print(f"时间索引: {time_idx}")
    print(f"{'='*60}\n")
    
    # 读取追踪数据获取台风位置
    track_df = pd.read_csv(track_file)
    if time_idx >= len(track_df):
        print(f"❌ 时次索引 {time_idx} 超出范围（最大: {len(track_df)-1}）")
        return None
    
    tc_lat = track_df.iloc[time_idx]['lat']
    tc_lon = track_df.iloc[time_idx]['lon']
    tc_time = track_df.iloc[time_idx]['time']
    
    print(f"台风位置 (时次 {time_idx}, {tc_time}): {tc_lat:.2f}°N, {tc_lon:.2f}°E\n")
    
    # 初始化提取器
    extractor = TCEnvironmentalSystemsExtractor(nc_file, track_file)
    
    try:
        # 提取锋面系统
        print("提取锋面系统(Frontal System)...")
        frontal_result = extractor.extract_frontal_system(time_idx, tc_lat, tc_lon)
        
        if frontal_result:
            print("✓ 锋面系统提取成功")
            print(f"  描述: {frontal_result['description']}")
            print(f"  位置: {frontal_result['position']['lat']:.2f}°N, {frontal_result['position']['lon']:.2f}°E")
            print(f"  强度: {frontal_result['intensity']['value']} {frontal_result['intensity']['unit']} ({frontal_result['intensity']['level']})")
            print(f"  影响: {frontal_result['properties']['impact']}")
            
            # 形状信息
            if 'coordinates' in frontal_result['shape']:
                coords = frontal_result['shape']['coordinates']
                print(f"\n  边界信息:")
                print(f"    点数: {coords['total_points']}")
                print(f"    跨度: 经度 {coords['span_deg'][0]:.2f}°, 纬度 {coords['span_deg'][1]:.2f}°")
                print(f"    范围: 经度 [{coords['lon_range'][0]:.2f}, {coords['lon_range'][1]:.2f}]")
                print(f"    范围: 纬度 [{coords['lat_range'][0]:.2f}, {coords['lat_range'][1]:.2f}]")
                
                # 合理性检查
                lon_span = coords['span_deg'][0]
                lat_span = coords['span_deg'][1]
                print(f"\n  合理性检查:")
                
                # 检查1: 边界跨度不应过大（锋面应在台风周围800-1200km，约7-11度）
                if lon_span > 30 or lat_span > 30:
                    print(f"    ⚠️  警告: 边界跨度过大 (经度:{lon_span:.1f}°, 纬度:{lat_span:.1f}°)")
                    print(f"        期望: 锋面带跨度应小于30°")
                else:
                    print(f"    ✓ 边界跨度合理 (经度:{lon_span:.1f}°, 纬度:{lat_span:.1f}°)")
                
                # 检查2: 边界中心应接近台风位置（在搜索半径内）
                center_lat = np.mean(coords['lat_range'])
                center_lon = np.mean(coords['lon_range'])
                lat_diff = abs(center_lat - tc_lat)
                lon_diff = abs(center_lon - tc_lon)
                
                if lat_diff > 15 or lon_diff > 15:
                    print(f"    ⚠️  警告: 锋面位置离台风中心较远")
                    print(f"        中心差异: 纬度 {lat_diff:.1f}°, 经度 {lon_diff:.1f}°")
                else:
                    print(f"    ✓ 锋面位置在台风周围范围内")
                
                # 检查3: 温度梯度值应该合理
                gradient_value = frontal_result['intensity']['value']
                if gradient_value < 0.1 or gradient_value > 100:
                    print(f"    ⚠️  警告: 温度梯度值可能异常 ({gradient_value}×10⁻⁵ °C/m)")
                else:
                    print(f"    ✓ 温度梯度值在合理范围内 ({gradient_value}×10⁻⁵ °C/m)")
            else:
                print(f"\n  边界信息: {frontal_result['shape']['description']}")
        else:
            print("ℹ️  未检测到锋面系统")
        
        # 保存结果
        nc_name = Path(nc_file).stem
        output_subdir = Path(output_dir) / "frontal"
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{nc_name}_time{time_idx}_frontal.json"
        
        result = {
            "file": nc_file,
            "track_file": track_file,
            "time_idx": time_idx,
            "tc_position": {"lat": float(tc_lat), "lon": float(tc_lon), "time": tc_time},
            "frontal_system": numpy_to_python(frontal_result) if frontal_result else None
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 结果已保存到: {output_file}")
        return frontal_result
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_all_files(output_dir="data/testExtract"):
    """测试所有NC文件"""
    nc_files = [
        "data/AURO_v100_IFS_2025061000_f000_f240_06.nc",
        "data/FOUR_v200_GFS_2020093012_f000_f240_06.nc",
        "data/PANG_v100_IFS_2022032900_f000_f240_06.nc"
    ]
    
    print("\n" + "="*60)
    print("批量测试锋面系统提取")
    print("="*60)
    
    results = {}
    for nc_file in nc_files:
        if not Path(nc_file).exists():
            print(f"\n⚠️  跳过不存在的文件: {nc_file}")
            continue
        
        # 测试第0时次
        result = test_single_file(nc_file, time_idx=0, output_dir=output_dir)
        results[nc_file] = result
    
    # 统计汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    total = len(results)
    detected = sum(1 for r in results.values() if r is not None)
    
    print(f"总文件数: {total}")
    print(f"检测到锋面: {detected}")
    print(f"未检测到: {total - detected}")
    
    for nc_file, result in results.items():
        nc_name = Path(nc_file).stem
        if result:
            intensity = result['intensity']['value']
            level = result['intensity']['level']
            print(f"  {nc_name}: ✓ 强度={intensity:.2f}×10⁻⁵°C/m ({level})")
        else:
            print(f"  {nc_name}: - 未检测到")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="锋面系统提取测试")
    parser.add_argument("--nc_file", help="NC文件路径")
    parser.add_argument("--track_file", help="Track文件路径（可选，将自动查找）")
    parser.add_argument("--time_idx", type=int, default=0, help="时间索引（默认:0）")
    parser.add_argument("--output_dir", default="data/testExtract", help="输出目录")
    parser.add_argument("--all", action="store_true", help="测试所有文件")
    
    args = parser.parse_args()
    
    if args.all:
        test_all_files(args.output_dir)
    elif args.nc_file:
        test_single_file(args.nc_file, args.track_file, args.time_idx, args.output_dir)
    else:
        parser.print_help()
        print("\n示例:")
        print("  # 测试单个文件")
        print("  python test_frontal_extraction.py --nc_file data/AURO_v100_IFS_2025061000_f000_f240_06.nc")
        print("\n  # 测试所有文件")
        print("  python test_frontal_extraction.py --all")


if __name__ == "__main__":
    main()
