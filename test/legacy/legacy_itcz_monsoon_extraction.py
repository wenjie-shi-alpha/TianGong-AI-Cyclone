#!/usr/bin/env python3
"""
ITCZ和季风槽提取测试脚本

用于测试改进后的热带辐合带(ITCZ)和季风槽提取功能。
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
    """测试单个NC文件的ITCZ和季风槽提取"""
    
    # 自动查找track文件
    if track_file is None:
        track_file = find_track_file(nc_file)
        if track_file is None:
            print(f"❌ 找不到对应的track文件: {nc_file}")
            return None, None
    
    print(f"\n{'='*60}")
    print(f"测试文件: {nc_file}")
    print(f"追踪文件: {track_file}")
    print(f"时间索引: {time_idx}")
    print(f"{'='*60}\n")
    
    # 读取追踪数据获取台风位置
    track_df = pd.read_csv(track_file)
    if time_idx >= len(track_df):
        print(f"❌ 时次索引 {time_idx} 超出范围（最大: {len(track_df)-1}）")
        return None, None
    
    tc_lat = track_df.iloc[time_idx]['lat']
    tc_lon = track_df.iloc[time_idx]['lon']
    tc_time = track_df.iloc[time_idx]['time']
    
    print(f"台风位置 (时次 {time_idx}, {tc_time}): {tc_lat:.2f}°N, {tc_lon:.2f}°E\n")
    
    # 初始化提取器
    extractor = TCEnvironmentalSystemsExtractor(nc_file, track_file)
    
    try:
        # 1. 提取ITCZ
        print("1. 提取热带辐合带(ITCZ)...")
        itcz_result = extractor.extract_intertropical_convergence_zone(
            time_idx, tc_lat, tc_lon
        )
        
        if itcz_result:
            print("✓ ITCZ提取成功")
            print(f"  描述: {itcz_result['description']}")
            print(f"  位置: {itcz_result['position']['lat']}°, {itcz_result['position'].get('lon_range', 'N/A')}")
            print(f"  辐合强度: {itcz_result['intensity']['value']} {itcz_result['intensity']['unit']} ({itcz_result['intensity']['level']})")
            print(f"  距台风中心: {itcz_result['properties']['distance_to_tc_km']:.1f} km")
            print(f"  影响程度: {itcz_result['properties']['influence']}")
            
            # 合理性检查
            print("\n  合理性检查:")
            checks = []
            
            # 检查1: 辐合强度应为正值且在合理范围内
            conv_value = itcz_result['intensity']['value']
            if 0 < conv_value < 50:
                checks.append("✓ 辐合强度在合理范围 (0-50 ×10⁻⁵ s⁻¹)")
            else:
                checks.append(f"✗ 辐合强度异常: {conv_value}")
            
            # 检查2: ITCZ纬度应在热带范围内
            itcz_lat = itcz_result['position']['lat']
            if -25 < itcz_lat < 25:
                checks.append(f"✓ ITCZ纬度在热带范围 ({itcz_lat}°)")
            else:
                checks.append(f"✗ ITCZ纬度超出热带范围: {itcz_lat}°")
            
            # 检查3: 距离计算合理
            dist_km = itcz_result['properties']['distance_to_tc_km']
            dist_deg = itcz_result['properties']['distance_to_tc_deg']
            expected_km = abs(dist_deg * 111)
            if abs(dist_km - expected_km) < expected_km * 0.3:
                checks.append(f"✓ 距离计算合理 ({dist_km:.0f} km ≈ {expected_km:.0f} km)")
            else:
                checks.append(f"⚠ 距离计算可能有误: {dist_km:.0f} km vs {expected_km:.0f} km")
            
            # 检查4: 边界坐标（如果存在）
            if 'boundary_coordinates' in itcz_result:
                coords = itcz_result['boundary_coordinates']
                checks.append(f"✓ 提供了边界坐标 ({len(coords)} 个点)")
            else:
                checks.append("⚠ 未提供边界坐标")
            
            for check in checks:
                print(f"    {check}")
        else:
            print("✗ ITCZ提取失败或未检测到")
        
        # 2. 提取季风槽
        print("\n2. 提取季风槽(Monsoon Trough)...")
        monsoon_result = extractor.extract_monsoon_trough(
            time_idx, tc_lat, tc_lon
        )
        
        if monsoon_result:
            print("✓ 季风槽提取成功")
            print(f"  描述: {monsoon_result['description']}")
            print(f"  槽底位置: {monsoon_result['position']['lat']}°, {monsoon_result['position']['lon']}°")
            print(f"  涡度强度: {monsoon_result['intensity']['value']} {monsoon_result['intensity']['unit']} ({monsoon_result['intensity']['level']})")
            print(f"  距台风中心: {monsoon_result['properties']['distance_to_tc_km']:.1f} km ({monsoon_result['properties']['direction_from_tc']})")
            print(f"  槽轴长度: {monsoon_result['shape']['axis_length_km']:.0f} km")
            print(f"  风场特征: {monsoon_result['properties']['zonal_wind_pattern']}")
            print(f"  影响程度: {monsoon_result['properties']['influence']}")
            
            # 合理性检查
            print("\n  合理性检查:")
            checks = []
            
            # 检查1: 涡度强度应为正值且在合理范围内
            vort_value = monsoon_result['intensity']['value']
            if 0 < vort_value < 50:
                checks.append(f"✓ 涡度强度在合理范围 (0-50 ×10⁻⁵ s⁻¹)")
            else:
                checks.append(f"✗ 涡度强度异常: {vort_value}")
            
            # 检查2: 槽底纬度应在热带范围内
            trough_lat = monsoon_result['position']['lat']
            if -30 < trough_lat < 30:
                checks.append(f"✓ 槽底纬度在热带范围 ({trough_lat}°)")
            else:
                checks.append(f"✗ 槽底纬度超出热带范围: {trough_lat}°")
            
            # 检查3: 距离应在搜索范围内
            dist_km = monsoon_result['properties']['distance_to_tc_km']
            if dist_km < 1500:
                checks.append(f"✓ 距离在搜索范围内 ({dist_km:.0f} km < 1500 km)")
            else:
                checks.append(f"⚠ 距离超出搜索范围: {dist_km:.0f} km")
            
            # 检查4: 槽轴长度合理
            axis_length = monsoon_result['shape']['axis_length_km']
            if 0 < axis_length < 5000:
                checks.append(f"✓ 槽轴长度合理 ({axis_length:.0f} km)")
            else:
                checks.append(f"⚠ 槽轴长度可能异常: {axis_length:.0f} km")
            
            # 检查5: 边界坐标（如果存在）
            if 'boundary_coordinates' in monsoon_result:
                coords = monsoon_result['boundary_coordinates']
                checks.append(f"✓ 提供了边界坐标 ({len(coords)} 个点)")
            else:
                checks.append("⚠ 未提供边界坐标")
            
            for check in checks:
                print(f"    {check}")
        else:
            print("✗ 季风槽提取失败或未检测到")
        
        # 3. 保存结果
        output_path = Path(output_dir)
        
        # 保存ITCZ结果
        if itcz_result:
            itcz_dir = output_path / "itcz"
            itcz_dir.mkdir(parents=True, exist_ok=True)
            
            filename = Path(nc_file).stem
            output_file = itcz_dir / f"{filename}_itcz.json"
            
            # 转换numpy类型为Python原生类型
            itcz_serializable = numpy_to_python(itcz_result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(itcz_serializable, f, indent=2, ensure_ascii=False)
            print(f"\n✓ ITCZ结果已保存至: {output_file}")
        
        # 保存季风槽结果
        if monsoon_result:
            monsoon_dir = output_path / "monsoonTrough"
            monsoon_dir.mkdir(parents=True, exist_ok=True)
            
            filename = Path(nc_file).stem
            output_file = monsoon_dir / f"{filename}_monsoon.json"
            
            # 转换numpy类型为Python原生类型
            monsoon_serializable = numpy_to_python(monsoon_result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(monsoon_serializable, f, indent=2, ensure_ascii=False)
            print(f"✓ 季风槽结果已保存至: {output_file}")
        
        return itcz_result, monsoon_result
        
    finally:
        extractor.close()


def test_all_files(output_dir="data/testExtract"):
    """测试所有预定义的NC文件"""
    
    # 预定义的测试用例（文件路径）
    test_cases = [
        "data/AURO_v100_IFS_2025061000_f000_f240_06.nc",
        "data/FOUR_v200_GFS_2020093012_f000_f240_06.nc",
        "data/PANG_v100_IFS_2022032900_f000_f240_06.nc",
    ]
    
    print("\n" + "="*60)
    print("开始批量测试ITCZ和季风槽提取")
    print("="*60)
    
    results = []
    for nc_file in test_cases:
        if not Path(nc_file).exists():
            print(f"\n⚠ 跳过不存在的文件: {nc_file}")
            continue
        
        itcz_result, monsoon_result = test_single_file(nc_file, None, 0, output_dir)
        results.append({
            'file': nc_file,
            'itcz': itcz_result is not None,
            'monsoon': monsoon_result is not None,
        })
    
    # 汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    print(f"总测试文件数: {len(results)}")
    print(f"ITCZ检测成功: {sum(1 for r in results if r['itcz'])}/{len(results)}")
    print(f"季风槽检测成功: {sum(1 for r in results if r['monsoon'])}/{len(results)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="ITCZ和季风槽提取测试")
    parser.add_argument("--nc_file", help="NetCDF文件路径")
    parser.add_argument("--track_file", help="追踪结果CSV文件路径（可选，自动查找）")
    parser.add_argument("--time_idx", type=int, default=0, help="时间索引(默认0)")
    parser.add_argument("--output", default="data/testExtract", help="输出目录")
    parser.add_argument("--all", action="store_true", help="测试所有预定义文件")
    
    args = parser.parse_args()
    
    if args.all:
        test_all_files(args.output)
    elif args.nc_file:
        test_single_file(
            args.nc_file,
            args.track_file,
            args.time_idx,
            args.output
        )
    else:
        parser.print_help()
        print("\n示例用法:")
        print("  # 测试单个文件（自动查找track文件）")
        print("  python test_itcz_monsoon_extraction.py --nc_file data/AURO_v100_IFS_2025061000_f000_f240_06.nc")
        print("\n  # 测试单个文件（指定track文件）")
        print("  python test_itcz_monsoon_extraction.py --nc_file data/AURO_v100_IFS_2025061000_f000_f240_06.nc --track_file data/test/tracks/track_xxx.csv")
        print("\n  # 测试所有文件")
        print("  python test_itcz_monsoon_extraction.py --all")


if __name__ == "__main__":
    main()
