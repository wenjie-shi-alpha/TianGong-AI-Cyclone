#!/usr/bin/env python3
"""
高空辐散提取测试脚本

测试修改后的高空辐散提取功能，使用球面散度公式和圆域平均
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from environment_extractor import TCEnvironmentalSystemsExtractor


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


def test_divergence(nc_file, track_file, output_file, time_idx=0):
    """
    测试高空辐散提取
    
    Args:
        nc_file: NetCDF文件路径
        track_file: 追踪结果CSV文件路径
        output_file: 输出JSON文件路径
        time_idx: 要测试的时次索引
    """
    print(f"\n{'='*70}")
    print(f"高空辐散提取测试")
    print(f"{'='*70}")
    print(f"数据文件: {nc_file}")
    print(f"追踪文件: {track_file}")
    print(f"时次索引: {time_idx}")
    
    try:
        # 读取追踪数据
        track_df = pd.read_csv(track_file)
        print(f"\n追踪数据: {len(track_df)} 个时次")
        print(track_df.head())
        
        if time_idx >= len(track_df):
            print(f"\n❌ 时次索引 {time_idx} 超出范围（最大: {len(track_df)-1}）")
            return
        
        # 获取指定时次的台风位置
        tc_lat = track_df.iloc[time_idx]['lat']
        tc_lon = track_df.iloc[time_idx]['lon']
        tc_time = track_df.iloc[time_idx]['time']
        
        print(f"\n{'='*70}")
        print(f"时次 {time_idx}: {tc_time}")
        print(f"台风位置: {tc_lat}°N, {tc_lon}°E")
        print(f"{'='*70}")
        
        # 创建提取器
        extractor = TCEnvironmentalSystemsExtractor(nc_file, track_file)
        
        # 提取高空辐散
        print(f"\n提取高空辐散...")
        divergence = extractor.extract_upper_level_divergence(time_idx, tc_lat, tc_lon)
        
        if divergence is None:
            print("❌ 高空辐散提取失败")
            return
        
        print(f"\n✓ 高空辐散提取成功!")
        print(f"\n{'='*70}")
        print(f"高空辐散信息:")
        print(f"{'='*70}")
        
        # 转换numpy类型为Python原生类型
        divergence_serializable = numpy_to_python(divergence)
        print(json.dumps(divergence_serializable, indent=2, ensure_ascii=False))
        
        # 验证关键字段
        print(f"\n{'='*70}")
        print(f"关键字段验证:")
        print(f"{'='*70}")
        
        # 检查位置信息
        position = divergence.get('position', {})
        radius_km = position.get('radius_km', 0)
        print(f"✓ 计算范围: 台风中心周围 {radius_km} 公里圆域")
        
        # 检查强度信息
        intensity = divergence.get('intensity', {})
        avg_value = intensity.get('average_value', 0)
        max_value = intensity.get('max_value', 0)
        level = intensity.get('level', '')
        print(f"✓ 平均散度: {avg_value:.2f} ×10⁻⁵ s⁻¹")
        print(f"✓ 最大散度: {max_value:.2f} ×10⁻⁵ s⁻¹")
        print(f"✓ 强度等级: {level}")
        
        # 检查辐散中心位置
        div_center = divergence.get('divergence_center', {})
        if div_center:
            center_lat = div_center.get('lat', 0)
            center_lon = div_center.get('lon', 0)
            distance = div_center.get('distance_to_tc_km', 0)
            direction = div_center.get('direction', '')
            print(f"✓ 最大辐散中心: ({center_lat}°N, {center_lon}°E)")
            print(f"✓ 距台风中心: {distance:.1f} 公里 ({direction}方向)")
        
        # 检查影响评估
        properties = divergence.get('properties', {})
        impact = properties.get('impact', '')
        favorable = properties.get('favorable_development', False)
        center_offset = properties.get('center_offset', False)
        print(f"✓ 影响评估: {impact}")
        print(f"✓ 有利发展: {'是' if favorable else '否'}")
        if center_offset:
            print(f"⚠️ 辐散中心偏移: 辐散中心与台风中心存在明显偏移")
        
        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            "test_info": {
                "nc_file": str(nc_file),
                "track_file": str(track_file),
                "time_idx": time_idx,
                "time": tc_time,
                "tc_position": {"lat": float(tc_lat), "lon": float(tc_lon)}
            },
            "divergence": divergence_serializable
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存到: {output_path}")
        
        # 验证结果的合理性
        print(f"\n{'='*70}")
        print(f"合理性检验:")
        print(f"{'='*70}")
        
        # 1. 检查散度数值范围
        if abs(avg_value) > 50:
            print(f"⚠️ 警告: 平均散度值 ({avg_value:.2f}) 超出正常范围 (±50)")
        else:
            print(f"✓ 平均散度值在合理范围内")
        
        # 2. 检查最大值与平均值的关系
        if max_value < avg_value:
            print(f"⚠️ 警告: 最大散度值小于平均散度值（可能存在问题）")
        else:
            print(f"✓ 最大散度值大于等于平均散度值")
        
        # 3. 检查辐散中心距离
        if distance > radius_km:
            print(f"⚠️ 警告: 辐散中心距离 ({distance:.1f} km) 超出计算半径 ({radius_km} km)")
        else:
            print(f"✓ 辐散中心在计算范围内")
        
        # 4. 检查物理意义
        if avg_value > 0:
            print(f"✓ 高空辐散为正值，有利于台风发展")
        else:
            print(f"⚠️ 高空辐散为负值（辐合），不利于台风发展")
        
        print(f"\n{'='*70}")
        print(f"测试完成")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_files():
    """测试多个文件"""
    
    # 定义测试用例
    test_cases = [
        {
            "nc_file": "data/AURO_v100_IFS_2025061000_f000_f240_06.nc",
            "track_file": "data/test/tracks/track_2025162N15114_AURO_v100_IFS_2025061000_f000_f240_06.csv",
            "output_file": "data/testExtract/divergence/AURO_divergence.json",
            "time_idx": 0
        },
        {
            "nc_file": "data/FOUR_v200_GFS_2020093012_f000_f240_06.nc",
            "track_file": "data/test/tracks/track_2020270N17159_FOUR_v200_GFS_2020093012_f000_f240_06.csv",
            "output_file": "data/testExtract/divergence/FOUR_divergence.json",
            "time_idx": 0
        },
        {
            "nc_file": "data/PANG_v100_IFS_2022032900_f000_f240_06.nc",
            "track_file": "data/test/tracks/track_2022088N09116_PANG_v100_IFS_2022032900_f000_f240_06.csv",
            "output_file": "data/testExtract/divergence/PANG_divergence.json",
            "time_idx": 0
        }
    ]
    
    # 运行所有测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n")
        print(f"{'#'*70}")
        print(f"# 测试用例 {i}/{len(test_cases)}")
        print(f"{'#'*70}")
        
        nc_file = Path(test_case["nc_file"])
        track_file = Path(test_case["track_file"])
        
        # 检查文件是否存在
        if not nc_file.exists():
            print(f"⚠️ 跳过: 数据文件不存在 - {nc_file}")
            continue
        
        if not track_file.exists():
            print(f"⚠️ 跳过: 追踪文件不存在 - {track_file}")
            continue
        
        test_divergence(
            nc_file=nc_file,
            track_file=track_file,
            output_file=test_case["output_file"],
            time_idx=test_case["time_idx"]
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试高空辐散提取功能')
    parser.add_argument('--nc-file', type=str, help='NetCDF文件路径')
    parser.add_argument('--track-file', type=str, help='追踪结果CSV文件路径')
    parser.add_argument('--output', type=str, help='输出JSON文件路径')
    parser.add_argument('--time-idx', type=int, default=0, help='时次索引')
    parser.add_argument('--all', action='store_true', help='测试所有文件')
    
    args = parser.parse_args()
    
    if args.all:
        test_multiple_files()
    elif args.nc_file and args.track_file and args.output:
        test_divergence(
            nc_file=Path(args.nc_file),
            track_file=Path(args.track_file),
            output_file=args.output,
            time_idx=args.time_idx
        )
    else:
        print("使用方式:")
        print("  测试单个文件: python test_divergence_extraction.py --nc-file <nc文件> --track-file <track文件> --output <输出文件> [--time-idx <时次>]")
        print("  测试所有文件: python test_divergence_extraction.py --all")
        print("\n运行所有测试...")
        test_multiple_files()
