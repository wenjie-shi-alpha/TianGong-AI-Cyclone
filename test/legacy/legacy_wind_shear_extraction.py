#!/usr/bin/env python3
"""
风切变提取测试脚本

测试修改后的风切变提取功能
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
    if isinstance(obj, np.integer):
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


def test_wind_shear(nc_file, track_file, output_file, time_idx=0):
    """
    测试风切变提取
    
    Args:
        nc_file: NetCDF文件路径
        track_file: 追踪结果CSV文件路径
        output_file: 输出JSON文件路径
        time_idx: 要测试的时次索引
    """
    print(f"\n{'='*70}")
    print(f"风切变提取测试")
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
        
        # 提取风切变
        print(f"\n提取风切变...")
        wind_shear = extractor.extract_vertical_wind_shear(time_idx, tc_lat, tc_lon)
        
        if wind_shear is None:
            print("❌ 风切变提取失败")
            return
        
        print(f"\n✓ 风切变提取成功!")
        print(f"\n{'='*70}")
        print(f"风切变信息:")
        print(f"{'='*70}")
        
        # 转换numpy类型为Python原生类型
        wind_shear_serializable = numpy_to_python(wind_shear)
        print(json.dumps(wind_shear_serializable, indent=2, ensure_ascii=False))
        
        # 验证关键字段
        print(f"\n{'='*70}")
        print(f"关键字段验证:")
        print(f"{'='*70}")
        
        # 检查计算方法
        calc_method = wind_shear.get('properties', {}).get('calculation_method', '')
        print(f"✓ 计算方法: {calc_method}")
        if '500km' in calc_method or '面积平均' in calc_method:
            print(f"  ✓ 使用了500km圆域面积平均")
        else:
            print(f"  ⚠️  未明确显示使用500km圆域")
        
        # 检查强度值
        intensity = wind_shear.get('intensity', {}).get('value', 0)
        print(f"\n✓ 风切变强度: {intensity} m/s")
        print(f"  等级: {wind_shear.get('intensity', {}).get('level', 'N/A')}")
        
        # 检查方向
        direction = wind_shear.get('properties', {}).get('direction_from_deg', 0)
        print(f"\n✓ 风切变方向: {direction}° (来向)")
        
        # 检查矢量分量
        shear_u = wind_shear.get('properties', {}).get('shear_vector_mps', {}).get('u', 0)
        shear_v = wind_shear.get('properties', {}).get('shear_vector_mps', {}).get('v', 0)
        print(f"\n✓ 矢量分量: u={shear_u} m/s, v={shear_v} m/s")
        
        # 检查描述
        description = wind_shear.get('description', '')
        print(f"\n✓ 描述: {description}")
        
        # 保存结果
        result = {
            'test_info': {
                'nc_file': str(nc_file),
                'track_file': str(track_file),
                'time_idx': time_idx,
                'tc_time': str(tc_time),
                'tc_position': {'lat': float(tc_lat), 'lon': float(tc_lon)},
            },
            'wind_shear': numpy_to_python(wind_shear),
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✓ 结果已保存到: {output_file}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='风切变提取测试')
    parser.add_argument('--nc_file', required=True, help='NetCDF文件路径')
    parser.add_argument('--track_file', required=True, help='追踪CSV文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--time_idx', type=int, default=0, help='时次索引（默认0）')
    
    args = parser.parse_args()
    
    test_wind_shear(args.nc_file, args.track_file, args.output, args.time_idx)
