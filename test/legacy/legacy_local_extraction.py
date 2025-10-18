#!/usr/bin/env python3
"""
本地数据测试脚本 - 验证气旋追踪和环境场提取功能

使用方法:
    python3 test_local_extraction.py --nc_file data/AURO_v100_IFS_2025061000_f000_f240_06.nc
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from initial_tracker import Tracker, track_file_with_initials
from environment_extractor import TCEnvironmentalSystemsExtractor
import xarray as xr
import pandas as pd


def test_cyclone_tracking(nc_file, initial_lat=None, initial_lon=None):
    """
    测试气旋路径追踪功能
    
    Args:
        nc_file: NetCDF文件路径
        initial_lat: 初始纬度（可选，如不提供则自动搜索）
        initial_lon: 初始经度（可选，如不提供则自动搜索）
    
    Returns:
        追踪结果DataFrame
    """
    print(f"\n{'='*60}")
    print(f"测试1: 气旋路径追踪")
    print(f"{'='*60}")
    print(f"数据文件: {nc_file}")
    
    try:
        # 打开数据集查看基本信息
        ds = xr.open_dataset(nc_file)
        print(f"\n数据集信息:")
        print(f"  - 时间维度: {len(ds.time)} 个时次")
        print(f"  - 纬度范围: {float(ds.lat.min().values):.2f}°N 到 {float(ds.lat.max().values):.2f}°N")
        print(f"  - 经度范围: {float(ds.lon.min().values):.2f}°E 到 {float(ds.lon.max().values):.2f}°E")
        print(f"  - 变量列表: {list(ds.data_vars.keys())[:10]}...")  # 显示前10个变量
        
        # 如果提供了初始点，使用它；否则让追踪器自动搜索
        if initial_lat is not None and initial_lon is not None:
            print(f"\n使用指定初始点: ({initial_lat}°N, {initial_lon}°E)")
            initial_points = pd.DataFrame({
                'lat': [initial_lat],
                'lon': [initial_lon],
                'name': ['TEST_TC'],
                'time': [ds.time[0].values]
            })
        else:
            print(f"\n⚠️  警告: 未提供初始点，请使用 --initial_lat 和 --initial_lon 参数")
            print(f"   示例: python3 {sys.argv[0]} --nc_file {nc_file} --initial_lat 15.0 --initial_lon 130.0")
            return None
        
        # 执行追踪
        print(f"\n开始追踪...")
        results = track_file_with_initials(nc_file, initial_points)
        
        print(f"\n追踪完成!")
        print(f"  - 追踪到 {len(results)} 个时次的位置")
        print(f"\n前5个时次的追踪结果:")
        print(results.head())
        
        return results
        
    except Exception as e:
        print(f"\n❌ 追踪失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_environment_extraction(nc_file, track_results, time_idx=0):
    """
    测试环境场系统提取功能
    
    Args:
        nc_file: NetCDF文件路径
        track_results: 追踪结果DataFrame
        time_idx: 要测试的时次索引
    
    Returns:
        提取的环境场数据字典
    """
    print(f"\n{'='*60}")
    print(f"测试2: 环境场系统提取")
    print(f"{'='*60}")
    
    if track_results is None or len(track_results) == 0:
        print("❌ 没有追踪结果，无法进行环境场提取")
        return None
    
    try:
        # 获取指定时次的气旋位置
        tc_lat = float(track_results.iloc[time_idx]['lat'])
        tc_lon = float(track_results.iloc[time_idx]['lon'])
        
        print(f"\n测试时次: {time_idx}")
        print(f"气旋中心: ({tc_lat:.2f}°N, {tc_lon:.2f}°E)")
        
        # 创建环境场提取器
        extractor = TCEnvironmentalSystemsExtractor(nc_file)
        
        # 提取各个环境场系统
        print(f"\n开始提取环境场系统...")
        
        all_systems = {}
        
        # 1. 副热带高压和引导气流
        print(f"\n  [1/9] 提取副热带高压和引导气流...")
        try:
            steering = extractor.extract_steering_system(time_idx, tc_lat, tc_lon)
            all_systems['steering_system'] = steering
            if steering:
                print(f"    ✓ 副高中心: ({steering['position']['center']['lat']:.1f}°N, "
                      f"{steering['position']['center']['lon']:.1f}°E)")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['steering_system'] = None
        
        # 2. 垂直风切变
        print(f"\n  [2/9] 提取垂直风切变...")
        try:
            shear = extractor.extract_vertical_wind_shear(time_idx, tc_lat, tc_lon)
            all_systems['wind_shear'] = shear
            if shear:
                print(f"    ✓ 切变强度: {shear['intensity']['magnitude']:.1f} m/s")
                print(f"    ✓ 切变方向: {shear['intensity']['direction']:.1f}°")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['wind_shear'] = None
        
        # 3. 海洋热含量（重点测试）
        print(f"\n  [3/9] 提取海洋热含量...")
        try:
            ocean = extractor.extract_ocean_heat_content(time_idx, tc_lat, tc_lon)
            all_systems['ocean_heat'] = ocean
            if ocean:
                print(f"    ✓ 平均SST: {ocean['intensity']['sst_mean']:.2f}°C")
                if 'boundary_coordinates' in ocean and ocean['boundary_coordinates']:
                    coords = ocean['boundary_coordinates']
                    lats = [c[1] for c in coords]
                    lons = [c[0] for c in coords]
                    print(f"    ✓ 边界坐标数量: {len(coords)}")
                    print(f"    ✓ 纬度范围: {min(lats):.2f}°N 到 {max(lats):.2f}°N (跨度 {max(lats)-min(lats):.2f}°)")
                    print(f"    ✓ 经度范围: {min(lons):.2f}°E 到 {max(lons):.2f}°E (跨度 {max(lons)-min(lons):.2f}°)")
                    
                    # 检查边界是否合理（应在气旋中心±5度内）
                    if abs(max(lats) - tc_lat) > 5 or abs(min(lats) - tc_lat) > 5:
                        print(f"    ⚠️  警告: 纬度边界超出预期范围（气旋中心±5°）")
                    if abs(max(lons) - tc_lon) > 5 or abs(min(lons) - tc_lon) > 5:
                        print(f"    ⚠️  警告: 经度边界超出预期范围（气旋中心±5°）")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['ocean_heat'] = None
        
        # 4. 高空辐散
        print(f"\n  [4/9] 提取高空辐散...")
        try:
            divergence = extractor.extract_upper_level_divergence(time_idx, tc_lat, tc_lon)
            all_systems['upper_divergence'] = divergence
            if divergence:
                print(f"    ✓ 辐散强度: {divergence['intensity']['divergence_value']:.2e} s⁻¹")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['upper_divergence'] = None
        
        # 5. 热带辐合带
        print(f"\n  [5/9] 提取热带辐合带...")
        try:
            itcz = extractor.extract_intertropical_convergence_zone(time_idx, tc_lat, tc_lon)
            all_systems['itcz'] = itcz
            if itcz:
                print(f"    ✓ ITCZ位置: {itcz['position']['center']['lat']:.1f}°N")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['itcz'] = None
        
        # 6. 西风槽
        print(f"\n  [6/9] 提取西风槽...")
        try:
            trough = extractor.extract_westerly_trough(time_idx, tc_lat, tc_lon)
            all_systems['westerly_trough'] = trough
            if trough:
                print(f"    ✓ 槽中心: ({trough['position']['center']['lat']:.1f}°N, "
                      f"{trough['position']['center']['lon']:.1f}°E)")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['westerly_trough'] = None
        
        # 7. 锋面系统
        print(f"\n  [7/9] 提取锋面系统...")
        try:
            frontal = extractor.extract_frontal_system(time_idx, tc_lat, tc_lon)
            all_systems['frontal_system'] = frontal
            if frontal:
                print(f"    ✓ 锋面类型: {frontal.get('type', 'unknown')}")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['frontal_system'] = None
        
        # 8. 季风槽
        print(f"\n  [8/9] 提取季风槽...")
        try:
            monsoon = extractor.extract_monsoon_trough(time_idx, tc_lat, tc_lon)
            all_systems['monsoon_trough'] = monsoon
            if monsoon:
                print(f"    ✓ 季风槽中心: ({monsoon['position']['center']['lat']:.1f}°N, "
                      f"{monsoon['position']['center']['lon']:.1f}°E)")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['monsoon_trough'] = None
        
        # 9. 对流层中层湿度
        print(f"\n  [9/9] 提取对流层中层湿度...")
        try:
            humidity = extractor.extract_mid_level_humidity(time_idx, tc_lat, tc_lon)
            all_systems['mid_level_humidity'] = humidity
            if humidity:
                mean_rh = humidity['intensity'].get('mean_rh', 'N/A')
                print(f"    ✓ 平均湿度: {mean_rh}%")
        except Exception as e:
            print(f"    ❌ 提取失败: {e}")
            all_systems['mid_level_humidity'] = None
        
        print(f"\n✓ 环境场提取完成!")
        
        return all_systems
        
    except Exception as e:
        print(f"\n❌ 环境场提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_results(track_results, env_systems):
    """
    验证结果的合理性
    
    Args:
        track_results: 追踪结果
        env_systems: 环境场系统数据
    """
    print(f"\n{'='*60}")
    print(f"测试3: 结果验证")
    print(f"{'='*60}")
    
    issues = []
    
    # 验证追踪结果
    if track_results is not None and len(track_results) > 0:
        print(f"\n✓ 气旋追踪验证:")
        print(f"  - 追踪时次数: {len(track_results)}")
        
        # 检查位置是否在合理范围内（西北太平洋）
        lat_range = (track_results['lat'].min(), track_results['lat'].max())
        lon_range = (track_results['lon'].min(), track_results['lon'].max())
        print(f"  - 纬度范围: {float(lat_range[0]):.1f}°N 到 {float(lat_range[1]):.1f}°N")
        print(f"  - 经度范围: {float(lon_range[0]):.1f}°E 到 {float(lon_range[1]):.1f}°E")
        
        if lat_range[0] < 0 or lat_range[1] > 60:
            issues.append("气旋纬度超出典型范围（0-60°N）")
        if lon_range[0] < 100 or lon_range[1] > 180:
            issues.append("气旋经度超出西北太平洋范围（100-180°E）")
    else:
        issues.append("未能成功追踪气旋")
    
    # 验证环境场系统
    if env_systems:
        print(f"\n✓ 环境场系统验证:")
        
        # 重点检查海洋热含量边界
        if 'ocean_heat' in env_systems and env_systems['ocean_heat']:
            ocean = env_systems['ocean_heat']
            if 'boundary_coordinates' in ocean and ocean['boundary_coordinates']:
                coords = ocean['boundary_coordinates']
                lats = [c[1] for c in coords]
                lons = [c[0] for c in coords]
                
                tc_lat = track_results.iloc[0]['lat'] if track_results is not None else None
                tc_lon = track_results.iloc[0]['lon'] if track_results is not None else None
                
                lat_span = max(lats) - min(lats)
                lon_span = max(lons) - min(lons)
                
                print(f"  - 海洋热含量边界跨度: {lat_span:.1f}° × {lon_span:.1f}°")
                
                # 边界应该在气旋中心周围合理范围内（通常<10°）
                if lat_span > 10 or lon_span > 10:
                    issues.append(f"海洋热含量边界过大（跨度 {lat_span:.1f}° × {lon_span:.1f}°），"
                                 "可能是全球等值线问题")
                
                if tc_lat and tc_lon:
                    if abs(max(lats) - tc_lat) > 10 or abs(min(lats) - tc_lat) > 10:
                        issues.append("海洋热含量边界纬度远离气旋中心")
                    if abs(max(lons) - tc_lon) > 10 or abs(min(lons) - tc_lon) > 10:
                        issues.append("海洋热含量边界经度远离气旋中心")
        
        # 检查其他系统
        systems_found = sum(1 for v in env_systems.values() if v is not None)
        print(f"  - 成功提取的系统数: {systems_found}/{len(env_systems)}")
        
        for system_name, system_data in env_systems.items():
            if system_data is None:
                print(f"    ⚠️  {system_name}: 未检测到")
    else:
        issues.append("环境场提取完全失败")
    
    # 汇总结果
    print(f"\n{'='*60}")
    print(f"验证结果汇总")
    print(f"{'='*60}")
    
    if len(issues) == 0:
        print(f"\n✅ 所有测试通过！未发现问题。")
    else:
        print(f"\n⚠️  发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    return len(issues) == 0


def save_results(track_results, env_systems, output_file):
    """
    保存测试结果到JSON文件
    
    Args:
        track_results: 追踪结果
        env_systems: 环境场系统数据
        output_file: 输出文件路径
    """
    try:
        output = {
            'test_time': datetime.now().isoformat(),
            'tracking_results': track_results.to_dict('records') if track_results is not None else None,
            'environmental_systems': env_systems
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✓ 测试结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='测试本地NetCDF数据的气旋追踪和环境场提取功能'
    )
    parser.add_argument(
        '--nc_file',
        default='data/AURO_v100_IFS_2025061000_f000_f240_06.nc',
        help='NetCDF文件路径'
    )
    parser.add_argument(
        '--initial_lat',
        type=float,
        default=None,
        help='初始纬度（可选）'
    )
    parser.add_argument(
        '--initial_lon',
        type=float,
        default=None,
        help='初始经度（可选）'
    )
    parser.add_argument(
        '--time_idx',
        type=int,
        default=0,
        help='测试的时次索引（默认为0，即初始时刻）'
    )
    parser.add_argument(
        '--output',
        default='test_results.json',
        help='结果输出文件路径'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"气旋追踪与环境场提取 - 本地测试")
    print(f"{'='*60}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试1: 气旋追踪
    track_results = test_cyclone_tracking(
        args.nc_file,
        args.initial_lat,
        args.initial_lon
    )
    
    # 测试2: 环境场提取
    env_systems = None
    if track_results is not None and len(track_results) > 0:
        env_systems = test_environment_extraction(
            args.nc_file,
            track_results,
            args.time_idx
        )
    
    # 测试3: 结果验证
    validation_passed = validate_results(track_results, env_systems)
    
    # 保存结果
    save_results(track_results, env_systems, args.output)
    
    print(f"\n{'='*60}")
    print(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 返回退出码
    sys.exit(0 if validation_passed else 1)


if __name__ == '__main__':
    main()
