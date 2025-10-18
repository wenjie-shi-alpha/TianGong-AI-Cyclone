#!/usr/bin/env python3
"""
使用真实历史观测初始点测试气旋追踪和环境场提取
结果保存到data/test目录
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import pandas as pd
import xarray as xr
from environment_extractor import TCEnvironmentalSystemsExtractor
from initial_tracker import track_file_with_initials


# NC文件与历史台风的映射关系
NC_TO_STORM_MAPPING = {
    'AURO_v100_IFS_2025061000_f000_f240_06.nc': {
        'storm_id': '2025162N15114',
        'storm_name': 'WUTIP',
    },
    'FOUR_v200_GFS_2020093012_f000_f240_06.nc': {
        'storm_id': '2020270N17159',
        'storm_name': 'KUJIRA',
    },
    'PANG_v100_IFS_2022032900_f000_f240_06.nc': {
        'storm_id': '2022088N09116',
        'storm_name': 'UNNAMED',
    }
}


def load_typhoon_data(csv_path='input/western_pacific_typhoons_superfast.csv'):
    """加载历史台风数据"""
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def find_initial_point(nc_path, typhoon_data, storm_info):
    """从历史数据中找到初始点"""
    # 获取NC文件时间范围
    ds = xr.open_dataset(nc_path)
    start_time = pd.Timestamp(ds.time.values[0])
    ds.close()
    
    # 筛选台风数据
    storm_data = typhoon_data[typhoon_data['storm_id'] == storm_info['storm_id']].copy()
    
    if len(storm_data) == 0:
        return None
    
    # 找最接近的观测
    time_diff = abs(storm_data['datetime'] - start_time)
    closest_idx = time_diff.idxmin()
    closest = storm_data.loc[closest_idx]
    
    # 构造DataFrame
    return pd.DataFrame({
        'storm_id': [storm_info['storm_id']],
        'datetime': [closest['datetime']],
        'dt': [closest['datetime']],
        'latitude': [float(closest['latitude'])],
        'longitude': [float(closest['longitude'])],
        'max_wind_usa': [float(closest['max_wind_usa']) if pd.notna(closest.get('max_wind_usa')) else None],
        'min_pressure_usa': [float(closest['min_pressure_usa']) if pd.notna(closest.get('min_pressure_usa')) else None],
    })


def track_cyclone(nc_path, initial_points, output_dir):
    """追踪气旋"""
    track_output_dir = Path(output_dir) / "tracks"
    track_output_dir.mkdir(parents=True, exist_ok=True)
    
    return track_file_with_initials(
        nc_path=Path(nc_path),
        all_points=initial_points,
        output_dir=track_output_dir,
        max_storms=None,
        time_window_hours=24
    )


def extract_environments(nc_path, track_csv, output_dir):
    """提取环境场系统"""
    track_df = pd.read_csv(track_csv)
    
    if len(track_df) == 0:
        return None
    
    # 创建临时文件
    temp_csv = Path(output_dir) / f"temp_{track_csv.name}"
    track_df.to_csv(temp_csv, index=False)
    
    # 创建提取器
    extractor = TCEnvironmentalSystemsExtractor(
        forecast_data_path=str(nc_path),
        tc_tracks_path=str(temp_csv)
    )
    
    # 提取所有点
    all_results = []
    
    systems_to_extract = [
        ('steering_system', 'extract_steering_system'),
        ('wind_shear', 'extract_vertical_wind_shear'),
        ('ocean_heat', 'extract_ocean_heat_content'),
        ('upper_divergence', 'extract_upper_level_divergence'),
        ('itcz', 'extract_intertropical_convergence_zone'),
        ('westerly_trough', 'extract_westerly_trough'),
        ('frontal_system', 'extract_frontal_system'),
        ('monsoon_trough', 'extract_monsoon_trough'),
    ]
    
    for idx, row in track_df.iterrows():
        point_systems = {}
        
        for key, method_name in systems_to_extract:
            try:
                method = getattr(extractor, method_name)
                result = method(idx, float(row['lat']), float(row['lon']))
                point_systems[key] = result
            except Exception:
                point_systems[key] = None
        
        all_results.append({
            'time': pd.to_datetime(row['time']).isoformat(),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'environmental_systems': point_systems
        })
    
    extractor.close()
    
    if temp_csv.exists():
        temp_csv.unlink()
    
    return {
        'nc_file': Path(nc_path).name,
        'track_file': track_csv.name,
        'storm_id': NC_TO_STORM_MAPPING.get(Path(nc_path).name, {}).get('storm_id', 'unknown'),
        'storm_name': NC_TO_STORM_MAPPING.get(Path(nc_path).name, {}).get('storm_name', 'unknown'),
        'total_points': len(all_results),
        'points': all_results
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print("基于真实观测初始点的气旋追踪和环境场提取")
    print("="*70 + "\n")
    
    # 设置路径
    data_dir = Path("data")
    output_dir = data_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载历史数据
    print("正在加载历史台风数据...")
    typhoon_data = load_typhoon_data()
    print(f"  ✓ 加载了 {len(typhoon_data)} 条记录\n")
    
    # 处理所有NC文件
    nc_files = [
        data_dir / "AURO_v100_IFS_2025061000_f000_f240_06.nc",
        data_dir / "FOUR_v200_GFS_2020093012_f000_f240_06.nc",
        data_dir / "PANG_v100_IFS_2022032900_f000_f240_06.nc",
    ]
    
    results = []
    
    for i, nc_file in enumerate(nc_files, 1):
        if not nc_file.exists():
            continue
            
        storm_info = NC_TO_STORM_MAPPING.get(nc_file.name)
        if not storm_info:
            continue
        
        print(f"\n[{i}/3] 处理: {nc_file.name}")
        print(f"  台风: {storm_info['storm_name']} ({storm_info['storm_id']})")
        
        # 找初始点
        print("  查找初始点...")
        initial_points = find_initial_point(nc_file, typhoon_data, storm_info)
        
        if initial_points is None:
            print("  ✗ 未找到初始点")
            continue
        
        print(f"  ✓ 初始点: ({initial_points.iloc[0]['latitude']:.1f}°N, "
              f"{initial_points.iloc[0]['longitude']:.1f}°E)")
        
        # 追踪
        print("  追踪气旋...")
        track_files = track_cyclone(nc_file, initial_points, output_dir)
        
        if not track_files:
            print("  ✗ 追踪失败")
            continue
        
        print(f"  ✓ 生成轨迹: {track_files[0].name}")
        
        # 提取环境场
        print("  提取环境场...")
        env_result = extract_environments(nc_file, track_files[0], output_dir)
        
        if env_result:
            # 保存环境场结果
            env_file = output_dir / f"env_systems_{track_files[0].stem}.json"
            with open(env_file, 'w', encoding='utf-8') as f:
                json.dump(env_result, f, ensure_ascii=False, indent=2, default=str)
            print(f"  ✓ 保存环境场: {env_file.name}")
            
            results.append({
                'nc_file': nc_file.name,
                'storm_id': storm_info['storm_id'],
                'storm_name': storm_info['storm_name'],
                'track_file': track_files[0].name,
                'env_file': env_file.name,
                'points_count': len(env_result['points'])
            })
    
    # 保存摘要
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'real_historical_initial_points',
            'total_processed': len(results),
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    # 打印总结
    print("\n" + "="*70)
    print("处理完成")
    print("="*70)
    print(f"\n处理文件数: {len(results)}")
    print(f"\n输出目录: {output_dir}/")
    print("  ├── tracks/           # 气旋追踪CSV文件")
    print("  ├── env_systems_*.json  # 环境场提取JSON文件")
    print("  └── processing_summary.json")
    
    for r in results:
        print(f"\n  {r['nc_file']}:")
        print(f"    台风: {r['storm_name']} ({r['storm_id']})")
        print(f"    轨迹: {r['track_file']}")
        print(f"    环境场: {r['env_file']}")
        print(f"    路径点数: {r['points_count']}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
