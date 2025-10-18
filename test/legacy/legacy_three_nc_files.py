#!/usr/bin/env python3
"""
测试三个NC文件的气旋追踪和环境场提取
读取data目录下的三个NC文件，运行追踪生成CSV，然后按点提取天气系统
结果保存到data/test目录
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import pandas as pd
import xarray as xr
from environment_extractor import TCEnvironmentalSystemsExtractor
from initial_tracker import track_file_with_initials


def find_initial_points_from_nc(nc_path):
    """
    从NC文件中自动寻找可能的气旋初始点
    通过分析第一个时次的海平面气压场找到低压中心
    
    Returns:
        DataFrame with initial points
    """
    print(f"\n  正在从数据中搜索可能的气旋初始点...")
    
    try:
        ds = xr.open_dataset(nc_path)
        
        # 获取第一个时次的数据
        first_time = ds.time.values[0]
        
        # 读取海平面气压
        if 'msl' in ds.variables:
            msl = ds['msl'].isel(time=0).values
        elif 'sp' in ds.variables:
            msl = ds['sp'].isel(time=0).values
        else:
            print(f"    ⚠️  未找到气压变量，使用默认初始点")
            ds.close()
            return create_default_initial_points(nc_path, first_time)
        
        lat = ds.latitude.values if 'latitude' in ds.coords else ds.lat.values
        lon = ds.longitude.values if 'longitude' in ds.coords else ds.lon.values
        
        # 只在热带区域寻找（5-35°N）
        tropical_mask = (lat >= 5) & (lat <= 35)
        lat_tropical = lat[tropical_mask]
        
        if len(lat_tropical) == 0:
            print(f"    ⚠️  数据不覆盖热带区域，使用默认初始点")
            ds.close()
            return create_default_initial_points(nc_path, first_time)
        
        # 在热带区域找最低气压点
        msl_tropical = msl[tropical_mask, :]
        min_idx = msl_tropical.argmin()
        lat_idx, lon_idx = divmod(min_idx, msl_tropical.shape[1])
        
        init_lat = float(lat_tropical[lat_idx])
        init_lon = float(lon[lon_idx])
        min_pressure = float(msl_tropical[lat_idx, lon_idx]) / 100.0  # 转换为hPa
        
        ds.close()
        
        # 只有当气压低于1010 hPa时才认为可能是气旋
        if min_pressure < 1010:
            print(f"    ✓ 找到可能的气旋中心: ({init_lat:.2f}°N, {init_lon:.2f}°E)")
            print(f"      气压: {min_pressure:.1f} hPa")
            
            return pd.DataFrame({
                'storm_id': [f'AUTO_{Path(nc_path).stem}'],
                'datetime': [pd.Timestamp(first_time)],
                'latitude': [init_lat],
                'longitude': [init_lon],
                'max_wind_usa': [None],
                'min_pressure_usa': [min_pressure],
            })
        else:
            print(f"    ⚠️  未找到明显的低压中心（最低气压 {min_pressure:.1f} hPa），使用默认初始点")
            return create_default_initial_points(nc_path, first_time)
            
    except Exception as e:
        print(f"    ⚠️  自动搜索失败: {e}，使用默认初始点")
        return create_default_initial_points(nc_path, first_time)


def create_default_initial_points(nc_path, first_time):
    """
    创建默认的初始点（基于文件名猜测）
    """
    # 根据文件名选择合适的初始点
    filename = Path(nc_path).name.upper()
    
    if 'AURO' in filename:
        # AURO 案例 - 可能在北太平洋
        init_lat, init_lon = 15.0, 130.0
    elif 'FOUR' in filename or 'PANG' in filename:
        # FOUR/PANG 案例
        init_lat, init_lon = 18.0, 135.0
    else:
        # 默认西北太平洋位置
        init_lat, init_lon = 15.0, 135.0
    
    print(f"    使用默认初始点: ({init_lat:.2f}°N, {init_lon:.2f}°E)")
    
    return pd.DataFrame({
        'storm_id': [f'DEFAULT_{Path(nc_path).stem}'],
        'datetime': [pd.Timestamp(first_time)],
        'latitude': [init_lat],
        'longitude': [init_lon],
        'max_wind_usa': [None],
        'min_pressure_usa': [None],
    })


def track_cyclone(nc_path, output_dir):
    """
    对单个NC文件进行气旋追踪
    
    Args:
        nc_path: NC文件路径
        output_dir: 输出目录
    
    Returns:
        追踪结果CSV文件路径列表
    """
    print(f"\n{'='*60}")
    print(f"步骤1: 气旋路径追踪")
    print(f"{'='*60}")
    print(f"NC文件: {nc_path}")
    
    try:
        # 自动寻找初始点
        initial_points = find_initial_points_from_nc(nc_path)
        
        if initial_points is None or len(initial_points) == 0:
            print(f"  ❌ 无法确定初始点")
            return []
        
        # 确保DataFrame有所需的'dt'列
        if 'dt' not in initial_points.columns:
            initial_points['dt'] = pd.to_datetime(initial_points['datetime'])
        
        # 执行追踪
        print(f"\n  开始追踪...")
        track_output_dir = Path(output_dir) / "tracks"
        track_output_dir.mkdir(parents=True, exist_ok=True)
        
        written_files = track_file_with_initials(
            nc_path=Path(nc_path),
            all_points=initial_points,
            output_dir=track_output_dir,
            max_storms=None,
            time_window_hours=24  # 放宽时间窗口
        )
        
        if written_files:
            print(f"\n  ✓ 追踪完成，生成 {len(written_files)} 个轨迹文件:")
            for f in written_files:
                print(f"    - {f.name}")
        else:
            print(f"\n  ⚠️  未生成轨迹文件")
        
        return written_files
        
    except Exception as e:
        print(f"\n  ❌ 追踪失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_environmental_systems(nc_path, track_csv, output_dir):
    """
    基于追踪结果提取环境场系统
    
    Args:
        nc_path: NC文件路径
        track_csv: 追踪结果CSV文件
        output_dir: 输出目录
    
    Returns:
        提取结果字典
    """
    print(f"\n{'='*60}")
    print(f"步骤2: 环境场系统提取")
    print(f"{'='*60}")
    print(f"轨迹文件: {track_csv.name}")
    
    try:
        # 读取追踪结果
        track_df = pd.read_csv(track_csv)
        print(f"\n  轨迹点数: {len(track_df)}")
        
        if len(track_df) == 0:
            print(f"  ⚠️  轨迹文件为空")
            return None
        
        # 创建临时CSV文件供TCEnvironmentalSystemsExtractor使用
        temp_track_csv = Path(output_dir) / f"temp_{track_csv.name}"
        track_df.to_csv(temp_track_csv, index=False)
        
        # 创建提取器
        print(f"\n  初始化环境场提取器...")
        extractor = TCEnvironmentalSystemsExtractor(
            forecast_data_path=str(nc_path),
            tc_tracks_path=str(temp_track_csv)
        )
        
        # 按时间点提取所有系统
        all_results = []
        
        print(f"\n  开始逐点提取环境场系统...")
        for idx, row in track_df.iterrows():
            tc_lat = float(row['lat'])
            tc_lon = float(row['lon'])
            tc_time = pd.to_datetime(row['time'])
            
            print(f"\n  [{idx+1}/{len(track_df)}] 时间: {tc_time}, 位置: ({tc_lat:.2f}°N, {tc_lon:.2f}°E)")
            
            point_systems = {}
            
            # 提取各个系统
            systems_to_extract = [
                ('steering_system', 'extract_steering_system', '副热带高压'),
                ('wind_shear', 'extract_vertical_wind_shear', '垂直风切变'),
                ('ocean_heat', 'extract_ocean_heat_content', '海洋热含量'),
                ('upper_divergence', 'extract_upper_level_divergence', '高空辐散'),
                ('itcz', 'extract_intertropical_convergence_zone', '热带辐合带'),
                ('westerly_trough', 'extract_westerly_trough', '西风槽'),
                ('frontal_system', 'extract_frontal_system', '锋面系统'),
                ('monsoon_trough', 'extract_monsoon_trough', '季风槽'),
                ('mid_level_humidity', 'extract_mid_level_humidity', '中层湿度'),
            ]
            
            for key, method_name, cn_name in systems_to_extract:
                try:
                    method = getattr(extractor, method_name)
                    result = method(idx, tc_lat, tc_lon)
                    point_systems[key] = result
                    if result:
                        print(f"    ✓ {cn_name}")
                    else:
                        print(f"    - {cn_name} (未检测到)")
                except Exception as e:
                    print(f"    ✗ {cn_name} 提取失败: {e}")
                    point_systems[key] = None
            
            all_results.append({
                'time': tc_time.isoformat(),
                'lat': tc_lat,
                'lon': tc_lon,
                'environmental_systems': point_systems
            })
        
        # 关闭提取器
        extractor.close()
        
        # 删除临时文件
        if temp_track_csv.exists():
            temp_track_csv.unlink()
        
        print(f"\n  ✓ 完成所有 {len(all_results)} 个点的环境场提取")
        
        return {
            'nc_file': Path(nc_path).name,
            'track_file': track_csv.name,
            'total_points': len(all_results),
            'extraction_time': datetime.now().isoformat(),
            'points': all_results
        }
        
    except Exception as e:
        print(f"\n  ❌ 环境场提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_single_nc_file(nc_path, output_dir):
    """
    处理单个NC文件：追踪 + 环境场提取
    
    Args:
        nc_path: NC文件路径
        output_dir: 输出目录
    
    Returns:
        处理结果摘要
    """
    nc_path = Path(nc_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# 处理文件: {nc_path.name}")
    print(f"{'#'*60}")
    
    results_summary = {
        'nc_file': nc_path.name,
        'start_time': datetime.now().isoformat(),
        'status': 'processing',
        'tracks': [],
        'environmental_systems': []
    }
    
    # 步骤1: 气旋追踪
    track_files = track_cyclone(nc_path, output_dir)
    results_summary['tracks'] = [f.name for f in track_files]
    
    if not track_files:
        print(f"\n⚠️  未能生成轨迹，跳过环境场提取")
        results_summary['status'] = 'no_tracks'
        results_summary['end_time'] = datetime.now().isoformat()
        return results_summary
    
    # 步骤2: 对每个轨迹文件进行环境场提取
    env_results = []
    for track_csv in track_files:
        env_result = extract_environmental_systems(nc_path, track_csv, output_dir)
        if env_result:
            # 保存环境场提取结果
            env_output_file = output_dir / f"env_systems_{track_csv.stem}.json"
            with open(env_output_file, 'w', encoding='utf-8') as f:
                json.dump(env_result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n  ✓ 环境场结果已保存: {env_output_file.name}")
            env_results.append(env_output_file.name)
    
    results_summary['environmental_systems'] = env_results
    results_summary['status'] = 'completed' if env_results else 'completed_no_env'
    results_summary['end_time'] = datetime.now().isoformat()
    
    return results_summary


def main():
    """
    主函数：处理所有三个NC文件
    """
    print(f"\n{'='*70}")
    print(f"测试三个NC文件的气旋追踪和环境场提取")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 定义输入输出路径
    data_dir = Path("data")
    output_dir = data_dir / "test"
    
    # 三个NC文件
    nc_files = [
        data_dir / "AURO_v100_IFS_2025061000_f000_f240_06.nc",
        data_dir / "FOUR_v200_GFS_2020093012_f000_f240_06.nc",
        data_dir / "PANG_v100_IFS_2022032900_f000_f240_06.nc",
    ]
    
    # 检查文件是否存在
    print(f"检查NC文件:")
    existing_files = []
    for nc_file in nc_files:
        if nc_file.exists():
            print(f"  ✓ {nc_file.name}")
            existing_files.append(nc_file)
        else:
            print(f"  ✗ {nc_file.name} (不存在)")
    
    if not existing_files:
        print(f"\n❌ 没有找到任何NC文件，退出")
        return
    
    print(f"\n将处理 {len(existing_files)} 个文件")
    print(f"输出目录: {output_dir}\n")
    
    # 处理每个文件
    all_results = []
    for i, nc_file in enumerate(existing_files, 1):
        print(f"\n{'='*70}")
        print(f"进度: [{i}/{len(existing_files)}]")
        print(f"{'='*70}")
        
        result = process_single_nc_file(nc_file, output_dir)
        all_results.append(result)
    
    # 保存总体摘要
    summary_file = output_dir / "processing_summary.json"
    summary = {
        'test_time': datetime.now().isoformat(),
        'total_files': len(existing_files),
        'processed_files': len(all_results),
        'results': all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    
    # 打印最终摘要
    print(f"\n{'='*70}")
    print(f"处理完成摘要")
    print(f"{'='*70}")
    print(f"\n总体统计:")
    print(f"  - 处理文件数: {len(all_results)}")
    
    total_tracks = sum(len(r.get('tracks', [])) for r in all_results)
    total_env = sum(len(r.get('environmental_systems', [])) for r in all_results)
    
    print(f"  - 生成轨迹文件: {total_tracks}")
    print(f"  - 生成环境场文件: {total_env}")
    
    print(f"\n详细结果:")
    for result in all_results:
        print(f"\n  {result['nc_file']}:")
        print(f"    状态: {result['status']}")
        print(f"    轨迹文件: {len(result.get('tracks', []))}")
        print(f"    环境场文件: {len(result.get('environmental_systems', []))}")
    
    print(f"\n输出目录结构:")
    print(f"  {output_dir}/")
    print(f"    ├── tracks/          # 气旋追踪CSV文件")
    print(f"    ├── env_systems_*.json  # 环境场提取结果")
    print(f"    └── processing_summary.json  # 处理摘要")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
