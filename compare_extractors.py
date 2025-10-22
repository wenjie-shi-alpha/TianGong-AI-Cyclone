#!/usr/bin/env python3
"""
对比脚本：比较 environment_extractor 和 cds.py 中的提取算法输出
使用数据文件: data/AURO_v100_IFS_2025061000_f000_f240_06.nc

注意：此脚本直接使用已下载的NC文件，绕过CDS下载功能
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入environment_extractor
from environment_extractor import TCEnvironmentalSystemsExtractor


def load_nc_file(nc_path):
    """加载NC文件并返回基本信息"""
    print(f"\n{'='*80}")
    print(f"📂 加载数据文件: {nc_path}")
    print(f"{'='*80}")
    
    ds = xr.open_dataset(nc_path)
    
    print(f"\n数据集维度: {dict(ds.dims)}")
    print(f"数据变量: {list(ds.data_vars.keys())}")
    print(f"坐标: {list(ds.coords.keys())}")
    
    # 获取时间范围
    if 'time' in ds.coords:
        times = ds.time.values
        print(f"时间范围: {times[0]} 至 {times[-1]}")
        print(f"时间步数: {len(times)}")
    
    # 获取空间范围
    if 'latitude' in ds.coords:
        lats = ds.latitude.values
        lons = ds.longitude.values
        print(f"纬度范围: {lats.min():.2f}° 至 {lats.max():.2f}°")
        print(f"经度范围: {lons.min():.2f}° 至 {lons.max():.2f}°")
    
    return ds


def extract_with_environment_extractor(nc_path):
    """使用 environment_extractor (extractSyst.py) 进行提取"""
    print(f"\n{'='*80}")
    print("🔬 方法1: 使用 environment_extractor (extractSyst.py)")
    print(f"{'='*80}")
    
    # 创建临时的tracks文件
    import pandas as pd
    import tempfile
    
    # 为测试创建一个临时的台风路径文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_tracks = f.name
        # 创建测试用的台风路径数据
        df = pd.DataFrame({
            'time': ['2025-06-10 00:00:00'],
            'lat': [20.0],
            'lon': [130.0],
            'particle': [1],
            'time_idx': [0]
        })
        df.to_csv(temp_tracks, index=False)
    
    try:
        extractor = TCEnvironmentalSystemsExtractor(
            forecast_data_path=str(nc_path),
            tc_tracks_path=temp_tracks,
            enable_detailed_shape_analysis=True
        )
        
        # 使用第一个时间点和测试位置
        time_idx = 0
        test_lat = 20.0  # 测试纬度
        test_lon = 130.0  # 测试经度
        
        ds = xr.open_dataset(nc_path)
        if 'time' in ds.coords:
            test_time = pd.Timestamp(ds.time.values[time_idx])
        else:
            test_time = pd.Timestamp('2025-06-10 00:00:00')
        
        print(f"\n测试参数:")
        print(f"  时间: {test_time}")
        print(f"  时间索引: {time_idx}")
        print(f"  台风位置: ({test_lat}°N, {test_lon}°E)")
        
        # 调用各个提取方法
        systems = []
        extraction_methods = [
            ("SubtropicalHigh", lambda: extractor.extract_steering_system(time_idx, test_lat, test_lon)),
            ("VerticalWindShear", lambda: extractor.extract_vertical_wind_shear(time_idx, test_lat, test_lon)),
            ("OceanHeatContent", lambda: extractor.extract_ocean_heat_content(time_idx, test_lat, test_lon)),
            ("UpperLevelDivergence", lambda: extractor.extract_upper_level_divergence(time_idx, test_lat, test_lon)),
            ("ITCZ", lambda: extractor.extract_intertropical_convergence_zone(time_idx, test_lat, test_lon)),
            ("WesterlyTrough", lambda: extractor.extract_westerly_trough(time_idx, test_lat, test_lon)),
            ("FrontalSystem", lambda: extractor.extract_frontal_system(time_idx, test_lat, test_lon)),
            ("MonsoonTrough", lambda: extractor.extract_monsoon_trough(time_idx, test_lat, test_lon)),
        ]
        
        print(f"\n正在提取各个环境系统...")
        for system_name, extraction_func in extraction_methods:
            try:
                print(f"  - 提取 {system_name}...", end=" ")
                system = extraction_func()
                if system:
                    systems.append(system)
                    print("✓")
                else:
                    print("(无)")
            except Exception as e:
                print(f"✗ ({str(e)[:50]})")
        
        print(f"\n✅ 提取完成!")
        print(f"成功提取 {len(systems)} 个环境系统:")
        for i, system in enumerate(systems, 1):
            system_name = system.get('system_name', 'Unknown')
            print(f"  {i}. {system_name}")
        
        # 清理临时文件
        extractor.close()
        Path(temp_tracks).unlink(missing_ok=True)
        
        return systems
    except Exception as e:
        print(f"\n❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        if Path(temp_tracks).exists():
            Path(temp_tracks).unlink(missing_ok=True)
        return None


def extract_with_cds_extractor(nc_path):
    """使用 cds.py 中的 CDSEnvironmentExtractor 提取算法"""
    print(f"\n{'='*80}")
    print("🔬 方法2: 使用 cds.py (CDSEnvironmentExtractor)")
    print(f"{'='*80}")
    
    # 创建临时的tracks文件
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_tracks = f.name
        df = pd.DataFrame({
            'datetime': ['2025-06-10 00:00:00'],
            'latitude': [20.0],
            'longitude': [130.0],
            'storm_id': ['TEST001']
        })
        df.to_csv(temp_tracks, index=False)
    
    try:
        # 绕过cdsapi依赖：临时替换cdsapi模块
        import types
        mock_cdsapi = types.ModuleType('cdsapi')
        
        class MockClient:
            def __init__(self, *args, **kwargs):
                pass
            def retrieve(self, *args, **kwargs):
                pass
        
        mock_cdsapi.Client = MockClient
        sys.modules['cdsapi'] = mock_cdsapi
        
        # 现在可以安全导入CDSEnvironmentExtractor
        from cds import CDSEnvironmentExtractor
        
        print("\n✅ 成功绕过 cdsapi 依赖")
        
        # 创建提取器实例（会跳过CDS API检查）
        extractor = CDSEnvironmentExtractor(
            tracks_file=temp_tracks,
            output_dir="./temp_cds_output",
            cleanup_intermediate=False
        )
        
        # 直接加载已下载的NC文件
        print(f"\n📂 加载NC文件: {nc_path}")
        success = extractor.load_era5_data(
            single_file=str(nc_path),
            pressure_file=None
        )
        
        if not success:
            print("❌ 加载数据失败")
            Path(temp_tracks).unlink(missing_ok=True)
            return None
        
        # 使用第一个时间点和测试位置
        time_idx = 0
        test_lat = 20.0
        test_lon = 130.0
        
        ds = xr.open_dataset(nc_path)
        if 'time' in ds.coords:
            test_time = pd.Timestamp(ds.time.values[time_idx])
        else:
            test_time = pd.Timestamp('2025-06-10 00:00:00')
        
        print(f"\n测试参数:")
        print(f"  时间: {test_time}")
        print(f"  时间索引: {time_idx}")
        print(f"  台风位置: ({test_lat}°N, {test_lon}°E)")
        
        # 调用提取方法
        print(f"\n正在使用 CDSEnvironmentExtractor 提取环境系统...")
        systems = extractor.extract_environmental_systems(
            time_point=test_time,
            tc_lat=test_lat,
            tc_lon=test_lon
        )
        
        print(f"\n✅ 提取完成!")
        print(f"成功提取 {len(systems)} 个环境系统:")
        for i, system in enumerate(systems, 1):
            system_name = system.get('system_name', 'Unknown')
            print(f"  {i}. {system_name}")
        
        # 清理
        Path(temp_tracks).unlink(missing_ok=True)
        
        # 恢复sys.modules
        if 'cdsapi' in sys.modules:
            del sys.modules['cdsapi']
        
        return systems
        
    except Exception as e:
        print(f"\n❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        if Path(temp_tracks).exists():
            Path(temp_tracks).unlink(missing_ok=True)
        
        # 恢复sys.modules
        if 'cdsapi' in sys.modules:
            del sys.modules['cdsapi']
        
        return None


def compare_systems(systems1, systems2):
    """比较两个提取结果"""
    print(f"\n{'='*80}")
    print("📊 对比分析")
    print(f"{'='*80}")
    
    if systems1 is None or systems2 is None:
        print("⚠️ 无法进行对比，因为至少有一个提取失败")
        return
    
    # 按系统名称组织
    systems1_dict = {s.get('system_name', 'Unknown'): s for s in systems1}
    systems2_dict = {s.get('system_name', 'Unknown'): s for s in systems2}
    
    all_system_names = set(systems1_dict.keys()) | set(systems2_dict.keys())
    
    print(f"\n系统数量对比:")
    print(f"  environment_extractor: {len(systems1)} 个系统")
    print(f"  CDSEnvironmentExtractor: {len(systems2)} 个系统")
    
    print(f"\n详细对比:")
    for system_name in sorted(all_system_names):
        print(f"\n  【{system_name}】")
        
        in1 = system_name in systems1_dict
        in2 = system_name in systems2_dict
        
        if in1 and in2:
            print(f"    ✅ 两者都提取到此系统")
            s1 = systems1_dict[system_name]
            s2 = systems2_dict[system_name]
            
            # 比较描述
            desc1 = s1.get('description', '')
            desc2 = s2.get('description', '')
            if desc1 and desc2:
                print(f"    描述相似度: {_text_similarity(desc1, desc2):.1%}")
            
            # 比较位置（如果有）
            if 'position' in s1 and 'position' in s2:
                pos1 = s1['position']
                pos2 = s2['position']
                
                # 尝试提取纬度和经度（可能在不同的键中）
                lat1 = None
                lon1 = None
                lat2 = None
                lon2 = None
                
                # 尝试多种可能的键名
                for key in ['lat', 'latitude', 'center_lat']:
                    if key in pos1 and isinstance(pos1[key], (int, float)):
                        lat1 = float(pos1[key])
                        break
                    elif 'center_of_mass' in pos1 and isinstance(pos1['center_of_mass'], dict):
                        if 'lat' in pos1['center_of_mass']:
                            lat1 = float(pos1['center_of_mass']['lat'])
                            break
                
                for key in ['lon', 'longitude', 'center_lon']:
                    if key in pos1 and isinstance(pos1[key], (int, float)):
                        lon1 = float(pos1[key])
                        break
                    elif 'center_of_mass' in pos1 and isinstance(pos1['center_of_mass'], dict):
                        if 'lon' in pos1['center_of_mass']:
                            lon1 = float(pos1['center_of_mass']['lon'])
                            break
                
                for key in ['lat', 'latitude', 'center_lat']:
                    if key in pos2 and isinstance(pos2[key], (int, float)):
                        lat2 = float(pos2[key])
                        break
                    elif 'center_of_mass' in pos2 and isinstance(pos2['center_of_mass'], dict):
                        if 'lat' in pos2['center_of_mass']:
                            lat2 = float(pos2['center_of_mass']['lat'])
                            break
                
                for key in ['lon', 'longitude', 'center_lon']:
                    if key in pos2 and isinstance(pos2[key], (int, float)):
                        lon2 = float(pos2[key])
                        break
                    elif 'center_of_mass' in pos2 and isinstance(pos2['center_of_mass'], dict):
                        if 'lon' in pos2['center_of_mass']:
                            lon2 = float(pos2['center_of_mass']['lon'])
                            break
                
                if lat1 is not None and lat2 is not None and lon1 is not None and lon2 is not None:
                    lat_diff = abs(lat1 - lat2)
                    lon_diff = abs(lon1 - lon2)
                    print(f"    位置差异: 纬度 {lat_diff:.2f}°, 经度 {lon_diff:.2f}°")
            
            # 比较强度（如果有）
            if 'intensity' in s1 and 'intensity' in s2:
                int1 = s1['intensity']
                int2 = s2['intensity']
                if 'value' in int1 and 'value' in int2:
                    val_diff = abs(int1['value'] - int2['value'])
                    val_pct = val_diff / max(abs(int1['value']), abs(int2['value']), 1e-10) * 100
                    print(f"    强度差异: {val_diff:.2f} ({val_pct:.1f}%)")
        
        elif in1:
            print(f"    ⚠️ 仅 environment_extractor 提取到")
        else:
            print(f"    ⚠️ 仅 CDSEnvironmentExtractor 提取到")
    
    print(f"\n结构对比:")
    _compare_structure(systems1_dict, systems2_dict)


def _text_similarity(text1, text2):
    """简单的文本相似度计算"""
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def _compare_structure(dict1, dict2):
    """比较两个字典的结构"""
    common_systems = set(dict1.keys()) & set(dict2.keys())
    
    if not common_systems:
        print("  没有共同的系统可以比较结构")
        return
    
    # 选择一个共同的系统来比较结构
    system_name = list(common_systems)[0]
    s1 = dict1[system_name]
    s2 = dict2[system_name]
    
    print(f"\n  以 {system_name} 为例比较JSON结构:")
    
    keys1 = set(s1.keys())
    keys2 = set(s2.keys())
    
    common_keys = keys1 & keys2
    only_in1 = keys1 - keys2
    only_in2 = keys2 - keys1
    
    print(f"    共同字段: {sorted(common_keys)}")
    if only_in1:
        print(f"    仅在 environment_extractor: {sorted(only_in1)}")
    if only_in2:
        print(f"    仅在 CDSEnvironmentExtractor: {sorted(only_in2)}")


def convert_to_json_serializable(obj):
    """将NumPy类型转换为Python原生类型，使其可JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_results(systems1, systems2, output_dir="./comparison_output"):
    """保存对比结果到JSON文件"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if systems1:
        file1 = output_path / f"environment_extractor_{timestamp}.json"
        # 转换numpy类型为Python原生类型
        systems1_serializable = convert_to_json_serializable(systems1)
        with open(file1, 'w', encoding='utf-8') as f:
            json.dump(systems1_serializable, f, indent=2, ensure_ascii=False)
        print(f"\n💾 environment_extractor 结果已保存: {file1}")
    
    if systems2:
        file2 = output_path / f"cds_extractor_{timestamp}.json"
        systems2_serializable = convert_to_json_serializable(systems2)
        with open(file2, 'w', encoding='utf-8') as f:
            json.dump(systems2_serializable, f, indent=2, ensure_ascii=False)
        print(f"💾 CDSEnvironmentExtractor 结果已保存: {file2}")
    
    # 保存对比报告
    report = {
        "timestamp": timestamp,
        "nc_file": "data/AURO_v100_IFS_2025061000_f000_f240_06.nc",
        "environment_extractor": {
            "system_count": len(systems1) if systems1 else 0,
            "systems": [s.get('system_name', 'Unknown') for s in systems1] if systems1 else []
        },
        "cds_extractor": {
            "system_count": len(systems2) if systems2 else 0,
            "systems": [s.get('system_name', 'Unknown') for s in systems2] if systems2 else []
        }
    }
    
    report_file = output_path / f"comparison_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾 对比报告已保存: {report_file}")


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print("🔍 环境系统提取算法对比工具")
    print(f"{'='*80}")
    print("\n说明：")
    print("  - 方法1: environment_extractor (extractSyst.py 的底层实现)")
    print("  - 方法2: cds.py (CDSEnvironmentExtractor)")
    print("  - 两者使用相同的输入数据和参数进行提取")
    print("  - 脚本会绕过 CDS 下载功能，直接使用本地NC文件")
    
    nc_path = Path("data/AURO_v100_IFS_2025061000_f000_f240_06.nc")
    
    if not nc_path.exists():
        print(f"\n❌ 错误: 数据文件不存在: {nc_path}")
        print("请确保文件路径正确")
        sys.exit(1)
    
    # 加载并显示文件信息
    ds = load_nc_file(nc_path)
    
    # 方法1: environment_extractor
    systems1 = extract_with_environment_extractor(nc_path)
    
    # 方法2: cds.py (CDSEnvironmentExtractor)
    systems2 = extract_with_cds_extractor(nc_path)
    
    # 对比结果
    if systems1 and systems2:
        compare_systems(systems1, systems2)
    elif systems1 and not systems2:
        print(f"\n{'='*80}")
        print("📊 仅 environment_extractor 提取成功")
        print(f"{'='*80}")
        print(f"\n提取到 {len(systems1)} 个环境系统:")
        for i, system in enumerate(systems1, 1):
            system_name = system.get('system_name', 'Unknown')
            desc = system.get('description', '')[:100]
            print(f"  {i}. {system_name}")
            if desc:
                print(f"     {desc}...")
    elif systems2 and not systems1:
        print(f"\n{'='*80}")
        print("📊 仅 CDSEnvironmentExtractor 提取成功")
        print(f"{'='*80}")
        print(f"\n提取到 {len(systems2)} 个环境系统:")
        for i, system in enumerate(systems2, 1):
            system_name = system.get('system_name', 'Unknown')
            desc = system.get('description', '')[:100]
            print(f"  {i}. {system_name}")
            if desc:
                print(f"     {desc}...")
    
    # 保存结果
    save_results(systems1, systems2)
    
    print(f"\n{'='*80}")
    print("✅ 对比完成!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
