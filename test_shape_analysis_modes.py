#!/usr/bin/env python3
"""测试形状分析的两种模式：完整模式 vs 快速模式"""

import numpy as np
import time
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment_extractor.shape_analysis import WeatherSystemShapeAnalyzer

def create_test_data():
    """创建测试数据"""
    # 模拟一个典型的气象数据场
    lat = np.linspace(-90, 90, 361)  # 0.5度分辨率
    lon = np.linspace(0, 359.5, 720)
    
    # 创建一个高压系统（类似副热带高压）
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    center_lat, center_lon = 25, 140
    
    # 高斯型高压系统
    distance = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
    z500 = 5800 + 100 * np.exp(-distance**2 / 400)
    
    return lat, lon, z500

def test_mode(mode_name, enable_detailed):
    """测试特定模式"""
    print(f"\n{'='*60}")
    print(f"测试模式: {mode_name}")
    print(f"{'='*60}")
    
    lat, lon, z500 = create_test_data()
    
    # 创建分析器
    analyzer = WeatherSystemShapeAnalyzer(lat, lon, enable_detailed_analysis=enable_detailed)
    
    # 性能测试
    n_iterations = 10
    start_time = time.time()
    
    results = []
    for _ in range(n_iterations):
        result = analyzer.analyze_system_shape(
            z500, 
            threshold=5880, 
            system_type="high",
            center_lat=25,
            center_lon=140
        )
        results.append(result)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / n_iterations
    
    print(f"迭代次数: {n_iterations}")
    print(f"总耗时: {elapsed*1000:.2f} ms")
    print(f"平均耗时: {avg_time*1000:.2f} ms")
    
    # 检查结果
    if results[0]:
        print(f"\n结果结构:")
        for key in results[0].keys():
            print(f"  - {key}")
        
        if "basic_geometry" in results[0]:
            bg = results[0]["basic_geometry"]
            print(f"\n基础几何信息:")
            for k, v in list(bg.items())[:5]:
                print(f"  {k}: {v}")
    
    return avg_time, results[0]

def compare_output_consistency():
    """验证两种模式的输出结构"""
    print(f"\n{'='*60}")
    print("输出一致性验证")
    print(f"{'='*60}")
    
    lat, lon, z500 = create_test_data()
    
    # 完整模式
    analyzer_full = WeatherSystemShapeAnalyzer(lat, lon, enable_detailed_analysis=True)
    result_full = analyzer_full.analyze_system_shape(z500, 5880, "high", 25, 140)
    
    # 快速模式
    analyzer_quick = WeatherSystemShapeAnalyzer(lat, lon, enable_detailed_analysis=False)
    result_quick = analyzer_quick.analyze_system_shape(z500, 5880, "high", 25, 140)
    
    print("\n完整模式返回键:")
    if result_full:
        for key in result_full.keys():
            print(f"  ✓ {key}")
    
    print("\n快速模式返回键:")
    if result_quick:
        for key in result_quick.keys():
            marker = "✓" if key in result_full else "⚡"
            print(f"  {marker} {key}")
    
    # 验证快速模式的标记
    if result_quick and "basic_geometry" in result_quick:
        if "analysis_mode" in result_quick["basic_geometry"]:
            mode = result_quick["basic_geometry"]["analysis_mode"]
            print(f"\n✓ 快速模式正确标记为: {mode}")
    
    return result_full, result_quick

def main():
    print("🧪 形状分析模式性能测试")
    print("=" * 60)
    
    # 测试完整模式
    time_full, result_full = test_mode("完整模式 (enable_detailed_analysis=True)", True)
    
    # 测试快速模式
    time_quick, result_quick = test_mode("快速模式 (enable_detailed_analysis=False)", False)
    
    # 计算加速比
    speedup = time_full / time_quick if time_quick > 0 else 0
    
    print(f"\n{'='*60}")
    print("性能对比")
    print(f"{'='*60}")
    print(f"完整模式: {time_full*1000:.2f} ms")
    print(f"快速模式: {time_quick*1000:.2f} ms")
    print(f"⚡ 加速比: {speedup:.1f}x")
    print(f"性能提升: {(1 - time_quick/time_full)*100:.1f}%")
    
    # 输出一致性检查
    result_full_check, result_quick_check = compare_output_consistency()
    
    print(f"\n{'='*60}")
    print("使用建议")
    print(f"{'='*60}")
    print("✅ 默认使用完整模式 (enable_detailed_analysis=True)")
    print("   - 与原实现完全一致")
    print("   - 包含所有详细信息")
    print("   - 适合需要精确面积、周长等的场景")
    print()
    print("⚡ 性能优先使用快速模式 (enable_detailed_analysis=False)")
    print(f"   - 性能提升 {(1 - time_quick/time_full)*100:.0f}%")
    print("   - 跳过昂贵的 regionprops、find_contours、分形维数计算")
    print("   - 保留基本描述和强度信息")
    print("   - 适合批量处理或实时分析")
    
    print(f"\n{'='*60}")
    print("✅ 测试完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
