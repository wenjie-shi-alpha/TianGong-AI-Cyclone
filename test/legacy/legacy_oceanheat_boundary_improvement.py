#!/usr/bin/env python3
"""
测试脚本：验证 OceanHeat 边界提取改进

测试改进点：
1. 边界闭合性（连通区域标注法）
2. 曲率自适应采样（保留暖涡/冷涡特征）
3. 关键特征点标注（极值点、暖涡、相对台风位置）
4. 边界度量完整性
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor


def analyze_boundary_quality(result, case_name):
    """
    分析边界提取质量
    
    Args:
        result: 提取结果字典
        case_name: 案例名称
    """
    print(f"\n{'='*80}")
    print(f"案例: {case_name}")
    print(f"{'='*80}")
    
    if not result or "shape" not in result:
        print("❌ 提取失败\n")
        return {
            "case": case_name,
            "success": False
        }
    
    shape = result["shape"]
    
    # 1. 基本信息
    print(f"\n📍 基本信息:")
    print(f"   系统: {result.get('system_name', 'N/A')}")
    print(f"   边界类型: {shape.get('boundary_type', 'unknown')}")
    print(f"   台风位置: ({result['position']['lat']:.2f}, {result['position']['lon']:.2f})")
    
    # 2. 边界坐标
    boundary = shape.get("warm_water_boundary_26.5C", [])
    print(f"\n🔗 边界坐标:")
    print(f"   总点数: {len(boundary)}")
    
    closure_dist = 0
    if boundary and len(boundary) > 0:
        first = boundary[0]
        last = boundary[-1]
        closure_dist = np.sqrt((last[0]-first[0])**2 + (last[1]-first[1])**2)
        print(f"   首尾坐标: ({first[0]:.2f}, {first[1]:.2f}) -> ({last[0]:.2f}, {last[1]:.2f})")
        print(f"   首尾距离: {closure_dist:.4f}°")
    
    # 3. 边界度量
    analysis = {
        "case": case_name,
        "success": True,
        "total_points": len(boundary),
        "closure_distance_deg": closure_dist
    }
    
    if "boundary_metrics" in shape:
        metrics = shape["boundary_metrics"]
        is_closed = metrics.get('is_closed', False)
        
        print(f"\n📊 边界度量:")
        print(f"   闭合性: {'✅ 闭合' if is_closed else '❌ 未闭合'}")
        print(f"   总点数: {metrics.get('total_points', 0)}")
        print(f"   周长: {metrics.get('perimeter_km', 0):.1f} km")
        print(f"   方位角覆盖: {metrics.get('angle_coverage_deg', 0):.1f}°")
        print(f"   平均点间距: {metrics.get('avg_point_spacing_km', 0):.1f} km")
        print(f"   提取方法: {metrics.get('extraction_method', 'unknown')}")
        
        if "warm_water_area_approx_km2" in metrics:
            print(f"   暖水区面积: {metrics['warm_water_area_approx_km2']:.0f} km²")
        
        analysis.update({
            "is_closed": is_closed,
            "perimeter_km": metrics.get('perimeter_km', 0),
            "angle_coverage_deg": metrics.get('angle_coverage_deg', 0),
            "avg_spacing_km": metrics.get('avg_point_spacing_km', 0),
            "extraction_method": metrics.get('extraction_method', 'unknown'),
            "area_km2": metrics.get('warm_water_area_approx_km2', 0)
        })
    
    # 4. 边界特征
    feature_count = 0
    if "boundary_features" in shape:
        features = shape["boundary_features"]
        print(f"\n🎯 边界特征:")
        
        # 极值点
        if "extreme_points" in features:
            extreme = features["extreme_points"]
            print(f"   ✅ 极值点: 4个")
            print(f"      最北: ({extreme['northernmost']['lon']:.2f}, {extreme['northernmost']['lat']:.2f})")
            print(f"      最南: ({extreme['southernmost']['lon']:.2f}, {extreme['southernmost']['lat']:.2f})")
            print(f"      最东: ({extreme['easternmost']['lon']:.2f}, {extreme['easternmost']['lat']:.2f})")
            print(f"      最西: ({extreme['westernmost']['lon']:.2f}, {extreme['westernmost']['lat']:.2f})")
            feature_count += 4
        
        # 相对台风
        if "tc_relative_points" in features:
            tc_rel = features["tc_relative_points"]
            if "nearest_to_tc" in tc_rel and "farthest_from_tc" in tc_rel:
                nearest = tc_rel["nearest_to_tc"]
                farthest = tc_rel["farthest_from_tc"]
                print(f"   ✅ 相对台风关键点: 2个")
                print(f"      最近点: ({nearest['lon']:.2f}, {nearest['lat']:.2f}), 距离={nearest['distance_km']:.1f} km")
                print(f"      最远点: ({farthest['lon']:.2f}, {farthest['lat']:.2f}), 距离={farthest['distance_km']:.1f} km")
                feature_count += 2
                
                analysis.update({
                    "nearest_distance_km": nearest['distance_km'],
                    "farthest_distance_km": farthest['distance_km']
                })
        
        # 暖涡
        if "warm_eddy_centers" in features:
            eddies = features["warm_eddy_centers"]
            if eddies:
                print(f"   ✅ 暖涡中心: {len(eddies)}个")
                for i, eddy in enumerate(eddies, 1):
                    print(f"      暖涡{i}: ({eddy['lon']:.2f}, {eddy['lat']:.2f}), 曲率={eddy['curvature']:.6f}")
                feature_count += len(eddies)
                analysis["warm_eddy_count"] = len(eddies)
        
        # 冷涡
        if "cold_intrusion_points" in features:
            cold = features["cold_intrusion_points"]
            if cold:
                print(f"   ✅ 冷水侵入: {len(cold)}个")
                for i, c in enumerate(cold, 1):
                    print(f"      冷涡{i}: ({c['lon']:.2f}, {c['lat']:.2f}), 曲率={c['curvature']:.6f}")
                feature_count += len(cold)
                analysis["cold_intrusion_count"] = len(cold)
        
        analysis["total_features"] = feature_count
        print(f"\n   📈 特征点总数: {feature_count}个")
    
    # 5. 描述
    print(f"\n📝 系统描述:")
    desc = result.get('description', 'N/A')
    # 分行显示长描述
    if len(desc) > 80:
        words = desc.split('，')
        for word in words:
            print(f"   {word}{'，' if word != words[-1] else ''}")
    else:
        print(f"   {desc}")
    
    print()
    return analysis


def test_three_cases():
    """
    测试三个案例：AURO, FOUR, PANG
    """
    test_cases = [
        {
            "name": "AURO",
            "nc_file": "data/AURO_v100_IFS_2025061000_f000_f240_06.nc",
            "time_idx": 0,
            "tc_lat": 20.0,
            "tc_lon": 130.0
        },
        {
            "name": "FOUR",
            "nc_file": "data/FOUR_v200_GFS_2020093012_f000_f240_06.nc",
            "time_idx": 0,
            "tc_lat": 15.0,
            "tc_lon": 135.0
        },
        {
            "name": "PANG",
            "nc_file": "data/PANG_v100_IFS_2022032900_f000_f240_06.nc",
            "time_idx": 0,
            "tc_lat": 10.0,
            "tc_lon": 140.0
        }
    ]
    
    print("\n" + "="*80)
    print("OceanHeat 边界提取改进测试")
    print("测试内容：边界闭合性、特征点标注、曲率自适应采样")
    print("="*80)
    
    results = []
    
    for case in test_cases:
        try:
            # 创建临时CSV文件（用于初始化）
            csv_file = "input/western_pacific_typhoons_superfast.csv"
            
            # 创建提取器
            extractor = TCEnvironmentalSystemsExtractor(case["nc_file"], csv_file)
            
            # 提取海洋热含量
            result = extractor.extract_ocean_heat_content(
                case["time_idx"],
                case["tc_lat"],
                case["tc_lon"],
                radius_deg=2.0
            )
            
            # 分析结果
            analysis = analyze_boundary_quality(result, case["name"])
            results.append(analysis)
            
        except Exception as e:
            print(f"\n❌ 案例 {case['name']} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "case": case["name"],
                "success": False,
                "error": str(e)
            })
    
    # 生成对比表
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    
    print(f"\n{'案例':<10} {'成功':<8} {'闭合':<8} {'点数':<6} {'周长(km)':<10} {'覆盖度':<10} {'特征点':<8}")
    print("-" * 80)
    
    for r in results:
        if r["success"]:
            status = "✅"
            closed = "✅" if r.get("is_closed", False) else "❌"
            points = r.get("total_points", 0)
            perimeter = r.get("perimeter_km", 0)
            coverage = r.get("angle_coverage_deg", 0)
            features = r.get("total_features", 0)
            
            print(f"{r['case']:<10} {status:<8} {closed:<8} {points:<6} {perimeter:<10.1f} {coverage:<10.1f}° {features:<8}")
        else:
            print(f"{r['case']:<10} ❌")
    
    # 成功率统计
    success_count = sum(1 for r in results if r["success"])
    closed_count = sum(1 for r in results if r.get("is_closed", False))
    
    print("\n" + "="*80)
    print(f"测试统计:")
    print(f"  总案例数: {len(results)}")
    print(f"  成功提取: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"  边界闭合: {closed_count}/{success_count} ({closed_count/success_count*100:.1f}%)")
    
    if success_count > 0:
        avg_points = np.mean([r.get("total_points", 0) for r in results if r["success"]])
        avg_features = np.mean([r.get("total_features", 0) for r in results if r["success"]])
        avg_perimeter = np.mean([r.get("perimeter_km", 0) for r in results if r["success"]])
        
        print(f"  平均点数: {avg_points:.1f}")
        print(f"  平均特征点: {avg_features:.1f}")
        print(f"  平均周长: {avg_perimeter:.1f} km")
    
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = test_three_cases()
    
    # 保存结果
    output_file = "test_results_oceanheat_boundary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_file}")
