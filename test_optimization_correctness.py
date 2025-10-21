#!/usr/bin/env python3
"""验证优化后的向量化函数与原始逐点实现的数值一致性"""

import numpy as np
import sys

def original_haversine_loop(lats, lons):
    """原始的循环实现"""
    total = 0.0
    for i in range(1, len(lats)):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lats[i-1], lons[i-1], lats[i], lons[i]])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        total += R * c
    return total

def optimized_haversine_vectorized(lats, lons):
    """优化后的向量化实现"""
    if len(lats) < 2:
        return 0.0
    
    R = 6371.0
    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    lon1 = np.radians(lons[:-1])
    lon2 = np.radians(lons[1:])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    distances = R * c
    return float(np.sum(distances))

def original_curvature_loop(coords):
    """原始的曲率计算循环实现"""
    curvatures = []
    for i in range(len(coords)):
        prev_idx = (i - 1) % len(coords)
        next_idx = (i + 1) % len(coords)
        p1 = np.array(coords[prev_idx])
        p2 = np.array(coords[i])
        p3 = np.array(coords[next_idx])
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p3 - p1)
        curvature = cross / denom if denom > 1e-10 else 0.0
        curvatures.append(curvature)
    return np.array(curvatures)

def optimized_curvature_vectorized(coords):
    """优化后的向量化曲率计算"""
    coords_array = np.array(coords)
    n = len(coords_array)
    
    p_prev = np.roll(coords_array, 1, axis=0)
    p_curr = coords_array
    p_next = np.roll(coords_array, -1, axis=0)
    
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    v3 = p_next - p_prev
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    norm_v3 = np.linalg.norm(v3, axis=1)
    
    denom = norm_v1 * norm_v2 * norm_v3
    curvatures = np.where(denom > 1e-10, cross / denom, 0.0)
    return curvatures

def test_haversine():
    """测试 Haversine 距离计算"""
    print("=" * 60)
    print("测试 Haversine 距离计算")
    print("=" * 60)
    
    # 测试案例1: 简单路径
    lats = np.array([30.0, 30.5, 31.0, 31.5])
    lons = np.array([120.0, 120.5, 121.0, 121.5])
    
    result_original = original_haversine_loop(lats, lons)
    result_optimized = optimized_haversine_vectorized(lats, lons)
    
    print(f"原始实现: {result_original:.10f} km")
    print(f"优化实现: {result_optimized:.10f} km")
    print(f"差异: {abs(result_original - result_optimized):.2e} km")
    print(f"相对误差: {abs(result_original - result_optimized) / result_original * 100:.6f}%")
    
    assert np.isclose(result_original, result_optimized, rtol=1e-10), "Haversine 结果不一致!"
    print("✅ Haversine 测试通过\n")
    
    # 测试案例2: 更复杂的路径
    np.random.seed(42)
    lats2 = np.linspace(20, 40, 100) + np.random.randn(100) * 0.1
    lons2 = np.linspace(100, 140, 100) + np.random.randn(100) * 0.1
    
    result_original2 = original_haversine_loop(lats2, lons2)
    result_optimized2 = optimized_haversine_vectorized(lats2, lons2)
    
    print(f"复杂路径 (100点):")
    print(f"原始实现: {result_original2:.10f} km")
    print(f"优化实现: {result_optimized2:.10f} km")
    print(f"差异: {abs(result_original2 - result_optimized2):.2e} km")
    
    assert np.isclose(result_original2, result_optimized2, rtol=1e-10), "复杂路径结果不一致!"
    print("✅ 复杂路径测试通过\n")

def test_curvature():
    """测试曲率计算"""
    print("=" * 60)
    print("测试曲率计算")
    print("=" * 60)
    
    # 测试案例: 圆形路径
    theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    coords = np.column_stack([np.cos(theta), np.sin(theta)])
    
    result_original = original_curvature_loop(coords)
    result_optimized = optimized_curvature_vectorized(coords)
    
    print(f"圆形路径 (50点):")
    print(f"原始实现平均曲率: {np.mean(np.abs(result_original)):.10f}")
    print(f"优化实现平均曲率: {np.mean(np.abs(result_optimized)):.10f}")
    print(f"最大差异: {np.max(np.abs(result_original - result_optimized)):.2e}")
    print(f"相对误差: {np.max(np.abs((result_original - result_optimized) / (result_original + 1e-10))) * 100:.6f}%")
    
    assert np.allclose(result_original, result_optimized, rtol=1e-10, atol=1e-12), "曲率结果不一致!"
    print("✅ 曲率测试通过\n")

def performance_benchmark():
    """性能基准测试"""
    import time
    
    print("=" * 60)
    print("性能基准测试")
    print("=" * 60)
    
    # 生成大规模测试数据
    np.random.seed(42)
    n_points = 10000
    lats = np.linspace(20, 40, n_points) + np.random.randn(n_points) * 0.1
    lons = np.linspace(100, 140, n_points) + np.random.randn(n_points) * 0.1
    
    # Haversine 性能测试
    print(f"\nHaversine 距离计算 ({n_points} 点):")
    
    start = time.time()
    for _ in range(10):
        _ = original_haversine_loop(lats, lons)
    time_original = (time.time() - start) / 10
    
    start = time.time()
    for _ in range(10):
        _ = optimized_haversine_vectorized(lats, lons)
    time_optimized = (time.time() - start) / 10
    
    speedup = time_original / time_optimized
    print(f"原始实现: {time_original*1000:.2f} ms")
    print(f"优化实现: {time_optimized*1000:.2f} ms")
    print(f"⚡ 加速比: {speedup:.1f}x")
    
    # 曲率性能测试
    coords = np.column_stack([lats[:1000], lons[:1000]])
    print(f"\n曲率计算 (1000 点):")
    
    start = time.time()
    for _ in range(10):
        _ = original_curvature_loop(coords)
    time_original = (time.time() - start) / 10
    
    start = time.time()
    for _ in range(10):
        _ = optimized_curvature_vectorized(coords)
    time_optimized = (time.time() - start) / 10
    
    speedup = time_original / time_optimized
    print(f"原始实现: {time_original*1000:.2f} ms")
    print(f"优化实现: {time_optimized*1000:.2f} ms")
    print(f"⚡ 加速比: {speedup:.1f}x")

if __name__ == "__main__":
    try:
        test_haversine()
        test_curvature()
        performance_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ 所有优化验证测试通过！")
        print("数值精度: 保持一致 (误差 < 1e-10)")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
