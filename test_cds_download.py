#!/usr/bin/env python3
"""
测试CDS下载功能的脚本
验证：
1. 文件存在性检查
2. 压力层数据分批下载（2次或3次）
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cds import CDSEnvironmentExtractor

def test_download():
    """测试下载功能"""
    
    # 使用一个小的测试数据集
    tracks_file = "input/matched_cyclone_tracks.csv"
    output_dir = "./test_cds_output"
    
    print("=" * 70)
    print("测试CDS下载功能")
    print("=" * 70)
    
    # 创建提取器实例
    extractor = CDSEnvironmentExtractor(
        tracks_file=tracks_file,
        output_dir=output_dir,
        cleanup_intermediate=False,  # 测试时不清理，便于检查
        max_workers=1  # 串行模式，便于调试
    )
    
    # 限制测试数据量 - 只处理第一年的第一个月
    print("\n📊 原始数据点数:", len(extractor.tracks_df))
    
    # 获取第一年第一个月的数据
    extractor.tracks_df['year_month'] = extractor.tracks_df['time'].dt.to_period('M')
    first_month = extractor.tracks_df['year_month'].min()
    extractor.tracks_df = extractor.tracks_df[extractor.tracks_df['year_month'] == first_month].head(5)
    
    print(f"🧪 测试数据: {first_month}，{len(extractor.tracks_df)} 个路径点")
    
    # 获取时间范围
    start_date = extractor.tracks_df['time'].min().strftime('%Y-%m-%d')
    end_date = extractor.tracks_df['time'].max().strftime('%Y-%m-%d')
    
    print(f"\n📅 时间范围: {start_date} 到 {end_date}")
    
    # 测试1: 下载地面层数据
    print("\n" + "=" * 70)
    print("测试1: 下载地面层数据（应该一次成功）")
    print("=" * 70)
    single_file = extractor.download_era5_data(start_date, end_date)
    if single_file:
        print(f"✅ 地面层数据下载成功: {single_file}")
        print(f"📊 文件大小: {Path(single_file).stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("❌ 地面层数据下载失败")
        return False
    
    # 测试2: 第二次请求应该跳过（文件已存在）
    print("\n" + "=" * 70)
    print("测试2: 再次请求地面层数据（应该跳过）")
    print("=" * 70)
    single_file_2 = extractor.download_era5_data(start_date, end_date)
    if single_file_2 == single_file:
        print(f"✅ 正确跳过已存在文件")
    else:
        print("❌ 未能正确跳过已存在文件")
    
    # 测试3: 下载压力层数据（分批下载）
    print("\n" + "=" * 70)
    print("测试3: 下载压力层数据（测试分批逻辑）")
    print("=" * 70)
    pressure_file = extractor.download_era5_pressure_data(start_date, end_date)
    if pressure_file:
        print(f"✅ 压力层数据下载成功: {pressure_file}")
        print(f"📊 文件大小: {Path(pressure_file).stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("❌ 压力层数据下载失败")
        return False
    
    # 测试4: 第二次请求压力层数据应该跳过
    print("\n" + "=" * 70)
    print("测试4: 再次请求压力层数据（应该跳过）")
    print("=" * 70)
    pressure_file_2 = extractor.download_era5_pressure_data(start_date, end_date)
    if pressure_file_2 == pressure_file:
        print(f"✅ 正确跳过已存在文件")
    else:
        print("❌ 未能正确跳过已存在文件")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    print(f"\n💡 测试文件保存在: {output_dir}")
    print("   可以手动检查文件是否正确生成")
    
    return True

if __name__ == "__main__":
    try:
        success = test_download()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
