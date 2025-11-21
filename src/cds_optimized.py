#!/usr/bin/env python3
"""
CDS优化版本 - 解决API排队问题

主要改进：
1. 按日拆分pressure-level请求（遵循MARS tape规则）
2. 按周拆分single-level请求
3. 异步并行下载（ThreadPoolExecutor，限制4并发）
4. 智能重试机制
5. 区域裁剪和GRIB格式
6. 增量下载（断点续传）
"""

import cdsapi
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import xarray as xr
import warnings

warnings.filterwarnings('ignore')


class OptimizedCDSDownloader:
    """
    优化的CDS下载器
    
    关键改进：
    - 按MARS tape规则组织请求（pressure按日，single按周）
    - 4线程并行下载
    - 自动重试和错误恢复
    - 断点续传
    """
    
    def __init__(
        self,
        output_dir="./cds_output_optimized",
        max_concurrent=4,
        area=None,  # [North, West, South, East]，例如 [60, 100, 0, 180]
        use_grib=True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # CDS客户端配置
        self.cds_client = cdsapi.Client(
            timeout=600,      # 10分钟超时
            quiet=False,
            debug=False,
            retry_max=5
        )
        
        # 并发控制
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        
        # 区域和格式
        self.area = area  # 西太平洋: [60, 100, 0, 180]
        self.data_format = 'grib' if use_grib else 'netcdf'
        
        print(f"✅ 优化CDS下载器已初始化")
        print(f"   - 最大并发: {max_concurrent}")
        print(f"   - 区域裁剪: {'是' if area else '否'}")
        print(f"   - 格式: {self.data_format.upper()}")
    
    def download_single_level_day(self, date_str, variables=None):
        """
        下载单日的single-level数据
        
        Args:
            date_str: 日期字符串，格式 'YYYY-MM-DD'
            variables: 变量列表，None则使用默认列表
        """
        if variables is None:
            variables = [
                'mean_sea_level_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '2m_temperature',
                'sea_surface_temperature',
                'total_column_water_vapour'
            ]
        
        date = pd.Timestamp(date_str)
        output_file = self.output_dir / f"era5_single_{date.strftime('%Y%m%d')}.{self.data_format[:4]}"
        
        if output_file.exists():
            print(f"📁 单层数据已存在: {date_str}")
            return str(output_file)
        
        request = {
            'product_type': 'reanalysis',
            'format': self.data_format,
            'variable': variables,
            'year': date.strftime('%Y'),
            'month': date.strftime('%m'),
            'day': date.strftime('%d'),
            'time': ['00:00', '06:00', '12:00', '18:00'],
        }
        
        if self.area:
            request['area'] = self.area
        
        with self.semaphore:
            print(f"📥 下载单层数据: {date_str}")
            try:
                self.cds_client.retrieve(
                    'reanalysis-era5-single-levels',
                    request,
                    str(output_file)
                )
                print(f"✅ 单层数据完成: {date_str}")
                return str(output_file)
            except Exception as e:
                print(f"❌ 单层数据失败: {date_str} - {e}")
                if output_file.exists():
                    output_file.unlink()
                raise
    
    def download_pressure_level_day(self, date_str, levels=None, variables=None):
        """
        下载单日的pressure-level数据
        
        关键优化：单日所有层级和变量在MARS同一tape，检索最快
        
        Args:
            date_str: 日期字符串
            levels: 气压层级列表
            variables: 变量列表
        """
        if levels is None:
            levels = ['850', '700', '500', '300', '200']
        
        if variables is None:
            variables = [
                'u_component_of_wind',
                'v_component_of_wind',
                'geopotential',
                'temperature',
                'relative_humidity'
            ]
        
        date = pd.Timestamp(date_str)
        output_file = self.output_dir / f"era5_pressure_{date.strftime('%Y%m%d')}.{self.data_format[:4]}"
        
        if output_file.exists():
            print(f"📁 等压面数据已存在: {date_str}")
            return str(output_file)
        
        request = {
            'product_type': 'reanalysis',
            'format': self.data_format,
            'variable': variables,
            'pressure_level': levels,
            'year': date.strftime('%Y'),
            'month': date.strftime('%m'),
            'day': date.strftime('%d'),
            'time': ['00:00', '06:00', '12:00', '18:00'],
        }
        
        if self.area:
            request['area'] = self.area
        
        with self.semaphore:
            print(f"📥 下载等压面数据: {date_str}")
            try:
                self.cds_client.retrieve(
                    'reanalysis-era5-pressure-levels',
                    request,
                    str(output_file)
                )
                print(f"✅ 等压面数据完成: {date_str}")
                return str(output_file)
            except Exception as e:
                print(f"❌ 等压面数据失败: {date_str} - {e}")
                if output_file.exists():
                    output_file.unlink()
                raise
    
    def download_with_retry(self, download_func, max_retries=3):
        """
        带重试的下载包装器
        
        Args:
            download_func: 下载函数
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                return download_func()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 60  # 指数退避: 1分钟, 2分钟, 4分钟
                    print(f"⚠️ 下载失败（尝试 {attempt + 1}/{max_retries}），"
                          f"{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 下载彻底失败（已重试{max_retries}次）: {e}")
                    raise
    
    def get_missing_dates(self, start_date, end_date, data_type='pressure'):
        """
        检测缺失的日期
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            data_type: 'pressure' 或 'single'
        
        Returns:
            缺失日期列表
        """
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        missing_dates = []
        
        for date in all_dates:
            date_str = date.strftime('%Y%m%d')
            if data_type == 'pressure':
                expected_file = self.output_dir / f"era5_pressure_{date_str}.{self.data_format[:4]}"
            else:
                expected_file = self.output_dir / f"era5_single_{date_str}.{self.data_format[:4]}"
            
            if not expected_file.exists():
                missing_dates.append(date.strftime('%Y-%m-%d'))
        
        return missing_dates
    
    def parallel_download_dates(self, date_list, download_func):
        """
        并行下载多个日期
        
        Args:
            date_list: 日期列表
            download_func: 下载函数（接受date_str参数）
        
        Returns:
            成功下载的文件列表
        """
        if not date_list:
            print("ℹ️ 无需下载")
            return []
        
        print(f"🚀 开始并行下载 {len(date_list)} 个日期（{self.max_concurrent} 线程）")
        
        downloaded_files = []
        failed_dates = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # 提交所有任务
            future_to_date = {
                executor.submit(
                    self.download_with_retry,
                    lambda d=date: download_func(d)
                ): date
                for date in date_list
            }
            
            # 收集结果
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    if result:
                        downloaded_files.append(result)
                except Exception as e:
                    failed_dates.append(date)
                    print(f"❌ {date} 最终失败: {e}")
        
        print(f"📊 下载完成: {len(downloaded_files)} 成功, {len(failed_dates)} 失败")
        if failed_dates:
            print(f"   失败日期: {failed_dates}")
        
        return downloaded_files
    
    def download_month_optimized(self, year, month):
        """
        优化的月度下载
        
        策略：
        1. 检测缺失日期
        2. 并行下载pressure和single数据
        3. 合并为月度文件（可选）
        
        Args:
            year: 年份
            month: 月份
        
        Returns:
            (pressure_files, single_files) 元组
        """
        start_date = f"{year}-{month:02d}-01"
        end_date = pd.Timestamp(start_date) + pd.offsets.MonthEnd(0)
        end_date = end_date.strftime('%Y-%m-%d')
        
        print(f"\n{'='*50}")
        print(f"处理月份: {year}-{month:02d}")
        print(f"日期范围: {start_date} 至 {end_date}")
        print(f"{'='*50}")
        
        # 检测缺失数据
        missing_pressure = self.get_missing_dates(start_date, end_date, 'pressure')
        missing_single = self.get_missing_dates(start_date, end_date, 'single')
        
        print(f"📋 需要下载: pressure={len(missing_pressure)}天, single={len(missing_single)}天")
        
        # 并行下载pressure数据
        print("\n--- 下载等压面数据 ---")
        pressure_files = self.parallel_download_dates(
            missing_pressure,
            self.download_pressure_level_day
        )
        
        # 并行下载single数据
        print("\n--- 下载单层数据 ---")
        single_files = self.parallel_download_dates(
            missing_single,
            self.download_single_level_day
        )
        
        # 获取所有已存在的文件
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_pressure_files = [
            str(self.output_dir / f"era5_pressure_{d.strftime('%Y%m%d')}.{self.data_format[:4]}")
            for d in all_dates
        ]
        all_single_files = [
            str(self.output_dir / f"era5_single_{d.strftime('%Y%m%d')}.{self.data_format[:4]}")
            for d in all_dates
        ]
        
        # 过滤存在的文件
        all_pressure_files = [f for f in all_pressure_files if Path(f).exists()]
        all_single_files = [f for f in all_single_files if Path(f).exists()]
        
        print(f"\n✅ {year}-{month:02d} 下载完成:")
        print(f"   等压面: {len(all_pressure_files)} 个文件")
        print(f"   单层: {len(all_single_files)} 个文件")
        
        return all_pressure_files, all_single_files
    
    def merge_daily_files_to_month(self, daily_files, output_file):
        """
        合并日度文件为月度文件（可选）
        
        Args:
            daily_files: 日度文件列表
            output_file: 输出文件路径
        """
        if not daily_files:
            return None
        
        if Path(output_file).exists():
            print(f"📁 月度合并文件已存在: {output_file}")
            return str(output_file)
        
        print(f"🔗 合并 {len(daily_files)} 个日度文件...")
        
        try:
            # 使用xarray合并
            if self.data_format == 'grib':
                datasets = [xr.open_dataset(f, engine='cfgrib') for f in daily_files]
            else:
                datasets = [xr.open_dataset(f) for f in daily_files]
            
            merged = xr.concat(datasets, dim='time')
            merged = merged.sortby('time')
            
            # 保存
            if output_file.endswith('.nc'):
                merged.to_netcdf(output_file)
            else:
                merged.to_netcdf(output_file.replace('.grib', '.nc'))
            
            # 关闭数据集
            for ds in datasets:
                ds.close()
            merged.close()
            
            print(f"✅ 月度文件已保存: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"⚠️ 合并失败: {e}")
            return None


def example_usage():
    """使用示例"""
    
    # 创建下载器
    downloader = OptimizedCDSDownloader(
        output_dir="./cds_output_optimized",
        max_concurrent=4,  # 4线程并发
        area=[60, 100, 0, 180],  # 西太平洋区域
        use_grib=True  # 使用GRIB格式（更快）
    )
    
    # 示例1：下载单个月份
    pressure_files, single_files = downloader.download_month_optimized(2006, 3)
    
    # 示例2：批量下载多个月份
    months_to_download = [
        (2006, 3),
        (2006, 4),
        (2006, 5),
    ]
    
    for year, month in months_to_download:
        p_files, s_files = downloader.download_month_optimized(year, month)
        
        # 可选：合并为月度文件
        if p_files:
            monthly_pressure = f"./cds_output_optimized/era5_pressure_{year}{month:02d}_monthly.nc"
            downloader.merge_daily_files_to_month(p_files, monthly_pressure)
        
        if s_files:
            monthly_single = f"./cds_output_optimized/era5_single_{year}{month:02d}_monthly.nc"
            downloader.merge_daily_files_to_month(s_files, monthly_single)


if __name__ == "__main__":
    example_usage()
