#!/usr/bin/env python3
"""
多模型AI预报数据下载器：根据CSV文件中的日期下载对应的AI预报数据
支持 FourCastNetv2-small, Pangu-Weather, 和 GraphCast 三种模型
AI预报数据每天发布两次，分别为00和12小时
"""

import os
import sys
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import threading
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("multi_model_ai_forecast_downloader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class MultiModelAIForecastDownloader:
    """多模型AI预报数据下载器"""

    def __init__(self, csv_file: str, download_dir: str = "./AIForecast"):
        """
        初始化下载器

        Args:
            csv_file: 包含台风数据的CSV文件路径
            download_dir: 下载目录
        """
        self.csv_file = csv_file
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

        # 定义AI模型及其可用时间范围（包括GFS和IFS初始场）
        self.models = {
            # GFS初始场模型
            "FOUR_v200_GFS": {
                "name": "FourCastNetv2-small",
                "initial_condition": "GFS",
                "start_date": "20201001",  # 2020年10月开始
                "description": "FourCastNetv2-small model with GFS initial conditions (from Oct 2020)",
            },
            "PANG_v100_GFS": {
                "name": "Pangu-Weather",
                "initial_condition": "GFS",
                "start_date": "20201001",  # 2020年10月开始
                "description": "Pangu-Weather model with GFS initial conditions (from Oct 2020)",
            },
            "GRAP_v100_GFS": {
                "name": "GraphCast",
                "initial_condition": "GFS",
                "start_date": "20220101",  # 2022年1月开始
                "description": "GraphCast model with GFS initial conditions (from Jan 2022)",
            },
            # IFS初始场模型
            "FOUR_v200_IFS": {
                "name": "FourCastNetv2-small",
                "initial_condition": "IFS",
                "start_date": "20201001",  # 2020年10月开始
                "description": "FourCastNetv2-small model with IFS initial conditions (from Oct 2020)",
            },
            "PANG_v100_IFS": {
                "name": "Pangu-Weather",
                "initial_condition": "IFS",
                "start_date": "20201001",  # 2020年10月开始
                "description": "Pangu-Weather model with IFS initial conditions (from Oct 2020)",
            },
            "GRAP_v100_IFS": {
                "name": "GraphCast",
                "initial_condition": "IFS",
                "start_date": "20220101",  # 2022年1月开始
                "description": "GraphCast model with IFS initial conditions (from Jan 2022)",
            },
        }

        self.s3_base = "s3://noaa-oar-mlwp-data"

    def extract_dates_from_csv(self) -> set:
        """从CSV文件中提取所有独特的日期"""
        logger.info(f"从CSV文件提取日期: {self.csv_file}")

        try:
            # 读取CSV文件
            df = pd.read_csv(self.csv_file)

            # 提取年月日列并创建日期字符串
            dates = set()
            for _, row in df.iterrows():
                year = int(row["year"])
                month = int(row["month"])
                day = int(row["day"])

                # 格式化为YYYYMMDD
                date_str = f"{year:04d}{month:02d}{day:02d}"
                dates.add(date_str)

            logger.info(f"提取到 {len(dates)} 个独特日期")
            return dates

        except Exception as e:
            logger.error(f"提取日期失败: {e}")
            raise

    def check_model_availability(self, model_code: str) -> dict:
        """检查指定模型的可用年份"""
        try:
            s3_path = f"{self.s3_base}/{model_code}/"
            cmd = ["aws", "s3", "ls", s3_path, "--no-sign-request"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                years = []
                for line in result.stdout.strip().split("\n"):
                    if "PRE" in line and "/" in line:
                        year = line.split()[-1].replace("/", "")
                        if year.isdigit() and len(year) == 4:
                            years.append(year)

                return {"available": True, "years": sorted(years), "total_years": len(years)}
            else:
                logger.warning(f"模型 {model_code} 不可用: {result.stderr}")
                return {"available": False, "years": [], "total_years": 0}

        except Exception as e:
            logger.error(f"检查模型 {model_code} 可用性时出错: {e}")
            return {"available": False, "years": [], "total_years": 0}

    def check_data_availability(self, model_code: str, date_str: str, hour: str) -> bool:
        """检查指定模型、日期和时间的数据是否存在"""
        # 构建S3路径
        year = date_str[:4]
        month_day = date_str[4:]

        filename = f"{model_code}_{date_str}{hour}_f000_f240_06.nc"
        s3_path = f"{self.s3_base}/{model_code}/{year}/{month_day}/{filename}"

        # 使用aws s3 ls检查文件是否存在
        try:
            cmd = ["aws", "s3", "ls", s3_path, "--no-sign-request"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"检查文件 {s3_path} 时出错: {e}")
            return False

    def get_file_size_s3(self, s3_path: str) -> int:
        """获取S3文件大小"""
        try:
            cmd = [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                "noaa-oar-mlwp-data",
                "--key",
                s3_path.replace("s3://noaa-oar-mlwp-data/", ""),
                "--no-sign-request",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                import json

                data = json.loads(result.stdout)
                return data.get("ContentLength", 0)
            return 0
        except Exception as e:
            logger.warning(f"获取文件大小失败: {e}")
            return 0

    def download_with_progress(self, s3_path: str, local_path: Path, filename: str) -> bool:
        """使用进度条下载文件"""
        temp_path = None
        try:
            # 获取文件大小
            file_size = self.get_file_size_s3(s3_path)
            if file_size == 0:
                logger.warning(f"无法获取文件大小，使用标准下载: {filename}")
                return self.download_without_progress(s3_path, local_path)

            # 创建临时文件路径
            temp_path = local_path.with_suffix(".tmp")

            # 构建aws s3 cp命令
            cmd = [
                "aws",
                "s3",
                "cp",
                s3_path,
                str(temp_path),
                "--no-sign-request",
                "--cli-read-timeout",
                "0",
                "--cli-connect-timeout",
                "60",
                "--quiet",  # 减少输出
            ]

            # 启动下载进程
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # 创建进度条
            pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc=f"下载 {filename}")

            # 监控下载进度
            start_time = time.time()
            last_size = 0
            stall_time = 0
            max_stall_time = 300  # 5分钟无进度则认为卡住
            download_complete_time = None

            while process.poll() is None:
                current_time = time.time()

                # 更新进度条
                if temp_path.exists():
                    current_size = temp_path.stat().st_size
                    pbar.n = current_size
                    pbar.refresh()

                    # 检查是否有进度
                    if current_size == last_size:
                        stall_time += 1
                    else:
                        stall_time = 0
                        last_size = current_size

                    # 如果下载完成，记录时间但继续等待进程结束
                    if current_size >= file_size and download_complete_time is None:
                        download_complete_time = current_time
                        logger.info(f"文件下载完成，等待AWS CLI完成后续操作...")

                # 检查超时条件
                if download_complete_time is None:
                    # 下载阶段：检查是否卡住太久
                    if stall_time > max_stall_time:
                        process.terminate()
                        pbar.close()
                        logger.warning(f"下载卡住超过{max_stall_time}秒: {filename}")
                        if temp_path and temp_path.exists():
                            temp_path.unlink()
                        return False
                else:
                    # 下载完成后：给AWS CLI额外10分钟时间完成操作
                    if current_time - download_complete_time > 600:
                        process.terminate()
                        pbar.close()
                        logger.warning(f"AWS CLI完成操作超时: {filename}")
                        if temp_path and temp_path.exists():
                            temp_path.unlink()
                        return False

                time.sleep(1)

            # 关闭进度条
            pbar.close()

            # 检查下载结果
            stdout, stderr = process.communicate()

            if process.returncode == 0 and temp_path.exists():
                # 重命名临时文件为最终文件
                temp_path.rename(local_path)
                logger.info(f"下载成功: {filename}")
                return True
            else:
                logger.warning(f"下载失败: {stderr}")
                if temp_path and temp_path.exists():
                    temp_path.unlink()
                return False

        except Exception as e:
            logger.error(f"下载过程中出错: {e}")
            if temp_path and temp_path.exists():
                temp_path.unlink()
            return False

    def download_without_progress(self, s3_path: str, local_path: Path) -> bool:
        """标准下载（无进度条）"""
        try:
            cmd = [
                "aws",
                "s3",
                "cp",
                s3_path,
                str(local_path),
                "--no-sign-request",
                "--cli-read-timeout",
                "0",
                "--cli-connect-timeout",
                "60",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时

            if result.returncode == 0:
                return True
            else:
                logger.warning(f"标准下载失败: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning("标准下载超时")
            return False
        except Exception as e:
            logger.error(f"标准下载出错: {e}")
            return False

    def download_model_data(
        self, model_code: str, date_str: str, hour: str, max_retries: int = 3
    ) -> bool:
        """
        下载指定模型、日期和时间的AI预报数据

        Args:
            model_code: 模型代码 (如 GRAP_v100_GFS)
            date_str: 日期字符串，格式YYYYMMDD
            hour: 小时，"00"或"12"
            max_retries: 最大重试次数

        Returns:
            bool: 下载是否成功
        """
        year = date_str[:4]
        month_day = date_str[4:]

        # 构建文件名和S3路径
        filename = f"{model_code}_{date_str}{hour}_f000_f240_06.nc"
        s3_path = f"{self.s3_base}/{model_code}/{year}/{month_day}/{filename}"

        # 创建按日期组织的下载目录
        date_dir = self.download_dir / date_str
        date_dir.mkdir(exist_ok=True)
        model_dir = date_dir / model_code
        model_dir.mkdir(exist_ok=True)
        local_path = model_dir / filename

        # 如果文件已存在，跳过下载
        if local_path.exists():
            logger.info(f"文件已存在，跳过下载: {filename}")
            return True

        # 下载文件
        for attempt in range(max_retries):
            logger.info(f"下载 {model_code} {filename} (尝试 {attempt + 1}/{max_retries})")

            # 使用带进度条的下载函数
            success = self.download_with_progress(s3_path, local_path, filename)

            if success:
                return True
            else:
                logger.warning(f"下载失败 (尝试 {attempt + 1})")
                # 清理可能存在的不完整文件
                if local_path.exists():
                    local_path.unlink()

        logger.error(f"下载失败，已达到最大重试次数: {filename}")
        return False

    def filter_dates_by_model(self, dates: set, model_code: str) -> list:
        """根据模型的可用时间范围过滤日期"""
        start_date = self.models[model_code]["start_date"]
        filtered_dates = [d for d in dates if d >= start_date]
        return sorted(filtered_dates)

    def filter_models_by_initial_condition(self, initial_condition: str) -> list:
        """
        根据初始场类型过滤模型

        Args:
            initial_condition: "GFS", "IFS", 或 "ALL"

        Returns:
            list: 过滤后的模型代码列表
        """
        if initial_condition.upper() == "ALL":
            return list(self.models.keys())
        elif initial_condition.upper() in ["GFS", "IFS"]:
            return [
                model_code
                for model_code in self.models.keys()
                if model_code.endswith(f"_{initial_condition.upper()}")
            ]
        else:
            logger.warning(f"未知的初始场类型: {initial_condition}，使用所有模型")
            return list(self.models.keys())

    def check_all_models_availability(self, initial_condition: str = "ALL"):
        """
        检查所有模型的可用性

        Args:
            initial_condition: 初始场类型过滤
        """
        logger.info("检查AI模型的可用性...")

        models_to_check = self.filter_models_by_initial_condition(initial_condition)
        logger.info(f"检查 {len(models_to_check)} 个模型 (初始场: {initial_condition})")

        for model_code in models_to_check:
            model_info = self.models[model_code]
            logger.info(f"\n检查模型: {model_info['name']} ({model_code})")
            logger.info(f"初始场: {model_info['initial_condition']}")
            logger.info(f"描述: {model_info['description']}")

            availability = self.check_model_availability(model_code)
            if availability["available"]:
                logger.info(f"✓ 模型可用，数据年份: {availability['years']}")
                logger.info(f"  总计 {availability['total_years']} 年的数据")
            else:
                logger.warning(f"✗ 模型不可用")

    def find_available_data(
        self,
        start_date: str = None,
        end_date: str = None,
        models: list = None,
        check_only: bool = False,
    ):
        """
        查找可用的AI预报数据

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            models: 要检查的模型列表，None表示检查所有模型
            check_only: 是否只检查不下载
        """
        # 提取所有日期
        all_dates = self.extract_dates_from_csv()

        # 应用日期过滤
        filtered_dates = sorted(list(all_dates))
        if start_date:
            filtered_dates = [d for d in filtered_dates if d >= start_date]
        if end_date:
            filtered_dates = [d for d in filtered_dates if d <= end_date]

        # 选择要处理的模型
        models_to_check = models if models else list(self.models.keys())

        logger.info(
            f"检查 {len(filtered_dates)} 个日期在 {len(models_to_check)} 个模型中的数据可用性"
        )

        # 统计结果
        results = {}

        for model_code in models_to_check:
            model_name = self.models[model_code]["name"]
            logger.info(f"\n处理模型: {model_name} ({model_code})")

            # 根据模型可用时间过滤日期
            model_dates = self.filter_dates_by_model(filtered_dates, model_code)
            logger.info(f"该模型适用的日期数量: {len(model_dates)}")

            if not model_dates:
                logger.warning(f"没有适用于模型 {model_code} 的日期")
                continue

            # 检查数据可用性
            available_data = []
            missing_data = []

            # 根据check_only决定处理的日期数量
            dates_to_process = model_dates[:10] if check_only else model_dates

            logger.info(f"处理 {len(dates_to_process)} 个日期...")

            for i, date_str in enumerate(dates_to_process):
                if i > 0 and i % 50 == 0:
                    logger.info(f"已处理 {i}/{len(dates_to_process)} 个日期")

                for hour in ["00", "12"]:
                    if self.check_data_availability(model_code, date_str, hour):
                        available_data.append(f"{date_str}{hour}")
                        if not check_only:
                            # 下载数据
                            success = self.download_model_data(model_code, date_str, hour)
                            if not success:
                                logger.warning(f"下载失败: {date_str}{hour}")
                    else:
                        missing_data.append(f"{date_str}{hour}")

            results[model_code] = {
                "name": model_name,
                "total_dates": len(model_dates),
                "processed_dates": len(dates_to_process),
                "available": len(available_data),
                "missing": len(missing_data),
            }

            logger.info(f"模型 {model_name} 结果:")
            logger.info(
                f"  处理的日期数: {results[model_code]['processed_dates']}/{results[model_code]['total_dates']}"
            )
            logger.info(f"  可用数据: {results[model_code]['available']}")
            logger.info(f"  缺失数据: {results[model_code]['missing']}")

        # 输出总结
        logger.info("\n=== 数据可用性总结 ===")
        for model_code, result in results.items():
            logger.info(
                f"{result['name']}: {result['available']}/{result['available']+result['missing']} 可用"
            )


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python multi_model_ai_forecast_downloader.py <csv_file> [command] [options]")
        print("命令:")
        print("  check [initial_condition] - 检查所有模型可用性")
        print("  find [start_date] [end_date] [initial_condition] - 查找可用数据")
        print("  download [start_date] [end_date] [model] [initial_condition] - 下载数据")
        print("参数:")
        print("  initial_condition: GFS 或 IFS 或 ALL (默认: ALL)")
        print("  model: 特定模型代码 (如 FOUR_v200, PANG_v100, GRAP_v100)")
        print("例如:")
        print(
            "  python multi_model_ai_forecast_downloader.py western_pacific_typhoons_superfast.csv check"
        )
        print(
            "  python multi_model_ai_forecast_downloader.py western_pacific_typhoons_superfast.csv check GFS"
        )
        print(
            "  python multi_model_ai_forecast_downloader.py western_pacific_typhoons_superfast.csv find 20210101 20211231 IFS"
        )
        print(
            "  python multi_model_ai_forecast_downloader.py western_pacific_typhoons_superfast.csv download 20220329 20220329 GRAP_v100 GFS"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "check"

    if not os.path.exists(csv_file):
        logger.error(f"CSV文件不存在: {csv_file}")
        sys.exit(1)

    # 创建下载器
    downloader = MultiModelAIForecastDownloader(csv_file)

    try:
        if command == "check":
            # 检查所有模型可用性
            initial_condition = sys.argv[3] if len(sys.argv) > 3 else "ALL"
            downloader.check_all_models_availability(initial_condition)

        elif command == "find":
            # 查找可用数据
            start_date = sys.argv[3] if len(sys.argv) > 3 else None
            end_date = sys.argv[4] if len(sys.argv) > 4 else None
            initial_condition = sys.argv[5] if len(sys.argv) > 5 else "ALL"
            models = downloader.filter_models_by_initial_condition(initial_condition)
            downloader.find_available_data(start_date, end_date, models, check_only=True)

        elif command == "download":
            # 下载数据
            start_date = sys.argv[3] if len(sys.argv) > 3 else None
            end_date = sys.argv[4] if len(sys.argv) > 4 else None
            model_base = sys.argv[5] if len(sys.argv) > 5 else None
            initial_condition = sys.argv[6] if len(sys.argv) > 6 else "ALL"

            # 构建完整的模型列表
            if model_base and initial_condition != "ALL":
                models = [f"{model_base}_{initial_condition}"]
            elif model_base:
                models = [f"{model_base}_GFS", f"{model_base}_IFS"]
            else:
                models = downloader.filter_models_by_initial_condition(initial_condition)

            downloader.find_available_data(start_date, end_date, models, check_only=False)

        else:
            logger.error(f"未知命令: {command}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"操作过程中出错: {e}")
        raise


if __name__ == "__main__":
    main()
