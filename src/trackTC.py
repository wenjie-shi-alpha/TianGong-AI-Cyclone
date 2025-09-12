#!/usr/bin/env python3
"""
统一热带气旋追踪系统 (CSV+S3重写版)

功能:
1. 读取 output/nc_file_urls.csv (含列: s3_url, model_prefix, init_time)
2. 针对每个 s3_url 匿名(UNSIGNED)下载到临时目录
3. 使用统一算法识别/追踪气旋, 输出结果到 data/ 目录
4. 处理完成后删除临时 NetCDF 文件

相较原版:
- 去除本地多模型目录扫描与对比功能
- 新增命令行参数过滤 (模型/起止日期/数量限制)
- 输出文件命名: tracks_<model_prefix>_<initTimeCompact>_<forecastTag>.csv
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
import trackpy as tp
import warnings
import tempfile
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import shutil
import re

warnings.filterwarnings("ignore")


class UnifiedTropicalCycloneTracker:
    def __init__(self, forecast_data_path, model_type="auto"):
        self.ds = xr.open_dataset(forecast_data_path)
        self.lat = self.ds.latitude.values if "latitude" in self.ds.coords else self.ds.lat.values
        self.lon = self.ds.longitude.values if "longitude" in self.ds.coords else self.ds.lon.values
        self.time_steps = len(self.ds.time)
        self.model_type = self._detect_model_type() if model_type == "auto" else model_type
        print(f"识别模型类型: {self.model_type}")
        self.lon_180 = np.where(self.lon > 180, self.lon - 360, self.lon)
        self.lat_spacing = np.abs(np.diff(self.lat).mean())
        self.lon_spacing = np.abs(np.diff(self.lon).mean())
        self.params = {
            "mslp_threshold": 1010.0,
            "vorticity_threshold": 3.0e-5,
            "search_radius_km": 278,
            "min_vorticity_avg": 5.0e-5,
            "min_warm_core": 0.5,
            "min_wind_speed": 8.0,
            "min_pressure_depth": 2.0,
            "max_search_radius_deg": 5.0,
            "min_track_duration": 4,
            "min_track_distance": 200,
        }
        self._check_data_availability()
        print(f"📊 加载数据: {self.time_steps}个时间步")
        print(
            f"🌍 区域范围: {self.lat.min():.1f}°-{self.lat.max():.1f}°N, "
            f"{self.lon.min():.1f}°-{self.lon.max():.1f}°E"
        )

    def _detect_model_type(self):
        data_vars = list(self.ds.data_vars.keys())
        if "z" in data_vars and "q" in data_vars:
            return "pangu"
        elif "w" in data_vars:
            return "graphcast"
        elif "r" in data_vars:
            return "fourcastnet"
        else:
            return "unknown"

    def _check_data_availability(self):
        self.has_mslp = "msl" in self.ds.data_vars
        self.has_u10 = "u10" in self.ds.data_vars
        self.has_v10 = "v10" in self.ds.data_vars
        self.has_temp = "t" in self.ds.data_vars
        self.has_u = "u" in self.ds.data_vars
        self.has_v = "v" in self.ds.data_vars
        self.has_geopotential = "z" in self.ds.data_vars
        self.has_omega = "w" in self.ds.data_vars
        self.has_tcwc = "tcwv" in self.ds.data_vars
        print("📋 数据可用性检查:")
        print(f"  MSLP: {'✅' if self.has_mslp else '❌'}")
        print(f"  10m风场: {'✅' if self.has_u10 and self.has_v10 else '❌'}")
        print(f"  温度: {'✅' if self.has_temp else '❌'}")
        print(f"  风场: {'✅' if self.has_u and self.has_v else '❌'}")
        print(f"  位势高度: {'✅' if self.has_geopotential else '❌'}")
        print(f"  垂直速度: {'✅' if self.has_omega else '❌'}")
        print(f"  水汽含量: {'✅' if self.has_tcwc else '❌'}")
        if not (self.has_mslp and self.has_u and self.has_v):
            raise ValueError("缺少必要数据(MSLP和风场)，无法进行气旋追踪")

    def _get_data_at_level(self, var_name, level_hPa, time_idx):
        if var_name not in self.ds.data_vars:
            return None
        var_data = self.ds[var_name]
        level_dim = None
        for dim in ["level", "isobaricInhPa", "pressure"]:
            if dim in var_data.dims:
                level_dim = dim
                break
        if level_dim is None:
            return (
                var_data.isel(time=time_idx).values if "time" in var_data.dims else var_data.values
            )
        levels = self.ds[level_dim].values
        level_idx = np.abs(levels - level_hPa).argmin()
        actual_level = levels[level_idx]
        if abs(actual_level - level_hPa) > 25:
            print(f"⚠️  警告: 找不到接近{level_hPa}hPa的层级，使用{actual_level}hPa")
        return var_data.isel(time=time_idx, **{level_dim: level_idx}).values

    def _calculate_vorticity(self, u, v):
        try:
            dx = np.gradient(self.lon_180) * 111000
            dy = np.gradient(self.lat) * 111000
            dvdx = np.gradient(v, axis=1) / dx[np.newaxis, :]
            dudy = np.gradient(u, axis=0) / dy[:, np.newaxis]
            return dvdx - dudy
        except Exception as e:
            print(f"⚠️ 涡度计算失败: {e}")
            return np.zeros_like(u)

    def _get_region_stats(self, field, center_lat_idx, center_lon_idx, radius_deg):
        radius_lat = int(radius_deg / self.lat_spacing)
        radius_lon = int(
            radius_deg / (self.lon_spacing * np.cos(np.deg2rad(self.lat[center_lat_idx])))
        )
        lat_min = max(0, center_lat_idx - radius_lat)
        lat_max = min(field.shape[0], center_lat_idx + radius_lat + 1)
        lon_min = max(0, center_lon_idx - radius_lon)
        lon_max = min(field.shape[1], center_lon_idx + radius_lon + 1)
        region = field[lat_min:lat_max, lon_min:lon_max]
        return np.mean(region), np.max(region), np.min(region)

    def find_candidates(self, time_idx):
        candidates = []
        if self.has_mslp:
            try:
                mslp = self.ds.msl.isel(time=time_idx).values / 100
                mslp_smooth = ndimage.gaussian_filter(mslp, sigma=1.5)
                local_minima = ndimage.minimum_filter(mslp_smooth, size=9) == mslp_smooth
                min_coords = np.where(local_minima)
                for i in range(len(min_coords[0])):
                    lat_idx, lon_idx = min_coords[0][i], min_coords[1][i]
                    pressure = mslp[lat_idx, lon_idx]
                    if pressure < self.params["mslp_threshold"]:
                        candidates.append(
                            {
                                "lat_idx": lat_idx,
                                "lon_idx": lon_idx,
                                "lat": self.lat[lat_idx],
                                "lon": self.lon[lon_idx],
                                "pressure": pressure,
                                "source": "mslp",
                            }
                        )
            except Exception as e:
                print(f"⚠️ MSLP候选点寻找失败: {e}")
        if self.has_u and self.has_v:
            try:
                u_850 = self._get_data_at_level("u", 850, time_idx)
                v_850 = self._get_data_at_level("v", 850, time_idx)
                if u_850 is not None and v_850 is not None:
                    vorticity = self._calculate_vorticity(u_850, v_850)
                    vorticity_smooth = ndimage.gaussian_filter(vorticity, sigma=1.0)
                    if np.mean(self.lat) >= 0:
                        local_maxima = (
                            ndimage.maximum_filter(vorticity_smooth, size=7) == vorticity_smooth
                        )
                    else:
                        local_maxima = (
                            ndimage.minimum_filter(vorticity_smooth, size=7) == vorticity_smooth
                        )
                    max_coords = np.where(local_maxima)
                    for i in range(len(max_coords[0])):
                        lat_idx, lon_idx = max_coords[0][i], max_coords[1][i]
                        vort_value = vorticity[lat_idx, lon_idx]
                        threshold = self.params["vorticity_threshold"]
                        if (np.mean(self.lat) >= 0 and vort_value > threshold) or (
                            np.mean(self.lat) < 0 and vort_value < -threshold
                        ):
                            candidates.append(
                                {
                                    "lat_idx": lat_idx,
                                    "lon_idx": lon_idx,
                                    "lat": self.lat[lat_idx],
                                    "lon": self.lon[lon_idx],
                                    "vorticity": vort_value,
                                    "source": "vorticity",
                                }
                            )
            except Exception as e:
                print(f"⚠️ 涡度候选点寻找失败: {e}")
        return self._merge_and_deduplicate_candidates(candidates)

    def _merge_and_deduplicate_candidates(self, candidates, distance_threshold=2.0):
        if not candidates:
            return []
        merged = []
        for candidate in candidates:
            is_duplicate = False
            for existing in merged:
                distance = np.sqrt(
                    (candidate["lat"] - existing["lat"]) ** 2
                    + (candidate["lon"] - existing["lon"]) ** 2
                )
                if distance < distance_threshold:
                    if "pressure" in candidate and "pressure" in existing:
                        if candidate["pressure"] < existing["pressure"]:
                            merged.remove(existing)
                            merged.append(candidate)
                    elif "vorticity" in candidate and "vorticity" in existing:
                        if abs(candidate["vorticity"]) > abs(existing["vorticity"]):
                            merged.remove(existing)
                            merged.append(candidate)
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(candidate)
        return merged

    def physical_diagnosis(self, candidates, time_idx):
        if not candidates:
            return []
        diagnosed = []
        mslp = self.ds.msl.isel(time=time_idx).values / 100 if self.has_mslp else None
        u_850 = self._get_data_at_level("u", 850, time_idx) if self.has_u and self.has_v else None
        v_850 = self._get_data_at_level("v", 850, time_idx) if self.has_u and self.has_v else None
        t_500 = self._get_data_at_level("t", 500, time_idx) if self.has_temp else None
        t_850 = self._get_data_at_level("t", 850, time_idx) if self.has_temp else None
        u10 = self.ds.u10.isel(time=time_idx).values if self.has_u10 else None
        v10 = self.ds.v10.isel(time=time_idx).values if self.has_v10 else None
        omega_850 = self._get_data_at_level("w", 850, time_idx) if self.has_omega else None
        for candidate in candidates:
            lat_idx, lon_idx = candidate["lat_idx"], candidate["lon_idx"]
            lat = candidate["lat"]
            if abs(lat) > 40:
                continue
            if u_850 is not None and v_850 is not None:
                vorticity = self._calculate_vorticity(u_850, v_850)
                vort_mean, _, _ = self._get_region_stats(vorticity, lat_idx, lon_idx, 2.5)
                vort_ok = (lat >= 0 and vort_mean > self.params["min_vorticity_avg"]) or (
                    lat < 0 and vort_mean < -self.params["min_vorticity_avg"]
                )
                if not vort_ok:
                    continue
            else:
                vort_mean = 0
                vort_ok = False
            warm_core_strength = 0
            warm_core_ok = True
            if t_500 is not None and t_850 is not None:
                t500_center = np.mean(
                    t_500[
                        max(0, lat_idx - 2) : min(t_500.shape[0], lat_idx + 3),
                        max(0, lon_idx - 2) : min(t_500.shape[1], lon_idx + 3),
                    ]
                )
                t500_env, _, _ = self._get_region_stats(t_500, lat_idx, lon_idx, 5.0)
                warm_core_strength = t500_center - t500_env
                warm_core_ok = warm_core_strength > self.params["min_warm_core"]
                if not warm_core_ok:
                    continue
            moisture_ok = True
            if self.has_tcwc:
                tcwv = self.ds.tcwv.isel(time=time_idx).values
                tcwv_center, _, _ = self._get_region_stats(tcwv, lat_idx, lon_idx, 1.0)
                tcwv_env, _, _ = self._get_region_stats(tcwv, lat_idx, lon_idx, 5.0)
                moisture_ok = tcwv_center > tcwv_env
            max_wind = 0
            if u10 is not None and v10 is not None:
                wind_speed = np.sqrt(u10**2 + v10**2)
                _, max_wind, _ = self._get_region_stats(
                    wind_speed, lat_idx, lon_idx, self.params["search_radius_km"] / 111.0
                )
            upward_motion = 0
            if omega_850 is not None:
                omega_center, _, _ = self._get_region_stats(omega_850, lat_idx, lon_idx, 1.0)
                upward_motion = -omega_center
            pressure_depth = 0
            if mslp is not None and "pressure" in candidate:
                pressure_env, _, _ = self._get_region_stats(mslp, lat_idx, lon_idx, 5.0)
                pressure_depth = pressure_env - candidate["pressure"]
            diagnosed_candidate = {
                **candidate,
                "time_idx": time_idx,
                "time": self.ds.time.isel(time=time_idx).values,
                "max_wind": max_wind,
                "warm_core_strength": warm_core_strength,
                "upward_motion": upward_motion,
                "pressure_depth": pressure_depth,
                "vorticity_avg": vort_mean,
                "vorticity_ok": vort_ok,
                "warm_core_ok": warm_core_ok,
                "moisture_ok": moisture_ok,
            }
            diagnosed_candidate["intensity"] = self._classify_intensity(
                diagnosed_candidate.get("pressure", 1013), max_wind
            )
            diagnosed.append(diagnosed_candidate)
        return diagnosed

    def track_cyclones(self, search_range=3.0, memory=3):
        print("\n🔍 开始统一算法涡旋识别与追踪...")
        all_candidates = []
        for time_idx in range(self.time_steps):
            print(f"处理时间步 {time_idx+1}/{self.time_steps}...", end=" ")
            candidates = self.find_candidates(time_idx)
            diagnosed = self.physical_diagnosis(candidates, time_idx)
            all_candidates.extend(diagnosed)
            print(f"找到 {len(diagnosed)} 个有效候选点")
        if not all_candidates:
            print("❌ 未找到任何有效候选涡旋")
            return pd.DataFrame(), pd.DataFrame()
        features_df = pd.DataFrame(all_candidates)
        print(f"\n📊 总共识别到 {len(features_df)} 个有效涡旋点")
        print("🔗 开始路径连接...")
        try:
            tracks = tp.link_df(
                features_df,
                search_range=search_range,
                memory=memory,
                pos_columns=["lat", "lon"],
                t_column="time_idx",
            )
            print(f"✅ 生成了 {tracks['particle'].nunique()} 条初始路径")
            valid_tracks = self._filter_tracks(tracks)
            return features_df, valid_tracks
        except Exception as e:
            print(f"❌ 路径连接失败: {e}")
            return features_df, pd.DataFrame()

    def _filter_tracks(self, tracks_df):
        valid_tracks = []
        for pid in tracks_df["particle"].unique():
            track = tracks_df[tracks_df["particle"] == pid]
            if len(track) < self.params["min_track_duration"]:
                continue
            if self._calculate_track_distance(track) < self.params["min_track_distance"]:
                continue
            max_wind = track["max_wind"].max() if "max_wind" in track.columns else 0
            if max_wind < 10.0:
                continue
            vort_ok_ratio = track["vorticity_ok"].mean() if "vorticity_ok" in track.columns else 0
            warm_core_ok_ratio = (
                track["warm_core_ok"].mean() if "warm_core_ok" in track.columns else 0
            )
            if vort_ok_ratio < 0.7 or warm_core_ok_ratio < 0.7:
                continue
            valid_tracks.append(track)
        return pd.concat(valid_tracks, ignore_index=True) if valid_tracks else pd.DataFrame()

    def _calculate_track_distance(self, track):
        """计算路径总距离（km）"""
        if len(track) < 2:
            return 0

        track_sorted = track.sort_values("time_idx")
        total_distance = 0

        for i in range(len(track_sorted) - 1):
            lat1, lon1 = track_sorted.iloc[i][["lat", "lon"]]
            lat2, lon2 = track_sorted.iloc[i + 1][["lat", "lon"]]

            # 简化的距离计算（km）
            distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111
            total_distance += distance

        return total_distance

    def _classify_intensity(self, pressure, max_wind):
        """分类气旋强度"""
        if max_wind >= 32.9:  # >= 64 knots
            return "Typhoon/Hurricane"
        elif max_wind >= 24.5:  # >= 48 knots
            return "Severe Tropical Storm"
        elif max_wind >= 17.2:  # >= 34 knots
            return "Tropical Storm"
        elif max_wind >= 10.8:  # >= 21 knots
            return "Tropical Depression"
        else:
            return "Weak Disturbance"

    def analyze_tracks(self, tracks_df):
        """分析追踪结果"""
        if tracks_df.empty:
            print("❌ 没有有效的气旋路径")
            return

        print(f"\n🌀 热带气旋路径分析结果")
        print("=" * 80)

        for particle_id in tracks_df["particle"].unique():
            track = tracks_df[tracks_df["particle"] == particle_id].sort_values("time_idx")

            # 计算路径统计信息
            duration = len(track)
            max_intensity = track["max_wind"].max() if "max_wind" in track.columns else 0
            min_pressure = track["pressure"].min() if "pressure" in track.columns else "N/A"

            # 计算移动距离和速度
            total_distance = self._calculate_track_distance(track)
            avg_speed = total_distance / (duration * 6) if duration > 1 else 0  # km/h

            # 强度变化
            if "max_wind" in track.columns:
                wind_trend = (
                    "增强" if track["max_wind"].iloc[-1] > track["max_wind"].iloc[0] else "减弱"
                )
            else:
                wind_trend = "未知"

            print(f"\n🌪️  气旋路径 {particle_id}:")
            print(f"  📅 持续时间: {duration} 个时间步 ({duration*6} 小时)")
            print(f"  💨 最大风速: {max_intensity:.1f} m/s")
            print(f"  🌊 最低气压: {min_pressure} hPa")
            print(f"  📏 移动距离: {total_distance:.0f} km")
            print(f"  🚀 平均速度: {avg_speed:.1f} km/h")
            print(f"  📈 强度趋势: {wind_trend}")
            print(
                f"  🏷️  最高强度: {track['intensity'].mode().iloc[0] if len(track) > 0 else 'Unknown'}"
            )

            # 显示路径轨迹
            print(f"  🗺️  路径轨迹:")
            for i, (_, point) in enumerate(track.head(5).iterrows()):
                time_str = (
                    pd.to_datetime(point["time"]).strftime("%Y-%m-%d %H:%M")
                    if "time" in point
                    else f"T+{point.get('time_idx', i)*6}h"
                )
                print(
                    f"    {time_str}: ({point['lat']:.1f}°N, {point['lon']:.1f}°E) - {point['intensity']}"
                )
            if len(track) > 5:
                print(f"    ... (共{len(track)}个点)")

            print("-" * 60)

    def plot_tracks(self, tracks_df, output_path=None, model_name="Unknown"):
        # 绘图功能已移除，占位以保持接口兼容
        print("⚠️ 绘图功能已移除，跳过 plot_tracks 调用")


def process_all_models():
    print("⚠️ 已移除多模型可视化与比较功能，仅保留核心追踪算法。")


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text)


def download_s3_public(s3_url: str, target_path: Path):
    if not s3_url.startswith("s3://"):
        raise ValueError(f"无效S3 URL: {s3_url}")
    bucket, key = s3_url[5:].split("/", 1)
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket, key, str(target_path))


def process_single_file(nc_path: Path, model_prefix: str, init_time: str, output_dir: Path):
    try:
        tracker = UnifiedTropicalCycloneTracker(str(nc_path), model_type="auto")
        features_df, tracks_df = tracker.track_cyclones()
        if tracks_df.empty:
            print("⚠️ 未找到轨迹, 跳过保存")
            return 0
        m = re.search(r"(f\d{3}_f\d{3}_\d{2})", nc_path.stem)
        forecast_tag = m.group(1) if m else "track"
        safe_prefix = sanitize_filename(model_prefix)
        safe_init = sanitize_filename(init_time.replace(":", "").replace("-", ""))
        csv_name = f"tracks_{safe_prefix}_{safe_init}_{forecast_tag}.csv"
        csv_path = output_dir / csv_name
        tracks_df.to_csv(csv_path, index=False)
        print(f"💾 保存轨迹: {csv_path}")
        return len(tracks_df["particle"].unique())
    except Exception as e:
        print(f"❌ 处理文件失败 {nc_path.name}: {e}")
        return 0


def process_from_csv(
    csv_path: Path,
    limit: int | None = None,
    model_filter: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"s3_url", "model_prefix", "init_time"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 缺少必要列: {required_cols - set(df.columns)}")
    if model_filter:
        df = df[df["model_prefix"].str.contains(model_filter)]
    if start_date:
        df = df[df["init_time"].str.startswith(start_date)]
    if end_date:
        df = df[df["init_time"] <= end_date + "T23:59:59"]
    if limit is not None:
        df = df.head(limit)
    print(f"📄 待处理文件数: {len(df)}")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="tc_tmp_"))
    print(f"🗂️ 临时目录: {tmp_root}")
    total_tracks = 0
    try:
        for idx, row in df.iterrows():
            s3_url = row["s3_url"]
            model_prefix = row["model_prefix"]
            init_time = row["init_time"]
            # ---- 新增: 在下载与处理前构造预期输出文件名, 若已存在则跳过 ----
            try:
                # 依据 S3 文件名提取 forecast_tag (与 process_single_file 保持一致)
                fname = Path(s3_url).name  # e.g. AURO_v100_GFS_2025061000_f000_f240_06.nc
                m = re.search(r"(f\d{3}_f\d{3}_\d{2})", Path(fname).stem)
                forecast_tag = m.group(1) if m else "track"
                safe_prefix = sanitize_filename(model_prefix)
                safe_init = sanitize_filename(init_time.replace(":", "").replace("-", ""))
                expected_csv = output_dir / f"tracks_{safe_prefix}_{safe_init}_{forecast_tag}.csv"
                if expected_csv.exists():
                    print(f"⏭️  已存在轨迹文件, 跳过: {expected_csv}")
                    continue
            except Exception as _e:
                # 不因构造输出文件名失败而中断, 继续按原逻辑下载处理
                print(f"⚠️ 构造输出文件名时出错(继续处理): {_e}")
            # ------------------------------------------------------------------
            print(f"\n⬇️  下载 {idx+1}/{len(df)}: {s3_url}")
            tmp_file = tmp_root / sanitize_filename(Path(s3_url).name)
            try:
                download_s3_public(s3_url, tmp_file)
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                continue
            track_count = process_single_file(tmp_file, model_prefix, init_time, output_dir)
            total_tracks += track_count
            try:
                tmp_file.unlink()
            except Exception:
                pass
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass
    print(f"\n✅ 完成. 总轨迹数(文件内去重): {total_tracks}")
    return total_tracks


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Tropical cyclone tracking from S3 URL list")
    p.add_argument(
        "--csv",
        default="output/nc_file_urls.csv",
        help="CSV file with s3_url, model_prefix, init_time",
    )
    p.add_argument("--limit", type=int, default=None, help="Process only first N files")
    p.add_argument("--model", default=None, help="Filter model_prefix (substring or regex)")
    p.add_argument("--start", default=None, help="Filter init_time start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="Filter init_time end date YYYY-MM-DD")
    # Plot option removed
    return p.parse_args()


def main():
    args = parse_args()
    process_from_csv(
        csv_path=Path(args.csv),
        limit=args.limit,
        model_filter=args.model,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
