"""整体分析与导出相关的逻辑。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


class AnalysisMixin:
    def analyze_and_export_as_json(self, output_dir="final_single_output"):
        try:
            return self._analyze_and_export_as_json(output_dir)
        finally:
            self.close()

    def _analyze_and_export_as_json(self, output_dir="final_single_output"):
        print("\n🔍 开始进行专家级环境场解译并构建JSON...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        existing_outputs = list(output_path.glob(f"{self.nc_stem}_TC_Analysis_*.json"))
        if existing_outputs:
            if "particle" in self.tc_tracks.columns:
                expected_particles = sorted(set(str(p) for p in self.tc_tracks["particle"].unique()))
            else:
                expected_particles = ["TC_01"]
            existing_particles = []
            for pfile in existing_outputs:
                stem = pfile.stem
                if stem.startswith(f"{self.nc_stem}_TC_Analysis_"):
                    pid = stem.replace(f"{self.nc_stem}_TC_Analysis_", "")
                    try:
                        if pfile.stat().st_size > 10:
                            existing_particles.append(pid)
                    except Exception:
                        pass
            if set(expected_particles).issubset(existing_particles):
                print(
                    f"⏩ 检测到当前NC对应的所有分析结果已存在于 '{output_path}' (共{len(existing_particles)}个)，跳过重算。"
                )
                return {pid: None for pid in expected_particles}

        if "particle" not in self.tc_tracks.columns:
            print("警告: 路径文件 .csv 中未找到 'particle' 列，将所有路径点视为单个台风事件。")
            self.tc_tracks["particle"] = "TC_01"

        tc_groups = self.tc_tracks.groupby("particle")
        all_typhoon_events = {}

        for tc_id, track_df in tc_groups:
            print(f"\n🌀 正在处理台风事件: {tc_id}")
            event_data = {
                "tc_id": str(tc_id),
                "analysis_time": datetime.now().isoformat(),
                "time_series": [],
            }

            for _, track_point in track_df.sort_values(by="time").iterrows():
                time_idx, lat, lon = (
                    int(track_point.get("time_idx", 0)),
                    track_point["lat"],
                    track_point["lon"],
                )
                print(f"  -> 分析时间点: {track_point['time'].strftime('%Y-%m-%d %H:%M')}")

                environmental_systems = []
                systems_to_extract = [
                    self.extract_steering_system,
                    self.extract_vertical_wind_shear,
                    self.extract_ocean_heat_content,
                    self.extract_upper_level_divergence,
                    self.extract_intertropical_convergence_zone,
                    self.extract_westerly_trough,
                    self.extract_frontal_system,
                    self.extract_monsoon_trough,
                ]

                for func in systems_to_extract:
                    system_obj = func(time_idx, lat, lon)
                    if system_obj:
                        environmental_systems.append(system_obj)

                event_data["time_series"].append(
                    {
                        "time": track_point["time"].isoformat(),
                        "time_idx": time_idx,
                        "tc_position": {"lat": lat, "lon": lon},
                        "tc_intensity_hpa": track_point.get("intensity", None),
                        "environmental_systems": environmental_systems,
                    }
                )

            non_empty_steps = sum(1 for item in event_data["time_series"] if item["environmental_systems"])
            if event_data["time_series"] and non_empty_steps == 0:
                print(
                    f"⚠️ 事件 {tc_id} 的环境系统在全部 {len(event_data['time_series'])} 个时次均为空。"
                    "通常表示输入NC缺少高空层变量(u/v/z/t/w)。"
                )

            all_typhoon_events[str(tc_id)] = event_data

        for tc_id, data in all_typhoon_events.items():
            json_filename = output_path / f"{self.nc_stem}_TC_Analysis_{tc_id}.json"
            print(f"💾 保存专家解译结果到: {json_filename}")

            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    val = float(obj)
                    if not np.isfinite(val):
                        return None
                    return val
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return obj

            def sanitize_inf_nan(o):
                if isinstance(o, dict):
                    return {k: sanitize_inf_nan(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [sanitize_inf_nan(v) for v in o]
                if isinstance(o, float):
                    if math.isinf(o) or math.isnan(o):
                        return None
                    return o
                return o

            converted_data = convert_numpy_types(data)
            converted_data = sanitize_inf_nan(converted_data)

            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ 所有台风事件解译完成，结果保存在: {output_path}")
        return all_typhoon_events


__all__ = ["AnalysisMixin"]
