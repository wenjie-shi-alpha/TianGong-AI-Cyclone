#!/usr/bin/env python3
"""Generate cyclone forecast training samples via OpenRouter completions.

The script consumes ``preprocessed_data/matched_samples.jsonl`` (produced by
``prepare_forecast_samples.py``), selects the first N aggregated samples, and for
each one builds a structured prompt that mirrors the workflow in
``specdataset.md``.  Every sample now contains multiple model forecasts (each
already reduced到单条最佳轨迹)，因此提示词会综合呈现多模式信息。

调用 OpenRouter 托管模型后，脚本将模型推理与最终预报写入标准 SFT 数据格式：

    {
      "instruction": "...系统提示词...",
      "input": "...观测+数值预测...",
      "output": "...思维链与预报...",
      "metadata": {...}
    }

The OpenRouter API key is loaded from a ``.env`` file located in the project
root.  The file must contain a line ``OPENROUTER_API_KEY=sk-...`` and is ignored
by git (already listed in the repository's ``.gitignore``).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

ISO_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

logger = logging.getLogger(__name__)

SYSTEM_NAME_LABELS: Dict[str, str] = {
    "SubtropicalHigh": "副热带高压",
    "VerticalWindShear": "垂直风切变",
    "OceanHeatContent": "海洋热含量",
    "UpperLevelDivergence": "高空辐散",
    "InterTropicalConvergenceZone": "热带辐合带",
    "WesterlyTrough": "西风槽",
    "FrontalSystem": "锋面系统",
    "MonsoonTrough": "季风槽",
    "LowLevelFlow": "低层风场",
    "AtmosphericStability": "大气稳定度",
    "BlockingHigh": "阻塞高压",
    "MaddenJulianOscillation": "MJO",
}

GENERATION_STRATEGIES: Dict[str, Dict[str, object]] = {
    # === 正向样本 (70-80%) ===
    "comprehensive": {
        "weight": 0.30,
        "temperature": 0.7,
        "top_p": 0.9,
        "sample_type": "positive",
        "system_prompt_addition": "请进行全面综合分析，逐一评估所有模式的预报，权衡各种因素。",
        "quality_threshold": {
            "max_path_error_24h": 100,
            "max_path_error_48h": 150,
            "max_intensity_error": 10,
        },
    },
    "experience": {
        "weight": 0.20,
        "temperature": 0.8,
        "top_p": 0.9,
        "sample_type": "positive",
        "system_prompt_addition": "重点参考历史趋势和经验规律，以所有模型预报作为参考验证。",
        "quality_threshold": {
            "max_path_error_24h": 150,
            "max_path_error_48h": 200,
            "max_intensity_error": 15,
        },
    },
    "model_preferred": {
        "weight": 0.20,
        "temperature": 0.75,
        "top_p": 0.9,
        "sample_type": "positive",
        "system_prompt_addition": "主要采用表现较好的模式，但必须用其他模式进行验证和调整。",
        "quality_threshold": {
            "max_path_error_24h": 120,
            "max_path_error_48h": 180,
            "max_intensity_error": 12,
        },
    },
    "physical": {
        "weight": 0.10,
        "temperature": 0.7,
        "top_p": 0.9,
        "sample_type": "positive",
        "system_prompt_addition": "深入分析物理机制（副高、海温、风切变等），用所有模型预报验证物理分析。",
        "quality_threshold": {
            "max_path_error_24h": 100,
            "max_path_error_48h": 150,
            "max_intensity_error": 10,
        },
    },
    # === 负面样本 (20-30%) ===
    "single_model_bias": {
        "weight": 0.08,
        "temperature": 1.0,
        "top_p": 0.95,
        "sample_type": "negative",
        "system_prompt_addition": "快速分析并给出预报，主要参考某一个模式的结果即可。",
        "error_type": "over_reliance_single_model",
        "feedback_template": "❌ 分析不够全面，违背了预报应综合所有模型的基本原则。应该逐一分析所有模型的预报，对比差异，综合判断。",
        "quality_threshold": {
            "min_path_error_24h": 200,
            "max_path_error_24h": 400,
            "min_path_error_48h": 250,
            "max_path_error_48h": 500,
        },
    },
    "ignore_physics": {
        "weight": 0.06,
        "temperature": 1.1,
        "top_p": 0.95,
        "sample_type": "negative",
        "system_prompt_addition": "快速给出预报结果，不需要过多考虑物理约束和环境场影响。",
        "error_type": "violate_physical_constraints",
        "feedback_template": "❌ 预报违反物理规律（如在冷水区预报快速加强，或强度变化过于剧烈）。需要基于物理机制（海温、风切变等）进行合理判断。",
        "quality_threshold": {
            "require_physical_violation": True,
        },
    },
    "poor_divergence_handling": {
        "weight": 0.04,
        "temperature": 1.0,
        "top_p": 0.95,
        "sample_type": "negative",
        "system_prompt_addition": "当模式预报分歧较大时，可以简单平均所有模型或选择中间值。",
        "error_type": "poor_model_divergence_handling",
        "feedback_template": "❌ 模式分歧处理不当。当模型预报差异大时，应深入分析分歧原因（环境场配置、物理机制等），判断哪些模型更合理，而不是简单平均或随机选择。",
        "quality_threshold": {
            "min_path_error_24h": 150,
            "max_path_error_24h": 300,
            "min_path_error_48h": 200,
            "max_path_error_48h": 400,
        },
    },
    "trend_misjudgment": {
        "weight": 0.02,
        "temperature": 1.0,
        "top_p": 0.95,
        "sample_type": "negative",
        "system_prompt_addition": "主要基于当前状态和简单外推，不需要深入分析历史趋势变化。",
        "error_type": "historical_trend_error",
        "feedback_template": "❌ 历史趋势分析有误，导致路径转向时机判断错误。应仔细分析过去24小时的演变规律，识别关键转折点。",
        "quality_threshold": {
            "min_path_error_24h": 150,
            "max_path_error_24h": 300,
        },
    },
}


def load_env(path: Path) -> Dict[str, str]:
    """Parse a minimal .env file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"找不到环境变量文件: {path}")
    variables: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            variables[key.strip()] = value.strip().strip('"').strip("'")
    return variables


def format_lat(lat: Optional[float]) -> str:
    if lat is None:
        return "未知"
    hemisphere = "N" if lat >= 0 else "S"
    return f"{abs(lat):.1f}°{hemisphere}"


def format_lon(lon: Optional[float]) -> str:
    if lon is None:
        return "未知"
    hemisphere = "E" if lon >= 0 else "W"
    return f"{abs(lon):.1f}°{hemisphere}"


def format_observation(point: Dict[str, object]) -> str:
    stamp = point.get("datetime")
    if isinstance(stamp, str):
        timestamp = stamp
    elif isinstance(stamp, datetime):
        timestamp = stamp.strftime(ISO_TIME_FORMAT)
    else:
        timestamp = "未知时间"
    lat = format_lat(_to_float(point.get("latitude")))
    lon = format_lon(_to_float(point.get("longitude")))
    wind = point.get("max_wind_wmo") or point.get("max_wind_usa")
    pressure = point.get("min_pressure_wmo") or point.get("min_pressure_usa")
    wind_txt = f"{wind:.0f} kt" if isinstance(wind, (int, float)) else "未知"
    pres_txt = f"{pressure:.0f} hPa" if isinstance(pressure, (int, float)) else "未知"
    speed = point.get("storm_speed")
    direction = point.get("storm_direction")
    motion_txt = "未知"
    if isinstance(speed, (int, float)) and isinstance(direction, (int, float)):
        motion_txt = f"{speed:.0f} kt / {direction:.0f}°"
    distance = point.get("distance_to_land")
    distance_txt = f"{distance:.0f} km" if isinstance(distance, (int, float)) else "未知"
    return (
        f"{timestamp} | 位置: {lat}, {lon} | 风速: {wind_txt} | 气压: {pres_txt} | "
        f"移动: {motion_txt} | 距离陆地: {distance_txt}"
    )


def _to_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_forecast_point(point: Dict[str, object]) -> str:
    stamp = point.get("datetime") or point.get("time")
    if isinstance(stamp, datetime):
        timestamp = stamp.strftime(ISO_TIME_FORMAT)
    else:
        timestamp = str(stamp)
    lat = format_lat(_to_float(point.get("lat")))
    lon = format_lon(_to_float(point.get("lon")))
    wind = point.get("wind")
    wind_txt = f"{wind:.1f} m/s" if isinstance(wind, (int, float)) else "未知"
    msl = point.get("msl")
    if isinstance(msl, (int, float)):
        pressure_txt = f"{msl/100:.1f} hPa"
    else:
        pressure_txt = "未知"
    return f"{timestamp} | 位置: {lat}, {lon} | 风速: {wind_txt} | 海平面气压: {pressure_txt}"


def _parse_any_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    adjusted = text
    if adjusted.endswith("Z"):
        adjusted = adjusted[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(adjusted)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _normalise_time_key(value: object) -> Optional[str]:
    dt = _parse_any_datetime(value)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M")


def _json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _system_label(system_name: object) -> str:
    name = str(system_name) if system_name is not None else ""
    return SYSTEM_NAME_LABELS.get(name, name or "未知系统")


def _format_environment_system(system: Dict[str, object], indent: str = "  ") -> List[str]:
    lines: List[str] = []
    name = system.get("system_name")
    lines.append(f"{indent}- 系统: {_system_label(name)} ({name})")
    description = system.get("description")
    if isinstance(description, str) and description.strip():
        lines.append(f"{indent}  描述: {description.strip()}")
    intensity = system.get("intensity")
    if intensity:
        lines.append(f"{indent}  强度: {_json_dumps(intensity)}")
    position = system.get("position")
    if position:
        lines.append(f"{indent}  位置: {_json_dumps(position)}")
    properties = system.get("properties")
    if properties:
        lines.append(f"{indent}  属性: {_json_dumps(properties)}")
    shape = system.get("shape")
    if shape:
        lines.append(f"{indent}  形态: {_json_dumps(shape)}")
    extras = {
        key: value
        for key, value in system.items()
        if key
        not in {
            "system_name",
            "description",
            "intensity",
            "position",
            "properties",
            "shape",
        }
    }
    if extras:
        lines.append(f"{indent}  其他: {_json_dumps(extras)}")
    return lines


def _format_environment_entry_detail(entry: Dict[str, object]) -> List[str]:
    time_text = entry.get("time", "未知时间")
    summary = entry.get("summary", "")
    header = f"{time_text}: {summary}" if summary else f"{time_text}:"
    lines: List[str] = [header]
    systems = entry.get("systems")
    if isinstance(systems, list) and systems:
        for system in systems:
            if isinstance(system, dict):
                lines.extend(_format_environment_system(system, indent="  "))
    else:
        lines.append("  - 无环境系统信息")
    return lines


def format_model_forecast_section(forecast: Dict[str, object]) -> str:
    model_info = forecast.get("model") or {}
    name = model_info.get("name", "未知模式")
    version = model_info.get("version", "")
    source = model_info.get("source", "")
    cycle = model_info.get("cycle_hours")
    model_desc = f"{name} {version} ({source})".strip()
    if isinstance(cycle, (int, float)):
        model_desc = f"{model_desc} / Δt {int(cycle)}h"

    particle_id = forecast.get("particle_id") or "未知"

    track = forecast.get("track") or {}
    track_points = track.get("points") or []

    env = forecast.get("environment") or {}
    initial_env = env.get("initial_summary")
    env_timeline = env.get("timeline") or env.get("forecast_summaries") or []
    env_timeline_text = env.get("timeline_text")

    alignment_entries = forecast.get("track_environment_alignment") or []
    alignment_map: Dict[str, Dict[str, object]] = {}
    for entry in alignment_entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("time")
        norm_key = _normalise_time_key(key) or (str(key) if key else None)
        if not norm_key:
            continue
        alignment_map[norm_key] = entry

    env_map: Dict[str, str] = {}
    env_system_map: Dict[str, List[Dict[str, object]]] = {}
    for entry in env_timeline:
        if not isinstance(entry, dict):
            continue
        key = _normalise_time_key(entry.get("time"))
        if not key:
            continue
        summary = entry.get("summary")
        if isinstance(summary, str) and summary.strip():
            env_map.setdefault(key, summary.strip())
        systems = entry.get("systems")
        if isinstance(systems, list):
            env_system_map[key] = systems

    section_lines: List[str] = [
        f"模式: {model_desc}",
        f"粒子ID: {particle_id}",
    ]

    if isinstance(initial_env, str) and initial_env.strip():
        section_lines.append(f"模式起报环境: {initial_env.strip()}")

    section_lines.append("轨迹预报：")
    if track_points:
        for point in track_points:
            base_line = f"  - {format_forecast_point(point)}"
            time_key = _normalise_time_key(point.get("datetime") or point.get("time"))
            env_summary = env_map.get(time_key) if time_key else None
            alignment_entry = alignment_map.get(time_key) if time_key else None
            if alignment_entry:
                env_summary = alignment_entry.get("environment_summary") or env_summary
            system_labels: List[str] = []
            systems = None
            if alignment_entry:
                systems = alignment_entry.get("environment_systems")
            if systems is None and time_key:
                systems = env_system_map.get(time_key)
            if isinstance(systems, list):
                for system in systems:
                    if isinstance(system, dict):
                        system_labels.append(_system_label(system.get("system_name")))
            if isinstance(env_summary, str) and env_summary.strip():
                base_line += f" | 环境: {env_summary.strip()}"
            if system_labels:
                base_line += " | 环境系统: " + ", ".join(system_labels)
            section_lines.append(base_line)
    else:
        section_lines.append("  - 缺少轨迹点。")

    section_lines.append("模式环境场摘要：")
    if env_timeline:
        for entry in env_timeline:
            if not isinstance(entry, dict):
                continue
            time_text = entry.get("time", "未知时间")
            summary_text = entry.get("summary", "")
            section_lines.append(f"  - {time_text}: {summary_text}")
    elif isinstance(env_timeline_text, str) and env_timeline_text.strip():
        section_lines.append(f"  - {env_timeline_text.strip()}")
    else:
        section_lines.append("  - 未提供环境摘要。")

    section_lines.append("模式环境系统详解：")
    if env_timeline:
        for entry in env_timeline:
            if not isinstance(entry, dict):
                continue
            for line in _format_environment_entry_detail(entry):
                section_lines.append(f"  {line}")
    elif isinstance(env_timeline_text, str) and env_timeline_text.strip():
        section_lines.append(f"  {env_timeline_text.strip()}")
    else:
        section_lines.append("  无可用环境系统信息。")

    return "\n".join(section_lines)


def build_prompt(sample: Dict[str, object]) -> str:
    history = sample.get("history") or []
    environment_info = sample.get("environment") or {}
    model_forecasts = sample.get("model_forecasts") or []

    history_txt = "\n".join(format_observation(point) for point in history) or "无可用观测。"

    real_env = (
        environment_info.get("analysis_text")
        or environment_info.get("analysis_summary")
        or "环境信息缺失。"
    )

    if model_forecasts:
        forecast_sections = [format_model_forecast_section(entry) for entry in model_forecasts]
        forecast_txt = "\n\n".join(forecast_sections)
    else:
        forecast_txt = "模式集合数据缺失。"

    storm_id = sample.get("storm_id", "未知编号")
    storm_name = sample.get("storm_name") or ""
    init_time = sample.get("init_time", "未知起报时间")

    instructions = textwrap.dedent(
        """\
        任务要求：
        1. 先在<think></think>标签内完成详细推理，覆盖：形势分析、历史趋势、模式对比、环境演变、综合判断、不确定性六个环节。
        2. 模式对比环节需逐一评估所有给定模式的路径、强度与环境差异，解释偏差来源与可信度。
        3. 检查各模式预报的气旋与环境系统交互是否符合热带气旋动力学原理（引导气流、能量供给、垂直风切变、外部系统等），指出合理与可能失衡之处。
        4. 推理过程中仅使用起报时刻可获取的信息（历史观测、当前环境、模式预报），严禁引用未来真实演变或真值数据。
        5. 给出未来24/48/72小时的路径（纬度、经度）、最大风速（m/s）和最低气压（hPa）预报。
        6. 明确阐述预报结论的依据，并指出至少两个主要不确定性来源。
        7. 将“最终预报”与“不确定性”小节置于<think></think>标签之外，并保持结构化列点。"""
    )

    sections = [
        f"台风编号: {storm_id} {storm_name}",
        f"预报起报时间: {init_time}",
        "",
        "【历史观测（过去24小时）】",
        history_txt,
        "",
        "【模式预报路径（各模式代表轨迹）】",
        forecast_txt,
        "",
        "【当前真实环境场分析】",
        real_env,
        "",
        instructions,
    ]
    return "\n".join(sections)


def build_prompt_with_strategy(sample: Dict[str, object], strategy_config: Dict[str, object]) -> str:
    """Inject strategy-specific guidance into the base prompt."""
    base_prompt = build_prompt(sample)
    addition = str(strategy_config.get("system_prompt_addition") or "").strip()
    if not addition:
        return base_prompt
    strategy_header = f"【分析策略】\n{addition}\n\n"
    if "任务要求：" in base_prompt:
        return base_prompt.replace("任务要求：", strategy_header + "任务要求：", 1)
    return base_prompt + "\n\n" + strategy_header


SYSTEM_INSTRUCTION = textwrap.dedent(
    """\
    输出结构化、可执行的预报建议。
    请遵守以下原则：
    - 所有详细推理必须置于<think></think>标签内，先完成思维链再输出最终答案。
    - 严格遵循用户提示中的六步推理框架，并检验模式预报与环境系统的相互作用是否符合热带气旋动力学。
    - 量化描述必须使用国际单位：风速 m/s、气压 hPa、距离 km。
    - 最终预报部分以“最终预报”小节罗列24/48/72小时的预报条目。
    - “不确定性”小节列出至少两条关键风险或情景假设。
    - 输出使用中文。"""
)


def select_generation_strategies(num_samples: int) -> List[str]:
    if num_samples <= 0:
        return []
    strategy_names = list(GENERATION_STRATEGIES.keys())
    if not strategy_names:
        return []
    weights = [float(GENERATION_STRATEGIES[name].get("weight", 1.0)) for name in strategy_names]
    return random.choices(strategy_names, weights=weights, k=num_samples)


@dataclass
class OpenRouterClient:
    api_key: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 3
    retry_delay: float = 2.0
    referer: Optional[str] = None
    site_title: Optional[str] = None

    def __post_init__(self) -> None:
        self.client = OpenAI(base_url=OPENROUTER_API_BASE, api_key=self.api_key)
        headers: Dict[str, str] = {}
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.site_title:
            headers["X-Title"] = self.site_title
        self._extra_headers = headers or None
        self._extra_body: Dict[str, object] = {}

    def chat(
        self,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ]

        attempt = 0
        while True:
            attempt += 1
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=top_p if top_p is not None else self.top_p,
                    extra_headers=self._extra_headers,
                    extra_body=self._extra_body,
                )
                choices = completion.choices or []
                if not choices:
                    raise RuntimeError("OpenRouter返回数据缺少choices字段")
                content = choices[0].message.content
                if not content:
                    raise RuntimeError("OpenRouter返回数据缺少message内容")
                return content
            except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
                logger.warning("调用OpenRouter失败 (%s/%s): %s", attempt, self.max_retries, exc)
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)


def read_samples(path: Path, limit: int) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(f"未找到匹配样本文件: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_dataset_record(
    sample: Dict[str, object],
    response: str,
    instruction: str,
    user_prompt: str,
    *,
    strategy_name: str,
    strategy_config: Dict[str, object],
    generation_index: int,
    temperature: float,
    top_p: float,
) -> Dict[str, object]:
    models: List[Dict[str, object]] = []
    for forecast in sample.get("model_forecasts") or []:
        if not isinstance(forecast, dict):
            continue
        model_info = forecast.get("model")
        if isinstance(model_info, dict):
            models.append(model_info)
    strategy_meta: Dict[str, object] = {
        "name": strategy_name,
        "sample_type": strategy_config.get("sample_type", "unknown"),
        "temperature": temperature,
        "top_p": top_p,
        "generation_index": generation_index,
        "system_guidance": strategy_config.get("system_prompt_addition"),
        "quality_threshold": strategy_config.get("quality_threshold"),
    }
    if "error_type" in strategy_config:
        strategy_meta["error_type"] = strategy_config["error_type"]
    if "feedback_template" in strategy_config:
        strategy_meta["feedback_template"] = strategy_config["feedback_template"]
    metadata = {
        "storm_id": sample.get("storm_id"),
        "storm_name": sample.get("storm_name"),
        "init_time": sample.get("init_time"),
        "model_count": len(models),
        "models": models,
        "strategy": strategy_meta,
    }
    return {
        "instruction": instruction,
        "input": user_prompt,
        "output": response,
        "metadata": metadata,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用OpenRouter生成气旋预报样本。")
    parser.add_argument(
        "--matched-file",
        type=Path,
        default=Path("preprocessed_data") / "matched_samples.jsonl",
        help="预处理后的匹配样本路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_data") / "forecast_decisions.jsonl",
        help="生成的预报数据集输出路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="从匹配样本中选择的样本数量",
    )
    parser.add_argument(
        "--samples-per-forecast",
        type=int,
        default=10,
        help="为每个预报时刻生成的样本数量",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help=".env 文件路径（包含 OPENROUTER_API_KEY）",
    )
    parser.add_argument(
        "--api-key-var",
        default="OPENROUTER_API_KEY",
        help="从 .env 中读取的API密钥变量名",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.5-flash",
        help="OpenRouter模型名称",
    )
    parser.add_argument(
        "--referer",
        default=None,
        help="可选：HTTP Referer，用于OpenRouter统计",
    )
    parser.add_argument(
        "--site-title",
        default=None,
        help="可选：站点标题，用于OpenRouter统计",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="top-p 截断",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="增加日志输出（-v 或 -vv）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="跳过实际API调用，输出占位文本（调试用）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于复现策略抽样结果）",
    )
    return parser.parse_args(argv)


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if args.seed is not None:
        random.seed(args.seed)

    samples = read_samples(args.matched_file, args.limit)
    if not samples:
        logger.error("未能从 %s 中读取任何样本", args.matched_file)
        sys.exit(1)

    instruction = "请根据观测数据和数值模式预报进行详细分析并给出预报。"

    ensure_parent_dir(args.output)

    if args.dry_run:
        logger.warning("启用 --dry-run，所有输出将使用占位文本。")
        client = None  # type: ignore[assignment]
    else:
        env_vars = load_env(args.env_file)
        api_key = env_vars.get(args.api_key_var)
        if not api_key:
            logger.error("在 %s 中找不到变量 %s", args.env_file, args.api_key_var)
            sys.exit(2)
        client = OpenRouterClient(
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            referer=args.referer,
            site_title=args.site_title,
        )

    total_records = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for sample_index, sample in enumerate(samples, 1):
            strategy_names = select_generation_strategies(args.samples_per_forecast)
            if not strategy_names:
                logger.warning("未选择到任何策略，跳过样本 %s", sample_index)
                continue
            for generation_index, strategy_name in enumerate(strategy_names, 1):
                strategy_config = GENERATION_STRATEGIES.get(strategy_name)
                if not strategy_config:
                    logger.warning(
                        "策略 %s 未在配置中找到，跳过。", strategy_name
                    )
                    continue
                user_prompt = build_prompt_with_strategy(sample, strategy_config)
                strategy_temperature = float(strategy_config.get("temperature", args.temperature))
                strategy_top_p = float(strategy_config.get("top_p", args.top_p))
                if args.dry_run:
                    response = (
                        f"[DRY-RUN 占位响应] 样本 {sample_index} - 策略 {strategy_name} #{generation_index}"
                    )
                else:
                    try:
                        response = client.chat(
                            user_prompt,
                            temperature=strategy_temperature,
                            top_p=strategy_top_p,
                        )
                    except APIError as exc:
                        logger.exception(
                            "生成样本 %s (策略 %s) 失败，API返回错误: %s",
                            sample_index,
                            strategy_name,
                            exc,
                        )
                        raise SystemExit(3) from exc
                    except Exception as exc:
                        logger.exception(
                            "生成样本 %s (策略 %s) 失败，发生意外错误: %s",
                            sample_index,
                            strategy_name,
                            exc,
                        )
                        raise SystemExit(3) from exc

                record = build_dataset_record(
                    sample,
                    response,
                    instruction,
                    user_prompt,
                    strategy_name=strategy_name,
                    strategy_config=strategy_config,
                    generation_index=generation_index,
                    temperature=strategy_temperature,
                    top_p=strategy_top_p,
                )
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1
            logger.info(
                "完成样本 %s/%s，累计生成 %s 条记录",
                sample_index,
                len(samples),
                total_records,
            )


if __name__ == "__main__":
    main()
