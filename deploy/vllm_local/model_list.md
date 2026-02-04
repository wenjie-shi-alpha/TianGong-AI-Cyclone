# 预置模型列表（ModelScope）

以下为**初步配置**的 ModelScope 模型 ID，可在 `config.env` 中切换 `MODEL_ID` 使用。

| 模型 | ModelScope 模型 ID | 量化/特性 | 备注 |
|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 70B 规模 | 需要多 GPU 或大显存 |
| Qwen3-32B-unsloth-bnb-4bit | `unsloth/Qwen3-32B-unsloth-bnb-4bit` | bitsandbytes 4bit | 若 ModelScope 仓库名不同请按官网替换 |
| Qwen3-VL-32B-Instruct-AWQ-4bit | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | AWQ 4bit，多模态 | 可能需要 `--trust-remote-code` |
