# 本地部署大语言模型（vLLM + ModelScope）

本目录提供**本地部署方案**，使用 **vLLM** 作为推理框架，模型从 **ModelScope** 下载并启动为服务，提供 OpenAI 兼容接口访问链接。

## 1. 技术路线（总体流程）

1. **ModelScope 下载模型**：使用 `modelscope` SDK 拉取模型到本地缓存或指定目录。  
2. **vLLM 加载模型**：vLLM 负责高性能推理与显存管理。  
3. **服务化暴露接口**：通过 `vllm serve` 启动 OpenAI 兼容服务，提供 `/v1/models`、`/v1/chat/completions` 等接口。  
4. **客户端访问**：使用 curl 或任何 OpenAI 兼容 SDK 访问。  

## 2. 可实现的功能

- **本地私有化部署**：模型权重与推理过程完全本地化。  
- **服务化访问**：以 API 形式对外提供推理能力。  
- **多模型切换**：通过修改 `MODEL_ID` 快速切换。  
- **支持量化模型**：例如 AWQ / 4bit 版本模型。  
- **可选鉴权**：设置 `API_KEY` 进行访问控制。  
- **可扩展配置**：通过 `EXTRA_ARGS` 传入 vLLM 额外参数（如 `--trust-remote-code`）。  

## 3. 目录结构

```
deploy/vllm_local/
├── config.env              # 配置文件
├── model_list.md           # 预置模型列表（ModelScope）
├── download_model.sh       # 模型下载（ModelScope）
├── start_server.sh         # 启动服务
├── stop_server.sh          # 停止服务
├── health_check.sh         # 健康检查
├── test_request.sh         # 测试请求
└── logs/                   # 日志目录
```

## 4. 环境准备

建议使用 Python 3.10+，GPU 环境效果最佳。

```bash
pip install vllm modelscope
```

如需下载私有/受限模型，请先登录 ModelScope（或在网页端完成授权）。

## 5. 配置说明

编辑 `config.env`：

- `MODEL_SOURCE=modelscope`
- `MODEL_ID`：ModelScope 模型 ID（见 `model_list.md`）
- `MODEL_DIR`：可选，模型下载到本地的绝对/相对目录
- `SERVED_MODEL_NAME`：可选，接口返回模型名
- `HOST` / `PORT`：服务监听地址
- `API_KEY`：可选，开启鉴权
- `EXTRA_ARGS`：额外 vLLM 参数（例如 `--trust-remote-code`）

## 6. 模型下载（ModelScope）

```bash
bash deploy/vllm_local/download_model.sh
```

- 如果设置了 `MODEL_DIR`，模型文件会下载到该目录。
- 否则使用 ModelScope 默认缓存路径。

## 7. 启动服务

前台启动（调试）：
```bash
bash deploy/vllm_local/start_server.sh --fg
```

后台启动（服务化）：
```bash
bash deploy/vllm_local/start_server.sh
```

启动后脚本会输出访问链接，例如：
```
http://127.0.0.1:8000/v1
```

## 8. 健康检查

```bash
bash deploy/vllm_local/health_check.sh
```

## 9. 发送测试请求

```bash
bash deploy/vllm_local/test_request.sh
```

## 10. 停止服务

```bash
bash deploy/vllm_local/stop_server.sh
```

## 11. 常见问题

- **显存不足**：换更小模型或使用 4bit / AWQ 版本。
- **模型无法加载**：尝试 `EXTRA_ARGS=--trust-remote-code`。
- **访问失败**：确认 `HOST` / `PORT` 与防火墙设置。

