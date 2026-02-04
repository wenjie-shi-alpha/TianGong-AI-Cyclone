# 污染天气系统算法服务（本地部署）

本服务将 `docs/pollution_systems/code` 中的污染天气系统算法封装为 **本地 HTTP 服务**，支持本机或局域网访问，便于与其他系统对接。

## 1. 功能概览

- 提供 **高压控制、锋面/切变线、西风槽、低层风场、稳定度** 等算法的 API 调用
- 支持 **本机/局域网访问**（默认绑定 `0.0.0.0`）
- 支持 **可选 API Key** 鉴权
- 以 **JSON** 方式传入格点场数据或点值

## 2. 目录结构

```
deploy/pollution_service/
├── app.py                # FastAPI 服务
├── config.env            # 配置
├── requirements.txt      # 依赖
├── start_server.sh       # 启动（前台/后台）
├── stop_server.sh        # 停止
├── health_check.sh       # 健康检查
├── test_request.sh       # 请求示例
└── logs/                 # 日志
```

## 3. 安装依赖

```bash
pip install -r deploy/pollution_service/requirements.txt
```

## 4. 配置说明

编辑 `deploy/pollution_service/config.env`：

- `HOST`：服务监听地址（默认 `0.0.0.0`，支持局域网访问）
- `PORT`：服务端口（默认 9010）
- `API_KEY`：可选，设置后请求需携带 `Authorization: Bearer <API_KEY>`

## 5. 启动服务

前台启动（调试）：
```bash
bash deploy/pollution_service/start_server.sh --fg
```

后台启动（常用）：
```bash
bash deploy/pollution_service/start_server.sh
```

启动成功后会输出访问地址，例如：
```
http://127.0.0.1:9010
```

## 6. 健康检查

```bash
bash deploy/pollution_service/health_check.sh
```

## 7. 发送测试请求

```bash
bash deploy/pollution_service/test_request.sh
```

## 8. API 说明（简要）

- `GET /health`：服务健康检查
- `POST /high-pressure`：高压控制识别
- `POST /frontal`：锋面/切变线识别
- `POST /westerly-trough`：西风槽识别
- `POST /low-level-flow`：低层风场评估
- `POST /stability`：近地层稳定度评估

请求与响应为 JSON，字段说明参考 `docs/pollution_systems/algorithms.md`。

## 9. 常见问题

- **外部访问失败**：检查防火墙是否放行 `PORT`，并确认 `HOST=0.0.0.0`。
- **数据量过大**：建议在业务侧先裁剪区域或做降采样。
- **算法返回为空**：说明该系统未被触发（阈值不满足或场特征不足）。
