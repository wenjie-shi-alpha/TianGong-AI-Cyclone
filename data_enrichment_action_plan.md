# 飓风气象大模型（CycloneCopilot）数据补充与认知增强提升行动计划

## 1. 项目背景与当前数据集的痛点

### 1.1 项目背景
目前我们正在使用预训练大语言模型（如 Qwen3 等），通过监督微调 (SFT) 和组策略优化 (GRPO) 训练一个“气象专家副手”模型。该模型的目标是：**根据当前时刻获取的各项气象参考数据，自动生成高度类似于 NOAA (美国国家飓风中心) 前线预报员所撰写的“官方预报讨论 (Forecast Discussion)”。**
这要求模型不仅能预测路径和强度，还必须在文本中展现出清晰的**逻辑推导链 (CoT, Chain of Thought)**，解释*为什么*做出这个预测以及*模型预报之间的差异在哪*。

### 1.2 当前数据集的构成与局限
在现阶段的 Pipeline 中，大模型的上下文 (`context`) 仅包含：
1. **Best Track (历史最佳路径)**：原始的时间、经纬度、风速、气压等基本状态。
2. **Model Runs (模式预测)**：单条或数条路径（如 GFS、ECMWF），以及沿途插值的底层气象要素。
3. **CDS Reanalysis (起报环境)**：利用图像算法从 ERA5 中抽取的抽象大尺度天气系统（如副高、季风槽的方位）。

### 1.3 核心痛点：信息输入层级的巨大“断层”
通过对海量 NOAA 历史讨论（如 2022 年飓风 IAN, 2023 年飓风 IDALIA）的 Review，我们发现真实的专家推理依赖于**高阶诊断层**数据，而目前的模型处于“盲猜”状态：
- **诊断级环境指标缺失**：专家依赖“极高海洋热含量 (OHC)”、“强垂直风切变 (VWS)”、“干空气侵入 (Dry Air)”等明确计算结果，而模型只能通过底层风压场进行困难的自我推导。
- **模式离散度 (Spread) 盲区**：专家根据“几十个模式预测的集中度”来决定预报的置信区间和语气，而模型目前仅看到一两个孤立模型的单条轨迹。
- **客观观测事实断层**：专家开篇总会引用极轨/静止卫星的特征（最低云顶亮温、微波结构的完整度、SAR 风场极值），而模型毫无此类输入。

---

## 2. 核心补强数据源详规：数据是否足够？它在哪？怎么用？

针对上述痛点，我们需要引入云端公开数据集。**目前的规划包含了 AWS 开放数据计划 (Registry of Open Data on AWS) 与 Copernicus 数据空间生态 (Copernicus Data Space Ecosystem, CDS)。从解决专家推理逻辑树的角度来看，这四类数据已经构成了极具物理意义和逻辑闭环的“充分条件”，足以支撑达到 SOTA (Nature-grade) 级别的专家推理基准。**

以下是具体数据源的详细坐标与使用指南：

### 2.0 本项目现有 NetCDF 提取与信息增益链路（必须对齐）
- 原有数据作为基线使用，不再重算/清洗/重写；本计划只新增数据并进行时空对齐与特征增量注入。
- 数据入口与流程: `generate_nc_urls.py` 生成 `output/nc_file_urls.csv` → `src/environment_extractor/cli.py`（`streaming_from_csv` 或 `--auto`）执行“下载→追踪→环境分析→清理”；`src/wb2_hres_pipeline.py` 可从 WeatherBench2 HRES Zarr 直接生成精简 NetCDF。
- 追踪所需变量 (initial_tracker): `msl/mslp`、`u10/10u`、`v10/10v`（用于最低气压定位与近地面风场结构判断）。
- 环境诊断所需变量 (environment_extractor): `z`(500hPa)、`u/v`(200/300/500/700/850/925hPa)、`t`(1000/850/500hPa)、`w`(700hPa，可缺省降级)、`sst/ts/t2/t2m`、`msl`。
- 变量名对齐: 参考 `src/wb2_hres_pipeline.py::_adapt_variables` 与 `src/initial_tracker/dataset_adapter.py` 的变量映射，确保进入 `TCEnvironmentalSystemsExtractor` 的变量名统一。
- 现有 NetCDF 输出: 轨迹 `track_single/*.csv` + 环境诊断 `final_single_output/*_TC_Analysis_*.json`。
- 多模式轨迹筛选逻辑: `prepare_forecast_samples.py` 按 particle 分组，选择与真值均距最小的轨迹作为每个模型代表，后续用于多模式对比与离散度评估。
- 信息增益压缩链路: `src/prepare_forecast_samples.py` 将环境 JSON 摘要化并与轨迹对齐；`src/generate_forecast_dataset.py` 组装 Prompt；`spec.md` 明确“可信字段优先、几何面积与距离数值剔除”的过滤策略。

### 2.0.1 NetCDF 系统级提取清单（现有实现）
- 副热带高压/引导气流: 输入 `z500` + `u/v(850-300)`；算法 区域高值识别+层平均/地转风；输出 质心、强度(gpm)、引导风矢量、边界坐标。
- 垂直风切变: 输入 `u/v`(200,850)；算法 500km 圆域平均矢量差；输出 强度、方向、影响级别。
- 海洋热含量: 输入 `sst/ts/t2/t2m`；算法 2°圆域平均 + 26.5°C 边界与暖涡特征；输出 SST等级与暖水边界。
- 高空辐散: 输入 `u/v`(200)；算法 散度计算+500km 平均；输出 平均辐散与最大辐散中心偏移。
- ITCZ: 输入 `u/v`(850) + `w`(700 可选)；算法 低层辐合带定位；输出 位置、强度、距离与影响级别。
- 西风槽: 输入 `z500` + `u/v`(500) + `u`(200 可选)；算法 负距平槽轴提取；输出 槽轴、槽底位置、强度与相对方位。
- 锋面系统: 输入 `t`(1000/850/500) + `u/v`(925 可选)；算法 厚度梯度+锋生指数；输出 锋面类型、位置与强度。
- 季风槽: 输入 `u/v`(850) + `msl`(可选)；算法 低层涡度阈值+槽轴长度；输出 槽底位置、涡度强度与影响级别。

### 2.0.2 时间尺度与时空对齐规则（新增数据必须遵循）
- 真实 Best Track: 3 小时分辨率（`input/western_pacific_typhoons_superfast.csv`）。
- 模式轨迹: 6 小时分辨率（`track_single/*.csv`）；环境系统输出按 `time_idx` 与轨迹时间一致。
- 现有对齐基线: `prepare_forecast_samples.py::_align_track_and_environment` 允许 ±3 小时窗口匹配环境与轨迹时间点。
- 新数据时间对齐: 以轨迹时间点为锚（T0/T0+6h/...），高频数据取最近时次或在窗口内聚合；低频数据取最近可用时次并记录时间偏差。
- 新数据空间对齐: 以轨迹点中心 (lat, lon) 作为参考，按物理含义做半径统计或邻域提取；半径优先与现有指标保持一致（如切变/辐散 500km、SST 2°圆域等）。
- 对齐输出要求: 每条新增特征必须带 `source_time`、`target_time`、`delta_t_hours`、`spatial_window`（半径/区域）与 `coverage_flag`，确保可追踪与可解释。

### 2.1 TC PRIMED 档案 (AWS Open Data)
- **解决痛点**：填补“环境诊断标量 (SHIPS)”空白。
- **数据位置 (Location)**：AWS `us-east-1` 区域的 S3 存储桶 `s3://noaa-nesdis-tcprimed-pds/`。完全匿名公开。
- **具体包含与作用**：提供 SHIPS 模型计算出的风切变 (`shrd`)、中低层湿度 (`rhlo`/`rhhi`)、海洋热含量 (`ohc`)、200hPa 散度 (`d200`) 等。专家每次讨论强度变化必引用这些指标（例如设定因果关系：“因为强风切变，所以无法快速增强”）。
- **如何获取与使用 (How to use)**：
  - **工具**：Python 库 `tcprimedapi` 或 `s3fs` + `xarray`。
  - **使用方式**：不全量下载！风暴的卫星过境文件 (`overpass`) 极大，但环境文件 (`env.nc`，每风暴约 70MB) 很小。直接使用 `xarray` 结合 `s3fs` 匿名挂载读取，通过 `storm_id` 和时间戳检索出对应的环境变量标量，组装成缓存 JSON。

### 2.2 NOAA ATCF 历史极简 FTP 数据集 (a-deck)
- **解决痛点**：提供多模式离散度 (Spread) 与综合置信水平评估。
- **数据位置 (Location)**：NOAA 官方对外 FTP/HTTP 服务 (例如 `https://ftp.nhc.noaa.gov/atcf/archive/`)。
- **具体包含与作用**：包含了历史上所有起报时刻、10-30 个不同预报模型（动力+统计+共识）的完整预测轨迹。计算不同模型在 48h/72h 的经纬度方差，直接输出“处于高度一致”或“分歧巨大（High Uncertainty）”。这直接指导模型 CoT 中的决策取舍语气。
- **如何获取与使用 (How to use)**：
  - **工具**：`wget`，`pandas`。
  - **使用方式**：直接写 Bash 脚本将历史年份的压缩包（全部不到 1GB）拉取到任意服务器本地。使用 `pandas` 解析纯文本文件，分组计算 24/48/72 小时的路径与强度离散度指标。

### 2.3 GOES-16/18 卫星影像数据 (AWS Open Data / GEE)
- **解决痛点**：提供开篇必备的实时观测证据（深对流、云顶冷却速度）。
- **数据位置 (Location)**：AWS 存储桶 `s3://noaa-goes16/` / `s3://noaa-goes18/`。或者在 Google Earth Engine (GEE) 的 `NOAA/GOES/16/MCMIPF` 数据集中。
- **具体包含与作用**：红外波段（Band 13）的亮温数据。提取台风中心的最低亮温（Min BT）和冷云像素占比（Cold Cloud Fraction）。这是预测早期快速增强 (RI) 的视觉证据。
- **如何获取与使用 (How to use)**：
  - **由于图像海量，不能下载源文件。**
  - **GEE API 方案**：利用 Python `earthengine-api` 传入中心经纬度和时间，在 GEE 云端画一个 500km 半径的 Geometry，直接调用 `ee.Reducer.min()` 让云端算出最低温度标量值，只返回一个数字（几十 Bytes）到本地微调服务器。

### 2.4 Copernicus Data Space (CDS) 多源卫星观测
- **解决痛点**：穿透深云层提供大风风速和强风圈的绝对尺寸，以及提供精准的暖涡定位。
- **数据位置 (Location)**：
  - 可通过 `CDSE API` 访问检索。
  - 核心处理环境：CDS 官方提供的**免费带网云端 JupyterLab**。
- **具体包含与作用**：
  - **Sentinel-1 (SAR)**：C波段合成孔径雷达提供的极其稀缺的高清海面层真正风速估计（Ocean Wind Vector）。填补微波/飞机探测的空白，使得系统能报出精准的 34/50/64kt 风圈半径。
  - **Sentinel-3 (SLSTR/SRAL)**：海面高度异常 (SLA)。用于发现海洋表层下隐藏的“暖水涡流 (Warm Eddies)”。这是极其高级的专家指标，遇到它意味着极大可能爆发 RI。
- **如何获取与使用 (How to use)**：
  - **云端内网处理**：在 CDS 免费提供的 JupyterLab 中挂载挂载项目目录，环境自带 `cdsapi` 和 `sentinelsat` 且处于数据内网。
  - 编写时空匹配脚本，当台风轨迹在这个时间进入卫星条带时，利用平台算力提取风速极大值或 SLA 正异常阈值，计算完后将极小体积的标量特征 JSON (例：`{SAR_max_wind: 85, R34_radius: 110nm, warm_eddy_detected: true}`) 发回我们自己的训练机。

---

## 3. 将海量数据注入大模型：Prompt 降维改造行动方案

我们不能把上百 G 的 NetCDF 一股脑塞给模型。核心思想是**“降维压缩 + 对齐注入”**：原有数据保持不变，仅对新增数据做时空对齐与特征抽取；最终在 `prepare_forecast_samples.py` 与 `generate_forecast_dataset.py` 的组装阶段把新增特征作为增量注入到 Prompt。

### 3.1 改造后大模型输入（极简认知增强模块示例）

```markdown
[Current Satellite & Objective Observations]
Deep Convection: Intense (Cold cloud fraction: 52%, Min Cloud Top Temp: -81°C)
SAR/Microwave: Center well-defined, R34=110nm, Max Estimated SAR Wind=85kt

[Environmental Diagnostics (SHIPS & Altimetry)]
Vertical Wind Shear: 12 kt (Highly Favorable for intensification)
Mid-level Relative Humidity: 68% (Moist)
Ocean Heat Content & SEA: >100 kJ/cm2, Sentinel-3 detected positive SLA eddy.
Net Environmental Evaluation: Potent Rapid Intensification environment.

[Model Guidance & Spread]
Models Evaluated: GFS, ECMWF, HWRF, HAFS, UKMET.
Track Spread at 72H: 140 NM (Moderate uncertainty)
Intensity Spread at 48H: 35 kt (High uncertainty - Dynamical models indicate RI, statistical models lower)
```

**有了以上 200 个单词的精准注入，大模型结合其优秀的语言能力，自然而然就能在其 CoT 思维链中吐出与 NOAA 高级别专家一模一样的分析论断。**

### 3.2 信息增益与可信字段筛选（对齐 `spec.md`，仅用于新增特征）
- 可信字段优先: `system_name`、`position`、`intensity`、`properties.steering_flow`、边界坐标（仅用于形态描述）。
- 明确剔除: 面积/周长/长短轴/质心距离等几何数值，含具体面积或距离的描述文本转为定性表述。
- 摘要策略: 先系统级摘要（如“切变强/弱”“SST等级”），再多模式对比（Spread/一致性），最后加入外部观测证据（GOES/SAR/OHC），并保留对齐元数据（`delta_t_hours`、`spatial_window`）。

---

## 4. 三阶段落地执行计划 (Rollout Roadmap)

将这项工作系统性落地，应遵循“由轻到重、由文本到云图像”的路线：

### Phase 1: 新增数据对齐与轻量特征抽取 (Week 1)
- **Task 0: 新增数据对齐规范落地 (极简单)**：落实 `source_time/target_time/delta_t_hours/spatial_window/coverage_flag` 等元数据；原有数据不改动。
- **Task 1: ATCF 离散度提取 (极简单)**：编写 bash/python 脚本使用普通的 `urllib`/`wget` 从 NOAA FTP 拉取 `a-deck`。使用 pandas 对历史台风所有模型预测在 24/48/72 小时的距离和风速算方差，生成全局 `atcf_spread_cache.json`。
- **Task 2: SHIPS 诊断标量提取 (中等)**：依托于本机，使用 `s3fs` 和 `xarray` 匿名直读 `s3://noaa-nesdis-tcprimed-pds/`。由于只读取 10 个环境变量的单一格点时序，内存占用极小。将其写入 `ships_environment_cache.json`，附带对齐元数据。
- **Task 3: 新增特征摘要与可信字段筛选 (中等)**：仅对新增特征执行“可信字段白名单 + 不可信字段剔除”，输出稳定的增量摘要，不触碰原有环境 JSON。

### Phase 2: 多云生态视觉降维与大模型对齐 (Week 2-3)
- **Task 4: 多平台卫星特征挖掘 (复杂但零成本)**：
  - **CDS 节点端**：在 Copernicus Data Space 的免费 JupyterLab 内布署 Sentinel-1 和 Sentinel-3 轨道拦截代码，将抽取后的极简特征发回主服务器。
  - **GEE 节点端**：使用本地机器建立 `earthengine-api` 认证，利用 Google 云海量算力针对 GOES-16 画圈求最低温度等降维运算，拉回特征文件。

### Phase 3: Prompt 增量注入与模型重训 (核心检验, Week 3-4)
- **Task 5: Prompt 重构与模型重训**：
  - 调整 `prepare_forecast_samples.py` 与 `generate_forecast_dataset.py` 的组装段落，仅在 Prompt 层加入新特征与离散度摘要，原有数据保持不变。
  - 利用合并了全部新特征的 `raw_forecast_dataset_sft_compact.jsonl` 重启 Qwen3 训练。
  - 评估大模型生成的 Forecast Discussion 是否真正做到了“有理有据（由于监测到暖涡且切变低，故支持RI；由于集成分散，故采用共识路径）”，而非先前的含糊其辞或盲猜模式。
