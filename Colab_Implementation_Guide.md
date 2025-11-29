# 在 Google Colab 上运行气旋追踪与环境分析指南

本文档详细说明如何在 Google Colab 环境中，利用 `gs://weatherbench2/datasets/hres` 数据集实现气旋路径追踪和天气系统提取。

## 1. 环境准备

在 Colab Notebook 中，首先需要安装必要的 Python 库以支持 Zarr 数据读取、GCS 访问以及项目依赖。

```python
# 安装依赖
!pip install xarray[complete] zarr gcsfs pandas netCDF4
```

此外，你需要将本项目代码上传至 Colab 或挂载 Google Drive，确保 Python path 包含 `src` 目录，以便导入现有的追踪和提取模块。

```python
import sys
import os
from google.colab import drive

# 挂载 Drive (如果代码在 Drive 中)
drive.mount('/content/drive')

# 设置项目路径 (假设项目在 Drive 的 TianGong-AI-Cyclone 目录下)
PROJECT_PATH = '/content/drive/MyDrive/TianGong-AI-Cyclone'
sys.path.append(os.path.join(PROJECT_PATH, 'src'))

# 验证导入
try:
    import initial_tracker
    print("项目模块导入成功")
except ImportError:
    print("请检查路径设置")
```

## 2. 数据访问 (WeatherBench 2)

WeatherBench 2 数据存储在 Google Cloud Storage (GCS) 的公共存储桶中。我们可以使用 `xarray` 配合 `zarr` 引擎直接流式读取，无需下载整个数据集。

### 2.1 数据集位置

根据 WeatherBench 2 文档，IFS HRES 数据位于：
- **分析场 (Analysis, t=0)**: `gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr` (推荐用于“真实”路径追踪)
- **预报场 (Forecast)**: `gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr`

### 2.2 读取与变量适配

现有的 `initial_tracker` 模块期望特定的变量名（如 `msl`, `u10`, `v10`, `z`）。WeatherBench 2 使用全名（如 `mean_sea_level_pressure`）。我们需要在读取后进行重命名。

```python
import xarray as xr

# 读取 Zarr 数据集 (以 HRES t=0 分析场为例)
# chunks=None 表示延迟加载，chunks='auto' 或具体数值用于 Dask 并行
ds_raw = xr.open_zarr(
    'gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr', 
    chunks='auto'
)

# 变量重命名映射
rename_map = {
    'mean_sea_level_pressure': 'msl',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    'geopotential': 'z',
    # 如果需要其他变量，在此添加
}

# 选择需要的变量并重命名
ds_adapted = ds_raw[list(rename_map.keys())].rename(rename_map)

# 处理单位差异 (如果需要)
# 注意：WeatherBench 2 的 geopotential 通常是 m^2/s^2，而某些应用可能期望位势高度 (m)
# 现有的 _DsAdapter 会自动处理部分单位，但需确认 z 的处理逻辑。
# 现有代码中 _DsAdapter 会检查 level > 2000 来判断是否为 Pa 并转为 hPa。
# WeatherBench 的 level 已经是 hPa (50, 100, ... 1000)。
```

## 3. 气旋追踪实现

利用现有的 `initial_tracker` 包，我们可以构建一个在内存中处理 Zarr 数据的追踪流程，替代原有的基于 NetCDF 文件的流程。

### 3.1 核心组件

*   **`_DsAdapter`**: 用于包装 `xarray.Dataset`，提供统一的访问接口。
*   **`Tracker`**: 核心追踪算法类。
*   **`_build_batch_from_ds`**: 从 Adapter 提取特定时间步的数据用于追踪。

### 3.2 实现步骤

1.  **加载初始点**: 读取 CSV 文件（如 `western_pacific_typhoons_superfast.csv`），获取气旋的初始时间、经纬度。
2.  **初始化 Adapter**: 使用适配后的 Zarr Dataset 创建 `_DsAdapter` 实例。
3.  **追踪循环**:
    *   对每个气旋，找到其在 Dataset 中的起始时间索引。
    *   初始化 `Tracker` 对象。
    *   按时间步循环（例如每6小时），调用 `_build_batch_from_ds` 获取当前时刻数据。
    *   调用 `tracker.step(batch)` 更新气旋位置。
    *   检查是否消散 (`tracker.dissipated`)。
4.  **保存结果**: 将 `tracker.results()` 返回的 DataFrame 保存为 CSV。

**关键代码逻辑示例 (伪代码):**

```python
from initial_tracker.dataset_adapter import _DsAdapter, _build_batch_from_ds
from initial_tracker.tracker import Tracker
import pandas as pd

# 1. 准备数据适配器
adapter = _DsAdapter.build(ds_adapted)

# 2. 读取初始点
initials = pd.read_csv('path/to/initials.csv')

# 3. 遍历气旋
for _, storm in initials.iterrows():
    # 获取初始信息
    init_time = pd.to_datetime(storm['time'])
    
    # 在 Dataset 中找到对应的时间索引
    # 注意：需处理时间匹配，可以使用 ds_adapted.time.sel(time=init_time, method='nearest')
    try:
        # 简化的时间查找逻辑
        time_idx = list(adapter.times).index(init_time)
    except ValueError:
        continue # 时间不在数据集中

    # 初始化追踪器
    tracker = Tracker(
        init_lat=storm['lat'],
        init_lon=storm['lon'],
        init_time=init_time
    )

    # 向后追踪 N 步
    for step in range(1, max_steps):
        current_idx = time_idx + step
        if current_idx >= len(adapter.times):
            break
            
        # 构建数据批次 (这一步会触发从 GCS 读取少量数据)
        batch = _build_batch_from_ds(ds_adapted, current_idx)
        
        try:
            tracker.step(batch)
        except Exception as e:
            print(f"追踪中断: {e}")
            break
            
        if tracker.dissipated:
            break
            
    # 保存结果
    result_df = tracker.results()
    result_df.to_csv(f"track_{storm['sid']}.csv")
```

## 4. 环境系统提取实现

环境提取逻辑在 `src/environment_extractor` 中。原逻辑 (`TCEnvironmentalSystemsExtractor`) 设计为处理单个 NetCDF 文件。在 Colab + Zarr 环境下，建议进行如下调整：

### 4.1 调整策略

由于 Zarr 数据是全局的（包含所有时间），而提取器通常针对特定气旋的生命史。

1.  **复用提取核心**: `TCEnvironmentalSystemsExtractor` 内部使用了 `xarray`。我们可以继承或修改该类，使其接受一个已经打开的 `xr.Dataset` (即我们的 `ds_adapted`)，而不是文件路径。
2.  **切片处理**: 为了提高效率，在进行提取前，可以根据气旋的生命史时间段和活动范围（经纬度框），对全局 Zarr 数据进行 `.sel()` 切片，将所需的小块数据加载到内存或作为较小的 Dask 任务处理。

### 4.2 步骤

1.  **读取追踪结果**: 使用上一步生成的轨迹 CSV。
2.  **定义提取范围**: 根据轨迹的 `min/max lat/lon` 和 `start/end time`，确定需要从 Zarr 中提取的数据子集。
3.  **执行提取**:
    *   调用 `environment_extractor` 中的算法（如急流识别、槽脊分析）。
    *   这些算法通常输入是 `xr.DataArray` (如 200hPa 风场，500hPa 位势高度)。
    *   直接从 `ds_adapted` 中提取对应层级的数据传入算法。

## 5. 性能优化建议 (Colab)

*   **内存管理**: Colab 免费版内存有限。不要尝试 `.load()` 整个数据集。利用 `xarray` 的惰性加载特性。
*   **区域切片**: 始终先在时间(time)和空间(lat/lon)上进行切片 (`.sel`, `.isel`)，然后再进行计算或 `.values` 转换。
*   **持久化**: 将中间结果（如轨迹 CSV）实时保存到 Google Drive，防止 Colab 会话断开导致数据丢失。

## 6. 总结

通过上述步骤，你可以在 Colab 上直接利用云端海量数据 (WeatherBench 2) 运行现有的气旋分析算法，无需下载 TB 级的气象文件，极大地提高了研究效率。
