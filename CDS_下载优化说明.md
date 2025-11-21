# CDS 数据下载优化说明

## 修改概述

针对 `src/cds.py` 脚本进行了优化，解决了以下问题：

1. **避免重复下载**：在请求数据之前检查文件是否已存在
2. **压力层数据分批下载**：压力层数据量大，采用分批策略避免请求失败
3. **自动重试机制**：先尝试一年2批次，失败则自动改为3批次

## 详细修改

### 1. 地面层数据下载 (`download_era5_data`)

**修改内容：**
- 在下载前检查目标文件是否已存在
- 如果文件存在，直接跳过下载并返回文件路径

**代码逻辑：**
```python
output_file = self.output_dir / f"era5_single_{start_date.replace('-', '')}_{end_date.replace('-', '')}.nc"

# 检查文件是否已存在
if output_file.exists():
    print(f"✅ ERA5单层数据已存在，跳过下载: {output_file}")
    self._downloaded_files.append(str(output_file))
    return str(output_file)
```

**优势：**
- 避免重复下载已有数据
- 节省API调用次数
- 加快处理速度

---

### 2. 压力层数据下载 (`download_era5_pressure_data`)

**主要改进：**

#### 2.1 文件存在性检查
与地面层数据相同，下载前先检查文件是否已存在。

#### 2.2 分批下载策略
压力层数据量大，直接按年下载容易失败。采用以下策略：

**策略流程：**
```
1. 初始尝试：一年分为 2 批次下载
   ├── 成功 → 合并文件 → 完成
   └── 失败 → 进入步骤2

2. 重试机制：一年分为 3 批次下载
   ├── 成功 → 合并文件 → 完成
   └── 失败 → 抛出异常
```

**批次划分方法：**
- 将月份平均分配到各批次
- 例如一年12个月：
  - 2批次：每批6个月（1-6月，7-12月）
  - 3批次：每批4个月（1-4月，5-8月，9-12月）

#### 2.3 临时文件管理
```python
# 生成临时文件
batch_output = self.output_dir / f"era5_pressure_..._part{batch_idx}.nc"

# 下载完成后检查
if batch_output.exists():
    print(f"批次 {batch_idx} 数据已存在，跳过")
    temp_files.append(str(batch_output))
    continue
```

#### 2.4 文件合并
```python
# 使用 xarray 合并多个批次文件
ds_list = [xr.open_dataset(temp_file) for temp_file in temp_files]
ds_merged = xr.concat(ds_list, dim='time')
ds_merged = ds_merged.sortby('time')  # 确保时间顺序
ds_merged.to_netcdf(str(output_file))
```

#### 2.5 失败重试与清理
```python
except Exception as e:
    # 清理失败批次的临时文件
    for temp_file in temp_files:
        if Path(temp_file).exists():
            Path(temp_file).unlink()
    
    # 判断是否需要增加批次数
    if num_splits == 2:
        num_splits = 3
        print("改用 3 批次重试...")
    else:
        raise RuntimeError("即使分为3批次也无法完成")
```

---

## 使用示例

### 基本使用
```python
from src.cds import CDSEnvironmentExtractor

extractor = CDSEnvironmentExtractor(
    tracks_file="input/matched_cyclone_tracks.csv",
    output_dir="./cds_output",
    cleanup_intermediate=True,
    max_workers=4
)

# 处理所有数据（自动分年下载、分月保存）
saved_files = extractor.process_all_tracks()
```

### 运行测试
```bash
# 运行测试脚本验证功能
python test_cds_download.py
```

---

## 工作流程

### 完整数据处理流程

```
第一阶段：按年份下载数据
├── 2006年
│   ├── 地面层数据（1次请求）
│   │   └── 检查文件 → 跳过/下载
│   └── 压力层数据（2-3次请求）
│       ├── 尝试2批次
│       │   ├── 批次1（1-6月）→ 检查 → 跳过/下载
│       │   └── 批次2（7-12月）→ 检查 → 跳过/下载
│       ├── 失败则3批次
│       │   ├── 批次1（1-4月）
│       │   ├── 批次2（5-8月）
│       │   └── 批次3（9-12月）
│       └── 合并临时文件
├── 2007年
│   └── ...
└── ...

第二阶段：按月处理和保存
├── 2006-03 → cds_environment_analysis_2006-03.json
├── 2006-04 → cds_environment_analysis_2006-04.json
└── ...
```

---

## 技术细节

### 1. 文件命名规范

**地面层数据：**
```
era5_single_YYYYMMDD_YYYYMMDD.nc
例如: era5_single_20060101_20061231.nc
```

**压力层数据：**
```
# 最终文件
era5_pressure_YYYYMMDD_YYYYMMDD.nc

# 临时分批文件
era5_pressure_YYYYMMDD_YYYYMMDD_part1.nc
era5_pressure_YYYYMMDD_YYYYMMDD_part2.nc
era5_pressure_YYYYMMDD_YYYYMMDD_part3.nc
```

### 2. 数据请求参数

**地面层变量：**
- mean_sea_level_pressure
- 10m_u_component_of_wind
- 10m_v_component_of_wind
- 2m_temperature
- sea_surface_temperature
- total_column_water_vapour

**压力层变量（850, 500, 200 hPa）：**
- u_component_of_wind
- v_component_of_wind
- geopotential
- temperature
- relative_humidity

**时间分辨率：**
- 每天4个时次：00:00, 06:00, 12:00, 18:00

### 3. 错误处理

**主要异常情况：**
1. 文件已存在 → 跳过下载
2. 网络请求失败 → 清理临时文件，增加批次数重试
3. 文件合并失败 → 保留临时文件便于调试

---

## 性能优化

### API 调用优化

**优化前：**
- 每年2次API调用（地面层1次 + 压力层1次）
- 压力层数据量大，经常失败

**优化后：**
- 地面层：1次/年（不变）
- 压力层：2-3次/年（根据数据量自适应）
- 文件存在检查避免重复调用

### 实际效果

**场景1：首次下载**
```
年份数：5年
API调用：
- 地面层：5次
- 压力层：10-15次（取决于是否需要3批次）
总计：15-20次
```

**场景2：部分数据已存在**
```
已有：2年地面层 + 1年压力层
跳过：3次API调用
实际：12-17次API调用
节省：~20%
```

---

## 注意事项

1. **CDS API配置**
   - 确保配置了正确的CDS API密钥
   - 在CDS JupyterLab环境中运行时会自动配置

2. **磁盘空间**
   - 每年数据约1-5 GB
   - 临时文件在合并后会自动清理
   - 使用 `cleanup_intermediate=True` 可在处理完后删除下载的NC文件

3. **网络稳定性**
   - 单个批次下载可能需要5-30分钟
   - 网络不稳定时建议使用3批次策略

4. **数据完整性**
   - 合并后的文件会按时间排序
   - 自动验证时间连续性

---

## 测试验证

运行测试脚本：
```bash
python test_cds_download.py
```

测试内容：
1. ✅ 地面层数据下载
2. ✅ 文件存在性检查（跳过重复下载）
3. ✅ 压力层数据分批下载
4. ✅ 临时文件合并
5. ✅ 错误处理和重试

---

## 更新日志

**版本：2025-10-31**
- ✅ 新增文件存在性检查
- ✅ 压力层数据支持分批下载（2或3批次）
- ✅ 自动重试机制
- ✅ 临时文件自动清理
- ✅ 改进错误提示信息

---

## 问题反馈

如遇到问题，请检查：
1. CDS API配置是否正确
2. 网络连接是否稳定
3. 磁盘空间是否充足
4. 查看详细日志输出定位问题

需要帮助请提供：
- 完整错误信息
- 使用的数据范围
- 系统环境信息
