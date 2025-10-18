# 测试脚本说明

本目录包含两个测试脚本，用于测试气旋追踪和环境场提取功能。两个脚本的主要区别在于**初始点的选择方式**。

## 脚本对比

| 特性 | test_three_nc_files.py | test_three_nc_files_with_real_tracks.py |
|------|------------------------|----------------------------------------|
| **初始点来源** | 自动搜索算法 | 历史观测数据 |
| **适用场景** | 预报数据自动分析 | 历史台风验证和研究 |
| **优点** | 无需人工指定，可发现所有气旋 | 追踪准确，可与真实路径对比 |
| **缺点** | 可能找到错误的低压系统 | 需要历史数据，不能用于未来预报 |
| **输出文件名** | `track_AUTO_*.csv` | `track_{storm_id}_*.csv` |

## 方法1：自动搜索算法 (test_three_nc_files.py)

### 工作原理
1. 读取NC文件的第一个时次
2. 在热带区域（5-35°N）搜索最低气压点
3. 如果气压低于1010 hPa，将其作为初始点
4. 从该初始点开始追踪

### 使用方法
```bash
python3 test_three_nc_files.py
```

### 输出示例
```
track_AUTO_FOUR_v200_GFS_2020093012_f000_f240_06_FOUR_v200_GFS_2020093012_f000_f240_06.csv
```

### 优点
- ✅ 完全自动化，无需任何输入
- ✅ 可以发现预报数据中的所有气旋系统
- ✅ 适合批量处理大量预报文件

### 缺点
- ⚠️ 可能找到错误的低压系统（如温带气旋）
- ⚠️ 初始点可能偏离目标台风的实际位置
- ⚠️ 不适合研究特定历史台风的特征

### 示例结果
- FOUR文件：在35°N, 196°E找到低压中心（气压998.6 hPa）
- 这实际上是中太平洋的另一个低压系统，不是台风KUJIRA

## 方法2：历史观测初始点 (test_three_nc_files_with_real_tracks.py)

### 工作原理
1. 从`input/western_pacific_typhoons_superfast.csv`加载历史观测数据
2. 根据NC文件名查找对应的台风ID
3. 找到与NC文件开始时间最接近的观测记录
4. 使用该观测位置作为初始点进行追踪

### NC文件与台风的映射关系
```python
NC_TO_STORM_MAPPING = {
    'AURO_v100_IFS_2025061000_f000_f240_06.nc': {
        'storm_id': '2025162N15114',
        'storm_name': 'WUTIP',
    },
    'FOUR_v200_GFS_2020093012_f000_f240_06.nc': {
        'storm_id': '2020270N17159',
        'storm_name': 'KUJIRA',
    },
    'PANG_v100_IFS_2022032900_f000_f240_06.nc': {
        'storm_id': '2022088N09116',
        'storm_name': 'UNNAMED',
    }
}
```

### 使用方法
```bash
python3 test_three_nc_files_with_real_tracks.py
```

### 输出示例
```
track_2020270N17159_FOUR_v200_GFS_2020093012_f000_f240_06.csv
```

### 优点
- ✅ 追踪结果准确，基于真实观测位置
- ✅ 可以与历史真实路径进行对比验证
- ✅ 适合研究特定台风的环境场特征
- ✅ 文件命名包含真实台风ID，便于识别

### 缺点
- ⚠️ 需要历史观测数据（`western_pacific_typhoons_superfast.csv`）
- ⚠️ 只能用于已发生的历史台风
- ⚠️ 需要手动配置NC文件与台风ID的映射关系
- ⚠️ 如果NC文件时间与观测时间相差过大，可能影响精度

### 示例结果
- FOUR文件：使用KUJIRA在41.2°N, 166.3°E的观测位置（2020-09-30 12:00）
- 追踪结果与真实台风路径一致

## 结果对比

### 台风KUJIRA (FOUR文件) 的两种追踪结果

#### 自动搜索方法
- **初始点**: 35.0°N, 196.0°E（中太平洋）
- **轨迹点数**: 13个
- **纬度范围**: 35°N - 61.6°N
- **经度范围**: 196°E - 210.7°E
- **追踪的系统**: 中太平洋的某个低压系统

#### 真实观测方法
- **初始点**: 41.2°N, 166.3°E（西太平洋）
- **轨迹点数**: 11个
- **纬度范围**: 34°N - 41.2°N
- **经度范围**: 166.3°E - 183.5°E
- **追踪的系统**: 真实的台风KUJIRA

**结论**: 两种方法追踪的是完全不同的气旋系统，相距约3368公里！

## 输出文件结构

### 共同输出
```
data/test/
├── tracks/                      # 气旋追踪CSV文件
│   ├── track_*.csv
├── env_systems_*.json           # 环境场提取结果
└── processing_summary*.json     # 处理摘要
```

### 自动搜索方法特有
- `processing_summary.json`
- 轨迹文件包含 `AUTO_` 前缀

### 真实观测方法特有
- `processing_summary_real_tracks.json`
- 轨迹文件以真实台风ID命名（如 `2020270N17159`）

## 使用建议

### 场景1：历史台风研究和模型验证
**推荐**: `test_three_nc_files_with_real_tracks.py`

适用于：
- 研究特定历史台风的环境场特征
- 验证模型对已知台风的追踪能力
- 分析预报与实际的偏差
- 构建训练数据集

示例：
```python
# 研究台风KUJIRA的垂直风切变特征
python3 test_three_nc_files_with_real_tracks.py
# 然后分析 env_systems_track_2020270N17159_*.json
```

### 场景2：预报数据批量分析
**推荐**: `test_three_nc_files.py`

适用于：
- 自动分析大量预报文件
- 发现预报数据中的所有气旋系统
- 不关心具体台风ID，只关心系统特征
- 实时预报分析

示例：
```bash
# 批量处理100个预报文件
for file in data/*.nc; do
    python3 test_three_nc_files.py --nc_file "$file"
done
```

### 场景3：实时台风预报
**推荐**: 混合方法

流程：
1. 从实时观测获取当前台风位置
2. 修改 `NC_TO_STORM_MAPPING` 添加新的映射
3. 运行 `test_three_nc_files_with_real_tracks.py`

示例：
```python
# 添加新的实时台风
NC_TO_STORM_MAPPING['GFS_2025101812_f000_f240.nc'] = {
    'storm_id': 'CURRENT_TYPHOON',
    'storm_name': 'MAWAR',
    # 使用最新观测位置
}
```

## 技术细节

### 初始点DataFrame格式要求

**输入给 `track_file_with_initials` 的DataFrame需要包含以下列**：
- `storm_id`: 台风ID
- `datetime`: 观测时间
- `dt`: pandas Timestamp格式的时间
- `latitude`: 纬度（**注意：不是init_lat**）
- `longitude`: 经度（**注意：不是init_lon**）
- `max_wind_usa`: 最大风速（可选）
- `min_pressure_usa`: 最低气压（可选）

**重要**: 函数内部会调用 `_select_initials_for_time` 来转换列名为 `init_lat`/`init_lon`。

### 时间窗口匹配

两个脚本都使用24小时的时间窗口来匹配初始点：

```python
time_window_hours=24  # 允许±24小时的时间差
```

如果历史观测时间与NC文件开始时间相差超过6小时，会显示警告。

## 常见问题

### Q1: 为什么自动搜索找到了错误的气旋？
A: 自动搜索只根据气压最低点，可能找到温带气旋、副热带低压等非热带气旋系统。

### Q2: 如何添加新的NC文件映射？
A: 编辑 `test_three_nc_files_with_real_tracks.py` 中的 `NC_TO_STORM_MAPPING` 字典。

### Q3: 历史观测数据在哪里？
A: `input/western_pacific_typhoons_superfast.csv`，包含574个历史台风的35398条观测记录。

### Q4: 两种方法的环境场提取有区别吗？
A: 环境场提取方法完全相同，区别只在初始点和追踪路径。

### Q5: 哪个方法更准确？
A: 对于历史台风，真实观测方法更准确。对于未知系统，只能使用自动搜索。

## 相关文档

- [追踪差异分析报告](docs/tracking_difference_analysis.md)
- [环境场提取说明](docs/environment_extraction.md)
- [海洋热含量边界问题分析](docs/ocean_heat_boundary_issue_analysis.md)

## 更新日志

- 2025-10-18: 创建 `test_three_nc_files_with_real_tracks.py` 脚本
- 2025-10-18: 添加详细的方法对比文档
- 2025-10-18: 完成台风ID映射配置

---

**作者**: TianGong-AI-Cyclone团队
**最后更新**: 2025-10-18
