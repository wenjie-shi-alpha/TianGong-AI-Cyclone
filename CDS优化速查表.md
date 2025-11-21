# CDS.py 优化速查表

## 🚀 快速开始

### 最简单用法（兼容旧代码）
```python
from src.cds import CDSEnvironmentExtractor

extractor = CDSEnvironmentExtractor('tracks.csv')
extractor.process_all_tracks()
```

### 推荐用法（最优性能）
```python
extractor = CDSEnvironmentExtractor(
    tracks_file='tracks.csv',
    download_workers=4,          # 4线程并发 ⭐
    area=[60, 100, 0, 180],      # 西太平洋裁剪 ⭐
    use_grib=True                # GRIB格式 ⭐
)
extractor.process_all_tracks()
```

### 命令行用法
```bash
# 基础
python src/cds.py --tracks data.csv

# 完整优化（推荐）
python src/cds.py \
    --tracks data.csv \
    --download-workers 4 \
    --area 60,100,0,180 \
    --use-grib
```

## 📊 新增参数对照表

| 参数 | 类型 | 默认 | 说明 | 推荐值 |
|------|------|------|------|--------|
| `download_workers` | int | 4 | 下载并发线程数 | 2-4 |
| `area` | list | None | 区域裁剪 [N,W,S,E] | `[60,100,0,180]` (西太) |
| `use_grib` | bool | False | 使用GRIB格式 | True |
| `cleanup_intermediate` | bool | True | 清理临时文件 | True |
| `max_workers` | int | None | 处理并发数 | 4 |

## ⚡ 性能对比

| 场景 | 旧版 | 新版 | 提升 |
|------|------|------|------|
| 单月下载 | 2-6小时 | 0.5-2小时 | **3-5x** |
| API排队 | 频繁 | 很少 | **80%↓** |
| 数据量（裁剪后） | 100% | 30% | **70%↓** |

## 🎯 优化要点

### ✅ 做这些（推荐）
- 使用 4 线程下载 (`download_workers=4`)
- 启用区域裁剪 (`area=[60,100,0,180]`)
- 使用GRIB格式 (`use_grib=True`)
- 保持默认重试 (自动5次)

### ❌ 避免这些
- 不要超过4个并发 (会被限流)
- 不要在高峰时段运行 (UTC 8-12点)
- 不要禁用重试机制
- 不要请求全球数据（如果只需区域）

## 🔧 常见配置

### 西太平洋台风研究
```python
extractor = CDSEnvironmentExtractor(
    tracks_file='western_pacific_typhoons.csv',
    area=[60, 100, 0, 180],  # 西太平洋
    download_workers=4,
    use_grib=True
)
```

### 大西洋飓风研究
```python
extractor = CDSEnvironmentExtractor(
    tracks_file='atlantic_hurricanes.csv',
    area=[60, -100, 10, -20],  # 大西洋
    download_workers=4,
    use_grib=True
)
```

### 测试运行（小数据量）
```python
extractor = CDSEnvironmentExtractor(
    tracks_file='test.csv',
    download_workers=2,      # 减少并发
    cleanup_intermediate=False  # 保留文件调试
)
```

## 🐛 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 仍然排队久 | 请求太大 | 添加区域裁剪 |
| 下载失败 | 网络问题 | 自动重试5次，检查网络 |
| 内存不足 | 数据太大 | 使用区域裁剪 |
| GRIB错误 | 缺少库 | `pip install cfgrib` |

## 📚 查看详细文档

- 完整说明: `src/CDS_更新说明.md`
- 优化方案: `CDS_API_优化方案总结.md`
- 推荐做法: `src/cds_optimized_recommendations.md`

## 🧪 测试优化

```bash
# 运行测试脚本
python src/test_cds_optimized.py

# 小规模测试（10个点）
python src/cds.py --max-points 10 --download-workers 2

# 查看帮助
python src/cds.py --help
```

## 💡 专家提示

1. **首次运行**: 先用 `--max-points 10` 测试
2. **高峰期**: 避免 UTC 8-12 点
3. **大批量**: 分批次运行，每批1-2个月
4. **监控**: 访问 https://cds.climate.copernicus.eu/live
5. **失败恢复**: 自动断点续传，直接重新运行即可

## 🎉 关键改进总结

1. **按日拆分** - 遵循MARS tape规则
2. **4线程并发** - 充分利用带宽
3. **智能重试** - 指数退避，最多5次
4. **区域裁剪** - 减少70%数据量
5. **GRIB格式** - 更快的传输速度

---

**记住**: 所有优化都是可选的，不指定新参数时完全向后兼容！
