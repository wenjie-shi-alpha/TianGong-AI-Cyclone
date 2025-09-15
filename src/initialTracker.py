"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

增强: 内置 Batch 适配和 CSV/NetCDF 管线
- 去除对 aurora.batch.Batch 的外部依赖, 在内部实现一个轻量 Batch 结构
- 新增从 NetCDF (AWS 下载的预报格点数据) 构造 Batch 的适配器
- 新增命令行/方法: 读取 input/western_pacific_typhoons_superfast.csv 的每个气旋起始点, 逐步追踪并输出路径 CSV
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, minimum_filter

__all__ = ["Tracker", "track_file_with_initials", "main"]

logger = logging.getLogger(__file__)


class NoEyeException(Exception):
    """Raised when no eye can be found."""


# -----------------------------
# 内置 Batch/Metadata 适配层
# -----------------------------

@dataclass
class _Metadata:
    lat: np.ndarray
    lon: np.ndarray
    time: List[pd.Timestamp]
    atmos_levels: List[int] | np.ndarray


@dataclass
class _SimpleBatch:
    """轻量 Batch 占位类, 提供与原始接口兼容的属性.

    约定的字段:
    - atmos_vars: 包含等压面变量的 dict, 如 {"z": np.ndarray[1,1,level,lat,lon]}
    - surf_vars: 地面变量 dict, 如 {"msl": np.ndarray[1,1,lat,lon], "10u": ..., "10v": ...}
    - static_vars: 静态变量 dict, 如 {"lsm": np.ndarray[lat,lon]}
    - metadata: _Metadata
    """

    atmos_vars: Dict[str, np.ndarray]
    surf_vars: Dict[str, np.ndarray]
    static_vars: Dict[str, np.ndarray]
    metadata: _Metadata

    def to(self, device: str) -> "_SimpleBatch":  # 兼容原接口; 不做实际设备迁移
        return self


def _to_0360(lon: np.ndarray) -> np.ndarray:
    """将 -180..180 转为 0..360 (若已是 0..360 则原样返回)."""
    lon = np.asarray(lon)
    if lon.min() < 0:
        lon = (lon + 360.0) % 360.0
    return lon


def _safe_get(ds: xr.Dataset, names: Iterable[str]) -> Optional[xr.DataArray]:
    for n in names:
        if n in ds.data_vars or n in ds.coords:
            return ds[n]
    return None


def _build_batch_from_ds(ds: xr.Dataset, time_idx: int) -> _SimpleBatch:
    """从 xarray Dataset 构建一个与原算法兼容的 `_SimpleBatch`.

    变量名映射与降维规则:
    - msl: 优先使用 ["msl", "mslp"]
    - 10m 风: 期望键为 "10u"/"10v"; 从 ["u10"/"v10", "10u"/"10v"] 中择一
    - 位势高度 z: 期望键 "z"; 层维度优先 ["level", "isobaricInhPa", "pressure"]
    - lsm: 若无, 用 0 数组替代
    - lat/lon: 坐标名兼容 ["latitude"/"longitude", "lat"/"lon"], lon 统一转为 0..360
    """
    # lat/lon
    lat_da = _safe_get(ds, ["latitude", "lat"])
    lon_da = _safe_get(ds, ["longitude", "lon"])
    if lat_da is None or lon_da is None:
        raise ValueError("Dataset 缺少经纬度坐标 (lat/lon 或 latitude/longitude)")
    lats = lat_da.values
    lons = _to_0360(lon_da.values)

    # msl
    msl_da = _safe_get(ds, ["msl", "mslp"])
    if msl_da is None:
        raise ValueError("Dataset 缺少海平面气压 (msl/mslp)")
    msl_2d = msl_da.isel(time=time_idx).values
    if msl_2d.ndim == 3:  # 兼容额外维 (如 ensemble)
        msl_2d = msl_2d[0]

    # 10m wind
    u10_da = _safe_get(ds, ["u10", "10u"])  # 标准多为 u10/v10
    v10_da = _safe_get(ds, ["v10", "10v"])
    if u10_da is None or v10_da is None:
        # 若无 10 米风, 用 0 替代, 只使用 MSL 逻辑
        u10_2d = np.zeros_like(msl_2d)
        v10_2d = np.zeros_like(msl_2d)
    else:
        u10_2d = u10_da.isel(time=time_idx).values
        v10_2d = v10_da.isel(time=time_idx).values
        if u10_2d.ndim == 3:
            u10_2d = u10_2d[0]
        if v10_2d.ndim == 3:
            v10_2d = v10_2d[0]

    # geopotential height or geopotential
    z_da = _safe_get(ds, ["z", "gh", "geopotential", "geopotential_height"])  # 宽松匹配
    atmos_levels: List[int] = []
    z_3d: Optional[np.ndarray] = None
    if z_da is not None:
        # 识别层维度
        level_dim = None
        for dim in ["level", "isobaricInhPa", "pressure", "isobaricInPa"]:
            if dim in z_da.dims:
                level_dim = dim
                break
        if level_dim is None:
            # 无层维; 视作只有一个层 (无法用 z700 兜底)
            z_2d = z_da.isel(time=time_idx).values
            if z_2d.ndim == 3:
                z_2d = z_2d[0]
            z_3d = z_2d[np.newaxis, ...]  # 1 x lat x lon
            atmos_levels = [700]
        else:
            levels_vals = z_da[level_dim].values
            # 单位归一: Pa -> hPa
            if levels_vals.max() > 2000:  # 以 Pa 表示
                levels_vals = (levels_vals / 100.0).astype(float)
            atmos_levels = [int(round(v)) for v in levels_vals.tolist()]
            z_sel = z_da.isel(time=time_idx)
            z_3d = z_sel.transpose(..., "latitude" if "latitude" in z_sel.dims else "lat", "longitude" if "longitude" in z_sel.dims else "lon").values  # type: ignore
            # 若维序顺序非 [level, lat, lon], 尝试调整
            if z_3d.ndim == 4:
                z_3d = z_3d[0]  # 兼容额外维度
            if z_3d.shape[0] != len(atmos_levels):
                # 尝试猜测 level 维在最后
                if z_3d.shape[-3] == len(atmos_levels):
                    z_3d = np.moveaxis(z_3d, -3, 0)

    # lsm
    lsm_da = _safe_get(ds, ["lsm", "land_sea_mask", "landmask"])  # 可选
    if lsm_da is not None:
        lsm = lsm_da.values
        if lsm.ndim == 3:
            lsm = lsm[0]
    else:
        lsm = np.zeros_like(msl_2d)

    # metadata.time 单元素 list
    # xarray 时间通常是 np.datetime64; 转 pandas.Timestamp 保持兼容
    time_val = pd.to_datetime(ds.time.values[time_idx]) if "time" in ds.coords else pd.Timestamp.now()
    metadata = _Metadata(lat=lats, lon=lons, time=[time_val], atmos_levels=atmos_levels or [700])

    # 包装为原算法期望的形状
    surf_vars = {
        "msl": msl_2d[np.newaxis, np.newaxis, ...],
        "10u": u10_2d[np.newaxis, np.newaxis, ...],
        "10v": v10_2d[np.newaxis, np.newaxis, ...],
    }
    atmos_vars: Dict[str, np.ndarray] = {}
    if z_3d is not None:
        # 期待形状 [1,1,levels,lat,lon]
        atmos_vars["z"] = z_3d[np.newaxis, np.newaxis, ...]

    static_vars = {"lsm": lsm}

    return _SimpleBatch(atmos_vars=atmos_vars, surf_vars=surf_vars, static_vars=static_vars, metadata=metadata)


def get_box(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
):
    """Get a square box for a variable."""
    # Make latitude selection.
    lat_mask = (lat_min <= lats) & (lats <= lat_max)
    box = variable[..., lat_mask, :]
    lats = lats[lat_mask]

    # Make longitude selection. Be careful when wrapping around.
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        lon_mask = (lon_min <= lons) & (lons <= lon_max)
        box = box[..., lon_mask]
        lons = lons[lon_mask]
    else:
        lon_mask1 = lon_min <= lons
        lon_mask2 = lons <= lon_max
        box = np.concatenate((box[..., lon_mask1], box[..., lon_mask2]), axis=-1)
        lons = np.concatenate((lons[lon_mask1], lons[lon_mask2]))

    return lats, lons, box


def havdist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two latitude-longitude coordinates."""
    lat1, lat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    lon1, lon2 = np.deg2rad(lon1), np.deg2rad(lon2)
    rad_earth_km = 6371
    inner = 1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))
    return 2 * rad_earth_km * np.arcsin(np.sqrt(0.5 * inner))


def get_closest_min(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    delta_lat: float = 5,
    delta_lon: float = 5,
    minimum_cap_size: int = 8,
) -> tuple[float, float]:
    """Get the minimum in `variable` that is closest to `lat` and `lon`."""
    # Create a box centred around the current latitude and longitude.
    lats, lons, box = get_box(
        variable,
        lats,
        lons,
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )

    # Smooth to avoid local minima due to noise.
    box = gaussian_filter(box, sigma=1)

    # Find local minima.
    local_minima = minimum_filter(box, size=(minimum_cap_size, minimum_cap_size)) == box

    # Remove minima at the edges: these occur when the tracker fails.
    local_minima[0, :] = 0
    local_minima[-1, :] = 0
    local_minima[:, 0] = 0
    local_minima[:, -1] = 0

    # If no local minima are left, no eye can be found. Try the next one.
    if local_minima.sum() == 0:
        raise NoEyeException()

    # Return the latitude and longitude of the closest local minimum.
    lat_inds, lon_inds = zip(*np.argwhere(local_minima))
    dists = havdist(lats[list(lat_inds)], lons[list(lon_inds)], lat, lon)
    i = np.argmin(dists)

    return lats[lat_inds[i]], lons[lon_inds[i]]


def extrapolate(lats: list[float], lons: list[float]) -> tuple[float, float]:
    """Guess an initial latitude and longitude by extrapolating `lats` and `lons`."""
    assert len(lats) == len(lons)
    if len(lats) == 0:
        raise ValueError("Cannot extrapolate from empty lists.")
    elif len(lats) == 1:
        return lats[0], lons[0]
    else:
        # Linearly extrapolate using the last eight points.
        lats = lats[-8:]
        lons = lons[-8:]
        n = len(lats)
        fit = np.polyfit(np.arange(n), np.stack((lats, lons), axis=-1), 1)
        return np.polyval(fit, n)


class Tracker:
    """Simple tropical cyclone tracker.

    This algorithm was originally designed and implemented by Anna Allen. This particular
    implementation is by Wessel Bruinsma and features various improvements over the original design.
    """

    def __init__(
        self,
        init_lat: float,
        init_lon: float,
        init_time: datetime,
    ) -> None:
        self.tracked_times: list[datetime] = [init_time]
        self.tracked_lats: list[float] = [init_lat]
        self.tracked_lons: list[float] = [init_lon]
        self.tracked_msls: list[float] = [np.nan]
        self.tracked_winds: list[float] = [np.nan]
        self.fails: int = 0

    def results(self) -> pd.DataFrame:
        """Assemble the track into a convenient DataFrame."""
        return pd.DataFrame(
            {
                "time": self.tracked_times,
                "lat": self.tracked_lats,
                "lon": self.tracked_lons,
                "msl": self.tracked_msls,
                "wind": self.tracked_winds,
            }
        )

    def step(self, batch: _SimpleBatch) -> None:
        """Track the next step.

        Args:
            batch: Prediction for a single time step (internal _SimpleBatch).
        """
        # Check that there is only one prediction time. We don't support batched tracking.
        if len(batch.metadata.time) != 1:
            raise RuntimeError("Predictions don't have batch size one.")

        # No need to do tracking on the GPU. It's cheap.
        batch = batch.to("cpu")

        # Extract the relevant variables from the prediction.
        # 700hPa: 允许使用最接近 700 的层
        z700 = None
        if "z" in batch.atmos_vars and len(batch.metadata.atmos_levels) > 0:
            levels = np.array(batch.metadata.atmos_levels)
            # 确保是数值
            try:
                levels_float = levels.astype(float)
                idx = int(np.argmin(np.abs(levels_float - 700)))
                z700 = batch.atmos_vars["z"][0, 0, idx]
            except Exception:
                z700 = None

        msl = batch.surf_vars["msl"][0, 0]
        u10 = batch.surf_vars["10u"][0, 0]
        v10 = batch.surf_vars["10v"][0, 0]
        wind = np.sqrt(u10 * u10 + v10 * v10)
        lsm = batch.static_vars["lsm"]
        lats = np.array(batch.metadata.lat)
        lons = np.array(batch.metadata.lon)
        time = batch.metadata.time[0]

        # Provide an initial guess by extrapolating.
        lat, lon = extrapolate(self.tracked_lats, self.tracked_lons)
        lat = max(min(lat, 90), -90)
        lon = lon % 360

        def is_clear(lat: float, lon: float, delta: float) -> bool:
            """Is a box centred at `lat` and `lon` with "radius" `delta` clear of land?"""
            _, _, lsm_box = get_box(
                lsm,
                lats,
                lons,
                lat - delta,
                lat + delta,
                lon - delta,
                lon + delta,
            )
            return lsm_box.max() < 0.5

        # Did we "snap" from the guess to a real nearby minimum?
        snap = False

        # Try MSL with increasingly small boxes.
        for delta in [5, 4, 3, 2, 1.5]:
            try:
                if is_clear(lat, lon, delta):
                    lat, lon = get_closest_min(
                        msl,
                        lats,
                        lons,
                        lat,
                        lon,
                        delta_lat=delta,
                        delta_lon=delta,
                    )
                    snap = True
                    break
            except NoEyeException:
                pass

        if not snap:
            # MSL didn't work. Try Z700. If it works, try to refine with MSL.
            if z700 is not None:
                try:
                    lat, lon = get_closest_min(
                        z700,
                        lats,
                        lons,
                        lat,
                        lon,
                        delta_lat=5,
                        delta_lon=5,
                    )
                    snap = True

                    for delta in [5, 4, 3, 2, 1.5]:
                        try:
                            if is_clear(lat, lon, delta):
                                lat, lon = get_closest_min(
                                    msl,
                                    lats,
                                    lons,
                                    lat,
                                    lon,
                                    delta_lat=delta,
                                    delta_lon=delta,
                                )
                                break
                        except NoEyeException:
                            pass
                except NoEyeException:
                    pass

        if not snap:
            self.fails += 1
            if len(self.tracked_lats) > 1:
                logger.info(f"Failed at time {time}. Extrapolating in a silly way.")
            else:
                raise NoEyeException("Completely failed at the first step.")

        self.tracked_times.append(time)
        self.tracked_lats.append(lat)
        self.tracked_lons.append(lon)

        # Extract minimum MSL and maximum wind speed from a crop around the TC.
        _, _, msl_crop = get_box(
            msl,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        _, _, wind_crop = get_box(
            wind,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        self.tracked_msls.append(msl_crop.min())
        self.tracked_winds.append(wind_crop.max())


# -----------------------------
# 追踪驱动: 从 CSV 初始点与 NetCDF 运行
# -----------------------------

def _load_initial_points(csv_path: Path) -> pd.DataFrame:
    """读取 CSV, 提取每个 storm_id 的起始点 (首条记录).

    需要列: storm_id, datetime, latitude, longitude
    返回列: storm_id, init_time(pd.Timestamp), init_lat, init_lon
    """
    df = pd.read_csv(csv_path)
    required = {"storm_id", "datetime", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 缺少必要列: {required - set(df.columns)}")
    # 将 datetime 转为 pandas 时间
    df["init_time"] = pd.to_datetime(df["datetime"])  # 源列保留
    firsts = df.sort_values(["storm_id", "init_time"]).groupby("storm_id", as_index=False).first()
    return firsts[["storm_id", "init_time", "latitude", "longitude"]].rename(
        columns={"latitude": "init_lat", "longitude": "init_lon"}
    )


def _inside_domain(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> bool:
    lon = lon % 360
    return (lats.min() - 1 <= lat <= lats.max() + 1) and (
        lons.min() - 1 <= lon <= lons.max() + 1
    )


def track_file_with_initials(
    nc_path: Path,
    initials: pd.DataFrame,
    output_dir: Path,
    max_storms: Optional[int] = None,
) -> List[Path]:
    """使用给定的初始点, 在一个 NetCDF 预报文件上逐步追踪并输出 CSV.

    返回生成的 CSV 路径列表.
    """
    ds = xr.open_dataset(nc_path)
    lat_da = _safe_get(ds, ["latitude", "lat"])
    lon_da = _safe_get(ds, ["longitude", "lon"])
    if lat_da is None or lon_da is None:
        raise ValueError(f"文件缺少经纬度: {nc_path}")
    lats = lat_da.values
    lons = _to_0360(lon_da.values)

    times = pd.to_datetime(ds.time.values) if "time" in ds.coords else [pd.Timestamp.now()]
    if len(times) == 0:
        raise ValueError(f"文件无时间维度: {nc_path}")

    written: List[Path] = []
    count = 0
    for _, row in initials.iterrows():
        if max_storms is not None and count >= max_storms:
            break
        storm_id = str(row["storm_id"]) if "storm_id" in row else f"storm_{count}"
        init_lat = float(row["init_lat"]) if "init_lat" in row else float(row["latitude"])  # type: ignore
        init_lon = float(row["init_lon"]) if "init_lon" in row else float(row["longitude"])  # type: ignore

        if not _inside_domain(init_lat, init_lon, lats, lons):
            continue

        # 初始化跟踪器; 使用文件的第一个时间作为起点
        tracker = Tracker(init_lat=init_lat, init_lon=init_lon, init_time=times[0])

        # 顺序遍历每个时间步, 构造 batch 并 step
        for ti in range(len(times)):
            batch = _build_batch_from_ds(ds, ti)
            try:
                tracker.step(batch)
            except NoEyeException as e:
                # 若第一步即失败则放弃该风暴; 非首步失败则停止延伸
                if ti == 0:
                    logger.info(f"{storm_id}: 首步失败, 跳过 ({e})")
                    tracker = None  # type: ignore
                break

        if tracker is None:
            continue

        # 保存结果
        out_df = tracker.results()
        output_dir.mkdir(parents=True, exist_ok=True)
        # 推断模型/起报标签
        stem = nc_path.stem
        out_name = f"track_{storm_id}_{stem}.csv"
        out_path = output_dir / out_name
        out_df.to_csv(out_path, index=False)
        written.append(out_path)
        count += 1
    ds.close()
    return written


def main():
    import argparse

    parser = argparse.ArgumentParser(description="基于初始点与 NetCDF 文件的热带气旋逐步追踪")
    parser.add_argument(
        "--initials_csv",
        default=str(Path("input") / "western_pacific_typhoons_superfast.csv"),
        help="包含每个气旋起始点的 CSV 路径",
    )
    parser.add_argument(
        "--nc_dir",
        default=str(Path("data") / "nc_files"),
        help="AWS 下载的 NetCDF 文件目录",
    )
    parser.add_argument(
        "--limit_storms",
        type=int,
        default=None,
        help="最多处理的气旋数量 (按 CSV 首行去重后的 storm_id)",
    )
    parser.add_argument(
        "--limit_files",
        type=int,
        default=None,
        help="最多处理的 NetCDF 文件数量",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path("track_output")),
        help="输出气旋追踪路径 CSV 目录",
    )
    args = parser.parse_args()

    initials = _load_initial_points(Path(args.initials_csv))
    nc_dir = Path(args.nc_dir)
    nc_files = sorted([p for p in nc_dir.glob("*.nc") if p.is_file()])
    if args.limit_files is not None:
        nc_files = nc_files[: args.limit_files]
    output_dir = Path(args.output_dir)

    logger.info(f"初始风暴数量: {len(initials)} | NetCDF 文件数: {len(nc_files)}")
    total_written: List[Path] = []
    for nc in nc_files:
        try:
            written = track_file_with_initials(nc, initials, output_dir, max_storms=args.limit_storms)
            total_written.extend(written)
            logger.info(f"{nc.name}: 写入 {len(written)} 条路径")
        except Exception as e:
            logger.exception(f"处理 {nc.name} 失败: {e}")

    print(f"完成: 共写入 {len(total_written)} 条路径到 {output_dir}")


if __name__ == "__main__":
    main()