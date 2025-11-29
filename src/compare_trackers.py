
import argparse
import logging
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from initial_tracker.tracker import Tracker
from initial_tracker.robust_tracker import RobustTracker
from initial_tracker.batching import _SimpleBatch, _Metadata
from initial_tracker.dataset_adapter import _DsAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_comparison(nc_path, init_lat, init_lon, init_time, output_dir):
    ds = xr.open_dataset(nc_path)
    adapter = _DsAdapter.build(ds)
    times = adapter.times
    lats = adapter.lats
    lons = adapter.lons
    
    # Initialize trackers
    tracker_std = Tracker(init_lat, init_lon, times[0])
    tracker_rob = RobustTracker(init_lat, init_lon, times[0])
    
    # Run loop
    for time_idx in range(len(times)):
        # Prepare batch
        msl_2d = adapter.msl_at(time_idx)
        u10_2d = adapter.u10_at(time_idx)
        v10_2d = adapter.v10_at(time_idx)
        z2d = adapter.z_near700_at(time_idx)
        
        surf_vars = {
            "msl": msl_2d[np.newaxis, np.newaxis, ...],
            "10u": u10_2d[np.newaxis, np.newaxis, ...],
            "10v": v10_2d[np.newaxis, np.newaxis, ...],
        }
        atmos_vars = {}
        if z2d is not None:
            atmos_vars["z"] = z2d[np.newaxis, np.newaxis, np.newaxis, ...]
            
        metadata = _Metadata(
            lat=lats,
            lon=lons,
            time=[times[time_idx]],
            atmos_levels=[adapter.z_level_near_700 or 700],
        )
        static_vars = {"lsm": adapter.lsm}
        
        # Add dummy temp/wind for standard tracker to avoid crash if it expects them
        # But wait, standard tracker checks if they are None.
        # However, to trigger the "warm core" logic in standard tracker, we need them.
        # Let's try to fetch them if available to be fair.
        t_200 = None
        t_850 = None
        u_850 = None
        v_850 = None
        
        if adapter.idx_200hpa is not None:
             t_200 = adapter.t_at_level(time_idx, adapter.idx_200hpa)
        if adapter.idx_850hpa is not None:
             t_850 = adapter.t_at_level(time_idx, adapter.idx_850hpa)
             u_850 = adapter.u_at_level(time_idx, adapter.idx_850hpa)
             v_850 = adapter.v_at_level(time_idx, adapter.idx_850hpa)

        batch = _SimpleBatch(
            atmos_vars=atmos_vars,
            surf_vars=surf_vars,
            static_vars=static_vars,
            metadata=metadata,
            t_200hpa=t_200,
            t_850hpa=t_850,
            u_850hpa=u_850,
            v_850hpa=v_850
        )
        
        # Step Standard
        try:
            tracker_std.step(batch)
        except Exception as e:
            logger.warning(f"Standard Tracker failed at step {time_idx}: {e}")
            
        # Step Robust
        try:
            tracker_rob.step(batch)
        except Exception as e:
            logger.warning(f"Robust Tracker failed at step {time_idx}: {e}")

    # Save results
    out_std = tracker_std.results()
    out_rob = tracker_rob.results()
    
    out_std.to_csv(output_dir / "track_standard.csv", index=False)
    out_rob.to_csv(output_dir / "track_robust.csv", index=False)
    
    print(f"Standard Track Length: {len(out_std)}")
    print(f"Robust Track Length: {len(out_rob)}")

if __name__ == "__main__":
    # Configuration for the test
    nc_file = Path("data/FOUR_v200_GFS_2020093012_f000_f240_06.nc")
    
    # Storm MARIE (EP182020) at 2020-09-30 12:00:00
    # From CSV: 14.1 N, 113.1 W
    init_lat = 14.1
    init_lon = -113.1
    init_time = pd.Timestamp("2020-09-30 12:00:00")
    
    output_dir = Path("output/comparison_marie")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running comparison for Storm MARIE...")
    print(f"File: {nc_file}")
    print(f"Init: {init_lat}N, {init_lon}W at {init_time}")
    
    run_comparison(nc_file, init_lat, init_lon, init_time, output_dir)
