from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import pyart

from netCDF4 import num2date
from pyart.core.transforms import cartesian_to_geographic

# NEW: parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------
# Paths/settings
# ------------------------------------------------------------------

base_dir = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/single_precip_sample/raw"
)

date_re = re.compile(r"^\d{8}$")
file_re = re.compile(r"^\d{6}\.mdv$")

# Separate folders for the 6-hour and 24-hour products.
OUT_FOLDER_6H = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/single_precip_sample/processed/6h"
)

OUT_FOLDER_24H = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/single_precip_sample/processed/24h"
)

regrid = True

EXAMPLE_GRID_PATH = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/example_grids/"
    "2026_radar_sample_grid_cropped_to_old_extent.nc"
)

regridding_method = "conservative"

var_in = "rate"
var_out = "rainfall_rate"

# Number of daily directories to process simultaneously.
N_WORKERS = 4

OVERWRITE = True

# ------------------------------------------------------------
# Completeness requirements
# ------------------------------------------------------------

# 5-minute data:
# 6 hours * 60 minutes / 5 minutes = 72 timestamps
EXPECTED_TIMESTEPS_6H = 72

# Four 6-hour periods make one 24-hour 06:00 -> 06:00 product
EXPECTED_TIMESTEPS_24H = 4

# Require at least 75% of the expected radar scans
MIN_COMPLETENESS = 0.75

MIN_TIMESTEPS_6H = int(
    np.ceil(
        EXPECTED_TIMESTEPS_6H
        * MIN_COMPLETENESS
    )
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_worker_regridder = None
_worker_target_ds = None

def initialise_worker():

    global _worker_target_ds

    _worker_target_ds = xr.open_dataset(
        EXAMPLE_GRID_PATH
    ).load()

def centres_to_bounds(coord):
    """Convert 1D cell-centre coordinates to cell-edge coordinates."""

    coord = np.asarray(coord)

    midpoints = (coord[:-1] + coord[1:]) / 2

    bounds = np.empty(len(coord) + 1)

    bounds[1:-1] = midpoints

    bounds[0] = coord[0] - (midpoints[0] - coord[0])

    bounds[-1] = coord[-1] + (coord[-1] - midpoints[-1])

    return bounds


def make_xesmf_grid(x, y, lon, lat, projection):
    """
    Construct an xESMF grid containing cell-centre and cell-corner
    longitude/latitude coordinates for conservative regridding.
    """

    # 1D cell edges in projected coordinates
    x_b = centres_to_bounds(x)
    y_b = centres_to_bounds(y)

    # Convert projected cell corners into 2D arrays
    xx_b, yy_b = np.meshgrid(x_b, y_b)

    # Convert cell corners to geographic coordinates
    lon_b, lat_b = cartesian_to_geographic(
        xx_b,
        yy_b,
        projection,
    )

    return xr.Dataset(
        coords={
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
            "lon_b": (("y_b", "x_b"), lon_b),
            "lat_b": (("y_b", "x_b"), lat_b),
        }
    )

def process_day(day_dir):

    global _worker_regridder
    global _worker_target_ds

    # If all expected outputs already exist, there is nothing
    # to process.
    # ----------------------------------------------------------

    day_timestamp = pd.to_datetime(
        day_dir.name,
        format="%Y%m%d",
    )

    expected_output_times = [
        day_timestamp + pd.Timedelta(hours=0),
        day_timestamp + pd.Timedelta(hours=6),
        day_timestamp + pd.Timedelta(hours=12),
        day_timestamp + pd.Timedelta(hours=18),
    ]

    expected_output_paths = [
        (
            OUT_FOLDER_6H
            / str(timestamp.year)
            / timestamp.strftime("%Y%m%d_%H.nc")
        )
        for timestamp in expected_output_times
    ]

    if (
        not OVERWRITE
        and all(
            path.is_file()
            for path in expected_output_paths
        )
    ):

        return (
            f"{day_dir.name}: "
            f"all 6h files already exist, skipping day"
        )

    #Start processing

    print(f"Looking in directory: {day_dir}")

    radar_projection = None

    # --------------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------------

    day_datasets = []

    day_fpaths = sorted(
        f
        for f in day_dir.iterdir()
        if f.is_file() and file_re.match(f.name)
    )

    bad_files = []

    for f in day_fpaths:

        try:

            radar = pyart.io.read_grid_mdv(
                f,
                file_field_names=True,
            )

            radar_time = num2date(
                radar.time["data"][0],
                units=radar.time["units"],
                calendar=radar.time.get(
                    "calendar",
                    "standard",
                ),
                only_use_cftime_datetimes=False,
            )

            x = radar.point_x["data"][0, 0, :]
            y = radar.point_y["data"][0, :, 0]

            lon, lat = radar.get_point_longitude_latitude(
                level=0
            )

            radar_data = radar.fields[var_in]["data"][0]

            if radar_projection is None:

                radar_projection = radar.projection.copy()

                radar_projection["lon_0"] = float(
                    radar.origin_longitude["data"][0]
                )

                radar_projection["lat_0"] = float(
                    radar.origin_latitude["data"][0]
                )

            ds = xr.Dataset(
                data_vars={
                    var_out: (
                        ("time", "y", "x"),
                        radar_data[np.newaxis, ...],
                    ),
                },
                coords={
                    "time": [np.datetime64(radar_time)],
                    "x": x,
                    "y": y,
                    "lon": (("y", "x"), lon),
                    "lat": (("y", "x"), lat),
                },
            )

            day_datasets.append(ds)

        except Exception as e:

            bad_files.append(
                (
                    f,
                    type(e).__name__,
                    str(e),
                )
            )

            print(
                f"{day_dir.name}: BAD FILE: "
                f"{f.name} "
                f"({type(e).__name__}: {e})"
            )

    # --------------------------------------------------------------
    # No files/data
    # --------------------------------------------------------------

    if not day_datasets:

        return (
            f"{day_dir.name}: no radar datasets found"
        )

    # --------------------------------------------------------------
    # CONCATENATION
    # --------------------------------------------------------------

    print(
        f"{day_dir.name}: "
        f"Concatenating {len(day_datasets)} datasets"
    )

    day_ds = xr.concat(
        day_datasets,
        dim="time",
    )

    day_ds = day_ds.sortby("time")

    # --------------------------------------------------------------
    # Number of radar timestamps in each 6-hour window
    # --------------------------------------------------------------

    time_counts = (
        xr.DataArray(
            np.ones(day_ds.sizes["time"]),
            coords={
                "time": day_ds.time,
            },
            dims="time",
        )
        .resample(
            time="6h",
            label="left",
        )
        .sum()
    )

    # --------------------------------------------------------------
    # 6-hour mean rainfall rate
    # --------------------------------------------------------------

    day_ds = (
        day_ds
        .resample(
            time="6h",
            label="left",
        )
        .mean("time")
    )

    # --------------------------------------------------------------
    # REGRIDDING
    # --------------------------------------------------------------

    if regrid:

        if _worker_regridder is None:

            print(
                f"{day_dir.name}: Creating regridder"
            )

            # Source grid
            source_grid = make_xesmf_grid(
                x=day_ds.x.values,
                y=day_ds.y.values,
                lon=day_ds.lon.values,
                lat=day_ds.lat.values,
                projection=radar_projection,
            )

            target_grid = make_xesmf_grid(
                x=_worker_target_ds.x.values,
                y=_worker_target_ds.y.values,
                lon=_worker_target_ds.lon.values,
                lat=_worker_target_ds.lat.values,
                projection=radar_projection,
            )

            _worker_regridder = xe.Regridder(
                source_grid,
                target_grid,
                method=regridding_method,
            )

        print(
            f"{day_dir.name}: Regridding"
        )

        day_ds = _worker_regridder(day_ds)

        # Give output grid its target projected coordinates
        day_ds = day_ds.assign_coords(
            x=("x", _worker_target_ds.x.values),
            y=("y", _worker_target_ds.y.values),
            lon=(
                ("y", "x"),
                _worker_target_ds.lon.values,
            ),
            lat=(
                ("y", "x"),
                _worker_target_ds.lat.values,
            ),
        )

        day_ds = day_ds.assign_coords(
            radar_centre_lon=float(
                radar.origin_longitude["data"][0]
            ),
            radar_centre_lat=float(
                radar.origin_latitude["data"][0]
            ),
        )

        day_ds.radar_centre_lon.attrs = {
            "long_name": "radar centre longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        }

        day_ds.radar_centre_lat.attrs = {
            "long_name": "radar centre latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        }

    # --------------------------------------------------------------
    # Save individual 6-hour products
    # --------------------------------------------------------------

    written = 0
    skipped = 0

    for t in day_ds.time.values:

        timestamp = pd.Timestamp(t)

        n_times = int(
            time_counts
            .sel(time=[t])
            .item()
        )

        # ----------------------------------------------------------
        # Output path
        # ----------------------------------------------------------

        date = timestamp.strftime("%Y%m%d")
        hour = timestamp.strftime("%H")

        year_out_folder = (
            OUT_FOLDER_6H
            / str(timestamp.year)
        )

        out_path = (
            year_out_folder
            / f"{date}_{hour}.nc"
        )

        # ----------------------------------------------------------
        # Check completeness
        # ----------------------------------------------------------

        if n_times < MIN_TIMESTEPS_6H:

            print(
                f"{day_dir.name}: "
                f"{timestamp}: "
                f"{n_times}/{EXPECTED_TIMESTEPS_6H} timestamps present "
                f"({100 * n_times / EXPECTED_TIMESTEPS_6H:.1f}%), "
                f"skipped"
            )

            skipped += 1
            continue

        # ----------------------------------------------------------
        # Skip existing file unless OVERWRITE=True
        # ----------------------------------------------------------

        if out_path.is_file() and not OVERWRITE:

            print(
                f"{day_dir.name}: "
                f"{out_path} already exists, skipping"
            )

            skipped += 1
            continue

        # ----------------------------------------------------------
        # Save
        # ----------------------------------------------------------

        print(
            f"{day_dir.name}: "
            f"{timestamp}: "
            f"{n_times}/{EXPECTED_TIMESTEPS_6H} timestamps present "
            f"({100 * n_times / EXPECTED_TIMESTEPS_6H:.1f}%), "
            f"saving"
        )

        year_out_folder.mkdir(
            exist_ok=True,
            parents=True,
        )

        ds_out = day_ds.sel(time=[t])

        ds_out.to_netcdf(out_path)

        written += 1

    return (
        f"{day_dir.name}: "
        f"{written} 6h files written, "
        f"{skipped} incomplete"
    )
# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


if __name__ == "__main__":

    print(f"OVERWRITE = {OVERWRITE}")

    print(
        f"Requiring at least "
        f"{MIN_TIMESTEPS_6H}/"
        f"{EXPECTED_TIMESTEPS_6H} "
        f"timestamps per 6h period "
        f"({MIN_COMPLETENESS:.0%})"
    )

    print(
        f"Regridding = {regrid}, "
        f"method = {regridding_method}"
    )

    # CREATE ALL 6-HOURLY FILES

    OUT_FOLDER_6H.mkdir(
        exist_ok=True,
        parents=True,
    )

    OUT_FOLDER_24H.mkdir(
        exist_ok=True,
        parents=True,
    )

    # --------------------------------------------------------------
    # Find daily directories
    # --------------------------------------------------------------

    day_dirs = sorted(
        day_dir
        for day_dir in base_dir.iterdir()
        if (
            day_dir.is_dir()
            and date_re.match(day_dir.name)
        )
    )

    print(
        f"Found {len(day_dirs)} daily directories"
    )

    print(
        f"Processing with {N_WORKERS} workers..."
    )

    # --------------------------------------------------------------
    # Process the daily directories in parallel.
    # --------------------------------------------------------------

    failed_days = []
    
    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=initialise_worker,
    ) as executor:

        futures = {

            executor.submit(
                process_day,
                day_dir,
            ): day_dir

            for day_dir in day_dirs
        }

        for future in as_completed(
            futures
        ):

            day_dir = futures[future]

            try:

                result = future.result()

                print(result)

            except Exception as e:

                print(
                    f"FAILED {day_dir}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                failed_days.append(
                    (
                        day_dir,
                        type(e).__name__,
                        str(e),
                    )
                )

    # --------------------------------------------------------------
    # Report failed days
    # --------------------------------------------------------------

    if failed_days:

        print(
            "\nWARNING: some daily "
            "directories failed:"
        )

        for (
            day_dir,
            error_type,
            error_msg,
        ) in failed_days:

            print(
                f"  {day_dir.name}: "
                f"{error_type}: "
                f"{error_msg}"
            )

    # ==============================================================
    # CREATE 24-HOUR 06:00 -> 06:00 PRODUCTS
    # ==============================================================

    print(
        "\nCreating 24-hour products..."
    )

    # --------------------------------------------------------------
    # Find all the 6-hour products we just created
    # --------------------------------------------------------------

    files = sorted(
        OUT_FOLDER_6H.glob("*/*.nc")
    )

    if not files:

        raise FileNotFoundError(
            f"No NetCDF files found in "
            f"{OUT_FOLDER_6H}"
        )

    print(
        f"Found {len(files)} 6-hour files"
    )

    # --------------------------------------------------------------
    # Open all 6-hour files
    # --------------------------------------------------------------

    ds_6h = xr.open_mfdataset(
        files,
        combine="by_coords",
    )

    data_6h = (
        ds_6h[var_out]
        .sortby("time")
    )

    # --------------------------------------------------------------
    # Remove duplicate timestamps, just in case
    # --------------------------------------------------------------

    _, unique_indices = np.unique(
        data_6h["time"].values,
        return_index=True,
    )

    data_6h = data_6h.isel(
        time=np.sort(unique_indices)
    )

    print(
        f"Found {data_6h.sizes['time']} "
        f"unique 6-hour timestamps"
    )

    # ==============================================================
    # Count available 6-hour periods
    # ==============================================================

    # Create one value for every available 6-hour timestep.
    #
    # This counts timestamps independently of whether individual
    # rainfall pixels contain NaNs.

    time_counts = xr.DataArray(
        np.ones(
            data_6h.sizes["time"],
            dtype=np.int8,
        ),
        coords={
            "time": data_6h["time"].values,
        },
        dims=["time"],
    )

    time_counts_shifted = (
        time_counts.assign_coords(
            time=(
                time_counts.time
                - pd.Timedelta("6h")
            )
        )
    )

    # Count the available 6-hour periods in each
    # 06:00 -> 06:00 window.

    counts_24h = (
        time_counts_shifted
        .resample(
            time="1D",
            closed="left",
            label="left",
        )
        .sum()
    )

    # Shift labels back to 06:00.

    counts_24h = counts_24h.assign_coords(
        time=(
            counts_24h.time
            + pd.Timedelta("6h")
        )
    )

    # ==============================================================
    # Calculate the 24-hour 06 -> 06 means
    # ==============================================================

    ds_shifted = data_6h.assign_coords(
        time=(
            data_6h.time
            - pd.Timedelta("6h")
        )
    )

    daily = (
        ds_shifted
        .resample(
            time="1D",
            closed="left",
            label="left",
        )
        .mean()
    )

    # Put the timestamps back at 06:00.

    daily = daily.assign_coords(
        time=(
            daily.time
            + pd.Timedelta("6h")
        )
    )

    # ==============================================================
    # Write 24-hour products
    # ==============================================================

    daily_written = 0
    daily_incomplete = 0
    daily_skipped = 0

    for time in daily.time.values:

        timestamp = pd.Timestamp(time)

        period_end = (
            timestamp
            + pd.Timedelta(days=1)
        )

        # ----------------------------------------------------------
        # Number of available 6-hour periods
        # ----------------------------------------------------------

        n_periods_value = (
            counts_24h
            .sel(time=time)
            .item()
        )

        if pd.isna(n_periods_value):

            n_periods = 0

        else:

            n_periods = int(
                n_periods_value
            )

        # ----------------------------------------------------------
        # Require ALL FOUR valid 6-hour products.
        #
        # Remember that the individual 6-hour files have already
        # passed the 75% raw-scan completeness test above.
        # ----------------------------------------------------------

        if n_periods != EXPECTED_TIMESTEPS_24H:

            print(
                f"Skipping 24h period "
                f"{timestamp:%Y-%m-%d %H:%M} -> "
                f"{period_end:%Y-%m-%d %H:%M}: "
                f"only {n_periods}/"
                f"{EXPECTED_TIMESTEPS_24H} "
                f"6h periods available"
            )

            daily_incomplete += 1

            continue

        # ----------------------------------------------------------
        # Extra timestamp sanity check from your second script
        # ----------------------------------------------------------

        expected_times = pd.DatetimeIndex([
            timestamp,
            timestamp + pd.Timedelta(hours=6),
            timestamp + pd.Timedelta(hours=12),
            timestamp + pd.Timedelta(hours=18),
        ])

        actual_times = pd.DatetimeIndex(
            pd.to_datetime(
                data_6h["time"]
                .sel(
                    time=slice(
                        timestamp,
                        (
                            period_end
                            - pd.Timedelta(
                                microseconds=1
                            )
                        ),
                    )
                )
                .values
            )
        )

        if not actual_times.equals(
            expected_times
        ):

            print(
                f"Skipping 24h period "
                f"{timestamp:%Y-%m-%d %H:%M} -> "
                f"{period_end:%Y-%m-%d %H:%M}: "
                f"unexpected 6h timestamps"
            )

            print(
                f"  Expected: "
                f"{list(expected_times)}"
            )

            print(
                f"  Found: "
                f"{list(actual_times)}"
            )

            daily_incomplete += 1

            continue

        # ----------------------------------------------------------
        # Output path
        # ----------------------------------------------------------

        year_dir = (
            OUT_FOLDER_24H
            / timestamp.strftime("%Y")
        )

        year_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_path = (
            year_dir
            / timestamp.strftime(
                "%Y%m%d_%H.nc"
            )
        )

        # ----------------------------------------------------------
        # Skip existing file unless OVERWRITE=True
        # ----------------------------------------------------------

        if out_path.is_file() and not OVERWRITE:

            print(
                f"{out_path} already exists, skipping"
            )

            daily_skipped += 1
            continue

        # ----------------------------------------------------------
        # Save
        # ----------------------------------------------------------

        year_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if out_path.is_file():
            print(f"Overwriting {out_path}")
        else:
            print(f"Saving {out_path}")

        (
            daily
            .sel(time=[time])
            .to_dataset(name=var_out)
            .to_netcdf(out_path)
        )

        daily_written += 1

    # --------------------------------------------------------------
    # Close input 6-hour files
    # --------------------------------------------------------------

    ds_6h.close()

    # ==============================================================
    # Final summary
    # ==============================================================

    print(
        "\n24-hour products:"
    )

    print(
        f"  Written: {daily_written}"
    )

    print(
        f"  Incomplete periods: "
        f"{daily_incomplete}"
    )

    if failed_days:

        print(
            f"\nWARNING: "
            f"{len(failed_days)} "
            f"daily directories failed."
        )

    print("\nDone")