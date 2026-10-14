import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
from xarray import SerializationWarning


warnings.filterwarnings(
    "ignore",
    category=SerializationWarning,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/new_radar_sample/raw"
)

OUT_DIR_6H = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/new_radar_sample/processed/6h"
)

OUT_DIR_24H = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/new_radar_sample/processed/24h"
)

example_grid_path = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/example_grids/"
    "2026_radar_sample_grid_cropped_to_old_extent.nc"
)


var = "qpe"
var_out = "rainfall_rate"


# ------------------------------------------------------------
# Parallel processing
# ------------------------------------------------------------

# Start conservatively because we're reading from
# a network filesystem.
N_WORKERS = 4


# ------------------------------------------------------------
# Output behaviour
# ------------------------------------------------------------

# True:
#     regenerate existing 6-hour and 24-hour files.
#
# False:
#     leave existing files untouched and skip them.
OVERWRITE = False


# ------------------------------------------------------------
# Completeness requirements
# ------------------------------------------------------------

# 5-minute data:
#
# 6 hours * 60 minutes / 5 minutes = 72 timestamps
EXPECTED_TIMESTEPS_6H = 72
EXPECTED_TIMESTEPS_24H = 4

# Require at least 90% of the expected radar scans.
MIN_COMPLETENESS = 0.90

MIN_TIMESTEPS_6H = int(
    np.ceil(
        EXPECTED_TIMESTEPS_6H
        * MIN_COMPLETENESS
    )
)


# ============================================================
# Process one daily directory
# ============================================================

def process_day(
    subdir,
    x_min,
    x_max,
    y_min,
    y_max,
):

    # --------------------------------------------------------
    # Find raw NetCDF files
    # --------------------------------------------------------

    files = sorted(
        f
        for f in subdir.rglob("*.nc")
        if f.is_file()
    )

    if not files:
        return f"{subdir.name}: no files"

    arrays = []
    bad_files = []

    # --------------------------------------------------------
    # Read individual radar files
    # --------------------------------------------------------

    for f in files:

        try:

            with xr.open_dataset(f) as ds:

                # --------------------------------------------
                # Check required variable
                # --------------------------------------------

                if var not in ds:

                    raise KeyError(
                        f"{var!r} not found in {f}"
                    )

                da = ds[var]

                # --------------------------------------------
                # Rename coordinates if present
                # --------------------------------------------

                rename_dict = {}

                if "longitude" in da.coords:
                    rename_dict[
                        "longitude"
                    ] = "central_lon"

                if "latitude" in da.coords:
                    rename_dict[
                        "latitude"
                    ] = "central_lat"

                if rename_dict:
                    da = da.rename(
                        rename_dict
                    )

                # --------------------------------------------
                # Check spatial coordinates
                # --------------------------------------------

                if (
                    "x" not in da.coords
                    or "y" not in da.coords
                ):

                    raise ValueError(
                        f"{f} does not contain "
                        f"x/y coordinates"
                    )

                # --------------------------------------------
                # Work out coordinate orientation
                # --------------------------------------------

                x_ascending = (
                    da["x"].values[0]
                    < da["x"].values[-1]
                )

                y_ascending = (
                    da["y"].values[0]
                    < da["y"].values[-1]
                )

                x_slice = (
                    slice(
                        x_min,
                        x_max,
                    )
                    if x_ascending
                    else slice(
                        x_max,
                        x_min,
                    )
                )

                y_slice = (
                    slice(
                        y_min,
                        y_max,
                    )
                    if y_ascending
                    else slice(
                        y_max,
                        y_min,
                    )
                )

                # --------------------------------------------
                # Crop BEFORE loading
                # --------------------------------------------

                da = da.sel(
                    x=x_slice,
                    y=y_slice,
                )

                # Force the actual NetCDF/HDF read while
                # inside the try/except.
                #
                # If the source file is corrupt, the error
                # will therefore be caught below.
                da = da.load()

                arrays.append(da)

        except Exception as e:

            bad_files.append(
                (
                    f,
                    type(e).__name__,
                    str(e),
                )
            )

    # --------------------------------------------------------
    # Report corrupt/unreadable files
    # --------------------------------------------------------

    if bad_files:

        print(
            f"{subdir.name}: WARNING - "
            f"skipped {len(bad_files)} "
            f"bad file(s)"
        )

        for (
            bad_file,
            error_type,
            error_msg,
        ) in bad_files:

            print(
                f"{subdir.name}: BAD FILE: "
                f"{bad_file.name} "
                f"({error_type}: "
                f"{error_msg})"
            )

    # --------------------------------------------------------
    # Make sure something was successfully read
    # --------------------------------------------------------

    if not arrays:

        return (
            f"{subdir.name}: "
            f"{len(files)} raw files, "
            f"0 readable files"
        )

    # ========================================================
    # Combine this day's timestamps
    # ========================================================

    full_data = xr.concat(
        arrays,
        dim="time",
    ).sortby("time")

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    _, unique_indices = np.unique(
        full_data.time.values,
        return_index=True,
    )

    full_data = full_data.isel(
        time=sorted(
            unique_indices
        )
    )

    n_readable = len(arrays)
    n_unique = full_data.sizes["time"]

    # ========================================================
    # 6-hour means
    # ========================================================

    # Number of available timestamps in each
    # 6-hour window.
    counts_6h = (
        full_data["time"]
        .resample(time="6h")
        .count()
    )

    # Mean of whatever valid timestamps are
    # available within each window.
    data_6h = (
        full_data
        .resample(time="6h")
        .mean()
    )

    written = 0
    overwritten = 0
    skipped_existing = 0
    skipped_incomplete = 0

    # --------------------------------------------------------
    # Write each 6-hour period
    # --------------------------------------------------------

    for time in data_6h.time.values:

        timestamp = pd.Timestamp(
            time
        )

        # --------------------------------------------
        # Number of actual radar scans
        # --------------------------------------------

        n_timesteps = int(
            counts_6h
            .sel(time=time)
            .item()
        )

        completeness = (
            n_timesteps
            / EXPECTED_TIMESTEPS_6H
        )

        # --------------------------------------------
        # Reject incomplete periods
        # --------------------------------------------

        if n_timesteps < MIN_TIMESTEPS_6H:

            # Work out where this period's output would be
            year_dir = (
                OUT_DIR_6H
                / timestamp.strftime("%Y")
            )

            out_path = (
                year_dir
                / timestamp.strftime("%Y%m%d_%H.nc")
            )

            print(
                f"{subdir.name}: skipping "
                f"{timestamp:%Y-%m-%d %H:%M} -- "
                f"{n_timesteps}/{EXPECTED_TIMESTEPS_6H} "
                f"timestamps "
                f"({completeness:.1%})"
            )

            # If doing a full overwrite/rebuild, remove any
            # stale output that no longer passes QC.
            if OVERWRITE and out_path.is_file():

                print(
                    f"{subdir.name}: removing existing "
                    f"incomplete 6h product: "
                    f"{out_path}"
                )

                out_path.unlink()

            skipped_incomplete += 1
            continue

        # --------------------------------------------
        # Output path
        # --------------------------------------------

        year_dir = (
            OUT_DIR_6H
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

        file_already_exists = (
            out_path.is_file()
        )

        # --------------------------------------------
        # Skip existing output unless overwrite=True
        # --------------------------------------------

        if (
            file_already_exists
            and not OVERWRITE
        ):

            skipped_existing += 1

            continue

        # --------------------------------------------
        # Prepare output
        # --------------------------------------------

        data_out = (
            data_6h
            .sel(time=[time])
            .to_dataset(
                name=var_out
            )
        )

        # --------------------------------------------
        # Write
        # --------------------------------------------

        data_out.to_netcdf(
            out_path
        )

        if file_already_exists:
            overwritten += 1
        else:
            written += 1

    # ========================================================
    # Return summary
    # ========================================================

    summary = (
        f"{subdir.name}: "
        f"{len(files)} raw files, "
        f"{n_readable} readable, "
        f"{n_unique} unique timestamps -> "
        f"{len(data_6h.time)} periods, "
        f"{written} written"
    )

    if OVERWRITE:
        summary += (
            f", {overwritten} overwritten"
        )
    else:
        summary += (
            f", {skipped_existing} "
            f"already existed"
        )

    summary += (
        f", {skipped_incomplete} incomplete"
    )

    if bad_files:

        summary += (
            f", {len(bad_files)} "
            f"bad file(s)"
        )

    return summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    OUT_DIR_6H.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_DIR_24H.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print(
        f"OVERWRITE = {OVERWRITE}"
    )

    print(
        f"Requiring at least "
        f"{MIN_TIMESTEPS_6H}/"
        f"{EXPECTED_TIMESTEPS_6H} "
        f"timestamps per 6h period "
        f"({MIN_COMPLETENESS:.0%})"
    )

    # ========================================================
    # Load crop bounds once
    # ========================================================

    with xr.open_dataset(
        example_grid_path
    ) as example_grid:

        if (
            "x" not in example_grid.coords
            or "y"
            not in example_grid.coords
        ):

            raise ValueError(
                "example_grid must contain "
                "x/y coordinates"
            )

        x_min = float(
            example_grid["x"].min()
        )

        x_max = float(
            example_grid["x"].max()
        )

        y_min = float(
            example_grid["y"].min()
        )

        y_max = float(
            example_grid["y"].max()
        )

    print(
        f"Crop bounds: "
        f"x={x_min:.3f} "
        f"to {x_max:.3f}, "
        f"y={y_min:.3f} "
        f"to {y_max:.3f}"
    )

    # ========================================================
    # Find daily directories
    # ========================================================

    subdirs = sorted(
        p
        for p in BASE_DIR.iterdir()
        if p.is_dir()
    )

    print(
        f"Found "
        f"{len(subdirs)} "
        f"daily directories"
    )

    print(
        f"Processing with "
        f"{N_WORKERS} workers..."
    )

    # ========================================================
    # Process days in parallel
    # ========================================================

    failed_days = []

    with ProcessPoolExecutor(
        max_workers=N_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                process_day,
                subdir,
                x_min,
                x_max,
                y_min,
                y_max,
            ): subdir

            for subdir in subdirs
        }

        for future in as_completed(
            futures
        ):

            subdir = futures[
                future
            ]

            try:

                result = (
                    future.result()
                )

                print(result)

            except Exception as e:

                print(
                    f"FAILED "
                    f"{subdir}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                failed_days.append(
                    (
                        subdir,
                        type(e).__name__,
                        str(e),
                    )
                )

    # --------------------------------------------------------
    # Report failed days
    # --------------------------------------------------------

    if failed_days:

        print(
            "\n"
            "WARNING: some daily "
            "directories failed:"
        )

        for (
            subdir,
            error_type,
            error_msg,
        ) in failed_days:

            print(
                f"  {subdir.name}: "
                f"{error_type}: "
                f"{error_msg}"
            )

    # ========================================================
    # Construct 24-hour 06 -> 06 products
    # ========================================================

    print("\nCreating 24-hour products...")

    files = sorted(
        OUT_DIR_6H.glob("*/*.nc")
    )

    if not files:
        raise FileNotFoundError(
            f"No NetCDF files found in {OUT_DIR_6H}"
        )

    print(
        f"Found {len(files)} 6-hour files"
    )

    # ========================================================
    # Open all 6-hour files
    # ========================================================

    ds_6h = xr.open_mfdataset(
        files,
        combine="by_coords",
    )

    data_6h = (
        ds_6h[var_out]
        .sortby("time")
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps, just in case
    # --------------------------------------------------------

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

    # ========================================================
    # Count available 6-hour periods
    # ========================================================

    # Create a simple 1D DataArray containing one value for
    # every available 6-hour timestep.
    #
    # This lets us count timestamps independently of whether
    # individual rainfall pixels contain NaNs.
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

    # Shift timestamps backwards by 6 hours so that
    # 06:00 -> 06:00 periods align with calendar days.
    time_counts_shifted = time_counts.assign_coords(
        time=time_counts.time - pd.Timedelta("6h")
    )

    # Count the number of available 6-hour periods
    # in each 06:00 -> 06:00 daily window.
    counts_24h = (
        time_counts_shifted
        .resample(
            time="1D",
            closed="left",
            label="left",
        )
        .sum()
    )

    # Shift the daily labels back to 06:00.
    counts_24h = counts_24h.assign_coords(
        time=counts_24h.time + pd.Timedelta("6h")
    )

    # ========================================================
    # Calculate 24-hour products
    # ========================================================

    ds_shifted = data_6h.assign_coords(time=data_6h.time - pd.Timedelta("6h"))

    daily = ds_shifted.resample(
                time="1D",
                closed="left",
                label="left",
            ).mean()

    daily = daily.assign_coords(time=daily.time + pd.Timedelta("6h"))

    # ========================================================
    # Write 24-hour products
    # ========================================================

    daily_written = 0
    daily_overwritten = 0
    daily_skipped = 0
    daily_incomplete = 0
    daily_deleted = 0

    for time in daily.time.values:

        timestamp = pd.Timestamp(time)

        period_end = (
            timestamp
            + pd.Timedelta(days=1)
        )

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        year_dir = (
            OUT_DIR_24H
            / timestamp.strftime("%Y")
        )

        out_path = (
            year_dir
            / timestamp.strftime("%Y%m%d_%H.nc")
        )

        # ----------------------------------------------------
        # Number of available 6-hour periods
        # ----------------------------------------------------

        n_periods_value = (
            counts_24h
            .sel(time=time)
            .item()
        )

        if pd.isna(n_periods_value):
            n_periods = 0
        else:
            n_periods = int(n_periods_value)

        # ----------------------------------------------------
        # Require all four 6-hour periods
        # ----------------------------------------------------

        if n_periods != EXPECTED_TIMESTEPS_24H:

            print(
                f"Skipping 24h period "
                f"{timestamp:%Y-%m-%d %H:%M} -> "
                f"{period_end:%Y-%m-%d %H:%M}: "
                f"only {n_periods}/"
                f"{EXPECTED_TIMESTEPS_24H} "
                f"6h periods available"
            )

            # If this is an overwrite/rebuild run, remove
            # any stale 24-hour product that no longer
            # satisfies the completeness requirement.
            if (
                OVERWRITE
                and out_path.is_file()
            ):

                print(
                    f"Removing existing incomplete "
                    f"24h product: {out_path}"
                )

                out_path.unlink()

                daily_deleted += 1

            daily_incomplete += 1

            continue

        # ----------------------------------------------------
        # Extra timestamp sanity check
        # ----------------------------------------------------

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
                        period_end
                        - pd.Timedelta(microseconds=1),
                    )
                )
                .values
            )
        )

        if not actual_times.equals(expected_times):

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

            if (
                OVERWRITE
                and out_path.is_file()
            ):

                print(
                    f"Removing existing invalid "
                    f"24h product: {out_path}"
                )

                out_path.unlink()

                daily_deleted += 1

            daily_incomplete += 1

            continue

        # ----------------------------------------------------
        # Valid period
        # ----------------------------------------------------

        year_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_already_exists = (
            out_path.is_file()
        )

        # ----------------------------------------------------
        # Skip existing file unless OVERWRITE=True
        # ----------------------------------------------------

        if (
            file_already_exists
            and not OVERWRITE
        ):

            print(
                f"{out_path} already exists, skipping"
            )

            daily_skipped += 1

            continue

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        if file_already_exists:

            print(
                f"Overwriting {out_path}"
            )

        else:

            print(
                f"Saving {out_path}"
            )

        (
            daily
            .sel(time=[time])
            .to_dataset(name=var_out)
            .to_netcdf(out_path)
        )

        if file_already_exists:
            daily_overwritten += 1
        else:
            daily_written += 1

    # ========================================================
    # Close input files
    # ========================================================

    ds_6h.close()

    # ========================================================
    # Final summary
    # ========================================================

    print("\n24-hour products:")

    print(
        f"  Written: {daily_written}"
    )

    if OVERWRITE:

        print(
            f"  Overwritten: "
            f"{daily_overwritten}"
        )

        print(
            f"  Invalid old files deleted: "
            f"{daily_deleted}"
        )

    else:

        print(
            f"  Existing/skipped: "
            f"{daily_skipped}"
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