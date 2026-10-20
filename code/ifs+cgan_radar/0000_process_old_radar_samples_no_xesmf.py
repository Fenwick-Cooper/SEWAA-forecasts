#Required conda environment (Mac/Linux):
# conda create -n radar_processing -c conda-forge --override-channels python=3.12 numpy pandas xarray dask scipy netcdf4 h5netcdf arm_pyart matplotlib bottleneck cftime -y
# conda activate radar_processing

from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
import pyart

from netCDF4 import num2date
from pyart.core.transforms import cartesian_to_geographic

from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------
# Paths/settings
# ------------------------------------------------------------------

base_dir = Path(
    "TO EDIT"
)

SKIP_DAYS = set(Path("dates.txt").read_text().splitlines())

# None = process all available days
# Set of dates = process only specified days
RETRY_DAYS = None
# RETRY_DAYS = {
#     "20190206",
#     "20190227",
#     "20190408",
#     "20190410",
# }

date_re = re.compile(r"^\d{8}$")
file_re = re.compile(r"^\d{6}\.mdv$")

# Separate folders for the 6-hour and 24-hour products.
OUT_FOLDER_6H = Path(
    # "/Volumes/Rain/processed_rwandan_radar/6h"
    "TO EDIT"
)

OUT_FOLDER_24H = Path(
    # "/Volumes/Rain/processed_rwandan_radar/24h"
    "TO EDIT"
)

regrid = True

EXAMPLE_GRID_PATH = Path(
    "eg_radar_grid.nc"
)

regridding_method = "conservative_cartesian"
# Minimum fraction of each target cell covered by valid radar data.
# 0.0 = accept any positive valid overlap.
# 1.0 = require complete coverage.
MIN_VALID_AREA_FRACTION = 0.0

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

MIN_TIMESTEPS_24H = int(
    np.ceil(EXPECTED_TIMESTEPS_24H * MIN_COMPLETENESS)
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

def calculate_1d_overlaps(source_centres, target_centres):
    """
    Calculate overlap lengths between source and target
    cells along one Cartesian coordinate axis.

    Returns a sparse matrix of shape
    (n_target, n_source).
    """

    source_centres = np.asarray(
        source_centres, dtype=np.float64
    )
    target_centres = np.asarray(
        target_centres, dtype=np.float64
    )

    def validate(coord, name):
        if coord.ndim != 1 or coord.size < 2:
            raise ValueError(
                f"{name} must contain at least two 1D centres"
            )

        if not np.all(np.isfinite(coord)):
            raise ValueError(
                f"{name} contains non-finite coordinates"
            )

        diff = np.diff(coord)

        if not (np.all(diff > 0) or np.all(diff < 0)):
            raise ValueError(
                f"{name} must be strictly monotonic"
            )

    validate(source_centres, "Source coordinates")
    validate(target_centres, "Target coordinates")

    source_reversed = source_centres[0] > source_centres[-1]
    target_reversed = target_centres[0] > target_centres[-1]

    source_sorted = (
        source_centres[::-1]
        if source_reversed else source_centres
    )
    target_sorted = (
        target_centres[::-1]
        if target_reversed else target_centres
    )

    source_edges = centres_to_bounds(source_sorted)
    target_edges = centres_to_bounds(target_sorted)

    rows = []
    cols = []
    values = []

    i = 0
    j = 0

    while i < len(source_sorted) and j < len(target_sorted):

        left = max(source_edges[i], target_edges[j])
        right = min(source_edges[i + 1], target_edges[j + 1])

        if right > left:
            source_index = (
                len(source_sorted) - 1 - i
                if source_reversed else i
            )
            target_index = (
                len(target_sorted) - 1 - j
                if target_reversed else j
            )

            rows.append(target_index)
            cols.append(source_index)
            values.append(right - left)

        if source_edges[i + 1] < target_edges[j + 1]:
            i += 1
        elif target_edges[j + 1] < source_edges[i + 1]:
            j += 1
        else:
            i += 1
            j += 1

    return csr_matrix(
        (values, (rows, cols)),
        shape=(len(target_centres), len(source_centres)),
    )


def make_conservative_weights(source_x, source_y,
                              target_x, target_y):
    """
    Construct sparse 2D cell-overlap area weights.

    Source and target must use the same Cartesian
    coordinate reference system.
    """

    from scipy.sparse import kron

    wx = calculate_1d_overlaps(source_x, target_x)
    wy = calculate_1d_overlaps(source_y, target_y)

    weights = kron(wy, wx, format="csr")

    return weights


def apply_conservative_regridding(
    data,
    weights,
    target_shape,
    target_cell_areas,
    min_valid_fraction=0.0,
):
    """
    Apply area-overlap weights to a rainfall field.

    data: NumPy array with shape (time, y, x)
    weights: sparse matrix of overlap areas
    target_shape: (target_ny, target_nx)

    Missing source values are excluded from the
    weighted mean.

    min_valid_fraction is the minimum fraction of
    the full target cell area containing valid data.
    """

    data = np.ma.filled(
        np.ma.asarray(data, dtype=np.float64),
        np.nan,
    )

    if data.ndim != 3:
        raise ValueError(
            "Expected data with dimensions (time, y, x)"
        )

    nt, ny, nx = data.shape

    if weights.shape != (
        target_shape[0] * target_shape[1],
        ny * nx,
    ):
        raise ValueError(
            "Regridding weights do not match data shape"
        )

    flat_data = data.reshape(nt, ny * nx)

    valid = np.isfinite(flat_data)

    clean_data = np.where(valid, flat_data, 0.0)

    # Area-weighted rainfall sums
    numerator = (weights @ clean_data.T).T

    # Total valid source area contributing to each target cell
    denominator = (
        weights @ valid.astype(np.float64).T
    ).T

    # Full target cell area, including any uncovered region
    # of the source domain
    total_target_area = np.asarray(
        target_cell_areas,
        dtype=np.float64,
    ).ravel()

    # Calculate weighted mean
    result = np.full_like(numerator, np.nan)

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )

    if min_valid_fraction > 0:
        fraction = np.divide(
            denominator,
            total_target_area[None, :],
            out=np.zeros_like(denominator),
            where=total_target_area[None, :] > 0,
        )

        result[fraction < min_valid_fraction] = np.nan

    return result.reshape(
        nt,
        target_shape[0],
        target_shape[1],
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

            #This is the start of the period
            radar_time = pd.Timestamp(
                radar.time["units"].split("since", 1)[1].strip()
            ).tz_localize(None)

            #This is the end of the period
            # radar_time = num2date(
            #     radar.time["data"][0],
            #     units=radar.time["units"],
            #     calendar=radar.time.get(
            #         "calendar",
            #         "standard",
            #     ),
            #     only_use_cftime_datetimes=False,
            # )

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
            np.ones(day_ds.sizes["time"], dtype=np.int32),
            coords={"time": day_ds.time},
            dims="time",
        )
        .resample(
            time="6h",
            label="left",
        )
        .sum()
        .fillna(0)
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

        source_x = day_ds.x.values
        source_y = day_ds.y.values

        target_x = _worker_target_ds.x.values
        target_y = _worker_target_ds.y.values

        if _worker_regridder is None:

            print(
                f"{day_dir.name}: Creating conservative "
                f"Cartesian regridding weights"
            )

            _worker_regridder = make_conservative_weights(
                source_x=source_x,
                source_y=source_y,
                target_x=target_x,
                target_y=target_y,
            )

        print(f"{day_dir.name}: Regridding")

        # Calculate actual target cell areas in projected units.
        target_x_edges = centres_to_bounds(target_x)
        target_y_edges = centres_to_bounds(target_y)

        target_cell_areas = (
            np.abs(np.diff(target_y_edges))[:, None]
            * np.abs(np.diff(target_x_edges))[None, :]
        )

        target_shape = (
            len(target_y),
            len(target_x),
        )

        regridded_data = apply_conservative_regridding(
            data=day_ds[var_out].values,
            weights=_worker_regridder,
            target_shape=target_shape,
            target_cell_areas=target_cell_areas,
            min_valid_fraction=MIN_VALID_AREA_FRACTION,
        )

        day_ds = xr.Dataset(
            data_vars={
                var_out: (
                    ("time", "y", "x"),
                    regridded_data,
                ),
            },
            coords={
                "time": day_ds.time.values,
                "x": target_x,
                "y": target_y,
                "lon": (
                    ("y", "x"),
                    _worker_target_ds.lon.values,
                ),
                "lat": (
                    ("y", "x"),
                    _worker_target_ds.lat.values,
                ),
            },
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

        count_value = time_counts.sel(time=t).item()

        n_times = (
            0 if pd.isna(count_value)
            else int(count_value)
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

    if RETRY_DAYS is not None:

        day_dirs = sorted(
            day_dir
            for day_dir in base_dir.iterdir()
            if (
                day_dir.is_dir()
                and day_dir.name in RETRY_DAYS
            )
        )

    else:

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

    if RETRY_DAYS is not None:
        print(f"Retrying days: {sorted(RETRY_DAYS)}")

    if SKIP_DAYS is not None:
        print("Removing days that have already been processed, from dates.txt")
        day_dirs = [d for d in day_dirs if d.name not in SKIP_DAYS]

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
                    f"{type(e).__name__}: {e}"
                )

                print(
                    "".join(
                        traceback.format_exception(
                            type(e),
                            e,
                            e.__traceback__,
                        )
                    )
                )

                failed_days.append(
                    (
                        day_dir,
                        type(e).__name__,
                        str(e),
                    )
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
        f
        for f in OUT_FOLDER_6H.glob("*/*.nc")
        if (
            f.is_file()
            and re.fullmatch(
                r"\d{8}_(00|06|12|18)\.nc",
                f.name,
            )
        )
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

    print("\nChecking 6-hour NetCDF files...")

    good_files = []
    bad_files = []

    for f in files:

        try:

            with xr.open_dataset(
                f,
                engine="netcdf4",
            ) as ds:

                if var_out not in ds:
                    raise ValueError(
                        f"Missing variable: {var_out}"
                    )

                if "time" not in ds.coords:
                    raise ValueError(
                        "Missing time coordinate"
                    )

                # Force an actual read, not just metadata inspection.
                ds[var_out].load()

            good_files.append(f)

        except Exception as e:

            bad_files.append(
                (
                    f,
                    type(e).__name__,
                    str(e),
                )
            )

            print(
                f"BAD NETCDF: {f}\n"
                f"  {type(e).__name__}: {e}"
            )

    print(
        f"\nValid files: {len(good_files)}"
    )

    print(
        f"Invalid files: {len(bad_files)}"
    )
    
    if not good_files:
        raise FileNotFoundError(
            "No valid 6-hour NetCDF files found."
        )

    ds_6h = xr.open_mfdataset(
        good_files,
        combine="by_coords",
        engine="netcdf4",
        parallel=False,
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

        if n_periods < MIN_TIMESTEPS_24H:

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

        # Check that all available timestamps are expected
        # and that at least 3 of the 4 periods are present.

        valid_times = actual_times.isin(expected_times).all()

        if (
            not valid_times
            or len(actual_times) < MIN_TIMESTEPS_24H
        ):

            print(
                f"Skipping 24h period "
                f"{timestamp:%Y-%m-%d %H:%M} -> "
                f"{period_end:%Y-%m-%d %H:%M}: "
                f"invalid or insufficient 6h timestamps"
            )

            print(f"  Expected: {list(expected_times)}")
            print(f"  Found: {list(actual_times)}")

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