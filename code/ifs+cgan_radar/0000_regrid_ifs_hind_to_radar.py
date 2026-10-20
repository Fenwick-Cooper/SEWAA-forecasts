# Load IFS hindcast GRIB files on reduced Gaussian grid,
# calculate ensemble mean / standard deviation,
# regrid to desired grid
# and write one NetCDF file per year / variable.
#
# Output structure:
#
#   parent_output_dir/
#       2014/
#           tp.nc
#           t2m.nc
#           ...
#       2015/
#           ...
#
# Each file contains:
#   time
#   fcst_valid_time
#   <variable>_mean
#   <variable>_sd
import os
import traceback
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.spatial import Delaunay
from pyproj import CRS, Transformer
# ============================================================
# Configuration
# ============================================================
# Hindcast GRIB files:
# one directory per variable, one file per initialisation date.
INPUT_DIR = Path("/network/group/aopp/predict/AWH024_COOPERNATH_IFS/" "cGAN_gefs/processed_IFS/IFSens_hindcasts")
# Output directory
PARENT_OUTPUT_DIR = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/IFS/hindcasts/2026_radar_sample_grid_cropped_to_old_extent"
)
# Example radar file defining target grid
TARGET_GRID_FILE = Path(
    "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/"
    "rwanda_radar_data/example_grids/"
    "2026_radar_sample_grid_cropped_to_old_extent.nc"
)
COARSENING_FACTOR = 9
# Hindcast years contained in each GRIB
YEARS = [2023]
# Expected steps
EXPECTED_NUM_VALID_TIMES = 40
# ============================================================
# Variable definitions
# ============================================================
VARS_SFC = {
    "mucape": {
        "grib_key": "mucape",
        "long_name": "Most unstable convective available potential energy",
        "units": "J kg**-1",
        "accumulated": False,
    },
    "tclw": {
        "grib_key": "tclw",
        "long_name": "Total column cloud liquid water",
        "units": "kg m**-2",
        "accumulated": False,
    },
    "tciw": {
        "grib_key": "tciw",
        "long_name": "Total column cloud ice water",
        "units": "kg m**-2",
        "accumulated": False,
    },
    "tcrw": {
        "grib_key": "tcrw",
        "long_name": "Total column rain water",
        "units": "kg m**-2",
        "accumulated": False,
    },
    "sp": {
        "grib_key": "sp",
        "long_name": "Surface pressure",
        "units": "Pa",
        "accumulated": False,
    },
    "tcw": {
        "grib_key": "tcw",
        "long_name": "Total column water",
        "units": "kg m**-2",
        "accumulated": False,
    },
    "tcwv": {
        "grib_key": "tcwv",
        "long_name": "Total column vertically-integrated water vapour",
        "units": "kg m**-2",
        "accumulated": False,
    },
    "lsp": {
        "grib_key": "lsp",
        "long_name": "Large-scale precipitation",
        "units": "m/6h",
        "accumulated": True,
        "accumulation_per_h": 1,
    },
    "cp": {
        "grib_key": "cp",
        "long_name": "Convective precipitation",
        "units": "m/6h",
        "accumulated": True,
        "accumulation_per_h": 1,
    },
    "u10": {
        "grib_key": "u10",
        "long_name": "10 metre U wind component",
        "units": "m s**-1",
        "accumulated": False,
    },
    "v10": {
        "grib_key": "v10",
        "long_name": "10 metre V wind component",
        "units": "m s**-1",
        "accumulated": False,
    },
    "t2m": {
        "grib_key": "t2m",
        "long_name": "2 metre temperature",
        "units": "K",
        "accumulated": False,
    },
    "ssr": {
        "grib_key": "ssr",
        "long_name": "Surface net solar radiation",
        "units": "J m**-2 / 6h",
        "accumulated": True,
        "accumulation_per_h": 3600,
    },
    "mcc": {
        "grib_key": "mcc",
        "long_name": "Medium cloud cover",
        "units": "(0 - 1)",
        "accumulated": False,
    },
    "tp": {
        "grib_key": "tp",
        "long_name": "Total precipitation",
        "units": "m/6h",
        "accumulated": True,
        "accumulation_per_h": 1,
    },
    "tcc": {
        "grib_key": "tcc",
        "long_name": "Total cloud cover",
        "units": "(0 - 1)",
        "accumulated": False,
    },
}
VARS_PL700 = {
    "u700": {
        "grib_key": "u",
        "long_name": "U component of wind at 700 hPa",
        "units": "m s**-1",
        "accumulated": False,
    },
    "v700": {
        "grib_key": "v",
        "long_name": "V component of wind at 700 hPa",
        "units": "m s**-1",
        "accumulated": False,
    },
}
# ============================================================
# Date handling
# ============================================================
def get_initialisation_dates():
    """
    Return the 2025 initialisation dates corresponding to the
    dates processed by the original script.
    Initialisation dates are:
        1st, 5th, 9th, 13th, ...
    but 6 April through 30 September are excluded.
    """
    dates = []
    for month in range(1, 13):
        for day in range(1, 32, 4):
            try:
                today = datetime(2025, month, day)
            except ValueError:
                continue
            # Exclude dates we have not fully downloaded yet, or with corrupt hindcasts
            if (month == 3 and day == 17) or (month == 10 and day < 6) or (4 <= month <= 9):
                continue
            dates.append(today)
    return dates
# ============================================================
# Target grid
# ============================================================
def load_target_grid():
    """Load and coarsen the Rwanda radar grid exactly as in the forecast script."""
    with xr.open_dataset(TARGET_GRID_FILE) as radar:
        radar_x = radar["x"].values.copy()
        radar_y = radar["y"].values.copy()
        radar_centre_lat = radar["central_lat"].values.copy()
        radar_centre_lon = radar["central_lon"].values.copy()
        if "lat" in radar.data_vars:
            radar_lat = radar["lat"].values.copy()
            radar_lon = radar["lon"].values.copy()
        else:
            radar_crs = CRS.from_proj4(
                f"+proj=aeqd +lat_0={radar_centre_lat} " f"+lon_0={radar_centre_lon} +datum=WGS84 +units=m"
            )
            to_lonlat = Transformer.from_crs(radar_crs, "EPSG:4326", always_xy=True)
            radar_xx, radar_yy = np.meshgrid(radar_x, radar_y)
            radar_lon, radar_lat = to_lonlat.transform(radar_xx, radar_yy)
    factor = COARSENING_FACTOR
    crop_x = len(radar_x) % factor
    crop_y = len(radar_y) % factor
    print("Cropping x by", crop_x)
    print("Cropping y by", crop_y)
    left_x = crop_x // 2
    right_x = crop_x - left_x
    if crop_x > 0:
        radar_x = radar_x[left_x : len(radar_x) - right_x]
        radar_lon = radar_lon[:, left_x : radar_lon.shape[1] - right_x]
        radar_lat = radar_lat[:, left_x : radar_lat.shape[1] - right_x]
    left_y = crop_y // 2
    right_y = crop_y - left_y
    if crop_y > 0:
        radar_y = radar_y[left_y : len(radar_y) - right_y]
        radar_lon = radar_lon[left_y : radar_lon.shape[0] - right_y, :]
        radar_lat = radar_lat[left_y : radar_lat.shape[0] - right_y, :]
    centre = factor // 2
    radar_x = radar_x[centre::factor]
    radar_y = radar_y[centre::factor]
    radar_lon = radar_lon[centre::factor, centre::factor]
    radar_lat = radar_lat[centre::factor, centre::factor]
    assert radar_lon.shape == radar_lat.shape
    assert radar_lon.shape == (len(radar_y), len(radar_x))
    return radar_x, radar_y, radar_lon, radar_lat, radar_centre_lon, radar_centre_lat
# ============================================================
# Interpolation
# ============================================================
def build_interpolation_mapping(
    source_lon,
    source_lat,
    target_lon,
    target_lat,
):
    """
    Precompute the linear interpolation mapping from the
    irregular/reduced Gaussian source grid to the regular target
    grid.
    This replaces repeatedly constructing LinearNDInterpolator.
    Returns
    -------
    vertices
        Shape (n_target, 3). Indices of source-grid vertices.
    weights
        Shape (n_target, 3). Barycentric interpolation weights.
    inside
        Boolean array indicating target points inside triangulation.
    target_shape
        (n_lat, n_lon)
    """
    source_lon = np.asarray(source_lon)
    source_lat = np.asarray(source_lat)
    # Convert 0..360 longitude to -180..180
    if np.nanmax(source_lon) > 180:
        source_lon = ((source_lon + 180) % 360) - 180
    points = np.column_stack((source_lon, source_lat))
    if target_lon.shape != target_lat.shape:
        raise ValueError("target_lon and target_lat must have the same shape")
    target_points = np.column_stack((target_lon.ravel(), target_lat.ravel()))
    print(
        f"Building Delaunay triangulation "
        f"({len(points):,} source points -> "
        f"{len(target_points):,} target points)"
    )
    print(
        f"Source lon: {source_lon.min():.2f} to {source_lon.max():.2f}, "
        f"lat: {source_lat.min():.2f} to {source_lat.max():.2f}",
        flush=True,
    )
    print(
        f"Target lon: {np.min(target_lon):.2f} to {np.max(target_lon):.2f}, "
        f"lat: {np.min(target_lat):.2f} to {np.max(target_lat):.2f}",
        flush=True,
    )
    tri = Delaunay(points)
    # Which triangle contains each target point?
    simplex = tri.find_simplex(target_points)
    inside = simplex >= 0
    n_target = len(target_points)
    vertices = np.zeros(
        (n_target, 3),
        dtype=np.int32,
    )
    weights = np.full(
        (n_target, 3),
        np.nan,
        dtype=np.float64,
    )
    # Only calculate weights for target points inside
    # the source triangulation.
    valid_simplex = simplex[inside]
    vertices[inside] = tri.simplices[valid_simplex]
    # scipy Delaunay transform:
    #
    # transform[simplex, :2] = transformation matrix
    # transform[simplex, 2]  = offset
    #
    transform = tri.transform[valid_simplex, :2, :]
    delta = target_points[inside] - tri.transform[valid_simplex, 2, :]
    bary_first_two = np.einsum(
        "nij,nj->ni",
        transform,
        delta,
    )
    weights[inside, :2] = bary_first_two
    weights[inside, 2] = 1.0 - bary_first_two[:, 0] - bary_first_two[:, 1]
    target_shape = target_lon.shape
    return (
        vertices,
        weights,
        inside,
        target_shape,
    )
def interpolate_fields(
    fields,
    vertices,
    weights,
    inside,
    target_shape,
):
    """
    Interpolate multiple fields simultaneously.
    Parameters
    ----------
    fields
        Array of shape:
            (n_fields, n_source_points)
        e.g. all forecast lead times.
    Returns
    -------
    result
        Shape:
            (n_fields, n_target_lat, n_target_lon)
    """
    fields = np.asarray(fields)
    n_fields = fields.shape[0]
    n_target = len(inside)
    result = np.full(
        (n_fields, n_target),
        np.nan,
        dtype=np.float32,
    )
    v = vertices[inside]
    w = weights[inside]
    # fields[:, v] has dimensions:
    #
    #     field x target_point x triangle_vertex
    #
    # Multiply by barycentric weights and sum over
    # triangle vertices.
    result[:, inside] = np.einsum(
        "fti,ti->ft",
        fields[:, v],
        w,
        optimize=True,
    ).astype(np.float32)
    return result.reshape(
        n_fields,
        target_shape[0],
        target_shape[1],
    )
# ============================================================
# Ensemble statistics
# ============================================================
def calculate_statistics(data, accumulated):
    """
    Calculate ensemble mean and standard deviation for all
    valid times simultaneously.
    Expected input shape:
        ensemble_member x valid_time x source_point
    This matches the indexing used by the original script.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(
            "Expected GRIB data with 3 dimensions " "(ensemble, valid_time, point), " f"but got shape {data.shape}"
        )
    if accumulated:
        increments = np.diff(
            data,
            axis=1,
        )
        mean_increment = np.mean(
            increments,
            axis=0,
            dtype=np.float64,
        )
        std_increment = np.std(
            increments,
            axis=0,
            ddof=1,
            dtype=np.float64,
        )
        n_valid_times = data.shape[1]
        n_points = data.shape[2]
        means = np.zeros(
            (n_valid_times, n_points),
            dtype=np.float32,
        )
        stds = np.zeros(
            (n_valid_times, n_points),
            dtype=np.float32,
        )
        means[:-1] = mean_increment
        stds[:-1] = std_increment
        # Final timestep deliberately remains zero,
        # matching original implementation.
    else:
        means = np.mean(
            data,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        stds = np.std(
            data,
            axis=0,
            ddof=1,
            dtype=np.float64,
        ).astype(np.float32)
    return means, stds
# ============================================================
# NetCDF handling
# ============================================================
def output_file_is_valid(path, key):
    if not path.exists():
        return False
    try:
        with nc.Dataset(path, "r") as ds:
            mean_shape = ds.variables[f"{key}_mean"].shape
            std_shape = ds.variables[f"{key}_sd"].shape
            if mean_shape != std_shape:
                return False
            # Require at least one time record
            if mean_shape[0] == 0:
                return False
            # Only set this after processing finishes
            if getattr(ds, "processing_complete", 0) != 1:
                return False
        return True
    except (OSError, KeyError, RuntimeError):
        return False
def create_output_file(
    output_file,
    key,
    metadata,
    target_x,
    target_y,
    target_lon,
    target_lat,
    central_lon,
    central_lat,
    num_valid_times,
):
    """Create one output NetCDF file on the radar x/y grid."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rootgrp = nc.Dataset(output_file, "w", format="NETCDF4")
    rootgrp.description = "IFS hindcast ensemble mean and standard deviation interpolated to the Rwanda radar grid."
    rootgrp.createDimension("y", len(target_y))
    rootgrp.createDimension("x", len(target_x))
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("valid_time", num_valid_times)
    x_data = rootgrp.createVariable("x", "f4", ("x",), zlib=False)
    y_data = rootgrp.createVariable("y", "f4", ("y",), zlib=False)
    lon_data = rootgrp.createVariable("longitude", "f4", ("y", "x"), zlib=False)
    lat_data = rootgrp.createVariable("latitude", "f4", ("y", "x"), zlib=False)
    central_lon_data = rootgrp.createVariable("central_lon", "f4", ())
    central_lat_data = rootgrp.createVariable("central_lat", "f4", ())
    x_data.units = "m"
    y_data.units = "m"
    lon_data.units = "degrees_east"
    lat_data.units = "degrees_north"
    x_data[:] = target_x
    y_data[:] = target_y
    lon_data[:] = target_lon
    lat_data[:] = target_lat
    central_lon_data[:] = central_lon
    central_lat_data[:] = central_lat
    time_data = rootgrp.createVariable("time", "f8", ("time",), zlib=False)
    time_data.units = "hours since 1900-01-01 00:00:00.0"
    valid_time_data = rootgrp.createVariable("fcst_valid_time", "f8", ("time", "valid_time"), zlib=False)
    valid_time_data.units = "hours since 1900-01-01 00:00:00.0"
    ensemble_mean = rootgrp.createVariable(f"{key}_mean", "f4", ("time", "valid_time", "y", "x"), zlib=True)
    ensemble_mean.units = metadata["units"]
    ensemble_mean.long_name = f'{metadata["long_name"]} ensemble mean'
    ensemble_std = rootgrp.createVariable(f"{key}_sd", "f4", ("time", "valid_time", "y", "x"), zlib=True)
    ensemble_std.units = metadata["units"]
    ensemble_std.long_name = f'{metadata["long_name"]} ensemble standard deviation'
    return {
        "root": rootgrp,
        "time": time_data,
        "valid_time": valid_time_data,
        "mean": ensemble_mean,
        "std": ensemble_std,
    }
# ============================================================
# GRIB filenames
# ============================================================
def get_grib_directory(key, grib_key, level):
    """
    Return directory containing GRIB files.
    """
    if level == "pl":
        return INPUT_DIR / grib_key / "grib"
    return INPUT_DIR / key / "grib"
def get_grib_path(
    init_date,
    key,
    grib_key,
    level,
):
    """
    Construct GRIB filename.
    """
    directory = get_grib_directory(
        key,
        grib_key,
        level,
    )
    if level == "pl":
        filename = f"{init_date:%Y-%m-%d}_" f"{grib_key}_700_hPa_14d.grib"
    else:
        filename = f"{init_date:%Y-%m-%d}_" f"{key}_14d.grib"
    return directory / filename
# ============================================================
# Main variable worker
# ============================================================
def process_variable(
    key,
    metadata,
    level,
    target_x,
    target_y,
    target_lon,
    target_lat,
    central_lon,
    central_lat,
):
    """
    Process one complete variable.
    This function is intended to run in its own process.
    Important difference from the original code:
        variable
            -> init date
                -> open GRIB ONCE
                    -> process every hindcast year
    rather than:
        year
            -> variable
                -> init date
                    -> open GRIB
    """
    grib_key = metadata["grib_key"]
    print(
        f"[{key}] Starting variable " f"(PID {os.getpid()})",
        flush=True,
    )
    init_dates = get_initialisation_dates()
    # --------------------------------------------------------
    # Determine which years actually need processing
    # --------------------------------------------------------
    years_to_process = []
    for year in YEARS:
        output_file = PARENT_OUTPUT_DIR / str(year) / f"{key}.nc"
        if output_file_is_valid(output_file, key):
            print(
                f"[{key}] {year} already complete -- skipping",
                flush=True,
            )
        else:
            if output_file.exists():
                print(
                    f"[{key}] {year} output invalid -- " "reprocessing",
                    flush=True,
                )
            years_to_process.append(year)
    if not years_to_process:
        print(
            f"[{key}] Everything already complete",
            flush=True,
        )
        return key
    # --------------------------------------------------------
    # Find first valid GRIB.
    #
    # A valid file must contain the expected 40 forecast steps.
    # Corrupt/incomplete files are skipped.
    # --------------------------------------------------------
    num_valid_times = EXPECTED_NUM_VALID_TIMES
    first_file = None
    for init_date in init_dates:
        candidate = get_grib_path(
            init_date,
            key,
            grib_key,
            level,
        )
        if not candidate.exists():
            continue
        try:
            with xr.open_dataset(
                candidate,
                engine="cfgrib",
                decode_timedelta=True,
            ) as ds:
                n_steps = ds.sizes.get("step")
                if n_steps != num_valid_times:
                    print(
                        f"[{key}] Skipping unsuitable reference GRIB "
                        f"{candidate.name}: expected "
                        f"{num_valid_times} steps but found {n_steps}",
                        flush=True,
                    )
                    continue
                source_lon = ds["longitude"].values.copy()
                source_lat = ds["latitude"].values.copy()
            first_file = candidate
            break
        except Exception as exc:
            print(
                f"[{key}] Skipping unreadable reference GRIB " f"{candidate.name}: {exc}",
                flush=True,
            )
    if first_file is None:
        raise FileNotFoundError(f"[{key}] No valid GRIB with " f"{num_valid_times} forecast steps found")
    print(
        f"[{key}] Using grid from {first_file.name} " f"({num_valid_times} forecast steps)",
        flush=True,
    )
    # --------------------------------------------------------
    # Build interpolation mapping ONCE for this variable.
    # --------------------------------------------------------
    (
        vertices,
        weights,
        inside,
        target_shape,
    ) = build_interpolation_mapping(
        source_lon,
        source_lat,
        target_lon,
        target_lat,
    )
    # Source-grid points actually used by the target grid
    needed_points = np.unique(vertices[inside])
    print(
        f"[{key}] Need {len(needed_points):,} of "
        f"{len(source_lon):,} source points "
        f"({100 * len(needed_points) / len(source_lon):.2f}%)",
        flush=True,
    )
    # Convert original source-grid indices into indices in the
    # compact needed_points array.
    compact_vertices = np.zeros_like(vertices)
    compact_vertices[inside] = np.searchsorted(
        needed_points,
        vertices[inside],
    )
    # From this point onwards interpolation uses compact indices.
    vertices = compact_vertices
    print(
        f"[{key}] Interpolation mapping ready; "
        f"{inside.sum():,}/{len(inside):,} target points "
        "inside triangulation",
        flush=True,
    )
    # --------------------------------------------------------
    # Open all required yearly output files.
    #
    # There are only 12, so this is reasonable and lets us
    # process every year while each GRIB is open.
    # --------------------------------------------------------
    outputs = {}
    had_errors = False
    try:
        for year in years_to_process:
            output_file = PARENT_OUTPUT_DIR / str(year) / f"{key}.nc"
            # Remove invalid old file before recreating it.
            if output_file.exists():
                output_file.unlink()
            outputs[year] = create_output_file(
                output_file,
                key,
                metadata,
                target_x,
                target_y,
                target_lon,
                target_lat,
                central_lon,
                central_lat,
                num_valid_times,
            )
            # Independent time index for each year
            outputs[year]["time_idx"] = 0
        # ----------------------------------------------------
        # Process initialisation dates.
        #
        # Each GRIB is opened ONCE.
        # ----------------------------------------------------
        for init_number, init_date in enumerate(
            init_dates,
            start=1,
        ):
            file_path = get_grib_path(
                init_date,
                key,
                grib_key,
                level,
            )
            if not file_path.exists():
                had_errors = True
                print(
                    f"[{key}] Missing: {file_path}",
                    flush=True,
                )
                continue
            print(
                f"[{key}] " f"{init_number}/{len(init_dates)} " f"{file_path.name}",
                flush=True,
            )
            try:
                with xr.open_dataset(
                    file_path,
                    engine="cfgrib",
                    decode_timedelta=True,
                ) as ds:
                    n_steps = ds.sizes.get("step")
                    if n_steps != num_valid_times:
                        raise ValueError(
                            f"{file_path}: corrupt/incomplete GRIB -- "
                            f"expected {num_valid_times} forecast steps "
                            f"but found {n_steps}"
                        )
                    # Pull year coordinate once.
                    hindcast_years = ds["time"].dt.year.values
                    # ------------------------------------------------
                    # Process every required hindcast year contained
                    # in this GRIB before closing it.
                    # ------------------------------------------------
                    for year in years_to_process:
                        mask = hindcast_years == year
                        if not np.any(mask):
                            print(
                                f"[{key}] " f"No {year} hindcast in " f"{file_path.name}",
                                flush=True,
                            )
                            continue
                        # Usually this should select the hindcast
                        # ensemble corresponding to one year.
                        year_ds = ds.isel(time=mask).squeeze("time")
                        # Check lead-time count is consistent.
                        this_num_valid_times = year_ds.sizes["step"]
                        if this_num_valid_times != num_valid_times:
                            raise ValueError(
                                f"{file_path}: expected "
                                f"{num_valid_times} valid times "
                                f"but found "
                                f"{this_num_valid_times}"
                            )
                        # --------------------------------------------
                        # Read data into memory ONCE.
                        # --------------------------------------------
                        field = year_ds[grib_key]
                        # Identify the source-point dimension.
                        # For your current data this should be the final dimension.
                        point_dim = field.dims[-1]
                        data = field.isel({point_dim: needed_points}).values
                        print(
                            f"[{key}] {year} data shape after subsetting: " f"{data.shape}",
                            flush=True,
                        )
                        # --------------------------------------------
                        # Ensemble statistics for ALL lead times.
                        # --------------------------------------------
                        means, stds = calculate_statistics(
                            data,
                            metadata["accumulated"],
                        )
                        # --------------------------------------------
                        # Interpolate ALL lead times simultaneously.
                        # --------------------------------------------
                        data_interp_mean = interpolate_fields(
                            means,
                            vertices,
                            weights,
                            inside,
                            target_shape,
                        )
                        data_interp_std = interpolate_fields(
                            stds,
                            vertices,
                            weights,
                            inside,
                            target_shape,
                        )
                        # --------------------------------------------
                        # Time coordinates
                        # --------------------------------------------
                        epoch = np.datetime64("1900-01-01T00:00:00")
                        dt_hours = (year_ds["time"].values - epoch) / np.timedelta64(1, "h")
                        valid_times = (year_ds["valid_time"].values - epoch) / np.timedelta64(1, "h")
                        # --------------------------------------------
                        # Write to this year's NetCDF
                        # --------------------------------------------
                        output = outputs[year]
                        time_idx = output["time_idx"]
                        output["time"][time_idx] = dt_hours
                        output["valid_time"][time_idx, :] = valid_times
                        output["mean"][time_idx, :, :, :] = data_interp_mean
                        output["std"][time_idx, :, :, :] = data_interp_std
                        output["time_idx"] += 1
                        print(
                            f"[{key}]   finished {year}",
                            flush=True,
                        )
            except Exception as exc:
                had_errors = True
                print(
                    f"[{key}] ERROR processing " f"{file_path}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
        if not had_errors:
            print(
                f"[{key}] Finished all GRIBs successfully",
                flush=True,
            )
            for output in outputs.values():
                output["root"].processing_complete = 1
                output["root"].sync()
        else:
            print(
                f"[{key}] Finished with errors -- " "outputs NOT marked complete",
                flush=True,
            )
    finally:
        # Always close NetCDF files, including after exceptions.
        for output in outputs.values():
            try:
                output["root"].close()
            except Exception:
                pass
    return key
# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variable",
        required=True,
        help="Variable to process, e.g. t2m, tp, u700",
    )
    args = parser.parse_args()
    PARENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    target_x, target_y, target_lon, target_lat, central_lon, central_lat = load_target_grid()
    key = args.variable
    if key in VARS_SFC:
        metadata = VARS_SFC[key]
        level = "sfc"
    elif key in VARS_PL700:
        metadata = VARS_PL700[key]
        level = "pl"
    else:
        raise ValueError(f"Unknown variable: {key}")
    process_variable(
        key,
        metadata,
        level,
        target_x,
        target_y,
        target_lon,
        target_lat,
        central_lon,
        central_lat,
    )
if __name__ == "__main__":
    main()