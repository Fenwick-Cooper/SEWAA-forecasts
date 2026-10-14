# Load IFS forecast GRIB files on reduced Gaussian grid,
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

import numpy as np
import netCDF4 as nc
import h5py
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import Delaunay
from os.path import exists
import os
from pathlib import Path
import traceback
import argparse
import hashlib

# Archive transition dates

HISTORICAL_END = datetime(2024, 6, 2)
OPERATIONAL_START = datetime(2024, 9, 23)

OPERATIONAL_DIR = Path(
    "/network/group/aopp/predict/AWH024_COOPERNATH_IFS/"
    "IFS/ICPAC_data/operational/7d/grib"
)

# Operational files contain 30 lead times: 0, 6, ..., 174 h.
OPERATIONAL_EXPECTED_VALID_TIMES = 30

year_folders_pl700 = {
    '2018': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2019': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2020': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2021': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2023': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens_2023', #For pl700
    '2024': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens_2023/2023_shruti',
}

year_folders_sfc = {
    '2018': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2019': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2020': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2021': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens-EUAF',
    '2023': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens_2023/2023_shruti', #For sfc
    '2024': '/network/group/aopp/predict/AWH024_COOPERNATH_IFS/IFS/IFSens_2023/2023_shruti',
}

main_output_dir = "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/rwanda_radar_data/IFS/2026_radar_sample_grid_cropped_to_old_extent"

os.makedirs(main_output_dir, exist_ok=True)
coarsening_factor = 9 #How much coarser the IFS grid is compared to the radar grid. IFS is \\~9km; radar is \\~1km

#Get grid from example radar file -- Rwanda
# import pyart
# base_dir = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/rwanda_radar_data/single_precip_sample/20211019")
# filestr = base_dir / "000035.mdv"
# radar = pyart.io.read_grid_mdv(
#     filestr,
#     file_field_names=True
# )

# radar_x = radar.x["data"]
# radar_y = radar.y["data"]
# radar_lon, radar_lat = radar.get_point_longitude_latitude(level=0)

#Get grid from example radar file -- Rwanda new sample
filestr = "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/rwanda_radar_data/example_grids/2026_radar_sample_grid_cropped_to_old_extent.nc"
radar = xr.load_dataset(filestr)
radar_x = radar["x"].values
radar_y = radar["y"].values
radar_centre_lat = radar["central_lat"].values
radar_centre_lon = radar["central_lon"].values

if "lat" in radar.data_vars:
    radar_lat = radar["lat"].values
    radar_lon = radar["lon"].values

else:
    #Calculate radar lat/lon from radar x/y
    # Local azimuthal equidistant projection centred on radar
    radar_crs = CRS.from_proj4(
        f"+proj=aeqd "
        f"+lat_0={radar_centre_lat} "
        f"+lon_0={radar_centre_lon} "
        f"+datum=WGS84 +units=m"
    )

    to_lonlat = Transformer.from_crs(
        radar_crs,
        "EPSG:4326",
        always_xy=True
    )

    # x and y are 1-D arrays in metres
    radar_xx, radar_yy = np.meshgrid(radar_x, radar_y)
    radar_lon, radar_lat = to_lonlat.transform(
        radar_xx,
        radar_yy
    )

factor = coarsening_factor
crop_x = (len(radar_x) % factor)
crop_y = (len(radar_y) % factor)

print("Cropping x by", crop_x)
print("Cropping y by", crop_y)

left_x = crop_x // 2
right_x = crop_x - left_x

if crop_x > 0:
    radar_x = radar_x[left_x:len(radar_x)-right_x]
    radar_lon = radar_lon[:, left_x:radar_lon.shape[1]-right_x]
    radar_lat = radar_lat[:, left_x:radar_lat.shape[1]-right_x]

left_y = crop_y // 2
right_y = crop_y - left_y

if crop_y > 0:
    radar_y = radar_y[left_y:len(radar_y)-right_y]
    radar_lon = radar_lon[left_y:radar_lon.shape[0]-right_y, :]
    radar_lat = radar_lat[left_y:radar_lat.shape[0]-right_y, :]

centre = factor // 2

radar_x = radar_x[centre::factor]
radar_y = radar_y[centre::factor]

radar_lon = radar_lon[centre::factor, centre::factor]
radar_lat = radar_lat[centre::factor, centre::factor]

assert radar_lon.shape == radar_lat.shape, "radar_lon and radar_lat must have the same shape"
assert radar_lon.shape[0] == len(radar_y) and radar_lon.shape[1] == len(radar_x), "radar_lon shape must match radar_y and radar_x lengths"
assert radar_lat.shape[0] == len(radar_y) and radar_lat.shape[1] == len(radar_x), "radar_lat shape must match radar_y and radar_x lengths"

#sfc variables
the_keys_sfc = ['tclw','tciw','tcrw','sp','tcw','tcwv','lsp','cp','u10','v10','t2m','ssr','mcc','tp']

grib_keys_sfc = the_keys_sfc

the_vars_sfc = {

    "cape":{
        "long_name":"Convective available potential energy",
        "units":"J kg**-1",
        "accumulated":False,
    },
    "tclw":{
        "long_name":"Total column cloud liquid water",
        "units":"kg m**-2",
        "accumulated":False,
    },
    "tciw":{
        "long_name":"Total column cloud ice water",
        "units":"kg m**-2",
        "accumulated":False,
    },
    "tcrw":{
        "long_name":"Total column rain water",
        "units":"kg m**-2",
        "accumulated":False,
    },
    "sp":{
        "long_name":"Surface pressure",
        "units":"Pa",
        "accumulated":False,
    },
    "tcw":{
        "long_name":"Total column water",
        "units":"kg m**-2",
        "accumulated":False,
    },
    "tcwv":{
        "long_name":"Total column vertically-integrated water vapour",
        "units":"kg m**-2",
        "accumulated":False,
    },
    "lsp":{
        "long_name":"Large-scale precipitation",
        "units":"m/6h",
        "accumulated":True,
        "accumulation_per_h":1,
    },
    "cp":{
        "long_name":"Convective precipitation",
        "units":"m/6h",
        "accumulated":True,
        "accumulation_per_h":1,
    },
    "u10":{
        "long_name":"10 metre U wind component",
        "units":"m s**-1",
        "accumulated":False,
    },
    "v10":{
        "long_name":"10 metre V wind component",
        "units":"m s**-1",
        "accumulated":False,
    },
    "t2m":{
        "long_name":"2 metre temperature",
        "units":"K",
        "accumulated":False,
    },
    "ssr":{
        "long_name":"Surface net solar radiation",
        "units":"J m**-2 / 6h",
        "accumulated":True,
        "accumulation_per_h":3600,
    },
    "mcc":{
        "long_name":"Medium cloud cover",
        "units":"(0 - 1)",
        "accumulated":False,
    },
    "tp":{
        "long_name":"Total precipitation",
        "units":"m/6h",
        "accumulated":True,
        "accumulation_per_h":1,
    },

}

# 700hPa: The variable names that we are extracting
the_keys_pl700 = ['u700','v700']
grib_keys_pl700 = ['u','v']  # Different from the netCDF keys

the_vars_pl700 = {

    "u700":{
        "long_name":"U component of wind at 700 hPa",
        "units":"m s**-1",
        "accumulated":False,
    },
    "v700":{
        "long_name":"V component of wind at 700 hPa",
        "units":"m s**-1",
        "accumulated":False,
    },
}

# Convert numpy.datetime64 to datetime
def datetime64_to_datetime(datetime64):

    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (datetime64 - unix_epoch) / one_second
    return datetime.fromtimestamp(seconds_since_epoch, tz=timezone.utc).replace(tzinfo=None)

def get_grid_key(source_lon, source_lat):
    """
    Return a compact fingerprint identifying an IFS source grid.
    This lets us reuse an interpolation mapping when subsequent GRIBs
    use exactly the same cropped source grid, while automatically
    rebuilding it if the historical/operational grid changes.
    """
    hasher = hashlib.sha1()

    hasher.update(np.asarray(source_lon).shape.__repr__().encode())
    hasher.update(np.ascontiguousarray(source_lon).view(np.uint8))
    hasher.update(np.ascontiguousarray(source_lat).view(np.uint8))

    return hasher.hexdigest()

def get_historical_year_dir(year, sfc_or_pl):
    """Return the historical GRIB directory for a year, or None if unavailable."""

    folders = year_folders_sfc if sfc_or_pl == "sfc" else year_folders_pl700
    folder = folders.get(str(year))

    if folder is None:
        return None

    grib_dir = Path(folder)
    grib_year_dir = grib_dir / str(year)

    if not grib_year_dir.exists():
        grib_year_dir = grib_dir

    return grib_year_dir

def get_grib_path(today, sfc_or_pl, historical_year_dir):
    """Return the GRIB path for one forecast initialisation date.
    Historical archive: through 2024-06-02, YYYYMMDD_\<type>_7d.grib
    Operational archive: from 2024-09-23, YYYYMMDD_00Z_\<type>_174h.grib
    The interval 2024-06-03 through 2024-09-22 is intentionally unavailable.
    """

    if today <= HISTORICAL_END:
        if historical_year_dir is None:
            return None

        return historical_year_dir / f"{today:%Y%m%d}_{sfc_or_pl}_7d.grib"
    if today >= OPERATIONAL_START:
        return OPERATIONAL_DIR / f"{today:%Y%m%d}_00Z_{sfc_or_pl}_174h.grib"
    
    return None



# Cache interpolation geometry so the same IFS grid is triangulated only once.
_interpolation_cache = {}

def build_interpolation_mapping(source_lon, source_lat, target_lon, target_lat, crop_padding=1.0):
    """Crop source points around target domain and precompute linear interpolation weights."""
    source_lon = np.asarray(source_lon)
    source_lat = np.asarray(source_lat)
    if np.nanmax(source_lon) > 180:
        source_lon = ((source_lon + 180) % 360) - 180

    lon_min = np.nanmin(target_lon) - crop_padding
    lon_max = np.nanmax(target_lon) + crop_padding
    lat_min = np.nanmin(target_lat) - crop_padding
    lat_max = np.nanmax(target_lat) + crop_padding

    crop_mask = (
        (source_lon >= lon_min) & (source_lon <= lon_max) &
        (source_lat >= lat_min) & (source_lat <= lat_max)
    )
    crop_indices = np.flatnonzero(crop_mask)
    cropped_lon = source_lon[crop_mask]
    cropped_lat = source_lat[crop_mask]
    points = np.column_stack((cropped_lon, cropped_lat))
    target_points = np.column_stack((target_lon.ravel(), target_lat.ravel()))

    print(f"Cropping IFS grid to {len(points):,}/{len(source_lon):,} points before triangulation")
    print(f"Crop bounds: lon {lon_min:.2f} to {lon_max:.2f}, lat {lat_min:.2f} to {lat_max:.2f}")

    tri = Delaunay(points)
    simplex = tri.find_simplex(target_points)
    inside = simplex >= 0
    if not np.all(inside):
        raise ValueError(
            f"Cropped IFS triangulation only contains {inside.sum()}/{inside.size} radar points; "
            "increase crop_padding."
        )

    vertices = np.zeros((len(target_points), 3), dtype=np.int32)
    weights = np.full((len(target_points), 3), np.nan, dtype=np.float64)
    valid_simplex = simplex[inside]
    vertices[inside] = tri.simplices[valid_simplex]
    transform = tri.transform[valid_simplex, :2, :]
    delta = target_points[inside] - tri.transform[valid_simplex, 2, :]
    bary_first_two = np.einsum("nij,nj->ni", transform, delta)
    weights[inside, :2] = bary_first_two
    weights[inside, 2] = 1.0 - bary_first_two[:, 0] - bary_first_two[:, 1]

    # Keep only cropped source points that actually contribute to target cells.
    needed_cropped_points = np.unique(vertices[inside])
    needed_source_points = crop_indices[needed_cropped_points]
    compact_vertices = np.zeros_like(vertices)
    compact_vertices[inside] = np.searchsorted(needed_cropped_points, vertices[inside])

    print(
        f"Interpolation mapping ready: {len(needed_source_points):,}/{len(source_lon):,} "
        f"IFS points actually needed; {inside.sum():,}/{inside.size:,} radar points inside"
    )
    return compact_vertices, weights, inside, target_lon.shape, needed_source_points


def interpolate_fields(fields, vertices, weights, inside, target_shape):
    """Interpolate all lead times at once using precomputed barycentric weights."""
    fields = np.asarray(fields)
    result = np.full((fields.shape[0], len(inside)), np.nan, dtype=np.float32)
    v = vertices[inside]
    w = weights[inside]
    result[:, inside] = np.einsum(
        "fti,ti->ft", fields[:, v], w, optimize=True
    ).astype(np.float32)
    return result.reshape(fields.shape[0], target_shape[0], target_shape[1])


def calculate_statistics(data, accumulated):
    """Calculate ensemble mean/std for every lead time in one vectorised operation."""
    data = np.asarray(data)
    if accumulated:
        increments = np.diff(data, axis=1)
        means = np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
        stds = np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
        means[:-1] = np.mean(increments, axis=0, dtype=np.float64)
        stds[:-1] = np.std(increments, axis=0, ddof=1, dtype=np.float64)
    else:
        means = np.mean(data, axis=0, dtype=np.float64).astype(np.float32)
        stds = np.std(data, axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    return means, stds


def process_year(year, the_keys, grib_keys, the_vars, sfc_or_pl='sfc'):

    historical_year_dir = get_historical_year_dir(year, sfc_or_pl)

    # Historical-only years require a configured historical directory.
    if year < OPERATIONAL_START.year and historical_year_dir is None:
        print(f"Year {year} not found in the historical archive for {sfc_or_pl}. Skipping.")
        return

    # Where to save the NetCDF files
    output_data_dir = f"{main_output_dir}/{str(year)}"
    os.makedirs(output_data_dir, exist_ok=True)

    # Create the NetCDF files
    for key_no in range(len(the_keys)):
        file_name = f"{the_keys[key_no]}.nc"
        # Create a new NetCDF file
        rootgrp = nc.Dataset(f"{output_data_dir}/{file_name}", "w", format="NETCDF4")
        the_vars[the_keys[key_no]]["rootgrp"] = rootgrp

        # Describe where this data comes from
        rootgrp.description = "IFS forecast mean and standard deviation, in 6h time steps."
        rootgrp.createDimension("y", len(radar_y))
        rootgrp.createDimension("x", len(radar_x))
        rootgrp.createDimension("time", None)
        rootgrp.createDimension("valid_time", None)
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
        
        # if radar_x / radar_y are 2D arrays:
        x_data[:] = radar_x
        y_data[:] = radar_y
        lon_data[:] = radar_lon
        lat_data[:] = radar_lat
        central_lon_data[:] = radar_centre_lon
        central_lat_data[:] = radar_centre_lat

        # Create the time variable
        the_vars[the_keys[key_no]]["time_data"] = rootgrp.createVariable("time", "f8", ("time"), zlib=False)
        the_vars[the_keys[key_no]]["time_data"].units = "hours since 1900-01-01 00:00:00.0"

        # Create the valid_time variable
        the_vars[the_keys[key_no]]["valid_time_data"] = rootgrp.createVariable(
            "fcst_valid_time", "f8", ("time", "valid_time"), zlib=False
        )

        the_vars[the_keys[key_no]]["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

        # Create the mean variable
        var_name = f"{the_keys[key_no]}_mean"
        the_vars[the_keys[key_no]]["ensemble_mean"] = rootgrp.createVariable(
            f"{var_name}", "f4", ("time", "valid_time", "y", "x"), zlib=True
        )
        the_vars[the_keys[key_no]]["ensemble_mean"].units = the_vars[the_keys[key_no]]["units"]
        the_vars[the_keys[key_no]]["ensemble_mean"].long_name = f'{the_vars[the_keys[key_no]]["long_name"]} ensemble mean'

        # Create the standard deviation variable
        var_name = f"{the_keys[key_no]}_sd"
        the_vars[the_keys[key_no]]["ensemble_std"] = rootgrp.createVariable(
            f"{var_name}", "f4", ("time", "valid_time", "y", "x"), zlib=True
        )
        the_vars[the_keys[key_no]]["ensemble_std"].units = the_vars[the_keys[key_no]]["units"]
        the_vars[the_keys[key_no]]["ensemble_std"].long_name = f'{the_vars[the_keys[key_no]]["long_name"]} ensemble standard deviation'


    # Start date

    today = datetime(year, 1, 1)
    time_idx = 0

    # For the current year, stop after today rather than logging future files as missing.

    tomorrow_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    ) + timedelta(days=1)

    end_date = min(datetime(year + 1, 1, 1), tomorrow_utc)

    while today < end_date:
        print(f"Processing {today}")
        file_path = get_grib_path(today, sfc_or_pl, historical_year_dir)
        
        if file_path is None:
            print(f"No archive configured for {today:%Y-%m-%d}; skipping.")
            today += timedelta(days=1)
            time_idx += 1
            continue

        if file_path.exists():
            try:
                with xr.open_dataset(file_path, engine="cfgrib") as ds:
                    num_ensemble_members = ds["number"].shape[0]
                    num_valid_times = len(ds["valid_time"])

                    if today >= OPERATIONAL_START and num_valid_times != OPERATIONAL_EXPECTED_VALID_TIMES:
                        print(
                            f"WARNING: expected {OPERATIONAL_EXPECTED_VALID_TIMES} valid times "
                            f"for operational file {file_path}, found {num_valid_times}"
                        )

                    global _interpolation_cache

                    lon = ds["longitude"].values
                    lat = ds["latitude"].values

                    # Normalise longitude before identifying the source grid.
                    # build_interpolation_mapping() performs the same conversion internally.
                    lon_for_key = np.asarray(lon)
                    if np.nanmax(lon_for_key) > 180:
                        lon_for_key = ((lon_for_key + 180) % 360) - 180

                    # Identify the actual IFS grid used by this GRIB.
                    grid_key = get_grid_key(lon_for_key, lat)

                    if grid_key not in _interpolation_cache:
                        print(
                            f"New IFS source grid detected "
                            f"({len(lon):,} source points) -- "
                            f"building interpolation mapping",
                            flush=True,
                        )

                        (
                            vertices,
                            weights,
                            inside,
                            target_shape,
                            needed_points,
                        ) = build_interpolation_mapping(
                            lon,
                            lat,
                            radar_lon,
                            radar_lat,
                        )

                        _interpolation_cache[grid_key] = {
                            "vertices": vertices,
                            "weights": weights,
                            "inside": inside,
                            "target_shape": target_shape,
                            "needed_points": needed_points,
                        }

                    else:
                        print(
                            "Reusing interpolation mapping for existing IFS grid",
                            flush=True,
                        )

                    mapping = _interpolation_cache[grid_key]

                    vertices = mapping["vertices"]
                    weights = mapping["weights"]
                    inside = mapping["inside"]
                    target_shape = mapping["target_shape"]
                    needed_points = mapping["needed_points"]

                    dt = datetime64_to_datetime(ds["time"].values) - datetime(1900, 1, 1)
                    dt_hours = (dt.days * 24 * 60 * 60 + dt.seconds) / (60 * 60)
                    valid_times = np.zeros(num_valid_times)
                    for idx in range(num_valid_times):
                        dt = datetime64_to_datetime(ds["valid_time"].values[idx]) - datetime(1900, 1, 1)
                        valid_times[idx] = (dt.days * 24 * 60 * 60 + dt.seconds) / (60 * 60)

                    for key_no in range(len(the_keys)):
                        grib_key = grib_keys[key_no]
                        if grib_key not in ds.data_vars:
                            print(f"Skipping missing variable: {grib_key}")
                            continue

                        the_vars[the_keys[key_no]]["time_data"][time_idx] = dt_hours
                        the_vars[the_keys[key_no]]["valid_time_data"][time_idx, :num_valid_times] = valid_times

                        # Read only source points that can contribute to the radar grid.
                        field = ds[grib_key]
                        point_dim = field.dims[-1]
                        data = field.isel({point_dim: needed_points}).values

                        # Calculate all ensemble statistics and interpolate all lead times at once.
                        means, stds = calculate_statistics(
                            data, the_vars[the_keys[key_no]]["accumulated"]
                        )
                        data_interp_mean = interpolate_fields(
                            means, vertices, weights, inside, target_shape
                        )
                        data_interp_std = interpolate_fields(
                            stds, vertices, weights, inside, target_shape
                        )

                        the_vars[the_keys[key_no]]["ensemble_mean"][time_idx, :num_valid_times, :, :] = data_interp_mean
                        the_vars[the_keys[key_no]]["ensemble_std"][time_idx, :num_valid_times, :, :] = data_interp_std

            except Exception as e:

                print(f"ERROR processing {file_path}: {e}")

                traceback.print_exc()

        else:

            print(f"Missing: {file_path}")

        today += timedelta(days=1)

        time_idx += 1

    # Close the netCDF files

    for key_no in range(len(the_keys)):

        the_vars[the_keys[key_no]]["rootgrp"].close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interpolate IFS ensemble forecasts onto the Rwanda radar grid.")

    parser.add_argument("--year", type=int, required=True, help="Year to process")

    args = parser.parse_args()

    print(f"Starting to process year {args.year}, sfc variables")

    process_year(args.year, the_keys=the_keys_sfc, grib_keys=grib_keys_sfc, the_vars=the_vars_sfc, sfc_or_pl="sfc")

    print(f"Starting to process year {args.year}, 700hPa variables")

    process_year(args.year, the_keys=the_keys_pl700, grib_keys=grib_keys_pl700, the_vars=the_vars_pl700, sfc_or_pl="pl700")
