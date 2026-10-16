#!/usr/bin/env python
# coding: utf-8

# Load and interpolate a grib file on a reduced Gaussian grid

import sys
import os
import numpy as np
import netCDF4 as nc
import h5py
from datetime import datetime, date, timedelta
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

# For debugging only
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Target radar grid
TARGET_GRID_FILE = (
    "../data/example_grids/eg_ifs_grid.nc"
)
GRIB_dir = "/home/c/cooperf/data/cGAN/ICPAC/forecasts/IFS/grib"

#How much coarser the IFS grid is compared to the radar grid. IFS is \\~9km; radar is \\~1km
#Should be the same as the cGAN downscaling factor
coarsening_factor = 1

# Where to save the NetCDF files
output_data_dir = "/home/c/cooperf/data/cGAN/Rwanda_radar/forecasts/IFS/interpolated"
os.makedirs(output_data_dir, exist_ok=True)

# Load radar grid
radar = xr.load_dataset(TARGET_GRID_FILE)

radar_x = radar["x"].values
radar_y = radar["y"].values

radar_lon = radar["lon"].values
radar_lat = radar["lat"].values

radar_centre_lon = radar["central_lon"].values
radar_centre_lat = radar["central_lat"].values

if coarsening_factor != 1:
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

# Date of forecast initialisation for the command line argument
date_str = f"{sys.argv[1]}"
todays_date = date(int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]))

# Time of forecast initialisation for the command line argument
time_str = f"{int(sys.argv[2]):02d}"

# Convert numpy.datetime64 to datetime
def datetime64_to_datetime(datetime64):
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (datetime64 - unix_epoch) / one_second
    return datetime.utcfromtimestamp(seconds_since_epoch)

# Load the data on pressure levels
file_name = f"{GRIB_dir}/{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{time_str}Z_pl700_174h.grib"
ds_pl = xr.load_dataset(file_name, engine="cfgrib")

# Load the data on single levels
file_name = f"{GRIB_dir}/{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{time_str}Z_sfc_174h.grib"
ds_sfc = xr.load_dataset(file_name, engine="cfgrib",backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})

# Load the mucape data
file_name = f"{GRIB_dir}/{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{time_str}Z_MUCAPE_174h.grib"
ds_mucape = xr.load_dataset(file_name, engine="cfgrib")

# The number of ensemble members in each forecast
num_ensemble_members = ds_sfc["number"].shape[0]

# Crop IFS points to within 1 degree of radar grid
padding = 1.0

lon_min = np.nanmin(radar_lon) - padding
lon_max = np.nanmax(radar_lon) + padding
lat_min = np.nanmin(radar_lat) - padding
lat_max = np.nanmax(radar_lat) + padding

# Surface grid
mask_sfc = (
    (ds_sfc["longitude"].values >= lon_min) &
    (ds_sfc["longitude"].values <= lon_max) &
    (ds_sfc["latitude"].values >= lat_min) &
    (ds_sfc["latitude"].values <= lat_max)
)

# Pressure-level grid
mask_pl = (
    (ds_pl["longitude"].values >= lon_min) &
    (ds_pl["longitude"].values <= lon_max) &
    (ds_pl["latitude"].values >= lat_min) &
    (ds_pl["latitude"].values <= lat_max)
)

# MUCAPE grid
mask_mucape = (
    (ds_mucape["longitude"].values >= lon_min) &
    (ds_mucape["longitude"].values <= lon_max) &
    (ds_mucape["latitude"].values >= lat_min) &
    (ds_mucape["latitude"].values <= lat_max)
)

# Construct cropped source points
points_sfc = np.column_stack((
    ds_sfc["longitude"].values[mask_sfc],
    ds_sfc["latitude"].values[mask_sfc]
))

points_pl = np.column_stack((
    ds_pl["longitude"].values[mask_pl],
    ds_pl["latitude"].values[mask_pl]
))

points_mucape = np.column_stack((
    ds_mucape["longitude"].values[mask_mucape],
    ds_mucape["latitude"].values[mask_mucape]
))

# Triangulate cropped grids
tri_sfc = Delaunay(points_sfc)
tri_pl = Delaunay(points_pl)
tri_mucape = Delaunay(points_mucape)

print("IFS source points after cropping:")
print("Surface:", len(points_sfc))
print("Pressure:", len(points_pl))
print("MUCAPE:", len(points_mucape))

print("Radar grid shape:", radar_lon.shape)

for name, tri in [
    ("Surface", tri_sfc),
    ("Pressure", tri_pl),
    ("MUCAPE", tri_mucape),
]:
    outside = tri.find_simplex(
        np.column_stack((radar_lon.ravel(), radar_lat.ravel()))
    ) < 0

    print(f"{name}: {outside.sum()} radar points outside triangulation")

# Generate the points we are interpolating to
lon_dest = radar_lon
lat_dest = radar_lat

# Check that the data is in the correct place
if False:

    # Create the interpolation function using cropped IFS points
    interp = LinearNDInterpolator(
        tri_sfc,
        np.mean(ds_sfc["sp"].values[:, 0, :][:, mask_sfc], axis=0)
    )

    # Interpolate onto radar grid
    sp_interp = interp(lon_dest, lat_dest)

    # Check interpolation coverage
    print("Interpolated shape:", sp_interp.shape)
    print("Fraction of NaNs:", np.mean(np.isnan(sp_interp)))

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)

    c = ax.pcolormesh(
        radar_lon, radar_lat, sp_interp,
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    cb = plt.colorbar(c, fraction=0.02)
    cb.ax.tick_params(labelsize=18)
    cb.set_label("Pa", size=18)

    plt.title("Surface pressure interpolated onto radar grid")
    plt.show()



# Time that the forecast starts
start_time = datetime64_to_datetime(ds_sfc["time"].values)

# Convert numpy datetime64 to timedelta since 1900
dt = start_time - datetime(1900,1,1)

# Convert timedelta to hours (ignoring microseconds)
dt_hours = (dt.days * 24*60*60 + dt.seconds) / (60*60)

# Create the list of valid times
num_valid_times = len(ds_sfc["valid_time"])
valid_hours = np.zeros(num_valid_times)
for i in range(num_valid_times):
    
    # Convert numpy datetime64 to timedelta since 1900
    valid_time = datetime64_to_datetime(ds_sfc['time'].values + ds_sfc['step'][i].values)
    valid_dt = valid_time - datetime(1900,1,1)
    
    # Convert timedelta to hours (ignoring microseconds)
    valid_hours[i] = (valid_dt.days * 24*60*60 + valid_dt.seconds) / (60*60)



# the_keys = ['mucape','tclw','tciw','tcrw','sp','tcw','tcwv','lsp','cp','u10','v10','t2m','ssr','mcc','tp','u','v']
# A subset of the keys are required for cGAN
the_keys = ['mucape','tclw','tciw','sp','tcw','tcwv','cp','t2m','ssr','tcc','tp','u','v']
the_vars = {
    "mucape":{
        "long_name":"Most unstable convective available potential energy",
        "units":"J kg**-1",
        "accumulated":False,
        "single_level":True,
    },
    "tclw":{
        "long_name":"Total column cloud liquid water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "tciw":{
        "long_name":"Total column cloud ice water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "tcrw":{
        "long_name":"Total column rain water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "sp":{
        "long_name":"Surface pressure",
        "units":"Pa",
        "accumulated":False,
        "single_level":True,
    },
    "tcw":{
        "long_name":"Total column water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "tcwv":{
        "long_name":"Total column vertically-integrated water vapour",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "lsp":{
        "long_name":"Large-scale precipitation",
        "units":"m hour**-1",
        "accumulated":True,
        "accumulation_per_h":1,
        "single_level":True,
    },
    "cp":{
        "long_name":"Convective precipitation",
        "units":"m (6 hour)**-1",
        "accumulated":True,
        "accumulation_per_h":1,
        "single_level":True,
    },
    "u10":{
        "long_name":"10 metre U wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":True,
    },
    "v10":{
        "long_name":"10 metre V wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":True,
    },
    "t2m":{
        "long_name":"2 metre temperature",
        "units":"K",
        "accumulated":False,
        "single_level":True,
    },
    "ssr":{
        "long_name":"Surface net solar radiation",
        "units":"J m**-2 (6 hour)**-1",
        "accumulated":True,
        "accumulation_per_h":3600,
        "single_level":True,
    },
    "mcc":{
        "long_name":"Medium cloud cover",
        "units":"(0 - 1)",
        "accumulated":False,
        "single_level":True,
    },
    "tcc":{
        "long_name":"Total cloud cover",
        "units":"(0 - 1)",
        "accumulated":False,
        "single_level":True,
    },
    "tp":{
        "long_name":"Total precipitation",
        "units":"m (6 hour)**-1",
        "accumulated":True,
        "accumulation_per_h":1,
        "single_level":True,
    },
    "u":{
        "long_name":"700 hPa U wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":False,
    },
    "v":{
        "long_name":"700 hPa V wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":False,
    },
}



# Create the NetCDF files
file_name = f"IFS_{date_str}_{time_str}Z.nc"

print(f"Saving data to {output_data_dir}/{file_name}")

# Create a new NetCDF file
rootgrp = nc.Dataset(f"{output_data_dir}/{file_name}", "w", format="NETCDF4")

# Describe where this data comes from
rootgrp.description = "IFS forecast ensemble mean and standard deviation interpolated onto the Rwanda radar grid."

# Create dimensions
x_dim = rootgrp.createDimension("x", len(radar_x))
y_dim = rootgrp.createDimension("y", len(radar_y))
time_dim = rootgrp.createDimension("time", None)
valid_time_dim = rootgrp.createDimension("valid_time", num_valid_times)

# Create radar x and y coordinates
x_data = rootgrp.createVariable("x", "f4", ("x",), zlib=False)
y_data = rootgrp.createVariable("y", "f4", ("y",), zlib=False)

x_data.units = "m"
y_data.units = "m"

x_data[:] = radar_x
y_data[:] = radar_y

# Create 2D latitude and longitude coordinates
longitude_data = rootgrp.createVariable(
    "longitude", "f4", ("y", "x"), zlib=False
)
latitude_data = rootgrp.createVariable(
    "latitude", "f4", ("y", "x"), zlib=False
)

longitude_data.units = "degrees_east"
latitude_data.units = "degrees_north"

longitude_data[:] = radar_lon
latitude_data[:] = radar_lat

# Store radar centre
central_lon_data = rootgrp.createVariable("central_lon", "f4", ())
central_lat_data = rootgrp.createVariable("central_lat", "f4", ())

central_lon_data[:] = radar_centre_lon
central_lat_data[:] = radar_centre_lat

# Create the time variable
time_data = rootgrp.createVariable("time", "f4", ("time"), zlib=False)
time_data.units = "hours since 1900-01-01 00:00:00.0"
time_data.description = "Time corresponding to forecast model start"
time_data[:] = dt_hours         # Write the forecast model start time

# Create the valid_time variable
valid_time_data = rootgrp.createVariable("valid_time", "f4", ("valid_time"), zlib=False)
valid_time_data.units = "hours since 1900-01-01 00:00:00.0"
valid_time_data.description = "Time corresponding to forecast prediction"
valid_time_data[:] = valid_hours  # Write the forecast model valid times

for key_no in range(len(the_keys)):
    print(f"Interpolating {the_keys[key_no]}")
    
    # Which .grib file should we be reading
    if (the_keys[key_no] == "mucape"):
        ds = ds_mucape
        tri = tri_mucape
        mask = mask_mucape
    elif the_vars[the_keys[key_no]]['single_level']:
        ds = ds_sfc
        tri = tri_sfc
        mask = mask_sfc
    else:
        ds = ds_pl
        tri = tri_pl
        mask = mask_pl
    
    data_mean_interp = np.zeros((num_valid_times, len(radar_y), len(radar_x)))
    data_std_interp = np.zeros((num_valid_times, len(radar_y), len(radar_x)))
    for valid_time_idx in range(num_valid_times):

        if (the_vars[the_keys[key_no]]["accumulated"]):
            
            # If this is the final accumulated value
            if (valid_time_idx == num_valid_times-1):
                values_mean = np.zeros(np.count_nonzero(mask))
                values_std = np.zeros(np.count_nonzero(mask))
            else:
                values = ds[the_keys[key_no]].values[:,valid_time_idx:valid_time_idx+2,mask]
                values_mean = np.mean(values[:,1,:] - values[:,0,:], axis=0)
                # XXX Should be np.var but is np.std for consistency with training data
                values_std = np.std(values[:,1,:] - values[:,0,:], axis=0, ddof=1)
        else:
            values = ds[the_keys[key_no]].values[:,valid_time_idx,mask]
            values_mean = np.mean(values, axis=0)
            # XXX Should be np.var but is np.std for consistency with training data
            values_std = np.std(values, axis=0, ddof=1)
        
        # Create the interpolation functions
        interp_mean = LinearNDInterpolator(tri, values_mean)
        interp_std = LinearNDInterpolator(tri, values_std)
        
        # Interpolate
        data_mean_interp[valid_time_idx,:,:] = interp_mean(lon_dest, lat_dest)
        data_std_interp[valid_time_idx,:,:] = interp_std(lon_dest, lat_dest)
    
    # Label 700 hPa fields as being at 700 hPa
    if the_keys[key_no] in ['u','v']:
        key_name = the_keys[key_no] + '700'  # On the 700 hPa level
    else:
        key_name = the_keys[key_no]
    
    # Create the mean variable
    var_name = f"{key_name}_ensemble_mean"
    ensemble_mean_data = rootgrp.createVariable(
        var_name, "f4", ("valid_time", "y", "x"), zlib=True
    )   
    ensemble_mean_data.units = the_vars[the_keys[key_no]]["units"]
    ensemble_mean_data.long_name = f'{the_vars[the_keys[key_no]]["long_name"]} ensemble mean'
    ensemble_mean_data[:] = data_mean_interp

    # Create the variance variable
    var_name = f"{key_name}_ensemble_standard_deviation"
    ensemble_std_data = rootgrp.createVariable(
        var_name, "f4", ("valid_time", "y", "x"), zlib=True
    )    
    ensemble_std_data.units = the_vars[the_keys[key_no]]["units"]
    ensemble_std_data.long_name = f'{the_vars[the_keys[key_no]]["long_name"]} ensemble standard deviation'
    ensemble_std_data[:] = data_std_interp
    
# Close the netCDF file
rootgrp.close()

# Test that the fields look reasonable
if False:

    # Load the new axes, times and a test surface pressure
    file_name = f"IFS_{date_str}_{time_str}Z.nc"
    nc_file = nc.Dataset(f"{output_data_dir}/{file_name}")

    time_test = np.array(nc_file["time"][:])
    valid_time_test = np.array(nc_file["valid_time"][:])

    x_test = np.array(nc_file["x"][:])
    y_test = np.array(nc_file["y"][:])

    latitude_test = np.array(nc_file["latitude"][:])
    longitude_test = np.array(nc_file["longitude"][:])

    sp_test = np.array(nc_file["sp_ensemble_mean"][:])

    nc_file.close()

    # Compare the times and radar coordinates
    print("Start time matches:",
          datetime(1900, 1, 1) + timedelta(hours=float(time_test[0])) == start_time)

    print("Maximum valid time difference:",
          np.max(np.abs(valid_time_test - valid_hours)))

    print("Maximum x difference:",
          np.max(np.abs(radar_x - x_test)))

    print("Maximum y difference:",
          np.max(np.abs(radar_y - y_test)))

    print("Maximum latitude difference:",
          np.max(np.abs(radar_lat - latitude_test)))

    print("Maximum longitude difference:",
          np.max(np.abs(radar_lon - longitude_test)))

    # Compare saved fields against independently interpolated fields
    for i in range(num_valid_times):

        # Create interpolation function using cropped IFS points
        interp = LinearNDInterpolator(
            tri_sfc,
            np.mean(ds_sfc["sp"].values[:, i, :][:, mask_sfc], axis=0)
        )

        # Interpolate onto radar grid
        sp_interp = interp(lon_dest, lat_dest)

        # Compare with saved NetCDF values
        valid = np.isfinite(sp_interp) & np.isfinite(sp_test[i, :, :])

        if np.any(valid):
            err = np.max(
                np.abs(
                    (sp_interp[valid] - sp_test[i, :, :][valid])
                    / sp_interp[valid]
                )
            )
            print(f"err[{i}] = {err}")
        else:
            print(f"err[{i}] = No valid points")





